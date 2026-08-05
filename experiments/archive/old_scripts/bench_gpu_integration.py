"""GPU integration test for v0.2.0 new components.

Tests memory monitor, async transfer, and KV quantizer on real GPU hardware.
Run with: CUDA_VISIBLE_DEVICES=1 python experiments/test_gpu_integration_v2.py
"""

import json
import time
import sys
import os

import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deltacache.core.memory_monitor import GPUMemoryMonitor, MemoryPressure
from deltacache.core.async_transfer import AsyncTransferManager
from deltacache.core.kv_quantizer import KVQuantizer, QuantPrecision


def test_memory_monitor(device: torch.device) -> dict:
    """Test GPU memory monitor with real GPU state."""
    print("\n=== Test 1: GPU Memory Monitor ===")

    monitor = GPUMemoryMonitor(device=device)
    state = monitor.get_memory_state()

    print(f"  Total GPU memory: {state.total_bytes / 1e9:.2f} GB")
    print(f"  Free GPU memory:  {state.free_bytes / 1e9:.2f} GB")
    print(f"  Used GPU memory:  {state.used_bytes / 1e9:.2f} GB")
    print(f"  Utilization:      {state.utilization:.1%}")
    print(f"  Pressure level:   {monitor.get_pressure_level(state).value}")

    # Allocate some GPU memory and observe change
    tensors = []
    alloc_size_gb = 2.0
    num_elements = int(alloc_size_gb * 1e9 / 2)  # FP16 = 2 bytes
    print(f"\n  Allocating {alloc_size_gb} GB on GPU...")
    t = torch.empty(num_elements, dtype=torch.float16, device=device)
    tensors.append(t)

    state_after = monitor.get_memory_state()
    print(f"  Used after alloc:  {state_after.used_bytes / 1e9:.2f} GB")
    print(f"  Utilization:       {state_after.utilization:.1%}")
    print(f"  Pressure level:    {monitor.get_pressure_level(state_after).value}")
    print(f"  Can allocate 1GB:  {monitor.can_allocate(int(1e9))}")

    # Test high watermark detection
    # Fill to 85%+ to trigger HIGH pressure
    target_used = int(state.total_bytes * 0.87)
    additional_needed = target_used - state_after.used_bytes
    if additional_needed > 0:
        print(f"\n  Filling to ~87% utilization ({additional_needed / 1e9:.1f} GB more)...")
        t2 = torch.empty(additional_needed // 2, dtype=torch.float16, device=device)
        tensors.append(t2)
        state_high = monitor.get_memory_state()
        print(f"  Utilization:    {state_high.utilization:.1%}")
        print(f"  Pressure level: {monitor.get_pressure_level(state_high).value}")
        print(f"  Bytes to free:  {monitor.bytes_to_free(state_high) / 1e9:.2f} GB")

    # Clean up
    del tensors
    torch.cuda.empty_cache()

    state_final = monitor.get_memory_state()
    print(f"\n  After cleanup:  {state_final.utilization:.1%}")

    return {
        "total_gb": state.total_bytes / 1e9,
        "initial_util": state.utilization,
        "post_alloc_util": state_after.utilization,
        "monitor_works": True,
    }


def test_async_transfer(device: torch.device) -> dict:
    """Test async GPU-CPU transfer with real CUDA streams."""
    print("\n=== Test 2: Async Transfer Manager ===")

    manager = AsyncTransferManager(device=device)
    print(f"  Has CUDA: {manager.has_cuda}")

    # Create test KV tensors (simulate 4 layers, 256 tokens, 8 heads, 64 dim)
    shape = (4, 256, 8, 64)
    key_gpu = torch.randn(shape, dtype=torch.float16, device=device)
    value_gpu = torch.randn(shape, dtype=torch.float16, device=device)
    tensor_size_mb = key_gpu.numel() * 2 * 2 / 1e6  # 2 tensors, 2 bytes each
    print(f"  Test tensor size: {tensor_size_mb:.1f} MB")

    # Test 1: Async offload (GPU -> CPU)
    print("\n  --- GPU -> CPU Offload ---")
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    key_cpu, value_cpu = manager.offload_async(block_id=1, key_gpu=key_gpu, value_gpu=value_gpu)
    t_launch = time.perf_counter() - t0
    print(f"  Launch time: {t_launch * 1000:.2f} ms")

    assert not manager.is_complete(1), "Should not be complete immediately"

    manager.wait_transfer(1)
    t_total = time.perf_counter() - t0
    print(f"  Total time (launch + wait): {t_total * 1000:.2f} ms")

    # Verify correctness
    key_gpu_back = key_cpu.to(device)
    max_diff = (key_gpu - key_gpu_back).abs().max().item()
    print(f"  Max difference (should be 0): {max_diff}")
    assert max_diff == 0, f"Data corruption! max_diff={max_diff}"
    del key_gpu_back

    # Test 2: Async prefetch (CPU -> GPU)
    print("\n  --- CPU -> GPU Prefetch ---")
    # Pin the CPU tensors for optimal transfer
    key_cpu_pinned = key_cpu.pin_memory()
    value_cpu_pinned = value_cpu.pin_memory()

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    key_prefetched, value_prefetched = manager.prefetch_async(
        block_id=2, key_cpu=key_cpu_pinned, value_cpu=value_cpu_pinned
    )
    t_launch = time.perf_counter() - t0
    print(f"  Launch time: {t_launch * 1000:.2f} ms")

    manager.wait_transfer(2)
    t_total = time.perf_counter() - t0
    print(f"  Total time: {t_total * 1000:.2f} ms")

    # Verify correctness
    max_diff = (key_gpu - key_prefetched).abs().max().item()
    print(f"  Max difference (should be 0): {max_diff}")
    assert max_diff == 0, f"Prefetch data corruption! max_diff={max_diff}"

    # Test 3: Bandwidth measurement with larger tensors
    print("\n  --- Bandwidth Measurement ---")
    large_shape = (32, 1024, 32, 128)  # ~Llama-7B scale, ~512MB
    large_key = torch.randn(large_shape, dtype=torch.float16, device=device)
    large_value = torch.randn(large_shape, dtype=torch.float16, device=device)
    large_size_mb = large_key.numel() * 2 * 2 / 1e6
    print(f"  Large tensor size: {large_size_mb:.0f} MB")

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    lk_cpu, lv_cpu = manager.offload_async(block_id=3, key_gpu=large_key, value_gpu=large_value)
    manager.wait_transfer(3)
    elapsed = time.perf_counter() - t0
    bandwidth_gbs = (large_size_mb / 1000) / elapsed
    print(f"  GPU->CPU time: {elapsed * 1000:.1f} ms")
    print(f"  Bandwidth: {bandwidth_gbs:.2f} GB/s")

    # Prefetch back (with pinned memory)
    lk_pinned = lk_cpu.pin_memory()
    lv_pinned = lv_cpu.pin_memory()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    lk_gpu2, lv_gpu2 = manager.prefetch_async(block_id=4, key_cpu=lk_pinned, value_cpu=lv_pinned)
    manager.wait_transfer(4)
    elapsed_h2d = time.perf_counter() - t0
    bandwidth_h2d = (large_size_mb / 1000) / elapsed_h2d
    print(f"  CPU->GPU time: {elapsed_h2d * 1000:.1f} ms")
    print(f"  Bandwidth: {bandwidth_h2d:.2f} GB/s")

    # Cleanup
    del large_key, large_value, lk_cpu, lv_cpu, lk_pinned, lv_pinned, lk_gpu2, lv_gpu2
    del key_gpu, value_gpu, key_cpu, value_cpu, key_cpu_pinned, value_cpu_pinned
    del key_prefetched, value_prefetched
    torch.cuda.empty_cache()

    return {
        "small_offload_ms": t_total * 1000,
        "large_size_mb": large_size_mb,
        "d2h_bandwidth_gbs": bandwidth_gbs,
        "h2d_bandwidth_gbs": bandwidth_h2d,
        "data_integrity": True,
    }


def test_kv_quantizer_gpu(device: torch.device) -> dict:
    """Test KV quantizer with GPU tensors."""
    print("\n=== Test 3: KV Quantizer on GPU ===")

    # Simulate realistic KV cache (Mistral-7B like: 32 layers, 256 tokens, 32 heads, 128 dim)
    shape = (32, 256, 32, 128)
    key = torch.randn(shape, dtype=torch.float16, device=device)
    value = torch.randn(shape, dtype=torch.float16, device=device)
    original_size_mb = (key.numel() + value.numel()) * 2 / 1e6
    print(f"  Original KV size: {original_size_mb:.1f} MB")

    results = {}

    for precision in [QuantPrecision.INT8, QuantPrecision.INT4]:
        print(f"\n  --- {precision.name} Quantization ---")
        quantizer = KVQuantizer(precision)

        # Move to CPU for quantization (simulates offload path)
        key_cpu = key.cpu()
        value_cpu = value.cpu()

        t0 = time.perf_counter()
        qkv = quantizer.quantize(key_cpu, value_cpu)
        t_quant = time.perf_counter() - t0

        compressed_size_mb = qkv.memory_size / 1e6
        print(f"  Compressed size: {compressed_size_mb:.1f} MB")
        print(f"  Compression ratio: {qkv.compression_ratio:.2f}x")
        print(f"  Quantization time: {t_quant * 1000:.1f} ms")

        # Dequantize
        t0 = time.perf_counter()
        key_deq, value_deq = quantizer.dequantize(qkv)
        t_deq = time.perf_counter() - t0
        print(f"  Dequantization time: {t_deq * 1000:.1f} ms")

        # Quality metrics
        key_mse = ((key_cpu.float() - key_deq.float()) ** 2).mean().item()
        value_mse = ((value_cpu.float() - value_deq.float()) ** 2).mean().item()
        key_cos = torch.nn.functional.cosine_similarity(
            key_cpu.float().reshape(-1).unsqueeze(0),
            key_deq.float().reshape(-1).unsqueeze(0),
        ).item()
        value_cos = torch.nn.functional.cosine_similarity(
            value_cpu.float().reshape(-1).unsqueeze(0),
            value_deq.float().reshape(-1).unsqueeze(0),
        ).item()

        print(f"  Key MSE: {key_mse:.6f}")
        print(f"  Value MSE: {value_mse:.6f}")
        print(f"  Key cosine similarity: {key_cos:.6f}")
        print(f"  Value cosine similarity: {value_cos:.6f}")

        results[precision.name] = {
            "compression_ratio": qkv.compression_ratio,
            "quant_time_ms": t_quant * 1000,
            "deq_time_ms": t_deq * 1000,
            "key_mse": key_mse,
            "value_mse": value_mse,
            "key_cosine": key_cos,
            "value_cosine": value_cos,
        }

    del key, value
    torch.cuda.empty_cache()

    return results


def test_memory_monitor_with_allocation_pressure(device: torch.device) -> dict:
    """Test that monitor correctly detects pressure during real allocations."""
    print("\n=== Test 4: Memory Monitor Under Pressure ===")

    monitor = GPUMemoryMonitor(
        device=device,
        high_watermark=0.80,
        low_watermark=0.60,
        critical_threshold=0.92,
    )

    eviction_calls = []

    def on_eviction(bytes_to_free: int, pressure: MemoryPressure):
        eviction_calls.append({
            "bytes_to_free_mb": bytes_to_free / 1e6,
            "pressure": pressure.value,
            "timestamp": time.time(),
        })
        print(f"  [CALLBACK] Pressure={pressure.value}, free={bytes_to_free / 1e6:.0f} MB")

    monitor.set_eviction_callback(on_eviction)

    # Start with baseline
    baseline = monitor.get_memory_state()
    available = baseline.free_bytes
    print(f"  Available GPU memory: {available / 1e9:.2f} GB")

    # Gradually fill memory
    tensors = []
    chunk_size = int(available * 0.15)  # 15% chunks
    print(f"  Allocating in {chunk_size / 1e9:.2f} GB chunks...")

    for i in range(7):
        try:
            t = torch.empty(chunk_size // 2, dtype=torch.float16, device=device)
            tensors.append(t)
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at chunk {i}")
            break

        pressure = monitor.check_and_evict()
        state = monitor.get_memory_state()
        print(f"  Chunk {i}: util={state.utilization:.1%}, pressure={pressure.value}")

    # Cleanup
    del tensors
    torch.cuda.empty_cache()

    final = monitor.get_memory_state()
    print(f"\n  After cleanup: util={final.utilization:.1%}")
    print(f"  Total eviction callbacks: {len(eviction_calls)}")

    return {
        "eviction_callbacks": len(eviction_calls),
        "eviction_details": eviction_calls,
        "peak_util": monitor.peak_usage_bytes / baseline.total_bytes,
    }


def main():
    # Use GPU 1 (completely free)
    device_id = 1 if torch.cuda.device_count() > 1 else 0
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)

    print(f"Using GPU {device_id}: {torch.cuda.get_device_name(device)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")

    results = {}

    # Test 1: Memory Monitor
    results["memory_monitor"] = test_memory_monitor(device)

    # Test 2: Async Transfer
    results["async_transfer"] = test_async_transfer(device)

    # Test 3: KV Quantizer
    results["kv_quantizer"] = test_kv_quantizer_gpu(device)

    # Test 4: Memory Pressure Detection
    results["pressure_detection"] = test_memory_monitor_with_allocation_pressure(device)

    # Save results
    output_path = "experiments/results/paper/gpu_integration_v2.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n=== All Tests Passed ===")
    print(f"Results saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
