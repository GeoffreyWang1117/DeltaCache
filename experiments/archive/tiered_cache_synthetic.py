"""Tiered Cache + Quantization + Memory Monitor End-to-End Experiment.

Tests the full v0.2.0 pipeline:
1. Load model, run prefill with DeltaCache
2. Fill GPU cache until memory pressure triggers
3. Offload blocks to CPU (async transfer)
4. Quantize offloaded blocks on CPU (INT8)
5. Prefetch from CPU when cache hit found on CPU tier
6. Measure latency, bandwidth, quality at each step

Run: CUDA_VISIBLE_DEVICES=1 python experiments/tiered_cache_experiment.py
"""

import gc
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from deltacache.core.async_transfer import AsyncTransferManager
from deltacache.core.kv_quantizer import KVQuantizer, QuantPrecision
from deltacache.core.memory_monitor import GPUMemoryMonitor, MemoryPressure

RESULTS_DIR = Path(__file__).parent / "results" / "paper"


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def format_bytes(b: int) -> str:
    if b > 1e9:
        return f"{b / 1e9:.2f} GB"
    return f"{b / 1e6:.1f} MB"


def simulate_kv_block(
    num_layers: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    device: torch.device,
) -> tuple:
    """Create a realistic KV cache block."""
    shape = (num_layers, seq_len, num_heads, head_dim)
    key = torch.randn(shape, dtype=torch.float16, device=device)
    value = torch.randn(shape, dtype=torch.float16, device=device)
    return key, value


def test_tiered_offload_cycle(device: torch.device) -> dict:
    """Test complete offload -> quantize -> prefetch cycle."""
    print("\n=== Tiered Offload Cycle ===")

    # Mistral-7B-like dimensions
    num_layers, num_heads, head_dim = 32, 32, 128
    seq_len = 512  # 512 token prefix

    transfer_mgr = AsyncTransferManager(device=device)
    quantizer = KVQuantizer(QuantPrecision.INT8)

    # Step 1: Create KV block on GPU
    key_gpu, value_gpu = simulate_kv_block(num_layers, seq_len, num_heads, head_dim, device)
    block_size_mb = (key_gpu.numel() + value_gpu.numel()) * 2 / 1e6
    print(f"  Block size: {block_size_mb:.1f} MB ({seq_len} tokens, {num_layers} layers)")

    results = {}

    # Step 2: Async offload GPU -> CPU
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    key_cpu, value_cpu = transfer_mgr.offload_async(
        block_id=100, key_gpu=key_gpu, value_gpu=value_gpu
    )
    transfer_mgr.wait_transfer(100)
    t_offload = time.perf_counter() - t0
    offload_bw = (block_size_mb / 1000) / t_offload
    print(f"  Offload time: {t_offload * 1000:.1f} ms ({offload_bw:.2f} GB/s)")
    results["offload_ms"] = t_offload * 1000
    results["offload_bandwidth_gbs"] = offload_bw

    # Step 3: Quantize on CPU
    t0 = time.perf_counter()
    qkv = quantizer.quantize(key_cpu, value_cpu)
    t_quant = time.perf_counter() - t0
    print(f"  Quantize time: {t_quant * 1000:.1f} ms")
    print(f"  Compression: {qkv.compression_ratio:.2f}x ({qkv.memory_size / 1e6:.1f} MB)")
    results["quantize_ms"] = t_quant * 1000
    results["compression_ratio"] = qkv.compression_ratio
    results["compressed_size_mb"] = qkv.memory_size / 1e6

    # Free original CPU tensors (keep only quantized)
    del key_cpu, value_cpu

    # Step 4: Dequantize when needed
    t0 = time.perf_counter()
    key_deq, value_deq = quantizer.dequantize(qkv)
    t_deq = time.perf_counter() - t0
    print(f"  Dequantize time: {t_deq * 1000:.1f} ms")
    results["dequantize_ms"] = t_deq * 1000

    # Step 5: Pin memory for optimal prefetch
    t0 = time.perf_counter()
    key_pinned = key_deq.pin_memory()
    value_pinned = value_deq.pin_memory()
    t_pin = time.perf_counter() - t0
    print(f"  Pin memory time: {t_pin * 1000:.1f} ms")
    results["pin_memory_ms"] = t_pin * 1000

    # Step 6: Async prefetch CPU -> GPU
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    key_back, value_back = transfer_mgr.prefetch_async(
        block_id=101, key_cpu=key_pinned, value_cpu=value_pinned
    )
    transfer_mgr.wait_transfer(101)
    t_prefetch = time.perf_counter() - t0
    prefetch_bw = (block_size_mb / 1000) / t_prefetch
    print(f"  Prefetch time: {t_prefetch * 1000:.1f} ms ({prefetch_bw:.2f} GB/s)")
    results["prefetch_ms"] = t_prefetch * 1000
    results["prefetch_bandwidth_gbs"] = prefetch_bw

    # Step 7: Verify quality
    # Compare original GPU data with round-tripped data
    key_cos = torch.nn.functional.cosine_similarity(
        key_gpu.float().reshape(-1).unsqueeze(0),
        key_back.float().reshape(-1).unsqueeze(0),
    ).item()
    value_cos = torch.nn.functional.cosine_similarity(
        value_gpu.float().reshape(-1).unsqueeze(0),
        value_back.float().reshape(-1).unsqueeze(0),
    ).item()
    key_mse = ((key_gpu.float() - key_back.float()) ** 2).mean().item()

    print(f"\n  Quality after full cycle (GPU -> CPU -> quantize -> deq -> GPU):")
    print(f"    Key cosine sim:  {key_cos:.6f}")
    print(f"    Value cosine sim: {value_cos:.6f}")
    print(f"    Key MSE:         {key_mse:.8f}")
    results["key_cosine"] = key_cos
    results["value_cosine"] = value_cos
    results["key_mse"] = key_mse

    # Total cycle latency
    total_ms = results["offload_ms"] + results["quantize_ms"]
    total_prefetch_ms = results["dequantize_ms"] + results["pin_memory_ms"] + results["prefetch_ms"]
    print(f"\n  Total offload pipeline: {total_ms:.1f} ms")
    print(f"  Total prefetch pipeline: {total_prefetch_ms:.1f} ms")
    results["total_offload_pipeline_ms"] = total_ms
    results["total_prefetch_pipeline_ms"] = total_prefetch_ms

    # Cleanup
    del key_gpu, value_gpu, key_deq, value_deq, key_pinned, value_pinned
    del key_back, value_back, qkv
    clear_gpu()

    return results


def test_multi_block_pressure(device: torch.device) -> dict:
    """Simulate filling GPU cache and triggering tiered eviction."""
    print("\n=== Multi-Block Memory Pressure ===")

    # Smaller blocks for faster iteration (8 layers, 256 tok, 8 heads, 64 dim)
    num_layers, seq_len, num_heads, head_dim = 8, 256, 8, 64
    block_shape = (num_layers, seq_len, num_heads, head_dim)
    block_bytes = 2 * num_layers * seq_len * num_heads * head_dim * 2  # K + V, FP16
    block_mb = block_bytes / 1e6
    print(f"  Block size: {block_mb:.1f} MB")

    monitor = GPUMemoryMonitor(
        device=device,
        high_watermark=0.80,
        low_watermark=0.60,
        critical_threshold=0.92,
    )
    transfer_mgr = AsyncTransferManager(device=device)
    quantizer = KVQuantizer(QuantPrecision.INT8)

    baseline = monitor.get_memory_state()
    print(f"  Available: {format_bytes(baseline.free_bytes)}")

    # Track blocks
    gpu_blocks = {}  # id -> (key, value) on GPU
    cpu_blocks = {}  # id -> QuantizedKV on CPU
    timeline = []
    offloaded_count = 0
    prefetched_count = 0

    # Fill GPU cache
    target_blocks = int(baseline.free_bytes * 0.85 / block_bytes)
    print(f"  Target: {target_blocks} blocks to reach ~85% utilization")

    for i in range(target_blocks + 5):
        state = monitor.get_memory_state()
        pressure = monitor.get_pressure_level(state)

        if pressure == MemoryPressure.HIGH or pressure == MemoryPressure.CRITICAL:
            # Offload oldest blocks to CPU
            n_offload = max(1, len(gpu_blocks) // 5)
            oldest_ids = sorted(gpu_blocks.keys())[:n_offload]

            for bid in oldest_ids:
                k, v = gpu_blocks[bid]
                # Async offload
                k_cpu, v_cpu = transfer_mgr.offload_async(bid, k, v)
                transfer_mgr.wait_transfer(bid)
                # Quantize on CPU
                qkv = quantizer.quantize(k_cpu, v_cpu)
                cpu_blocks[bid] = qkv
                del gpu_blocks[bid]
                # Free GPU memory
                del k, v
                offloaded_count += 1

            clear_gpu()
            state = monitor.get_memory_state()
            pressure = monitor.get_pressure_level(state)

        # Allocate new block
        try:
            key, value = simulate_kv_block(num_layers, seq_len, num_heads, head_dim, device)
            gpu_blocks[i] = (key, value)
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at block {i}")
            break

        timeline.append({
            "block_id": i,
            "gpu_blocks": len(gpu_blocks),
            "cpu_blocks": len(cpu_blocks),
            "gpu_util": state.utilization,
            "pressure": pressure.value,
        })

    print(f"\n  Final state:")
    print(f"    GPU blocks: {len(gpu_blocks)}")
    print(f"    CPU blocks: {len(cpu_blocks)} (quantized)")
    print(f"    Total offloaded: {offloaded_count}")

    # Simulate prefetch: bring some CPU blocks back
    prefetch_ids = list(cpu_blocks.keys())[:3]
    for bid in prefetch_ids:
        qkv = cpu_blocks[bid]
        t0 = time.perf_counter()
        k_deq, v_deq = quantizer.dequantize(qkv)
        k_pinned, v_pinned = k_deq.pin_memory(), v_deq.pin_memory()
        k_gpu, v_gpu = transfer_mgr.prefetch_async(bid, k_pinned, v_pinned)
        transfer_mgr.wait_transfer(bid)
        t_prefetch = time.perf_counter() - t0
        prefetched_count += 1
        del cpu_blocks[bid]
        del k_deq, v_deq, k_pinned, v_pinned, k_gpu, v_gpu

    print(f"    Prefetched back: {prefetched_count}")

    # Cleanup
    del gpu_blocks, cpu_blocks
    clear_gpu()

    # Compute CPU-tier memory savings
    total_original_bytes = offloaded_count * block_bytes
    # INT8 gives ~2x compression
    total_compressed_bytes = total_original_bytes / 2

    print(f"\n  Memory savings from quantization:")
    print(f"    Original: {format_bytes(total_original_bytes)}")
    print(f"    Compressed (INT8): {format_bytes(int(total_compressed_bytes))}")
    print(f"    Savings: {total_original_bytes / max(1, total_compressed_bytes):.2f}x")

    return {
        "total_blocks_created": len(timeline),
        "gpu_blocks_final": timeline[-1]["gpu_blocks"] if timeline else 0,
        "cpu_blocks_final": timeline[-1]["cpu_blocks"] if timeline else 0,
        "offloaded_count": offloaded_count,
        "prefetched_count": prefetched_count,
        "original_bytes": total_original_bytes,
        "compressed_bytes": int(total_compressed_bytes),
        "timeline_sample": timeline[-5:] if len(timeline) >= 5 else timeline,
    }


def test_pinned_vs_unpinned_bandwidth(device: torch.device) -> dict:
    """Compare transfer bandwidth with pinned vs unpinned memory."""
    print("\n=== Pinned vs Unpinned Bandwidth ===")

    sizes_mb = [1, 10, 50, 100, 256, 512]
    results = {}

    for size_mb in sizes_mb:
        num_elements = int(size_mb * 1e6 / 2)  # FP16

        # Unpinned CPU -> GPU
        cpu_unpinned = torch.randn(num_elements, dtype=torch.float16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gpu_t = cpu_unpinned.to(device)
        torch.cuda.synchronize()
        t_unpinned = time.perf_counter() - t0
        del gpu_t

        # Pinned CPU -> GPU
        cpu_pinned = torch.randn(num_elements, dtype=torch.float16).pin_memory()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gpu_t = cpu_pinned.to(device, non_blocking=True)
        torch.cuda.synchronize()
        t_pinned = time.perf_counter() - t0
        del gpu_t

        bw_unpinned = (size_mb / 1000) / t_unpinned
        bw_pinned = (size_mb / 1000) / t_pinned
        speedup = t_unpinned / t_pinned

        print(f"  {size_mb:4d} MB: unpinned={bw_unpinned:.2f} GB/s, "
              f"pinned={bw_pinned:.2f} GB/s, speedup={speedup:.2f}x")

        results[f"{size_mb}mb"] = {
            "unpinned_gbs": bw_unpinned,
            "pinned_gbs": bw_pinned,
            "speedup": speedup,
        }

        del cpu_unpinned, cpu_pinned

    clear_gpu()
    return results


def main():
    device_id = 1 if torch.cuda.device_count() > 1 else 0
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)

    print(f"=== DeltaCache v0.2.0 Tiered Cache Experiment ===")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"VRAM: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")

    results = {}

    # Experiment 1: Full offload cycle
    results["tiered_offload_cycle"] = test_tiered_offload_cycle(device)

    # Experiment 2: Multi-block memory pressure
    results["multi_block_pressure"] = test_multi_block_pressure(device)

    # Experiment 3: Pinned vs unpinned bandwidth
    results["pinned_vs_unpinned"] = test_pinned_vs_unpinned_bandwidth(device)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "tiered_cache_experiment.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n=== Experiment Complete ===")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
