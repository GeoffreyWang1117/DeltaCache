"""Quick benchmark command for DeltaCache core operations.

Runs standardized benchmarks without requiring model weights:
- Prefix tree lookup/insert throughput
- KV cache incremental computation (synthetic)
- Tiered cache offload/prefetch latency
- INT8/INT4 quantization quality and speed
- Memory monitor accuracy
"""

import json
import time
from pathlib import Path

import click
import torch

BLOCK_CONFIGS = {
    "small": (4, 128, 8, 64),    # ~4MB
    "medium": (16, 256, 16, 128), # ~256MB
    "large": (32, 512, 32, 128),  # ~2GB
}


def _format_bytes(b: float) -> str:
    if b >= 1e9:
        return f"{b / 1e9:.2f} GB"
    return f"{b / 1e6:.1f} MB"


@click.command("bench")
@click.option(
    "--device", "-d",
    default="cpu",
    type=click.Choice(["cpu", "cuda"]),
    help="Device to run on",
)
@click.option(
    "--size", "-s",
    default="small",
    type=click.Choice(["small", "medium", "large"]),
    help="Block size for benchmarks",
)
@click.option(
    "--output-json", "-o",
    type=click.Path(),
    help="Save results to JSON file",
)
@click.pass_context
def bench(ctx, device, size, output_json):
    """Run quick performance benchmarks on DeltaCache core operations.

    \b
    Tests core components without requiring model weights:
    - Prefix tree insert/lookup throughput
    - Incremental KV computation with synthetic data
    - Tiered cache offload and prefetch latency (GPU only)
    - INT8/INT4 quantization quality metrics

    \b
    Examples:
      deltacache bench                      # CPU benchmarks, small blocks
      deltacache bench -d cuda -s medium    # GPU benchmarks, medium blocks
      deltacache bench -d cuda -o bench.json
    """
    click.echo("DeltaCache v0.2.0 Benchmark Suite")
    click.echo("=" * 60)

    dev = torch.device(device)
    if device == "cuda" and not torch.cuda.is_available():
        raise click.ClickException("CUDA not available")
    if device == "cuda":
        click.echo(f"GPU: {torch.cuda.get_device_name(dev)}")
    click.echo(f"Device: {device}, Block size: {size}")
    click.echo()

    results = {
        "device": device,
        "block_size": size,
        "torch_version": torch.__version__,
    }
    if device == "cuda":
        results["gpu"] = torch.cuda.get_device_name(dev)

    # 1. Prefix tree throughput
    results["prefix_tree"] = bench_prefix_tree()

    # 2. Incremental computation
    results["incremental"] = bench_incremental(dev, size)

    # 3. Quantization
    results["quantization"] = bench_quantization(size)

    # 4. Tiered cache (GPU only)
    if device == "cuda":
        results["tiered_cache"] = bench_tiered_cache(dev, size)

    # Summary
    click.echo()
    click.echo("=" * 60)
    click.echo("BENCHMARK SUMMARY")
    click.echo("=" * 60)

    pt = results["prefix_tree"]
    click.echo(f"  Prefix Tree:   {pt['insert_ops_per_sec']:.0f} insert/s, "
               f"{pt['lookup_ops_per_sec']:.0f} lookup/s")

    inc = results["incremental"]
    click.echo(f"  Incremental:   cold={inc['cold_ms']:.1f}ms, "
               f"warm={inc['warm_avg_ms']:.1f}ms, "
               f"speedup={inc['speedup']:.1f}x, "
               f"reuse={inc['token_reuse_rate']:.0%}")

    q = results["quantization"]
    click.echo(f"  INT8 Quant:    {q['INT8']['compression']:.1f}x compression, "
               f"cosine={q['INT8']['key_cosine']:.6f}, "
               f"{q['INT8']['quant_ms']:.1f}ms")
    click.echo(f"  INT4 Quant:    {q['INT4']['compression']:.1f}x compression, "
               f"cosine={q['INT4']['key_cosine']:.6f}, "
               f"{q['INT4']['quant_ms']:.1f}ms")

    if "tiered_cache" in results:
        tc = results["tiered_cache"]
        click.echo(f"  Offload:       {tc['offload_ms']:.1f}ms "
                   f"({tc['offload_bandwidth_gbs']:.2f} GB/s)")
        click.echo(f"  Prefetch:      {tc['prefetch_ms']:.1f}ms "
                   f"({tc['prefetch_bandwidth_gbs']:.2f} GB/s)")

    if output_json:
        out = Path(output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2, default=str)
        click.echo(f"\nResults saved to {out}")


def bench_prefix_tree() -> dict:
    """Benchmark prefix tree insert and lookup throughput."""
    click.echo("[1/4] Prefix Tree Throughput")

    from deltacache.core.prefix_tree import PrefixTree
    from deltacache.core.cache_block import CacheBlock

    tree = PrefixTree()
    num_seqs = 1000
    prefix = list(range(100))  # shared 100-token prefix

    # Generate sequences with shared prefix + varying suffix
    sequences = []
    for i in range(num_seqs):
        suffix = list(range(1000 + i * 10, 1000 + i * 10 + 20))
        sequences.append(prefix + suffix)

    # Benchmark insert
    t0 = time.perf_counter()
    for seq in sequences:
        block = CacheBlock(
            key_cache=torch.randn(1, len(seq), 1, 1),
            value_cache=torch.randn(1, len(seq), 1, 1),
        )
        tree.insert(seq, block)
    insert_time = time.perf_counter() - t0

    # Benchmark lookup
    t0 = time.perf_counter()
    hits = 0
    for seq in sequences:
        result = tree.lookup(seq)
        if result.has_match:
            hits += 1
    lookup_time = time.perf_counter() - t0

    insert_ops = num_seqs / insert_time
    lookup_ops = num_seqs / lookup_time

    click.echo(f"  Insert: {insert_ops:.0f} ops/s ({insert_time*1000:.1f}ms for {num_seqs} seqs)")
    click.echo(f"  Lookup: {lookup_ops:.0f} ops/s ({lookup_time*1000:.1f}ms, {hits}/{num_seqs} hits)")

    return {
        "num_sequences": num_seqs,
        "prefix_length": len(prefix),
        "insert_ops_per_sec": insert_ops,
        "lookup_ops_per_sec": lookup_ops,
        "insert_total_ms": insert_time * 1000,
        "lookup_total_ms": lookup_time * 1000,
        "hit_rate": hits / num_seqs,
    }


def bench_incremental(device: torch.device, size: str) -> dict:
    """Benchmark incremental KV computation with synthetic data."""
    click.echo("[2/4] Incremental KV Computation")

    from deltacache import DeltaCacheManager, DeltaCacheConfig

    config = DeltaCacheConfig(
        num_layers=4,
        num_heads=4,
        head_dim=64,
        device=str(device),
        dtype="float32",
    )
    manager = DeltaCacheManager(config)

    # Synthetic KV compute function
    def compute_kv(input_ids, position_ids, past_key_values=None):
        seq_len = len(input_ids)
        key = torch.randn(
            config.num_layers, seq_len, config.num_heads, config.head_dim,
            device=device,
        )
        value = torch.randn(
            config.num_layers, seq_len, config.num_heads, config.head_dim,
            device=device,
        )
        return key, value

    prefix = list(range(200))
    suffixes = [list(range(1000 + i * 20, 1000 + i * 20 + 30)) for i in range(10)]

    # Cold run
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    manager.compute_incremental(prefix + suffixes[0], compute_kv)
    if device.type == "cuda":
        torch.cuda.synchronize()
    cold_ms = (time.perf_counter() - t0) * 1000

    # Warm runs
    warm_times = []
    for suffix in suffixes[1:]:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = manager.compute_incremental(prefix + suffix, compute_kv)
        if device.type == "cuda":
            torch.cuda.synchronize()
        warm_times.append((time.perf_counter() - t0) * 1000)

    warm_avg = sum(warm_times) / len(warm_times)
    speedup = cold_ms / warm_avg if warm_avg > 0 else 0
    stats = manager.get_stats()

    click.echo(f"  Cold:  {cold_ms:.1f}ms (first request, no cache)")
    click.echo(f"  Warm:  {warm_avg:.1f}ms avg (cached prefix reuse)")
    click.echo(f"  Speedup: {speedup:.1f}x, Token reuse: {stats.get('token_reuse_rate', 0):.0%}")

    return {
        "cold_ms": cold_ms,
        "warm_avg_ms": warm_avg,
        "warm_times_ms": warm_times,
        "speedup": speedup,
        "hit_rate": stats.get("hit_rate", 0),
        "token_reuse_rate": stats.get("token_reuse_rate", 0),
        "prefix_length": len(prefix),
        "suffix_length": len(suffixes[0]),
    }


def bench_quantization(size: str) -> dict:
    """Benchmark INT8/INT4 quantization quality and speed."""
    click.echo("[3/4] KV Quantization")

    from deltacache.core.kv_quantizer import KVQuantizer, QuantPrecision

    nl, sl, nh, hd = BLOCK_CONFIGS[size]
    key = torch.randn(nl, sl, nh, hd, dtype=torch.float16)
    value = torch.randn(nl, sl, nh, hd, dtype=torch.float16)
    orig_mb = (key.numel() + value.numel()) * 2 / 1e6

    results = {}
    for precision in [QuantPrecision.INT8, QuantPrecision.INT4]:
        quantizer = KVQuantizer(precision)

        t0 = time.perf_counter()
        qkv = quantizer.quantize(key, value)
        quant_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        key_deq, value_deq = quantizer.dequantize(qkv)
        deq_ms = (time.perf_counter() - t0) * 1000

        key_cos = torch.nn.functional.cosine_similarity(
            key.float().reshape(-1).unsqueeze(0),
            key_deq.float().reshape(-1).unsqueeze(0),
        ).item()
        key_mse = ((key.float() - key_deq.float()) ** 2).mean().item()

        click.echo(f"  {precision.name}: {qkv.compression_ratio:.1f}x compression, "
                   f"cosine={key_cos:.6f}, MSE={key_mse:.6f}, "
                   f"quant={quant_ms:.1f}ms, deq={deq_ms:.1f}ms")

        results[precision.name] = {
            "compression": qkv.compression_ratio,
            "key_cosine": key_cos,
            "key_mse": key_mse,
            "quant_ms": quant_ms,
            "deq_ms": deq_ms,
            "original_mb": orig_mb,
            "compressed_mb": qkv.memory_size / 1e6,
        }

    return results


def bench_tiered_cache(device: torch.device, size: str) -> dict:
    """Benchmark tiered cache offload and prefetch (GPU only)."""
    click.echo("[4/4] Tiered Cache Transfer")

    from deltacache.core.async_transfer import AsyncTransferManager

    nl, sl, nh, hd = BLOCK_CONFIGS[size]
    key_gpu = torch.randn(nl, sl, nh, hd, dtype=torch.float16, device=device)
    value_gpu = torch.randn(nl, sl, nh, hd, dtype=torch.float16, device=device)
    block_mb = (key_gpu.numel() + value_gpu.numel()) * 2 / 1e6

    manager = AsyncTransferManager(device=device)

    # Offload GPU -> CPU
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    key_cpu, value_cpu = manager.offload_async(0, key_gpu, value_gpu)
    manager.wait_transfer(0)
    offload_ms = (time.perf_counter() - t0) * 1000
    offload_bw = (block_mb / 1000) / (offload_ms / 1000)

    # Prefetch CPU -> GPU (with pinned memory)
    key_pinned = key_cpu.pin_memory()
    value_pinned = value_cpu.pin_memory()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    key_back, value_back = manager.prefetch_async(1, key_pinned, value_pinned)
    manager.wait_transfer(1)
    prefetch_ms = (time.perf_counter() - t0) * 1000
    prefetch_bw = (block_mb / 1000) / (prefetch_ms / 1000)

    click.echo(f"  Block size: {block_mb:.1f} MB")
    click.echo(f"  Offload (GPU→CPU): {offload_ms:.1f}ms ({offload_bw:.2f} GB/s)")
    click.echo(f"  Prefetch (CPU→GPU): {prefetch_ms:.1f}ms ({prefetch_bw:.2f} GB/s)")

    del key_gpu, value_gpu, key_cpu, value_cpu, key_pinned, value_pinned
    del key_back, value_back
    torch.cuda.empty_cache()

    return {
        "block_mb": block_mb,
        "offload_ms": offload_ms,
        "offload_bandwidth_gbs": offload_bw,
        "prefetch_ms": prefetch_ms,
        "prefetch_bandwidth_gbs": prefetch_bw,
    }
