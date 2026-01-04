#!/usr/bin/env python3
"""
GPU Memory Usage Analysis for DeltaCache - ICML 2026 P0 Experiment

This experiment measures and compares GPU memory consumption between:
1. Baseline (no cache) - fresh KV computation each time
2. DeltaCache - with KV caching and prefix tree

Key metrics:
- Peak GPU memory usage
- Memory overhead from prefix tree/cache management
- Memory-speedup tradeoff at different cache sizes
- Memory efficiency (speedup per GB)

Output: Figures and data for ICML paper
"""

import os
import gc
import sys
import json
import time
import statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict, field

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from deltacache import DeltaCacheManager, DeltaCacheConfig
from deltacache.hf_integration import LlamaStyleAdapter

RESULTS_DIR = Path(__file__).parent / "results" / "paper"


def clear_gpu():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_gpu_memory_mb() -> Dict[str, float]:
    """Get detailed GPU memory stats in MB."""
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "max_allocated": 0}

    torch.cuda.synchronize()
    return {
        "allocated": torch.cuda.memory_allocated() / (1024 ** 2),
        "reserved": torch.cuda.memory_reserved() / (1024 ** 2),
        "max_allocated": torch.cuda.max_memory_allocated() / (1024 ** 2),
    }


def reset_gpu_memory_stats():
    """Reset GPU memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


@dataclass
class MemoryMeasurement:
    """Single memory measurement."""
    scenario: str
    method: str  # 'baseline' or 'deltacache'
    prefix_tokens: int
    n_queries: int

    # Memory stats (MB)
    initial_memory_mb: float
    peak_memory_mb: float
    final_memory_mb: float
    cache_memory_mb: float = 0.0

    # Latency stats (ms)
    mean_latency_ms: float = 0.0
    total_time_ms: float = 0.0

    # Cache stats
    cache_hit_rate: float = 0.0
    token_reuse_rate: float = 0.0


@dataclass
class MemoryAnalysisResult:
    """Complete memory analysis result."""
    model_name: str
    device: str
    timestamp: str
    measurements: List[MemoryMeasurement] = field(default_factory=list)

    # Aggregate stats
    baseline_peak_mb: float = 0.0
    deltacache_peak_mb: float = 0.0
    memory_overhead_mb: float = 0.0
    memory_overhead_pct: float = 0.0
    speedup: float = 0.0
    memory_efficiency: float = 0.0  # speedup per GB overhead


def create_test_prompts(prefix_length: int, n_queries: int = 20) -> Tuple[str, List[str]]:
    """Create test prompts with fixed prefix and varying queries."""

    # Create a long prefix (documentation-style content)
    base_content = """
# Python Data Processing Library Documentation

## Overview
This library provides comprehensive tools for data processing, transformation, and analysis.
It supports various data formats including JSON, CSV, XML, and binary formats.

## Core Components

### DataLoader
The DataLoader class handles loading data from various sources:
- File system (local and network)
- Databases (SQL and NoSQL)
- API endpoints (REST and GraphQL)
- Streaming sources (Kafka, Redis)

Usage:
```python
loader = DataLoader(source="path/to/data.csv")
data = loader.load()
```

### DataTransformer
The DataTransformer applies transformations to data:
- Filtering and selection
- Aggregation and grouping
- Normalization and scaling
- Feature engineering

### DataValidator
Ensures data quality through validation:
- Schema validation
- Type checking
- Range validation
- Custom rules

## Advanced Features

### Parallel Processing
The library supports parallel data processing using:
- Multi-threading for I/O-bound tasks
- Multi-processing for CPU-bound tasks
- GPU acceleration for large-scale computations

### Caching
Intelligent caching reduces computation:
- In-memory LRU cache
- Disk-based persistent cache
- Distributed cache support

### Monitoring
Built-in monitoring capabilities:
- Performance metrics
- Resource utilization
- Error tracking

"""

    # Repeat content to reach desired prefix length
    multiplier = max(1, prefix_length // 100)
    prefix = (base_content * multiplier)[:prefix_length * 4]  # Approximate chars to tokens

    # Create varied queries
    query_templates = [
        "How do I use the DataLoader class?",
        "What are the supported data formats?",
        "Explain the parallel processing features.",
        "How does caching work in this library?",
        "What validation options are available?",
        "Show me an example of data transformation.",
        "How can I monitor performance?",
        "What databases are supported?",
        "Explain the DataTransformer class.",
        "How do I enable GPU acceleration?",
        "What is the schema validation syntax?",
        "How do I configure distributed caching?",
        "What streaming sources are supported?",
        "How do I handle large datasets?",
        "What are the memory optimization options?",
        "How do I set up custom validation rules?",
        "Explain the aggregation functions.",
        "What normalization methods are available?",
        "How do I load data from APIs?",
        "What are the error handling patterns?",
    ]

    queries = []
    for i in range(n_queries):
        queries.append(query_templates[i % len(query_templates)])

    return prefix, queries


def measure_baseline_memory(
    adapter,
    tokenizer,
    prefix: str,
    queries: List[str],
    scenario_name: str,
) -> MemoryMeasurement:
    """Measure memory for baseline (no cache) approach."""

    prefix_tokens = len(tokenizer.encode(prefix))

    clear_gpu()
    reset_gpu_memory_stats()

    initial_mem = get_gpu_memory_mb()

    latencies = []

    for query in tqdm(queries, desc=f"Baseline {scenario_name}"):
        prompt = f"{prefix}\n{query}"
        tokens = tokenizer.encode(prompt)

        torch.cuda.synchronize()
        start = time.perf_counter()

        # Compute KV cache from scratch
        _ = adapter.compute_kv_for_tokens(tokens)

        torch.cuda.synchronize()
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)

        # Clear after each query (simulates no caching)
        clear_gpu()

    peak_mem = get_gpu_memory_mb()
    final_mem = get_gpu_memory_mb()

    return MemoryMeasurement(
        scenario=scenario_name,
        method="baseline",
        prefix_tokens=prefix_tokens,
        n_queries=len(queries),
        initial_memory_mb=initial_mem["allocated"],
        peak_memory_mb=peak_mem["max_allocated"],
        final_memory_mb=final_mem["allocated"],
        mean_latency_ms=statistics.mean(latencies),
        total_time_ms=sum(latencies),
    )


def measure_deltacache_memory(
    adapter,
    tokenizer,
    config: DeltaCacheConfig,
    prefix: str,
    queries: List[str],
    scenario_name: str,
) -> MemoryMeasurement:
    """Measure memory for DeltaCache approach."""

    prefix_tokens = len(tokenizer.encode(prefix))

    clear_gpu()
    reset_gpu_memory_stats()

    initial_mem = get_gpu_memory_mb()

    # Create DeltaCache manager
    manager = DeltaCacheManager(config)

    after_init_mem = get_gpu_memory_mb()

    latencies = []
    total_matched = 0
    total_tokens = 0

    for query in tqdm(queries, desc=f"DeltaCache {scenario_name}"):
        prompt = f"{prefix}\n{query}"
        tokens = tokenizer.encode(prompt)
        total_tokens += len(tokens)

        torch.cuda.synchronize()
        start = time.perf_counter()

        result = manager.compute_incremental(tokens, adapter.compute_kv)

        torch.cuda.synchronize()
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)
        total_matched += result.matched_length

    peak_mem = get_gpu_memory_mb()

    # Get cache memory usage
    memory_stats = manager.get_memory_stats()
    cache_memory_mb = memory_stats.used_bytes / (1024 ** 2)

    # Get cache stats
    stats = manager.get_stats()
    hit_rate = stats.get("hit_rate", 0)

    final_mem = get_gpu_memory_mb()

    del manager
    clear_gpu()

    return MemoryMeasurement(
        scenario=scenario_name,
        method="deltacache",
        prefix_tokens=prefix_tokens,
        n_queries=len(queries),
        initial_memory_mb=initial_mem["allocated"],
        peak_memory_mb=peak_mem["max_allocated"],
        final_memory_mb=final_mem["allocated"],
        cache_memory_mb=cache_memory_mb,
        mean_latency_ms=statistics.mean(latencies),
        total_time_ms=sum(latencies),
        cache_hit_rate=hit_rate,
        token_reuse_rate=total_matched / total_tokens if total_tokens > 0 else 0,
    )


def run_memory_analysis(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device: str = "cuda:1",
    prefix_lengths: List[int] = [250, 500, 750, 1000, 1500],
    n_queries: int = 20,
) -> MemoryAnalysisResult:
    """Run comprehensive memory analysis."""

    print("\n" + "=" * 70)
    print("GPU MEMORY USAGE ANALYSIS")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Prefix lengths: {prefix_lengths}")
    print(f"Queries per prefix: {n_queries}")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    adapter = LlamaStyleAdapter.from_pretrained(model_name, device=device)
    tokenizer = adapter.tokenizer
    config = DeltaCacheConfig.for_model(model_name)
    config.device = device

    model_mem = get_gpu_memory_mb()
    print(f"Model loaded. GPU memory: {model_mem['allocated']:.1f} MB")

    result = MemoryAnalysisResult(
        model_name=model_name,
        device=device,
        timestamp=datetime.now().isoformat(),
    )

    try:
        for prefix_len in prefix_lengths:
            print(f"\n{'='*60}")
            print(f"Prefix length: {prefix_len} tokens")
            print("=" * 60)

            prefix, queries = create_test_prompts(prefix_len, n_queries)
            actual_prefix_tokens = len(tokenizer.encode(prefix))
            print(f"Actual prefix tokens: {actual_prefix_tokens}")

            scenario = f"prefix_{actual_prefix_tokens}"

            # Measure baseline
            baseline_result = measure_baseline_memory(
                adapter, tokenizer, prefix, queries, scenario
            )
            result.measurements.append(baseline_result)
            print(f"\nBaseline:")
            print(f"  Peak memory: {baseline_result.peak_memory_mb:.1f} MB")
            print(f"  Mean latency: {baseline_result.mean_latency_ms:.1f} ms")

            # Measure DeltaCache
            deltacache_result = measure_deltacache_memory(
                adapter, tokenizer, config, prefix, queries, scenario
            )
            result.measurements.append(deltacache_result)
            print(f"\nDeltaCache:")
            print(f"  Peak memory: {deltacache_result.peak_memory_mb:.1f} MB")
            print(f"  Cache memory: {deltacache_result.cache_memory_mb:.1f} MB")
            print(f"  Mean latency: {deltacache_result.mean_latency_ms:.1f} ms")
            print(f"  Token reuse: {deltacache_result.token_reuse_rate*100:.1f}%")

            # Calculate speedup
            speedup = baseline_result.mean_latency_ms / deltacache_result.mean_latency_ms
            print(f"  Speedup: {speedup:.2f}x")

    finally:
        del adapter
        clear_gpu()

    # Calculate aggregate stats
    baseline_measurements = [m for m in result.measurements if m.method == "baseline"]
    dc_measurements = [m for m in result.measurements if m.method == "deltacache"]

    if baseline_measurements and dc_measurements:
        result.baseline_peak_mb = max(m.peak_memory_mb for m in baseline_measurements)
        result.deltacache_peak_mb = max(m.peak_memory_mb for m in dc_measurements)
        result.memory_overhead_mb = result.deltacache_peak_mb - result.baseline_peak_mb
        result.memory_overhead_pct = (result.memory_overhead_mb / result.baseline_peak_mb * 100) if result.baseline_peak_mb > 0 else 0

        avg_baseline_latency = statistics.mean(m.mean_latency_ms for m in baseline_measurements)
        avg_dc_latency = statistics.mean(m.mean_latency_ms for m in dc_measurements)
        result.speedup = avg_baseline_latency / avg_dc_latency if avg_dc_latency > 0 else 0

        # Memory efficiency: speedup per GB of overhead
        if result.memory_overhead_mb > 0:
            result.memory_efficiency = result.speedup / (result.memory_overhead_mb / 1024)
        else:
            result.memory_efficiency = float('inf')

    return result


def run_cache_size_sweep(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device: str = "cuda:1",
    cache_sizes_mb: List[int] = [100, 200, 500, 1000, 2000],
    prefix_length: int = 1000,
    n_queries: int = 30,
) -> Dict:
    """Analyze memory-speedup tradeoff at different cache sizes."""

    print("\n" + "=" * 70)
    print("CACHE SIZE SWEEP ANALYSIS")
    print(f"Model: {model_name}")
    print(f"Cache sizes: {cache_sizes_mb} MB")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    adapter = LlamaStyleAdapter.from_pretrained(model_name, device=device)
    tokenizer = adapter.tokenizer

    prefix, queries = create_test_prompts(prefix_length, n_queries)
    prefix_tokens = len(tokenizer.encode(prefix))
    print(f"Prefix tokens: {prefix_tokens}")

    results = {
        "model": model_name,
        "prefix_tokens": prefix_tokens,
        "n_queries": n_queries,
        "cache_size_sweep": []
    }

    try:
        for cache_size_mb in cache_sizes_mb:
            print(f"\n{'='*40}")
            print(f"Cache size: {cache_size_mb} MB")
            print("=" * 40)

            clear_gpu()
            reset_gpu_memory_stats()

            # Configure with specific cache size
            config = DeltaCacheConfig.for_model(model_name)
            config.device = device
            config.gpu_memory_limit = cache_size_mb * 1024 * 1024  # Convert to bytes

            manager = DeltaCacheManager(config)

            latencies = []
            evictions = 0
            total_matched = 0
            total_tokens = 0

            for query in tqdm(queries, desc=f"Cache {cache_size_mb}MB"):
                prompt = f"{prefix}\n{query}"
                tokens = tokenizer.encode(prompt)
                total_tokens += len(tokens)

                torch.cuda.synchronize()
                start = time.perf_counter()

                result = manager.compute_incremental(tokens, adapter.compute_kv)

                torch.cuda.synchronize()
                latency = (time.perf_counter() - start) * 1000
                latencies.append(latency)
                total_matched += result.matched_length

            peak_mem = get_gpu_memory_mb()
            stats = manager.get_stats()

            sweep_result = {
                "cache_size_mb": cache_size_mb,
                "peak_memory_mb": peak_mem["max_allocated"],
                "actual_cache_mb": manager.get_memory_stats().used_bytes / (1024 ** 2),
                "mean_latency_ms": statistics.mean(latencies),
                "token_reuse_rate": total_matched / total_tokens if total_tokens > 0 else 0,
                "evictions": stats.get("evictions", 0),
                "hit_rate": stats.get("hit_rate", 0),
            }
            results["cache_size_sweep"].append(sweep_result)

            print(f"  Peak memory: {sweep_result['peak_memory_mb']:.1f} MB")
            print(f"  Actual cache: {sweep_result['actual_cache_mb']:.1f} MB")
            print(f"  Mean latency: {sweep_result['mean_latency_ms']:.1f} ms")
            print(f"  Token reuse: {sweep_result['token_reuse_rate']*100:.1f}%")

            del manager
            clear_gpu()

    finally:
        del adapter
        clear_gpu()

    return results


def generate_memory_figures(result: MemoryAnalysisResult, output_dir: Path):
    """Generate figures for the paper."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available, skipping figure generation")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by scenario
    baseline_data = {}
    dc_data = {}

    for m in result.measurements:
        if m.method == "baseline":
            baseline_data[m.prefix_tokens] = m
        else:
            dc_data[m.prefix_tokens] = m

    prefix_lengths = sorted(baseline_data.keys())

    # Figure 1: Peak Memory Comparison
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(prefix_lengths))
    width = 0.35

    baseline_mem = [baseline_data[p].peak_memory_mb for p in prefix_lengths]
    dc_mem = [dc_data[p].peak_memory_mb for p in prefix_lengths]

    bars1 = ax.bar(x - width/2, baseline_mem, width, label='Baseline', color='#2196F3')
    bars2 = ax.bar(x + width/2, dc_mem, width, label='DeltaCache', color='#4CAF50')

    ax.set_xlabel('Prefix Length (tokens)', fontsize=12)
    ax.set_ylabel('Peak GPU Memory (MB)', fontsize=12)
    ax.set_title('GPU Memory Usage: Baseline vs DeltaCache', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(prefix_lengths)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "memory_comparison.pdf", dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 2: Memory-Speedup Tradeoff
    fig, ax1 = plt.subplots(figsize=(8, 5))

    speedups = [baseline_data[p].mean_latency_ms / dc_data[p].mean_latency_ms for p in prefix_lengths]
    cache_mem = [dc_data[p].cache_memory_mb for p in prefix_lengths]

    color1 = '#4CAF50'
    ax1.set_xlabel('Prefix Length (tokens)', fontsize=12)
    ax1.set_ylabel('Speedup (×)', fontsize=12, color=color1)
    line1 = ax1.plot(prefix_lengths, speedups, 'o-', color=color1, linewidth=2, markersize=8, label='Speedup')
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = '#FF5722'
    ax2.set_ylabel('Cache Memory (MB)', fontsize=12, color=color2)
    line2 = ax2.plot(prefix_lengths, cache_mem, 's--', color=color2, linewidth=2, markersize=8, label='Cache Memory')
    ax2.tick_params(axis='y', labelcolor=color2)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    ax1.set_title('Memory-Speedup Tradeoff', fontsize=14)
    ax1.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "memory_speedup_tradeoff.pdf", dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 3: Token Reuse Rate
    fig, ax = plt.subplots(figsize=(8, 5))

    reuse_rates = [dc_data[p].token_reuse_rate * 100 for p in prefix_lengths]

    ax.bar(prefix_lengths, reuse_rates, color='#9C27B0', width=50)
    ax.set_xlabel('Prefix Length (tokens)', fontsize=12)
    ax.set_ylabel('Token Reuse Rate (%)', fontsize=12)
    ax.set_title('Cache Token Reuse Rate by Prefix Length', fontsize=14)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    for i, (p, r) in enumerate(zip(prefix_lengths, reuse_rates)):
        ax.annotate(f'{r:.1f}%', (p, r + 2), ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "token_reuse_rate.pdf", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nFigures saved to {output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="GPU Memory Usage Analysis")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--prefix-lengths", type=int, nargs="+", default=[250, 500, 750, 1000, 1500])
    parser.add_argument("--n-queries", type=int, default=20)
    parser.add_argument("--cache-sweep", action="store_true", help="Run cache size sweep")
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Run main memory analysis
    print("\n" + "#" * 70)
    print("# ICML 2026 - GPU MEMORY USAGE ANALYSIS")
    print(f"# {datetime.now().isoformat()}")
    print("#" * 70)

    result = run_memory_analysis(
        model_name=args.model,
        device=args.device,
        prefix_lengths=args.prefix_lengths,
        n_queries=args.n_queries,
    )

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Baseline peak memory:   {result.baseline_peak_mb:.1f} MB")
    print(f"DeltaCache peak memory: {result.deltacache_peak_mb:.1f} MB")
    print(f"Memory overhead:        {result.memory_overhead_mb:.1f} MB ({result.memory_overhead_pct:.1f}%)")
    print(f"Average speedup:        {result.speedup:.2f}x")
    print(f"Memory efficiency:      {result.memory_efficiency:.2f}x speedup per GB")

    # Save results
    model_short = args.model.split("/")[-1].lower().replace("-", "_")
    output_path = Path(args.output) if args.output else RESULTS_DIR / f"memory_analysis_{model_short}.json"

    with open(output_path, "w") as f:
        json.dump({
            "model_name": result.model_name,
            "device": result.device,
            "timestamp": result.timestamp,
            "summary": {
                "baseline_peak_mb": result.baseline_peak_mb,
                "deltacache_peak_mb": result.deltacache_peak_mb,
                "memory_overhead_mb": result.memory_overhead_mb,
                "memory_overhead_pct": result.memory_overhead_pct,
                "speedup": result.speedup,
                "memory_efficiency": result.memory_efficiency,
            },
            "measurements": [asdict(m) for m in result.measurements],
        }, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Generate figures
    generate_memory_figures(result, RESULTS_DIR / "figures")

    # Optional: cache size sweep
    if args.cache_sweep:
        sweep_result = run_cache_size_sweep(
            model_name=args.model,
            device=args.device,
        )

        sweep_path = RESULTS_DIR / f"cache_size_sweep_{model_short}.json"
        with open(sweep_path, "w") as f:
            json.dump(sweep_result, f, indent=2)
        print(f"\nCache sweep results saved to: {sweep_path}")


if __name__ == "__main__":
    main()
