"""Ablation Sensitivity Analysis for ICML 2026.

This script analyzes the sensitivity of DeltaCache to:
1. Prefix length scaling
2. Eviction policy comparison
3. Cache capacity limits
"""

import os
import gc
import json
import time
import statistics
from pathlib import Path
from datetime import datetime
from typing import List, Dict

import torch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from deltacache import DeltaCacheManager, DeltaCacheConfig
from deltacache.hf_integration import LlamaStyleAdapter

RESULTS_DIR = Path(__file__).parent / "results" / "paper"


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_gpu_mem():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 3)
    return 0


# =============================================================================
# Ablation 1: Prefix Length Scaling
# =============================================================================

def ablation_prefix_length(
    adapter,
    tokenizer,
    model_name: str,
    device: str,
    prefix_lengths: List[int] = [100, 250, 500, 750, 1000],
    n_queries: int = 15,
    n_runs: int = 2,
) -> Dict:
    """Test speedup scaling with prefix length."""
    print("\n" + "=" * 70)
    print("ABLATION 1: Prefix Length Scaling")
    print("=" * 70)

    # Long base text for prefix extraction
    base_text = """
    Machine learning is a field of artificial intelligence that uses statistical
    techniques to give computer systems the ability to learn from data without
    being explicitly programmed. The field evolved from pattern recognition and
    computational learning theory. Machine learning explores the study and
    construction of algorithms that can learn from and make predictions on data.
    Deep learning is a subset of machine learning that uses neural networks with
    multiple layers to learn hierarchical representations of data. The transformer
    architecture has revolutionized natural language processing and enables large
    language models to understand and generate human text. These models are trained
    on massive datasets and can perform a wide variety of tasks including translation,
    summarization, question answering, and code generation. The key innovation of
    transformers is the self-attention mechanism which allows the model to weigh
    the importance of different parts of the input when generating each output.
    """ * 20  # Repeat for enough tokens

    results = {"prefix_lengths": {}}

    for target_len in prefix_lengths:
        print(f"\n  Testing prefix length: {target_len} tokens...")

        # Create prefix
        prefix_tokens = tokenizer.encode(base_text)[:target_len]
        prefix_text = tokenizer.decode(prefix_tokens)
        actual_len = len(prefix_tokens)

        # Generate varying queries
        queries = [f"Question {i}: What is the main concept?" for i in range(n_queries)]

        all_dc_latencies = []
        all_hf_latencies = []

        for run in range(n_runs):
            # DeltaCache
            config = DeltaCacheConfig.for_model(model_name)
            config.device = device
            manager = DeltaCacheManager(config)

            for q in queries:
                prompt = f"{prefix_text}\n\n{q}\nAnswer:"
                tokens = tokenizer.encode(prompt)

                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = manager.compute_incremental(tokens, adapter.compute_kv)
                torch.cuda.synchronize()
                all_dc_latencies.append((time.perf_counter() - start) * 1000)

            del manager
            clear_gpu()

            # Baseline
            for q in queries:
                prompt = f"{prefix_text}\n\n{q}\nAnswer:"
                tokens = tokenizer.encode(prompt)

                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = adapter.compute_kv_for_tokens(tokens)
                torch.cuda.synchronize()
                all_hf_latencies.append((time.perf_counter() - start) * 1000)

            clear_gpu()

        dc_mean = statistics.mean(all_dc_latencies)
        hf_mean = statistics.mean(all_hf_latencies)
        speedup = hf_mean / dc_mean if dc_mean > 0 else 0

        # Cached speedup (exclude first query of each run)
        dc_cached = [all_dc_latencies[i] for i in range(len(all_dc_latencies)) if i % n_queries != 0]
        hf_cached = [all_hf_latencies[i] for i in range(len(all_hf_latencies)) if i % n_queries != 0]
        cached_speedup = statistics.mean(hf_cached) / statistics.mean(dc_cached) if dc_cached else 0

        results["prefix_lengths"][actual_len] = {
            "speedup": speedup,
            "cached_speedup": cached_speedup,
            "deltacache_ms": dc_mean,
            "baseline_ms": hf_mean,
        }

        print(f"    Actual: {actual_len} tokens")
        print(f"    Speedup: {speedup:.2f}x (cached: {cached_speedup:.2f}x)")

    return results


# =============================================================================
# Ablation 2: Eviction Policy Comparison
# =============================================================================

def ablation_eviction_policies(
    adapter,
    tokenizer,
    model_name: str,
    device: str,
    n_queries: int = 20,
    n_runs: int = 2,
) -> Dict:
    """Compare different eviction policies."""
    print("\n" + "=" * 70)
    print("ABLATION 2: Eviction Policy Comparison")
    print("=" * 70)

    policies = ["lru", "lfu", "composite", "tiered", "adaptive"]

    # Fixed prefix for fair comparison
    prefix_text = """
    This is a comprehensive knowledge base document about machine learning and
    artificial intelligence. It covers topics including supervised learning,
    unsupervised learning, reinforcement learning, neural networks, deep learning,
    transformer architectures, and large language models. The document also discusses
    practical applications in computer vision, natural language processing, and
    recommendation systems.
    """ * 5  # ~300 tokens

    queries = [f"Query {i}: Explain concept {i}." for i in range(n_queries)]

    results = {"policies": {}}

    for policy in policies:
        print(f"\n  Testing policy: {policy}...")

        all_dc_latencies = []
        all_hf_latencies = []

        for run in range(n_runs):
            config = DeltaCacheConfig.for_model(model_name)
            config.device = device
            config.eviction_policy = policy

            manager = DeltaCacheManager(config)

            for q in queries:
                prompt = f"{prefix_text}\n\n{q}\nAnswer:"
                tokens = tokenizer.encode(prompt)

                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = manager.compute_incremental(tokens, adapter.compute_kv)
                torch.cuda.synchronize()
                all_dc_latencies.append((time.perf_counter() - start) * 1000)

            del manager
            clear_gpu()

            # Baseline (only need once, same for all policies)
            if policy == policies[0]:
                for q in queries:
                    prompt = f"{prefix_text}\n\n{q}\nAnswer:"
                    tokens = tokenizer.encode(prompt)

                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    _ = adapter.compute_kv_for_tokens(tokens)
                    torch.cuda.synchronize()
                    all_hf_latencies.append((time.perf_counter() - start) * 1000)

                clear_gpu()

        dc_mean = statistics.mean(all_dc_latencies)

        # Use stored baseline if available
        if policy == policies[0]:
            hf_mean = statistics.mean(all_hf_latencies)
            results["baseline_ms"] = hf_mean
        else:
            hf_mean = results["baseline_ms"]

        speedup = hf_mean / dc_mean if dc_mean > 0 else 0

        results["policies"][policy] = {
            "speedup": speedup,
            "latency_ms": dc_mean,
        }

        print(f"    Speedup: {speedup:.2f}x")

    return results


# =============================================================================
# Ablation 3: Query Count Scaling
# =============================================================================

def ablation_query_count(
    adapter,
    tokenizer,
    model_name: str,
    device: str,
    query_counts: List[int] = [5, 10, 20, 50, 100],
    n_runs: int = 2,
) -> Dict:
    """Test how speedup scales with number of queries on same prefix."""
    print("\n" + "=" * 70)
    print("ABLATION 3: Query Count Scaling (Cache Warm-up)")
    print("=" * 70)

    # Fixed long prefix
    prefix_text = """
    Comprehensive guide to software engineering best practices including design
    patterns, clean code principles, testing strategies, continuous integration,
    deployment pipelines, and system architecture. Topics covered include SOLID
    principles, microservices, API design, database optimization, and security.
    """ * 8  # ~500 tokens

    results = {"query_counts": {}}

    for n_queries in query_counts:
        print(f"\n  Testing {n_queries} queries...")

        queries = [f"Q{i}: Explain topic {i % 10}." for i in range(n_queries)]

        all_dc_latencies = []
        all_hf_latencies = []

        for run in range(n_runs):
            config = DeltaCacheConfig.for_model(model_name)
            config.device = device
            manager = DeltaCacheManager(config)

            for q in queries:
                prompt = f"{prefix_text}\n\n{q}\nAnswer:"
                tokens = tokenizer.encode(prompt)

                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = manager.compute_incremental(tokens, adapter.compute_kv)
                torch.cuda.synchronize()
                all_dc_latencies.append((time.perf_counter() - start) * 1000)

            del manager
            clear_gpu()

            for q in queries:
                prompt = f"{prefix_text}\n\n{q}\nAnswer:"
                tokens = tokenizer.encode(prompt)

                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = adapter.compute_kv_for_tokens(tokens)
                torch.cuda.synchronize()
                all_hf_latencies.append((time.perf_counter() - start) * 1000)

            clear_gpu()

        dc_mean = statistics.mean(all_dc_latencies)
        hf_mean = statistics.mean(all_hf_latencies)
        speedup = hf_mean / dc_mean if dc_mean > 0 else 0

        # Cached queries (after first)
        dc_cached = all_dc_latencies[1::n_queries]
        hf_cached = all_hf_latencies[1::n_queries]
        cached_speedup = statistics.mean(hf_cached) / statistics.mean(dc_cached) if dc_cached else 0

        results["query_counts"][n_queries] = {
            "overall_speedup": speedup,
            "cached_speedup": cached_speedup,
            "deltacache_ms": dc_mean,
            "baseline_ms": hf_mean,
        }

        print(f"    Overall: {speedup:.2f}x, Cached: {cached_speedup:.2f}x")

    return results


# =============================================================================
# Main
# =============================================================================

def run_all_ablations(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device: str = "cuda",
    n_runs: int = 2,
) -> Dict:
    """Run all ablation studies."""
    print("\n" + "#" * 70)
    print("# ICML 2026 ABLATION SENSITIVITY ANALYSIS")
    print(f"# Model: {model_name}")
    print(f"# Timestamp: {datetime.now().isoformat()}")
    print("#" * 70)

    # Load model
    print("\nLoading model...")
    adapter = LlamaStyleAdapter.from_pretrained(model_name, device=device)
    tokenizer = adapter.tokenizer
    print(f"Model loaded. GPU memory: {get_gpu_mem():.2f} GB")

    results = {
        "metadata": {
            "model": model_name,
            "device": device,
            "n_runs": n_runs,
            "timestamp": datetime.now().isoformat(),
        },
        "ablations": {}
    }

    try:
        # Ablation 1: Prefix Length Scaling
        results["ablations"]["prefix_length"] = ablation_prefix_length(
            adapter, tokenizer, model_name, device,
            prefix_lengths=[100, 250, 500, 750, 1000],
            n_queries=15,
            n_runs=n_runs,
        )

        # Ablation 2: Eviction Policy Comparison
        results["ablations"]["eviction_policies"] = ablation_eviction_policies(
            adapter, tokenizer, model_name, device,
            n_queries=20,
            n_runs=n_runs,
        )

        # Ablation 3: Query Count Scaling
        results["ablations"]["query_count"] = ablation_query_count(
            adapter, tokenizer, model_name, device,
            query_counts=[5, 10, 20, 50],
            n_runs=n_runs,
        )

    finally:
        del adapter
        clear_gpu()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if "prefix_length" in results["ablations"]:
        print("\nPrefix Length Scaling:")
        for plen, data in results["ablations"]["prefix_length"]["prefix_lengths"].items():
            print(f"  {plen} tokens: {data['speedup']:.2f}x (cached: {data['cached_speedup']:.2f}x)")

    if "eviction_policies" in results["ablations"]:
        print("\nEviction Policies:")
        for policy, data in results["ablations"]["eviction_policies"]["policies"].items():
            print(f"  {policy}: {data['speedup']:.2f}x")

    if "query_count" in results["ablations"]:
        print("\nQuery Count Scaling:")
        for count, data in results["ablations"]["query_count"]["query_counts"].items():
            print(f"  {count} queries: {data['overall_speedup']:.2f}x (cached: {data['cached_speedup']:.2f}x)")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Ablation Sensitivity Analysis")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-runs", type=int, default=2)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    results = run_all_ablations(
        model_name=args.model,
        device=args.device,
        n_runs=args.n_runs,
    )

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.output:
        output_path = Path(args.output)
    else:
        model_short = args.model.split("/")[-1].lower().replace("-", "_")
        output_path = RESULTS_DIR / f"ablation_sensitivity_{model_short}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
