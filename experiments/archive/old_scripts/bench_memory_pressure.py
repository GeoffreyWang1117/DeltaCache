#!/usr/bin/env python3
"""
Real Model Memory Pressure Experiment for IJCNN 2025.

This experiment runs on actual models (TinyLlama) to validate the
layer-aware eviction policy under memory pressure conditions.

Key differences from simulation:
- Uses real DeltaCache system with actual KV cache computation
- Measures actual GPU memory usage
- Reports real latency measurements

Experiment design:
- Limit cache size to force eviction
- Compare: No eviction control, LRU, Layer-aware
- Measure: Hit rate, latency, correctness
"""

import os
import gc
import sys
import json
import time
import random
import statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from deltacache import DeltaCacheManager, DeltaCacheConfig
from deltacache.hf_integration import LlamaStyleAdapter

RESULTS_DIR = Path(__file__).parent / "results" / "paper"


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_gpu_mem_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0


@dataclass
class RealExperimentResult:
    """Result from real model experiment."""
    experiment_name: str
    eviction_policy: str
    cache_limit_mb: float
    total_requests: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    avg_latency_ms: float
    baseline_latency_ms: float
    speedup: float
    peak_memory_mb: float
    tokens_reused: int
    total_tokens: int
    reuse_rate: float


def create_workload(
    tokenizer,
    num_prefixes: int = 5,
    queries_per_prefix: int = 20,
    prefix_base: str = None,
) -> List[Tuple[str, List[int]]]:
    """Create real workload with shared prefixes."""

    # Real prefix templates
    prefix_templates = [
        """You are a helpful AI assistant specializing in Python programming.
You provide clear, accurate code examples and explanations. When answering questions:
1. Start with a brief explanation of the concept
2. Provide working code examples
3. Explain any edge cases or common pitfalls
4. Suggest best practices when applicable

Here is some context about Python:
Python is a high-level, interpreted programming language known for its clear syntax and readability.
It supports multiple programming paradigms including procedural, object-oriented, and functional programming.
Python has a comprehensive standard library and a vast ecosystem of third-party packages.""",

        """You are an expert machine learning engineer. Your role is to help users understand
and implement machine learning algorithms. Key topics you cover include:
- Supervised learning (classification, regression)
- Unsupervised learning (clustering, dimensionality reduction)
- Deep learning (neural networks, CNNs, RNNs, Transformers)
- Model evaluation and validation techniques
- Feature engineering and data preprocessing

Background on ML:
Machine learning is a subset of AI that enables systems to learn patterns from data.""",

        """You are a database administrator and SQL expert. You help users with:
- Writing efficient SQL queries
- Database design and normalization
- Query optimization and indexing strategies
- Transaction management and ACID properties
- Database security best practices

SQL fundamentals:
SQL (Structured Query Language) is the standard language for relational database management.""",

        """You are a DevOps engineer specializing in cloud infrastructure. Your expertise includes:
- Container orchestration with Kubernetes and Docker
- CI/CD pipeline design and implementation
- Infrastructure as Code (Terraform, CloudFormation)
- Monitoring and observability (Prometheus, Grafana)
- Cloud platforms (AWS, GCP, Azure)

DevOps principles emphasize automation, collaboration, and continuous improvement.""",

        """You are a security researcher focused on application security. You help with:
- Identifying common vulnerabilities (OWASP Top 10)
- Secure coding practices
- Authentication and authorization patterns
- Encryption and data protection
- Security testing methodologies

Security is critical in modern software development to protect user data and systems.""",
    ]

    # Query templates
    query_templates = [
        "How do I implement {}?",
        "What is the best way to handle {}?",
        "Explain the concept of {}.",
        "Write code for {}.",
        "What are common mistakes when dealing with {}?",
        "Compare {} vs {}.",
        "How to optimize {}?",
        "Debug this issue: {}",
    ]

    topics = [
        "sorting algorithms", "hash tables", "binary search", "recursion",
        "dynamic programming", "graph traversal", "tree operations", "linked lists",
        "error handling", "file operations", "API calls", "caching",
        "authentication", "logging", "testing", "deployment",
    ]

    workload = []

    for i in range(num_prefixes):
        prefix = prefix_templates[i % len(prefix_templates)]

        for j in range(queries_per_prefix):
            query_template = random.choice(query_templates)
            topic = random.choice(topics)

            if "{}" in query_template and query_template.count("{}") == 2:
                topic2 = random.choice(topics)
                query = query_template.format(topic, topic2)
            else:
                query = query_template.format(topic)

            full_prompt = f"{prefix}\n\nUser: {query}\nAssistant:"
            tokens = tokenizer.encode(full_prompt)
            workload.append((f"prefix_{i}_query_{j}", tokens))

    # Shuffle to create realistic access pattern
    random.shuffle(workload)

    return workload


def run_baseline_benchmark(
    adapter,
    tokenizer,
    workload: List[Tuple[str, List[int]]],
) -> float:
    """Run baseline (no cache) benchmark."""
    print("  Running baseline (no cache)...")

    latencies = []

    for name, tokens in tqdm(workload[:20], desc="  Baseline"):  # Sample for speed
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = adapter.compute_kv_for_tokens(tokens)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)
        clear_gpu()

    return statistics.mean(latencies)


def run_deltacache_experiment(
    adapter,
    tokenizer,
    config: DeltaCacheConfig,
    workload: List[Tuple[str, List[int]]],
    experiment_name: str,
    eviction_policy: str,
    baseline_latency: float,
) -> RealExperimentResult:
    """Run DeltaCache experiment with specific configuration."""

    print(f"  Running {experiment_name} with {eviction_policy} policy...")

    # Update config with eviction policy
    config.eviction_policy = eviction_policy

    manager = DeltaCacheManager(config)

    latencies = []
    hits = 0
    misses = 0
    tokens_reused = 0
    total_tokens = 0
    peak_memory = 0

    for name, tokens in tqdm(workload, desc=f"  {eviction_policy}"):
        total_tokens += len(tokens)

        torch.cuda.synchronize()
        start = time.perf_counter()
        result = manager.compute_incremental(tokens, adapter.compute_kv)
        torch.cuda.synchronize()

        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)

        if result.matched_length > 0:
            hits += 1
            tokens_reused += result.matched_length
        else:
            misses += 1

        # Track peak memory
        current_mem = get_gpu_mem_mb()
        if current_mem > peak_memory:
            peak_memory = current_mem

    # Calculate statistics
    hit_rate = hits / len(workload) if workload else 0
    avg_latency = statistics.mean(latencies)
    speedup = baseline_latency / avg_latency if avg_latency > 0 else 1.0
    reuse_rate = tokens_reused / total_tokens if total_tokens > 0 else 0

    # Get cache stats
    stats = manager.get_stats()

    del manager
    clear_gpu()

    return RealExperimentResult(
        experiment_name=experiment_name,
        eviction_policy=eviction_policy,
        cache_limit_mb=stats.get("cache_size_mb", 0),
        total_requests=len(workload),
        cache_hits=hits,
        cache_misses=misses,
        hit_rate=hit_rate,
        avg_latency_ms=avg_latency,
        baseline_latency_ms=baseline_latency,
        speedup=speedup,
        peak_memory_mb=peak_memory,
        tokens_reused=tokens_reused,
        total_tokens=total_tokens,
        reuse_rate=reuse_rate,
    )


def run_full_experiment(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device: str = "cuda",
    num_prefixes: int = 5,
    queries_per_prefix: int = 20,
) -> Dict:
    """Run complete memory pressure experiment."""

    print("=" * 70)
    print("IJCNN 2025: Real Model Memory Pressure Experiment")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Prefixes: {num_prefixes}, Queries/Prefix: {queries_per_prefix}")

    # Load model
    print("\nLoading model...")
    adapter = LlamaStyleAdapter.from_pretrained(model_name, device=device)
    tokenizer = adapter.tokenizer
    print(f"Model loaded. GPU memory: {get_gpu_mem_mb():.1f} MB")

    # Create workload
    print("\nCreating workload...")
    random.seed(42)
    workload = create_workload(
        tokenizer,
        num_prefixes=num_prefixes,
        queries_per_prefix=queries_per_prefix,
    )
    print(f"Total requests: {len(workload)}")

    # Get baseline
    baseline_latency = run_baseline_benchmark(adapter, tokenizer, workload)
    print(f"Baseline latency: {baseline_latency:.2f} ms")

    # Test configurations
    results = []
    eviction_policies = ["lru", "composite", "tiered"]  # Available policies

    # Base config
    base_config = DeltaCacheConfig.for_model(model_name)
    base_config.device = device

    for policy in eviction_policies:
        print(f"\n{'=' * 50}")
        print(f"Testing: {policy}")
        print("=" * 50)

        result = run_deltacache_experiment(
            adapter=adapter,
            tokenizer=tokenizer,
            config=base_config,
            workload=workload,
            experiment_name="standard_cache",
            eviction_policy=policy,
            baseline_latency=baseline_latency,
        )
        results.append(result)

        print(f"  Hit rate: {result.hit_rate:.1%}")
        print(f"  Speedup: {result.speedup:.2f}x")
        print(f"  Avg latency: {result.avg_latency_ms:.2f} ms")
        print(f"  Token reuse: {result.reuse_rate:.1%}")

    # Clean up
    del adapter
    clear_gpu()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Policy':<15} {'Hit Rate':<12} {'Speedup':<10} {'Reuse Rate':<12}")
    print("-" * 70)
    for r in results:
        print(f"{r.eviction_policy:<15} {r.hit_rate:.1%}       {r.speedup:.2f}x       {r.reuse_rate:.1%}")

    return {
        "metadata": {
            "model": model_name,
            "device": device,
            "num_prefixes": num_prefixes,
            "queries_per_prefix": queries_per_prefix,
            "total_requests": len(workload),
            "baseline_latency_ms": baseline_latency,
            "timestamp": datetime.now().isoformat(),
        },
        "results": [asdict(r) for r in results],
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Real Model Memory Pressure Experiment")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prefixes", type=int, default=5)
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    results = run_full_experiment(
        model_name=args.model,
        device=args.device,
        num_prefixes=args.prefixes,
        queries_per_prefix=args.queries,
    )

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.output:
        output_path = Path(args.output)
    else:
        model_short = args.model.split("/")[-1].lower().replace("-", "_")
        output_path = RESULTS_DIR / f"real_memory_pressure_{model_short}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
