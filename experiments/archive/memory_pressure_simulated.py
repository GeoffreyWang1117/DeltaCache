#!/usr/bin/env python3
"""
Memory pressure experiments for advanced eviction policies.

This experiment tests how different eviction policies perform when
the cache is under memory pressure and must make eviction decisions.

Key metrics:
- Cache hit rate under pressure
- Quality of eviction decisions (do we keep the right entries?)
- Speedup maintenance under pressure
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from deltacache.core import PrefixTree, CacheBlock, MemoryPool
from deltacache.eviction import (
    LRUEvictionPolicy,
    AttentionAwareEvictionPolicy,
    LayerAwareCachingPolicy,
    HierarchicalEvictionPolicy,
    create_advanced_eviction_policy,
)


@dataclass
class PressureExperimentResult:
    """Result under memory pressure."""
    policy_name: str
    cache_limit_mb: float
    total_requests: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    evictions: int
    avg_latency_ms: float
    speedup_vs_nocache: float


def simulate_workload(
    num_system_prompts: int = 10,
    queries_per_system: int = 20,
    system_prompt_tokens: int = 500,
    query_tokens: int = 50,
    zipf_alpha: float = 1.5,  # Skewness of access pattern
) -> List[Tuple[int, List[int]]]:
    """
    Simulate a realistic workload with Zipf-distributed access patterns.

    Some system prompts are accessed much more frequently than others,
    which is common in production deployments.

    Returns:
        List of (system_prompt_id, query_tokens) tuples
    """
    # Create token sequences
    system_prompts = {}
    for i in range(num_system_prompts):
        # Each system prompt has unique tokens
        system_prompts[i] = list(range(i * 10000, i * 10000 + system_prompt_tokens))

    # Generate workload with Zipf distribution
    workload = []
    total_queries = num_system_prompts * queries_per_system

    # Zipf distribution: P(rank k) ~ 1/k^alpha
    ranks = list(range(1, num_system_prompts + 1))
    probs = [1.0 / (r ** zipf_alpha) for r in ranks]
    total_prob = sum(probs)
    probs = [p / total_prob for p in probs]

    for _ in range(total_queries):
        # Choose system prompt based on Zipf distribution
        r = random.random()
        cumulative = 0.0
        chosen_sys = 0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                chosen_sys = i
                break

        # Generate unique query tokens
        query = list(range(random.randint(100000, 999999), random.randint(100000, 999999) + query_tokens))

        # Full sequence = system + query
        full_tokens = system_prompts[chosen_sys] + query

        workload.append((chosen_sys, full_tokens))

    return workload


def run_pressure_experiment(
    policy_name: str,
    workload: List[Tuple[int, List[int]]],
    cache_limit_mb: float,
    num_layers: int = 22,
    num_heads: int = 32,
    head_dim: int = 64,
) -> PressureExperimentResult:
    """Run experiment with specified cache limit and policy."""

    # Create policy
    if policy_name == "lru":
        from deltacache.eviction.policy import LRUEvictionPolicy
        policy = LRUEvictionPolicy()
    else:
        policy = create_advanced_eviction_policy(policy_name, num_layers=num_layers)

    # Create prefix tree and memory pool
    prefix_tree = PrefixTree()
    memory_pool = MemoryPool(
        gpu_limit=int(cache_limit_mb * 1024 * 1024),
        cpu_limit=int(cache_limit_mb * 1024 * 1024 * 2),  # 2x CPU limit
    )

    # Statistics
    cache_hits = 0
    cache_misses = 0
    evictions = 0
    total_time = 0.0

    # Track which system prompts are in cache
    cached_systems: Dict[int, int] = {}  # system_id -> access_count

    for sys_id, tokens in workload:
        start_time = time.perf_counter()

        # Look up cache
        lookup = prefix_tree.lookup(tokens)

        if lookup.has_match and lookup.kv_cache is not None:
            cache_hits += 1
            # Simulate using cached data (fast)
            time.sleep(0.001)  # 1ms simulated cache hit latency
        else:
            cache_misses += 1
            # Simulate full computation (slow)
            time.sleep(0.01)  # 10ms simulated computation latency

            # Create fake KV cache
            # Size: 2 * num_layers * seq_len * num_heads * head_dim * 2 bytes (fp16)
            seq_len = len(tokens)
            kv_size = 2 * num_layers * seq_len * num_heads * head_dim * 2

            # Check if we need to evict
            while memory_pool.gpu_used + kv_size > memory_pool.gpu_limit:
                candidates = policy.select_victims(
                    prefix_tree, memory_pool, kv_size
                )
                if not candidates:
                    break

                # Evict one candidate
                victim = candidates[0]
                if victim.node.cache_block:
                    block_id = victim.node.cache_block.block_id
                    memory_pool.free(block_id)
                    prefix_tree.remove_cache(block_id)
                    evictions += 1

            # Insert new cache
            if memory_pool.gpu_used + kv_size <= memory_pool.gpu_limit:
                key_cache = torch.zeros(
                    (num_layers, seq_len, num_heads, head_dim),
                    dtype=torch.float16,
                    device="cpu"
                )
                value_cache = torch.zeros_like(key_cache)
                block = CacheBlock(key_cache, value_cache)
                memory_pool.register(block)
                prefix_tree.insert(tokens, block)

        elapsed = time.perf_counter() - start_time
        total_time += elapsed

        # Update access tracking for attention-aware policy
        if hasattr(policy, 'record_attention') and lookup.matched_node:
            fake_attention = torch.rand(min(100, len(tokens)))
            policy.record_attention(id(lookup.matched_node), fake_attention)

        # Track system prompt access
        cached_systems[sys_id] = cached_systems.get(sys_id, 0) + 1

    hit_rate = cache_hits / len(workload)
    avg_latency = (total_time / len(workload)) * 1000

    # Baseline (no cache) would be 10ms per request
    baseline_latency = 10.0
    speedup = baseline_latency / avg_latency

    return PressureExperimentResult(
        policy_name=policy_name,
        cache_limit_mb=cache_limit_mb,
        total_requests=len(workload),
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        hit_rate=hit_rate,
        evictions=evictions,
        avg_latency_ms=avg_latency,
        speedup_vs_nocache=speedup,
    )


def main():
    print("=" * 60)
    print("Memory Pressure Eviction Policy Experiment")
    print("=" * 60)

    # Generate workload
    print("\nGenerating workload...")
    workload = simulate_workload(
        num_system_prompts=20,
        queries_per_system=50,
        system_prompt_tokens=500,
        query_tokens=50,
        zipf_alpha=1.5,
    )
    print(f"  Total requests: {len(workload)}")

    # Test different cache sizes (memory pressure levels)
    cache_sizes_mb = [50, 100, 200, 500]  # Different pressure levels
    policies = ["lru", "attention_aware", "layer_aware", "hierarchical"]

    all_results = []

    for cache_mb in cache_sizes_mb:
        print(f"\n{'=' * 60}")
        print(f"Cache Limit: {cache_mb} MB")
        print("=" * 60)

        for policy in policies:
            print(f"\n  Testing {policy}...")
            result = run_pressure_experiment(
                policy_name=policy,
                workload=workload,
                cache_limit_mb=cache_mb,
            )
            all_results.append(result)

            print(f"    Hit rate: {result.hit_rate:.2%}")
            print(f"    Evictions: {result.evictions}")
            print(f"    Speedup: {result.speedup_vs_nocache:.2f}x")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Hit Rate by Policy and Cache Size")
    print("=" * 80)
    print(f"{'Policy':<20}", end="")
    for size in cache_sizes_mb:
        print(f"{size}MB".center(12), end="")
    print()
    print("-" * 80)

    for policy in policies:
        print(f"{policy:<20}", end="")
        for size in cache_sizes_mb:
            result = next(
                r for r in all_results
                if r.policy_name == policy and r.cache_limit_mb == size
            )
            print(f"{result.hit_rate:.1%}".center(12), end="")
        print()

    print("=" * 80)

    # Save results
    output_path = Path("experiments/results/paper/memory_pressure_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump({
            "cache_sizes_mb": cache_sizes_mb,
            "policies": policies,
            "results": [asdict(r) for r in all_results],
        }, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
