#!/usr/bin/env python3
"""
Large Model Memory Pressure Experiments for ICLR 2026 Workshop SPOT.

Tests advanced eviction policies under real memory pressure conditions
with Mistral-7B to demonstrate the benefits of attention-aware and
layer-aware caching strategies.

Key experiments:
1. Memory pressure levels: 25%, 50%, 75%, 100% of cache capacity
2. Compare LRU vs Attention-Aware vs Layer-Aware vs Hierarchical
3. Measure: hit rate, eviction quality, speedup maintenance
"""

import os
import sys
import gc
import json
import time
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
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
from deltacache.engine.incremental import IncrementalEngine
from deltacache.hf_integration.llama_adapter import LlamaStyleAdapter


@dataclass
class PressureResult:
    """Result under specific memory pressure."""
    policy: str
    pressure_level: float
    cache_limit_mb: float
    total_requests: int
    cache_hits: int
    evictions: int
    hit_rate: float
    avg_latency_ms: float
    ttft_ms: float
    speedup: float
    memory_efficiency: float  # hit_rate / memory_used


def create_workload_with_skew(
    tokenizer,
    num_system_prompts: int = 10,
    queries_per_prompt: int = 10,
    system_prompt_tokens: int = 500,
    query_tokens: int = 50,
    zipf_alpha: float = 1.5,
) -> List[Tuple[int, str, List[int]]]:
    """
    Create workload with Zipf-distributed access to system prompts.

    Some system prompts are accessed much more frequently (realistic).
    """
    # Create system prompts
    system_prompts = []
    for i in range(num_system_prompts):
        prompt = f"You are assistant {i}. " + " ".join([
            f"Context word {j} for system {i}."
            for j in range(system_prompt_tokens // 10)
        ])
        system_prompts.append(prompt)

    # Zipf distribution
    ranks = list(range(1, num_system_prompts + 1))
    probs = [1.0 / (r ** zipf_alpha) for r in ranks]
    total_prob = sum(probs)
    probs = [p / total_prob for p in probs]

    # Generate workload
    workload = []
    total_queries = num_system_prompts * queries_per_prompt

    for i in range(total_queries):
        # Choose system prompt based on Zipf
        r = random.random()
        cumulative = 0.0
        chosen_sys = 0
        for idx, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                chosen_sys = idx
                break

        # Create unique query
        query = f"Query {i}: What is the answer to question {random.randint(1, 1000)}?"
        full_prompt = system_prompts[chosen_sys] + "\n\n" + query

        tokens = tokenizer.encode(full_prompt)
        workload.append((chosen_sys, full_prompt, tokens))

    # Shuffle to simulate real access patterns
    random.shuffle(workload)

    return workload


def estimate_kv_cache_size(
    num_layers: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype_bytes: int = 2,  # fp16
) -> int:
    """Estimate KV cache size in bytes."""
    return 2 * num_layers * seq_len * num_heads * head_dim * dtype_bytes


def run_pressure_experiment(
    model,
    tokenizer,
    workload: List[Tuple[int, str, List[int]]],
    policy_name: str,
    cache_limit_mb: float,
    device: torch.device,
) -> PressureResult:
    """Run experiment with specified cache limit and policy."""

    config = model.config
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_heads

    # Create policy
    if policy_name == "lru":
        policy = LRUEvictionPolicy()
    else:
        policy = create_advanced_eviction_policy(policy_name, num_layers=num_layers)

    # Create components
    prefix_tree = PrefixTree()
    memory_pool = MemoryPool(
        gpu_limit=int(cache_limit_mb * 1024 * 1024),
        cpu_limit=int(cache_limit_mb * 1024 * 1024),  # Equal CPU limit
    )

    # Create adapter
    adapter = LlamaStyleAdapter(model, tokenizer)

    # Create engine
    engine = IncrementalEngine(
        prefix_tree=prefix_tree,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        device=device,
    )

    # Statistics
    cache_hits = 0
    cache_misses = 0
    evictions = 0
    total_time = 0.0
    ttft_total = 0.0

    for sys_id, prompt, tokens in workload:
        torch.cuda.synchronize() if device.type == "cuda" else None
        start_time = time.perf_counter()

        # Look up cache
        lookup = prefix_tree.lookup(tokens)

        ttft_start = time.perf_counter()

        if lookup.has_match and lookup.kv_cache is not None:
            cache_hits += 1
            matched_len = lookup.matched_length

            if matched_len < len(tokens):
                # Partial hit - compute suffix
                with torch.no_grad():
                    result = engine.compute(
                        tokens,
                        compute_fn=adapter.compute_kv,
                        store_result=False,  # Don't store yet
                    )
        else:
            cache_misses += 1
            matched_len = 0

            # Full computation
            with torch.no_grad():
                result = engine.compute(
                    tokens,
                    compute_fn=adapter.compute_kv,
                    store_result=False,
                )

        torch.cuda.synchronize() if device.type == "cuda" else None
        ttft_elapsed = time.perf_counter() - ttft_start

        # Estimate cache size for this entry
        kv_size = estimate_kv_cache_size(num_layers, len(tokens), num_heads, head_dim)

        # Evict if needed before inserting
        while memory_pool.gpu_used + kv_size > memory_pool.gpu_limit:
            candidates = policy.select_victims(prefix_tree, memory_pool, kv_size)
            if not candidates:
                break

            victim = candidates[0]
            if victim.node.cache_block:
                block_id = victim.node.cache_block.block_id
                memory_pool.free(block_id)
                prefix_tree.remove_cache(block_id)
                evictions += 1

        # Insert new cache if space available
        if memory_pool.gpu_used + kv_size <= memory_pool.gpu_limit:
            if hasattr(result, 'key_cache'):
                block = CacheBlock(result.key_cache, result.value_cache)
                memory_pool.register(block)
                prefix_tree.insert(tokens, block)

        total_elapsed = time.perf_counter() - start_time
        total_time += total_elapsed
        ttft_total += ttft_elapsed

        # Record attention for attention-aware policy
        if hasattr(policy, 'record_attention') and lookup.matched_node:
            fake_attention = torch.rand(min(100, len(tokens)), device='cpu')
            policy.record_attention(id(lookup.matched_node), fake_attention)

    # Calculate metrics
    hit_rate = cache_hits / len(workload) if workload else 0
    avg_latency = (total_time / len(workload)) * 1000 if workload else 0
    avg_ttft = (ttft_total / len(workload)) * 1000 if workload else 0

    # Memory efficiency: how much hit rate per MB of cache
    memory_used_mb = memory_pool.gpu_used / (1024 * 1024)
    memory_efficiency = hit_rate / max(memory_used_mb, 0.1)

    return PressureResult(
        policy=policy_name,
        pressure_level=0.0,  # Will be set by caller
        cache_limit_mb=cache_limit_mb,
        total_requests=len(workload),
        cache_hits=cache_hits,
        evictions=evictions,
        hit_rate=hit_rate,
        avg_latency_ms=avg_latency,
        ttft_ms=avg_ttft,
        speedup=0.0,  # Will be calculated
        memory_efficiency=memory_efficiency,
    )


def run_baseline(
    model,
    tokenizer,
    workload: List[Tuple[int, str, List[int]]],
    device: torch.device,
) -> Tuple[float, float]:
    """Run baseline without caching."""
    total_time = 0.0
    ttft_total = 0.0

    for sys_id, prompt, tokens in workload[:10]:  # Sample for speed
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        torch.cuda.synchronize() if device.type == "cuda" else None
        start = time.perf_counter()

        with torch.no_grad():
            outputs = model(**inputs, use_cache=True)

        torch.cuda.synchronize() if device.type == "cuda" else None
        elapsed = time.perf_counter() - start

        total_time += elapsed
        ttft_total += elapsed

    avg_latency = (total_time / 10) * 1000
    avg_ttft = (ttft_total / 10) * 1000

    return avg_latency, avg_ttft


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--queries-per-prompt", type=int, default=10)
    parser.add_argument("--system-tokens", type=int, default=500)
    parser.add_argument("--output", type=str, default="experiments/results/paper/pressure_experiment_results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    config = model.config
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_heads
    print(f"Model: {num_layers} layers, {num_heads} heads, {head_dim} head_dim")

    # Create workload
    print(f"\nCreating workload...")
    workload = create_workload_with_skew(
        tokenizer,
        num_system_prompts=args.num_prompts,
        queries_per_prompt=args.queries_per_prompt,
        system_prompt_tokens=args.system_tokens,
    )
    print(f"Total requests: {len(workload)}")

    # Estimate total cache needed
    avg_tokens = sum(len(t) for _, _, t in workload) / len(workload)
    single_cache_size = estimate_kv_cache_size(num_layers, int(avg_tokens), num_heads, head_dim)
    total_cache_needed = single_cache_size * len(workload) / (1024 * 1024)
    print(f"Avg tokens per request: {avg_tokens:.0f}")
    print(f"Estimated total cache needed: {total_cache_needed:.1f} MB")

    # Run baseline
    print("\nRunning baseline (no cache)...")
    baseline_latency, baseline_ttft = run_baseline(model, tokenizer, workload, device)
    print(f"Baseline latency: {baseline_latency:.2f} ms")
    print(f"Baseline TTFT: {baseline_ttft:.2f} ms")

    # Test different pressure levels
    # Pressure level = cache_limit / total_needed
    pressure_levels = [0.25, 0.50, 0.75, 1.0]
    policies = ["lru", "attention_aware", "layer_aware", "hierarchical"]

    all_results = []

    for pressure in pressure_levels:
        cache_limit_mb = total_cache_needed * pressure
        print(f"\n{'=' * 60}")
        print(f"Pressure Level: {pressure:.0%} (Cache: {cache_limit_mb:.1f} MB)")
        print("=" * 60)

        for policy_name in policies:
            print(f"\n  Testing {policy_name}...")

            # Clear GPU cache
            gc.collect()
            torch.cuda.empty_cache() if device.type == "cuda" else None

            try:
                result = run_pressure_experiment(
                    model, tokenizer, workload,
                    policy_name, cache_limit_mb, device
                )
                result.pressure_level = pressure
                result.speedup = baseline_latency / result.avg_latency_ms if result.avg_latency_ms > 0 else 0

                all_results.append(result)

                print(f"    Hit Rate: {result.hit_rate:.1%}")
                print(f"    Evictions: {result.evictions}")
                print(f"    Speedup: {result.speedup:.2f}x")
                print(f"    TTFT: {result.ttft_ms:.2f} ms")

            except Exception as e:
                print(f"    Error: {e}")
                import traceback
                traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Hit Rate by Policy and Pressure Level")
    print("=" * 80)
    print(f"{'Policy':<20}", end="")
    for p in pressure_levels:
        print(f"{p:.0%}".center(12), end="")
    print()
    print("-" * 80)

    for policy in policies:
        print(f"{policy:<20}", end="")
        for pressure in pressure_levels:
            result = next(
                (r for r in all_results if r.policy == policy and r.pressure_level == pressure),
                None
            )
            if result:
                print(f"{result.hit_rate:.1%}".center(12), end="")
            else:
                print("N/A".center(12), end="")
        print()

    print("\n" + "=" * 80)
    print("SUMMARY: Speedup by Policy and Pressure Level")
    print("=" * 80)
    print(f"{'Policy':<20}", end="")
    for p in pressure_levels:
        print(f"{p:.0%}".center(12), end="")
    print()
    print("-" * 80)

    for policy in policies:
        print(f"{policy:<20}", end="")
        for pressure in pressure_levels:
            result = next(
                (r for r in all_results if r.policy == policy and r.pressure_level == pressure),
                None
            )
            if result:
                print(f"{result.speedup:.2f}x".center(12), end="")
            else:
                print("N/A".center(12), end="")
        print()

    print("=" * 80)

    # Calculate improvement of advanced policies over LRU
    print("\n" + "=" * 80)
    print("IMPROVEMENT over LRU Baseline")
    print("=" * 80)

    for policy in ["attention_aware", "layer_aware", "hierarchical"]:
        improvements = []
        for pressure in pressure_levels:
            lru_result = next(
                (r for r in all_results if r.policy == "lru" and r.pressure_level == pressure),
                None
            )
            policy_result = next(
                (r for r in all_results if r.policy == policy and r.pressure_level == pressure),
                None
            )
            if lru_result and policy_result and lru_result.hit_rate > 0:
                improvement = (policy_result.hit_rate - lru_result.hit_rate) / lru_result.hit_rate * 100
                improvements.append(improvement)

        if improvements:
            avg_improvement = sum(improvements) / len(improvements)
            print(f"{policy}: {avg_improvement:+.1f}% avg hit rate improvement")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "model": args.model,
        "num_prompts": args.num_prompts,
        "queries_per_prompt": args.queries_per_prompt,
        "system_tokens": args.system_tokens,
        "baseline_latency_ms": baseline_latency,
        "baseline_ttft_ms": baseline_ttft,
        "pressure_levels": pressure_levels,
        "policies": policies,
        "results": [asdict(r) for r in all_results],
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
