#!/usr/bin/env python3
"""
Experiments for advanced eviction policies.

This script compares the performance of:
1. Baseline LRU eviction
2. Attention-Aware eviction
3. Layer-Aware eviction
4. Hierarchical eviction (combined)

Metrics:
- Cache hit rate
- Speedup vs baseline
- Memory efficiency
- TTFT improvement
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from deltacache.core import PrefixTree, CacheBlock, HierarchicalMemoryManager
from deltacache.eviction import (
    LRUEvictionPolicy,
    AttentionAwareEvictionPolicy,
    LayerAwareCachingPolicy,
    HierarchicalEvictionPolicy,
    create_eviction_policy,
    create_advanced_eviction_policy,
)
from deltacache.engine.incremental import IncrementalEngine
from deltacache.hf_integration.llama_adapter import LlamaStyleAdapter


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""
    policy_name: str
    cache_hit_rate: float
    avg_latency_ms: float
    speedup: float
    memory_used_mb: float
    ttft_ms: float
    num_requests: int
    num_cache_hits: int
    num_evictions: int


def create_test_prompts(
    num_prompts: int = 100,
    system_prompt_length: int = 500,
    query_length: int = 50,
    num_system_prompts: int = 5,
) -> List[str]:
    """Create test prompts with shared prefixes."""
    # Create system prompts
    system_prompts = []
    for i in range(num_system_prompts):
        prompt = f"System prompt {i}: " + " ".join(
            [f"word{j}" for j in range(system_prompt_length // 2)]
        )
        system_prompts.append(prompt)

    # Create queries with system prompts
    prompts = []
    for i in range(num_prompts):
        sys_idx = i % num_system_prompts
        query = f"Query {i}: " + " ".join(
            [f"q{j}" for j in range(query_length // 2)]
        )
        prompts.append(system_prompts[sys_idx] + "\n" + query)

    return prompts


def run_baseline_experiment(
    model,
    tokenizer,
    prompts: List[str],
    device: torch.device,
) -> Tuple[float, float]:
    """Run baseline (no caching) experiment."""
    total_time = 0.0
    ttft_total = 0.0

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        torch.cuda.synchronize() if device.type == "cuda" else None
        start = time.perf_counter()

        with torch.no_grad():
            outputs = model(**inputs, use_cache=True)

        torch.cuda.synchronize() if device.type == "cuda" else None
        elapsed = time.perf_counter() - start

        total_time += elapsed
        ttft_total += elapsed

    avg_latency = (total_time / len(prompts)) * 1000  # ms
    avg_ttft = (ttft_total / len(prompts)) * 1000  # ms

    return avg_latency, avg_ttft


def run_cached_experiment(
    model,
    tokenizer,
    prompts: List[str],
    device: torch.device,
    policy_name: str,
    num_layers: int,
) -> ExperimentResult:
    """Run experiment with caching and specified eviction policy."""

    # Create eviction policy
    if policy_name == "lru":
        policy = create_eviction_policy("lru")
    elif policy_name == "attention_aware":
        policy = create_advanced_eviction_policy(
            "attention_aware",
            num_layers=num_layers,
            attention_weight=0.4,
            frequency_weight=0.3,
            recency_weight=0.3,
        )
    elif policy_name == "layer_aware":
        policy = create_advanced_eviction_policy(
            "layer_aware",
            num_layers=num_layers,
        )
    elif policy_name == "hierarchical":
        policy = create_advanced_eviction_policy(
            "hierarchical",
            num_layers=num_layers,
        )
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    # Create prefix tree and memory manager
    prefix_tree = PrefixTree()
    memory_manager = HierarchicalMemoryManager(
        gpu_limit=int(4 * 1024**3),  # 4GB
        cpu_limit=int(8 * 1024**3),  # 8GB
        device=device,
        enable_async=False,  # Sync for accurate timing
    )

    # Create incremental engine
    config = model.config
    engine = IncrementalEngine(
        prefix_tree=prefix_tree,
        num_layers=config.num_hidden_layers,
        num_heads=config.num_attention_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        device=device,
    )

    # Create model adapter
    adapter = LlamaStyleAdapter(model, tokenizer)

    # Run experiment
    total_time = 0.0
    ttft_total = 0.0
    cache_hits = 0
    num_evictions = 0

    for prompt in prompts:
        tokens = tokenizer.encode(prompt)

        torch.cuda.synchronize() if device.type == "cuda" else None
        start = time.perf_counter()

        # Look up cache
        lookup = prefix_tree.lookup(tokens)

        ttft_start = time.perf_counter()

        if lookup.has_match and lookup.kv_cache is not None:
            cache_hits += 1
            # Use cached KV
            cached_key, cached_value = lookup.kv_cache
            suffix_tokens = tokens[lookup.matched_length:]

            if suffix_tokens:
                # Compute only suffix
                with torch.no_grad():
                    result = engine.compute(
                        tokens,
                        compute_fn=adapter.compute_kv,
                        store_result=True,
                    )
        else:
            # Full computation
            with torch.no_grad():
                result = engine.compute(
                    tokens,
                    compute_fn=adapter.compute_kv,
                    store_result=True,
                )

        torch.cuda.synchronize() if device.type == "cuda" else None
        ttft_elapsed = time.perf_counter() - ttft_start
        total_elapsed = time.perf_counter() - start

        total_time += total_elapsed
        ttft_total += ttft_elapsed

        # Track memory and trigger eviction if needed
        memory_manager.record_access(id(lookup.matched_node) if lookup.matched_node else 0)

        # Simulate attention recording for attention-aware policy
        if hasattr(policy, 'record_attention') and lookup.matched_node:
            # Simulate attention scores (in real use, these come from model)
            fake_attention = torch.rand(100)
            policy.record_attention(id(lookup.matched_node), fake_attention)

        # Periodic maintenance
        if len(prompts) > 0 and (prompts.index(prompt) + 1) % 10 == 0:
            result = memory_manager.maintain()
            num_evictions += result.get("offloaded", 0) + result.get("deleted", 0)

    avg_latency = (total_time / len(prompts)) * 1000  # ms
    avg_ttft = (ttft_total / len(prompts)) * 1000  # ms
    cache_hit_rate = cache_hits / len(prompts)

    # Get memory stats
    stats = memory_manager.get_stats()

    return ExperimentResult(
        policy_name=policy_name,
        cache_hit_rate=cache_hit_rate,
        avg_latency_ms=avg_latency,
        speedup=0.0,  # Will be calculated later
        memory_used_mb=stats.used_bytes / (1024**2),
        ttft_ms=avg_ttft,
        num_requests=len(prompts),
        num_cache_hits=cache_hits,
        num_evictions=num_evictions,
    )


def main():
    parser = argparse.ArgumentParser(description="Advanced eviction policy experiments")
    parser.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Model to use",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=50,
        help="Number of prompts to test",
    )
    parser.add_argument(
        "--system-prompt-length",
        type=int,
        default=500,
        help="Length of system prompts in tokens",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/results/paper/advanced_policies_results.json",
        help="Output file",
    )
    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    num_layers = model.config.num_hidden_layers
    print(f"Model has {num_layers} layers")

    # Create test prompts
    print(f"Creating {args.num_prompts} test prompts...")
    prompts = create_test_prompts(
        num_prompts=args.num_prompts,
        system_prompt_length=args.system_prompt_length,
    )

    # Run baseline
    print("\nRunning baseline experiment (no caching)...")
    baseline_latency, baseline_ttft = run_baseline_experiment(
        model, tokenizer, prompts, device
    )
    print(f"  Baseline latency: {baseline_latency:.2f} ms")
    print(f"  Baseline TTFT: {baseline_ttft:.2f} ms")

    # Run experiments with different policies
    policies = ["lru", "attention_aware", "layer_aware", "hierarchical"]
    results = []

    for policy_name in policies:
        print(f"\nRunning experiment with {policy_name} policy...")
        try:
            result = run_cached_experiment(
                model, tokenizer, prompts, device, policy_name, num_layers
            )
            result.speedup = baseline_latency / result.avg_latency_ms
            results.append(result)

            print(f"  Cache hit rate: {result.cache_hit_rate:.2%}")
            print(f"  Avg latency: {result.avg_latency_ms:.2f} ms")
            print(f"  Speedup: {result.speedup:.2f}x")
            print(f"  TTFT: {result.ttft_ms:.2f} ms")
            print(f"  Evictions: {result.num_evictions}")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "model": args.model,
        "num_prompts": args.num_prompts,
        "system_prompt_length": args.system_prompt_length,
        "baseline_latency_ms": baseline_latency,
        "baseline_ttft_ms": baseline_ttft,
        "results": [asdict(r) for r in results],
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Summary table
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"{'Policy':<20} {'Hit Rate':<12} {'Speedup':<10} {'TTFT (ms)':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r.policy_name:<20} {r.cache_hit_rate:.2%}      {r.speedup:.2f}x       {r.ttft_ms:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
