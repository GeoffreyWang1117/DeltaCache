#!/usr/bin/env python3
"""IJCNN 2025: Memory Pressure Eviction Policy Comparison.

This experiment tests how different eviction policies perform when
the cache is under memory pressure. This is critical for demonstrating
the advantage of layer-aware eviction over simpler policies.

Key hypothesis: Layer-aware policies should outperform LRU/LFU when
cache capacity is limited and eviction decisions matter.

Experiment design:
- Cache capacities: 100%, 50%, 30%, 10% of working set size
- Policies: LRU, LFU, Attention-only, Layer+Attention
- Metrics: Hit rate, speedup, eviction quality
"""

import os
import gc
import sys
import json
import time
import math
import random
import statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

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


@dataclass
class EvictionResult:
    """Result of memory pressure experiment."""
    policy_name: str
    capacity_ratio: float
    total_requests: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    evictions: int
    avg_latency_ms: float
    speedup_vs_nocache: float
    tokens_saved: int


class SimulatedCacheEntry:
    """Simulated cache entry for eviction experiments."""
    def __init__(self, tokens: List[int], size_bytes: int, layer_weights: List[float]):
        self.tokens = tokens
        self.size_bytes = size_bytes
        self.access_count = 1
        self.last_access = time.time()
        self.attention_score = random.random()  # Simulated attention
        self.layer_weights = layer_weights
        self.layer_importance = sum(layer_weights) / len(layer_weights)


class SimulatedCache:
    """Simulated cache with configurable eviction policies."""

    def __init__(
        self,
        capacity_bytes: int,
        policy: str,
        num_layers: int = 22,
        k: float = 5.0,  # Layer weight steepness
        tau: float = 0.3,  # Layer weight transition point
    ):
        self.capacity_bytes = capacity_bytes
        self.policy = policy
        self.num_layers = num_layers
        self.k = k
        self.tau = tau

        self.entries: Dict[tuple, SimulatedCacheEntry] = {}
        self.current_size = 0
        self.eviction_count = 0

        # Precompute layer weights
        self.layer_weights = self._compute_layer_weights()

    def _compute_layer_weights(self) -> List[float]:
        """Compute sigmoid-based layer weights."""
        weights = []
        for i in range(self.num_layers):
            normalized_pos = i / (self.num_layers - 1) if self.num_layers > 1 else 0.5
            weight = 1.0 / (1.0 + math.exp(-self.k * (normalized_pos - self.tau)))
            weights.append(weight)
        return weights

    def _compute_eviction_score(self, entry: SimulatedCacheEntry) -> float:
        """Compute eviction priority score (lower = evict first)."""
        now = time.time()
        time_since_access = max(0.001, now - entry.last_access)

        if self.policy == "lru":
            # LRU: Only consider recency
            return entry.last_access

        elif self.policy == "lfu":
            # LFU: Only consider frequency
            return entry.access_count

        elif self.policy == "attention":
            # Attention-only: Use attention score + frequency
            recency_score = 1.0 / (1.0 + math.log1p(time_since_access))
            freq_score = math.log1p(entry.access_count)
            return entry.attention_score * 0.4 + freq_score * 0.3 + recency_score * 0.3

        elif self.policy == "layer_attention":
            # Layer + Attention: Full layer-aware scoring
            recency_score = 1.0 / (1.0 + math.log1p(time_since_access))
            freq_score = math.log1p(entry.access_count)
            base_score = entry.attention_score * 0.3 + freq_score * 0.3 + recency_score * 0.4
            return base_score * entry.layer_importance

        else:
            raise ValueError(f"Unknown policy: {self.policy}")

    def _evict_until_space(self, required_space: int) -> int:
        """Evict entries until we have enough space."""
        evictions = 0
        while self.current_size + required_space > self.capacity_bytes and self.entries:
            # Find victim
            victims = sorted(
                self.entries.items(),
                key=lambda x: self._compute_eviction_score(x[1])
            )

            if not victims:
                break

            victim_key, victim_entry = victims[0]
            self.current_size -= victim_entry.size_bytes
            del self.entries[victim_key]
            evictions += 1
            self.eviction_count += 1

        return evictions

    def lookup(self, tokens: List[int]) -> Tuple[bool, int]:
        """Look up tokens in cache. Returns (hit, matched_length)."""
        token_tuple = tuple(tokens)

        # Check for exact match
        if token_tuple in self.entries:
            entry = self.entries[token_tuple]
            entry.access_count += 1
            entry.last_access = time.time()
            return True, len(tokens)

        # Check for prefix matches
        best_match_len = 0
        best_match_key = None

        for key, entry in self.entries.items():
            # Check if key is a prefix of tokens
            key_len = len(key)
            if key_len <= len(tokens) and tokens[:key_len] == list(key):
                if key_len > best_match_len:
                    best_match_len = key_len
                    best_match_key = key

        if best_match_key is not None:
            entry = self.entries[best_match_key]
            entry.access_count += 1
            entry.last_access = time.time()
            return True, best_match_len

        return False, 0

    def insert(self, tokens: List[int], size_bytes: int):
        """Insert tokens into cache."""
        token_tuple = tuple(tokens)

        if token_tuple in self.entries:
            return  # Already exists

        # Evict if necessary
        self._evict_until_space(size_bytes)

        # Check if we can fit
        if self.current_size + size_bytes > self.capacity_bytes:
            return  # Can't fit even after eviction

        # Create entry
        entry = SimulatedCacheEntry(tokens, size_bytes, self.layer_weights)
        self.entries[token_tuple] = entry
        self.current_size += size_bytes


def generate_workload(
    num_prefixes: int = 10,
    queries_per_prefix: int = 30,
    prefix_tokens: int = 500,
    query_tokens: int = 50,
    zipf_alpha: float = 1.5,
) -> List[Tuple[int, List[int]]]:
    """Generate workload with Zipf-distributed prefix access."""
    # Create unique prefixes
    prefixes = {}
    for i in range(num_prefixes):
        prefixes[i] = list(range(i * 10000, i * 10000 + prefix_tokens))

    # Generate queries with Zipf distribution
    workload = []
    total_queries = num_prefixes * queries_per_prefix

    # Zipf probabilities
    ranks = list(range(1, num_prefixes + 1))
    probs = [1.0 / (r ** zipf_alpha) for r in ranks]
    total_prob = sum(probs)
    probs = [p / total_prob for p in probs]

    for _ in range(total_queries):
        # Choose prefix based on Zipf
        r = random.random()
        cumulative = 0.0
        chosen = 0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                chosen = i
                break

        # Create full sequence
        query = list(range(random.randint(100000, 999999),
                          random.randint(100000, 999999) + query_tokens))
        full_tokens = prefixes[chosen] + query
        workload.append((chosen, full_tokens))

    return workload


def estimate_kv_cache_size(
    seq_len: int,
    num_layers: int = 22,
    num_heads: int = 32,
    head_dim: int = 64,
    dtype_bytes: int = 2,  # fp16
) -> int:
    """Estimate KV cache size in bytes."""
    # 2 for K and V, num_layers, seq_len, num_heads, head_dim, dtype
    return 2 * num_layers * seq_len * num_heads * head_dim * dtype_bytes


def run_memory_pressure_experiment(
    policy: str,
    capacity_ratio: float,
    workload: List[Tuple[int, List[int]]],
    num_layers: int = 22,
    compute_latency_ms: float = 10.0,  # Simulated computation latency
    cache_latency_ms: float = 1.0,  # Simulated cache hit latency
) -> EvictionResult:
    """Run experiment with specific policy and capacity."""

    # Estimate working set size
    avg_seq_len = statistics.mean([len(tokens) for _, tokens in workload[:20]])
    single_entry_size = estimate_kv_cache_size(int(avg_seq_len), num_layers)

    # Count unique prefixes to estimate working set
    unique_prefixes = len(set(prefix_id for prefix_id, _ in workload))
    working_set_size = single_entry_size * unique_prefixes * 2  # 2x for safety

    # Create cache with limited capacity
    cache_capacity = int(working_set_size * capacity_ratio)
    cache = SimulatedCache(
        capacity_bytes=cache_capacity,
        policy=policy,
        num_layers=num_layers,
    )

    # Run workload
    hits = 0
    misses = 0
    total_latency = 0.0
    tokens_saved = 0

    for prefix_id, tokens in workload:
        hit, matched_len = cache.lookup(tokens)

        if hit:
            hits += 1
            total_latency += cache_latency_ms
            tokens_saved += matched_len
        else:
            misses += 1
            total_latency += compute_latency_ms

            # Insert into cache
            entry_size = estimate_kv_cache_size(len(tokens), num_layers)
            cache.insert(tokens, entry_size)

    # Compute metrics
    total_requests = len(workload)
    hit_rate = hits / total_requests if total_requests > 0 else 0
    avg_latency = total_latency / total_requests if total_requests > 0 else 0
    speedup = compute_latency_ms / avg_latency if avg_latency > 0 else 1

    return EvictionResult(
        policy_name=policy,
        capacity_ratio=capacity_ratio,
        total_requests=total_requests,
        cache_hits=hits,
        cache_misses=misses,
        hit_rate=hit_rate,
        evictions=cache.eviction_count,
        avg_latency_ms=avg_latency,
        speedup_vs_nocache=speedup,
        tokens_saved=tokens_saved,
    )


def run_full_experiment(
    num_runs: int = 3,
    capacity_ratios: List[float] = [1.0, 0.5, 0.3, 0.1],
    policies: List[str] = ["lru", "lfu", "attention", "layer_attention"],
) -> Dict:
    """Run full memory pressure experiment."""
    print("=" * 70)
    print("IJCNN 2025: Memory Pressure Eviction Experiment")
    print("=" * 70)

    # Generate workload
    print("\nGenerating workload...")
    workload = generate_workload(
        num_prefixes=15,
        queries_per_prefix=40,
        prefix_tokens=500,
        query_tokens=50,
        zipf_alpha=1.5,
    )
    print(f"  Total requests: {len(workload)}")
    print(f"  Unique prefixes: {len(set(p for p, _ in workload))}")

    all_results = []

    for capacity in capacity_ratios:
        print(f"\n{'=' * 70}")
        print(f"Cache Capacity: {capacity*100:.0f}%")
        print("=" * 70)

        for policy in policies:
            print(f"\n  Testing {policy}...")

            # Run multiple times and average
            run_results = []
            for run in range(num_runs):
                random.seed(42 + run)  # Reproducible randomness
                result = run_memory_pressure_experiment(
                    policy=policy,
                    capacity_ratio=capacity,
                    workload=workload,
                )
                run_results.append(result)

            # Average results
            avg_result = EvictionResult(
                policy_name=policy,
                capacity_ratio=capacity,
                total_requests=run_results[0].total_requests,
                cache_hits=int(statistics.mean(r.cache_hits for r in run_results)),
                cache_misses=int(statistics.mean(r.cache_misses for r in run_results)),
                hit_rate=statistics.mean(r.hit_rate for r in run_results),
                evictions=int(statistics.mean(r.evictions for r in run_results)),
                avg_latency_ms=statistics.mean(r.avg_latency_ms for r in run_results),
                speedup_vs_nocache=statistics.mean(r.speedup_vs_nocache for r in run_results),
                tokens_saved=int(statistics.mean(r.tokens_saved for r in run_results)),
            )

            all_results.append(avg_result)

            print(f"    Hit rate: {avg_result.hit_rate:.1%}")
            print(f"    Evictions: {avg_result.evictions}")
            print(f"    Speedup: {avg_result.speedup_vs_nocache:.2f}x")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Hit Rate by Policy and Capacity")
    print("=" * 80)
    print(f"{'Policy':<20}", end="")
    for cap in capacity_ratios:
        print(f"{cap*100:.0f}%".center(12), end="")
    print()
    print("-" * 80)

    for policy in policies:
        print(f"{policy:<20}", end="")
        for cap in capacity_ratios:
            result = next(
                r for r in all_results
                if r.policy_name == policy and r.capacity_ratio == cap
            )
            print(f"{result.hit_rate:.1%}".center(12), end="")
        print()

    print("\n" + "=" * 80)
    print("SUMMARY: Speedup by Policy and Capacity")
    print("=" * 80)
    print(f"{'Policy':<20}", end="")
    for cap in capacity_ratios:
        print(f"{cap*100:.0f}%".center(12), end="")
    print()
    print("-" * 80)

    for policy in policies:
        print(f"{policy:<20}", end="")
        for cap in capacity_ratios:
            result = next(
                r for r in all_results
                if r.policy_name == policy and r.capacity_ratio == cap
            )
            print(f"{result.speedup_vs_nocache:.2f}x".center(12), end="")
        print()

    # Compute layer-aware advantage
    print("\n" + "=" * 80)
    print("Layer-Aware Advantage over LRU")
    print("=" * 80)
    for cap in capacity_ratios:
        lru_result = next(r for r in all_results if r.policy_name == "lru" and r.capacity_ratio == cap)
        la_result = next(r for r in all_results if r.policy_name == "layer_attention" and r.capacity_ratio == cap)

        hit_improvement = (la_result.hit_rate - lru_result.hit_rate) / max(0.01, lru_result.hit_rate) * 100
        speedup_improvement = (la_result.speedup_vs_nocache - lru_result.speedup_vs_nocache) / max(0.01, lru_result.speedup_vs_nocache) * 100

        print(f"  {cap*100:.0f}% capacity: +{hit_improvement:.1f}% hit rate, +{speedup_improvement:.1f}% speedup")

    return {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_runs": num_runs,
            "workload_size": len(workload),
        },
        "capacity_ratios": capacity_ratios,
        "policies": policies,
        "results": [asdict(r) for r in all_results],
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Memory Pressure Eviction Experiment")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per configuration")
    parser.add_argument("--output", type=str, default=None, help="Output file path")

    args = parser.parse_args()

    results = run_full_experiment(num_runs=args.runs)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = RESULTS_DIR / "ijcnn_memory_pressure_results.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
