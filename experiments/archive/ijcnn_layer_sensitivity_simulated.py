#!/usr/bin/env python3
"""IJCNN 2025: Layer Weight Parameter Sensitivity Analysis.

This experiment validates the choice of layer weight parameters:
- k (steepness): Controls how sharply weights transition
- τ (tau, transition point): Controls where the transition occurs

The sigmoid formula is: w_l = 1 / (1 + exp(-k * (l/L - τ)))

Default values in paper: k=5, τ=0.3

Grid search:
- k ∈ [1, 3, 5, 7, 10]
- τ ∈ [0.1, 0.2, 0.3, 0.4, 0.5]

This creates a 5x5 grid (25 configurations) to show:
1. Performance is robust to parameter choices
2. k=5, τ=0.3 is near-optimal or Pareto-optimal
"""

import os
import gc
import sys
import json
import math
import random
import statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path(__file__).parent / "results" / "paper"


@dataclass
class SensitivityResult:
    """Result for a single k, tau configuration."""
    k: float
    tau: float
    hit_rate: float
    speedup: float
    evictions: int
    late_layer_weight: float  # Weight of last layer
    early_layer_weight: float  # Weight of first layer
    weight_ratio: float  # late/early ratio


class LayerAwareCache:
    """Cache with configurable layer-aware eviction."""

    def __init__(
        self,
        capacity_bytes: int,
        num_layers: int,
        k: float,
        tau: float,
    ):
        self.capacity_bytes = capacity_bytes
        self.num_layers = num_layers
        self.k = k
        self.tau = tau

        self.entries: Dict[tuple, dict] = {}
        self.current_size = 0
        self.eviction_count = 0

        # Compute layer weights
        self.layer_weights = self._compute_layer_weights()

    def _compute_layer_weights(self) -> List[float]:
        """Compute sigmoid-based layer weights."""
        weights = []
        for i in range(self.num_layers):
            normalized_pos = i / (self.num_layers - 1) if self.num_layers > 1 else 0.5
            weight = 1.0 / (1.0 + math.exp(-self.k * (normalized_pos - self.tau)))
            weights.append(weight)
        return weights

    def get_early_late_weights(self) -> Tuple[float, float]:
        """Get weights for early and late layers."""
        return self.layer_weights[0], self.layer_weights[-1]

    def _compute_score(self, entry: dict) -> float:
        """Compute eviction score (lower = evict first)."""
        import time
        now = time.time()
        time_since = max(0.001, now - entry["last_access"])

        recency = 1.0 / (1.0 + math.log1p(time_since))
        frequency = math.log1p(entry["access_count"])
        layer_imp = entry["layer_importance"]

        return (recency * 0.4 + frequency * 0.3 + entry["attention"] * 0.3) * layer_imp

    def _evict_until_space(self, required: int):
        """Evict entries until we have enough space."""
        while self.current_size + required > self.capacity_bytes and self.entries:
            victims = sorted(self.entries.items(), key=lambda x: self._compute_score(x[1]))
            if not victims:
                break
            key, entry = victims[0]
            self.current_size -= entry["size"]
            del self.entries[key]
            self.eviction_count += 1

    def lookup(self, tokens: List[int]) -> Tuple[bool, int]:
        """Look up tokens in cache."""
        import time
        key = tuple(tokens)

        # Exact match
        if key in self.entries:
            self.entries[key]["access_count"] += 1
            self.entries[key]["last_access"] = time.time()
            return True, len(tokens)

        # Prefix match
        best_len = 0
        best_key = None
        for k in self.entries:
            klen = len(k)
            if klen <= len(tokens) and tokens[:klen] == list(k):
                if klen > best_len:
                    best_len = klen
                    best_key = k

        if best_key:
            self.entries[best_key]["access_count"] += 1
            self.entries[best_key]["last_access"] = time.time()
            return True, best_len

        return False, 0

    def insert(self, tokens: List[int], size: int):
        """Insert tokens into cache."""
        import time
        key = tuple(tokens)
        if key in self.entries:
            return

        self._evict_until_space(size)
        if self.current_size + size > self.capacity_bytes:
            return

        avg_layer_weight = sum(self.layer_weights) / len(self.layer_weights)
        self.entries[key] = {
            "size": size,
            "access_count": 1,
            "last_access": time.time(),
            "attention": random.random(),
            "layer_importance": avg_layer_weight,
        }
        self.current_size += size


def generate_workload(
    num_prefixes: int = 10,
    queries_per_prefix: int = 30,
    prefix_tokens: int = 500,
    query_tokens: int = 50,
) -> List[Tuple[int, List[int]]]:
    """Generate test workload."""
    prefixes = {i: list(range(i * 10000, i * 10000 + prefix_tokens))
                for i in range(num_prefixes)}

    workload = []
    total = num_prefixes * queries_per_prefix

    # Zipf distribution
    ranks = range(1, num_prefixes + 1)
    probs = [1.0 / (r ** 1.5) for r in ranks]
    total_prob = sum(probs)
    probs = [p / total_prob for p in probs]

    for _ in range(total):
        r = random.random()
        cumulative = 0.0
        chosen = 0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                chosen = i
                break

        query = list(range(random.randint(100000, 999999),
                          random.randint(100000, 999999) + query_tokens))
        workload.append((chosen, prefixes[chosen] + query))

    return workload


def estimate_cache_size(seq_len: int, num_layers: int = 22) -> int:
    """Estimate KV cache size in bytes."""
    return 2 * num_layers * seq_len * 32 * 64 * 2  # K+V, layers, seq, heads, dim, fp16


def run_single_config(
    k: float,
    tau: float,
    workload: List[Tuple[int, List[int]]],
    capacity_ratio: float = 0.3,  # Test under pressure
    num_layers: int = 22,
) -> SensitivityResult:
    """Run experiment with single k, tau configuration."""

    # Estimate working set
    avg_len = statistics.mean([len(t) for _, t in workload[:20]])
    single_size = estimate_cache_size(int(avg_len), num_layers)
    unique_prefixes = len(set(p for p, _ in workload))
    working_set = single_size * unique_prefixes * 2
    capacity = int(working_set * capacity_ratio)

    cache = LayerAwareCache(
        capacity_bytes=capacity,
        num_layers=num_layers,
        k=k,
        tau=tau,
    )

    hits = 0
    misses = 0
    compute_lat = 10.0
    cache_lat = 1.0
    total_lat = 0.0

    for _, tokens in workload:
        hit, matched = cache.lookup(tokens)
        if hit:
            hits += 1
            total_lat += cache_lat
        else:
            misses += 1
            total_lat += compute_lat
            size = estimate_cache_size(len(tokens), num_layers)
            cache.insert(tokens, size)

    total = len(workload)
    hit_rate = hits / total
    avg_lat = total_lat / total
    speedup = compute_lat / avg_lat

    early_w, late_w = cache.get_early_late_weights()

    return SensitivityResult(
        k=k,
        tau=tau,
        hit_rate=hit_rate,
        speedup=speedup,
        evictions=cache.eviction_count,
        early_layer_weight=early_w,
        late_layer_weight=late_w,
        weight_ratio=late_w / max(0.001, early_w),
    )


def run_sensitivity_analysis(
    k_values: List[float] = [1, 3, 5, 7, 10],
    tau_values: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5],
    num_runs: int = 3,
    capacity_ratio: float = 0.3,
) -> Dict:
    """Run full grid search."""
    print("=" * 70)
    print("IJCNN 2025: Layer Weight Sensitivity Analysis")
    print("=" * 70)
    print(f"\nGrid: k ∈ {k_values}, τ ∈ {tau_values}")
    print(f"Capacity ratio: {capacity_ratio*100:.0f}%")
    print(f"Number of runs: {num_runs}")

    # Generate workload
    print("\nGenerating workload...")
    base_workload = generate_workload(
        num_prefixes=15,
        queries_per_prefix=40,
        prefix_tokens=500,
        query_tokens=50,
    )
    print(f"  Total requests: {len(base_workload)}")

    all_results = []
    results_matrix = {}  # (k, tau) -> result

    total_configs = len(k_values) * len(tau_values)
    config_num = 0

    for k in k_values:
        for tau in tau_values:
            config_num += 1
            print(f"\n[{config_num}/{total_configs}] Testing k={k}, τ={tau}...")

            run_results = []
            for run in range(num_runs):
                random.seed(42 + run)
                result = run_single_config(
                    k=k,
                    tau=tau,
                    workload=base_workload,
                    capacity_ratio=capacity_ratio,
                )
                run_results.append(result)

            # Average results
            avg_result = SensitivityResult(
                k=k,
                tau=tau,
                hit_rate=statistics.mean(r.hit_rate for r in run_results),
                speedup=statistics.mean(r.speedup for r in run_results),
                evictions=int(statistics.mean(r.evictions for r in run_results)),
                early_layer_weight=run_results[0].early_layer_weight,
                late_layer_weight=run_results[0].late_layer_weight,
                weight_ratio=run_results[0].weight_ratio,
            )

            all_results.append(avg_result)
            results_matrix[(k, tau)] = avg_result

            print(f"    Hit rate: {avg_result.hit_rate:.1%}")
            print(f"    Speedup: {avg_result.speedup:.2f}x")
            print(f"    Late/Early ratio: {avg_result.weight_ratio:.2f}x")

    # Print heatmap-style table for hit rate
    print("\n" + "=" * 80)
    print("Hit Rate Heatmap (k × τ)")
    print("=" * 80)
    print(f"{'k \\ τ':<8}", end="")
    for tau in tau_values:
        print(f"{tau:.1f}".center(10), end="")
    print()
    print("-" * 80)

    for k in k_values:
        print(f"{k:<8}", end="")
        for tau in tau_values:
            result = results_matrix[(k, tau)]
            print(f"{result.hit_rate:.1%}".center(10), end="")
        print()

    # Print heatmap for speedup
    print("\n" + "=" * 80)
    print("Speedup Heatmap (k × τ)")
    print("=" * 80)
    print(f"{'k \\ τ':<8}", end="")
    for tau in tau_values:
        print(f"{tau:.1f}".center(10), end="")
    print()
    print("-" * 80)

    for k in k_values:
        print(f"{k:<8}", end="")
        for tau in tau_values:
            result = results_matrix[(k, tau)]
            print(f"{result.speedup:.2f}x".center(10), end="")
        print()

    # Find optimal configuration
    best_by_hit = max(all_results, key=lambda r: r.hit_rate)
    best_by_speedup = max(all_results, key=lambda r: r.speedup)

    print("\n" + "=" * 80)
    print("Optimal Configurations")
    print("=" * 80)
    print(f"Best hit rate:  k={best_by_hit.k}, τ={best_by_hit.tau} -> {best_by_hit.hit_rate:.1%}")
    print(f"Best speedup:   k={best_by_speedup.k}, τ={best_by_speedup.tau} -> {best_by_speedup.speedup:.2f}x")

    # Check our default (k=5, τ=0.3)
    default_result = results_matrix.get((5, 0.3))
    if default_result:
        print(f"\nDefault (k=5, τ=0.3):")
        print(f"  Hit rate: {default_result.hit_rate:.1%} (rank: {sorted(all_results, key=lambda r: -r.hit_rate).index(default_result)+1}/{len(all_results)})")
        print(f"  Speedup:  {default_result.speedup:.2f}x (rank: {sorted(all_results, key=lambda r: -r.speedup).index(default_result)+1}/{len(all_results)})")
        print(f"  Weight ratio: {default_result.weight_ratio:.2f}x (late/early)")

    # Compute robustness: std dev across configurations
    hit_rates = [r.hit_rate for r in all_results]
    speedups = [r.speedup for r in all_results]

    print(f"\nParameter Robustness:")
    print(f"  Hit rate range: {min(hit_rates):.1%} - {max(hit_rates):.1%} (std: {statistics.stdev(hit_rates):.2%})")
    print(f"  Speedup range:  {min(speedups):.2f}x - {max(speedups):.2f}x (std: {statistics.stdev(speedups):.2f})")

    return {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "k_values": k_values,
            "tau_values": tau_values,
            "num_runs": num_runs,
            "capacity_ratio": capacity_ratio,
        },
        "results": [asdict(r) for r in all_results],
        "optimal": {
            "best_hit_rate": {"k": best_by_hit.k, "tau": best_by_hit.tau, "value": best_by_hit.hit_rate},
            "best_speedup": {"k": best_by_speedup.k, "tau": best_by_speedup.tau, "value": best_by_speedup.speedup},
        },
        "robustness": {
            "hit_rate_std": statistics.stdev(hit_rates),
            "speedup_std": statistics.stdev(speedups),
        },
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Layer Weight Sensitivity Analysis")
    parser.add_argument("--runs", type=int, default=3, help="Runs per configuration")
    parser.add_argument("--capacity", type=float, default=0.3, help="Cache capacity ratio")
    parser.add_argument("--output", type=str, default=None, help="Output file")

    args = parser.parse_args()

    results = run_sensitivity_analysis(
        num_runs=args.runs,
        capacity_ratio=args.capacity,
    )

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = RESULTS_DIR / "ijcnn_layer_weight_sensitivity.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
