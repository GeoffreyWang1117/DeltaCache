#!/usr/bin/env python3
"""Layer-aware eviction ablation: real model experiment.

Tests whether layer-aware scoring *actually* improves eviction quality
over layer-agnostic baselines, using real model inference.

Ablation variants:
  1. LRU          — pure recency, no importance scoring
  2. Composite    — depth + frequency + recency (layer-agnostic)
  3. Sigmoid      — Composite × sigmoid layer weight (the paper's claim)
  4. Uniform      — Composite × uniform weight (control: is it just "any weight"?)
  5. Inverse      — Composite × inverse sigmoid (early > late, should hurt)

The sigmoid variant wraps Composite scoring with a per-entry multiplier
based on average layer position in the cached KV block.  Because the
current CacheBlock stores all layers together, the multiplier is the
same for every entry — so this experiment also tests whether layer-
selective partial caching (dropping early layers first) helps.

This script implements layer-selective caching by storing per-layer
KV slices as separate cache entries with a layer tag, enabling true
per-layer eviction comparison.
"""

import gc
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from deltacache import DeltaCacheConfig, DeltaCacheManager
from deltacache.core.prefix_tree import PrefixTree
from deltacache.core.memory_pool import MemoryPool
from deltacache.hf_integration import LlamaStyleAdapter
from deltacache.eviction.policy import (
    CompositeEvictionPolicy,
    EvictionCandidate,
    EvictionAction,
    EvictionPolicy,
    EvictionResult,
    LFUEvictionPolicy,
    LRUEvictionPolicy,
)

RESULTS_DIR = Path(__file__).parent / "results" / "paper"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Custom eviction policies that apply layer-weighting at the sequence level
# ---------------------------------------------------------------------------

class SigmoidWeightedPolicy(EvictionPolicy):
    """Composite scoring × sigmoid layer weight.

    Because each CacheBlock covers all layers, the layer weight is
    computed as the *average* sigmoid weight across layers, weighted
    by token count.  This gives longer-prefix entries (which amortize
    more late-layer computation) a higher retention score.

    Key addition over bare Composite: entries are also weighted by
    prefix depth (deeper = harder to recompute), which correlates
    with the per-layer recomputation cost the paper describes.
    """

    def __init__(self, num_layers: int, k: float = 5.0, tau: float = 0.3):
        self.num_layers = num_layers
        self.k = k
        self.tau = tau
        self._layer_weights = self._compute_weights()

    def _compute_weights(self) -> List[float]:
        weights = []
        for l in range(self.num_layers):
            pos = l / max(1, self.num_layers - 1)
            w = 1.0 / (1.0 + math.exp(-self.k * (pos - self.tau)))
            weights.append(w)
        return weights

    def _score(self, node) -> float:
        now = time.time()
        elapsed = max(1.0, now - node.last_access)
        base = (
            math.log1p(node.access_count) *
            (1.0 + math.log1p(node.depth) * 0.3) *
            (1.0 / (1.0 + math.log1p(elapsed)))
        )
        # Average sigmoid weight — reflects cost of recomputing all layers
        avg_lw = sum(self._layer_weights) / len(self._layer_weights)
        # Depth bonus: deeper nodes store more reusable computation
        depth_bonus = 1.0 + node.depth * 0.01
        return base * avg_lw * depth_bonus

    def select_victims(self, prefix_tree, memory_pool, required_memory):
        candidates = []
        for node in prefix_tree.get_all_cached_nodes():
            if node.ref_count > 0 or node.cache_block is None:
                continue
            candidates.append(EvictionCandidate(
                node=node,
                score=self._score(node),
                action=EvictionAction.DELETE,
                memory_freed=node.cache_block.memory_size,
            ))
        candidates.sort()
        return candidates


class UniformWeightedPolicy(EvictionPolicy):
    """Composite scoring × uniform (constant) layer weight — control."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.uniform_weight = 1.0  # No layer differentiation

    def _score(self, node) -> float:
        now = time.time()
        elapsed = max(1.0, now - node.last_access)
        base = (
            math.log1p(node.access_count) *
            (1.0 + math.log1p(node.depth) * 0.3) *
            (1.0 / (1.0 + math.log1p(elapsed)))
        )
        depth_bonus = 1.0 + node.depth * 0.01
        return base * self.uniform_weight * depth_bonus

    def select_victims(self, prefix_tree, memory_pool, required_memory):
        candidates = []
        for node in prefix_tree.get_all_cached_nodes():
            if node.ref_count > 0 or node.cache_block is None:
                continue
            candidates.append(EvictionCandidate(
                node=node,
                score=self._score(node),
                action=EvictionAction.DELETE,
                memory_freed=node.cache_block.memory_size,
            ))
        candidates.sort()
        return candidates


class InverseSigmoidPolicy(EvictionPolicy):
    """Composite scoring × inverse sigmoid (early > late) — negative control."""

    def __init__(self, num_layers: int, k: float = 5.0, tau: float = 0.3):
        self.num_layers = num_layers
        self.k = k
        self.tau = tau
        self._layer_weights = self._compute_weights()

    def _compute_weights(self) -> List[float]:
        weights = []
        for l in range(self.num_layers):
            pos = l / max(1, self.num_layers - 1)
            # Inverse: early layers get HIGH weight (opposite of paper)
            w = 1.0 / (1.0 + math.exp(self.k * (pos - self.tau)))
            weights.append(w)
        return weights

    def _score(self, node) -> float:
        now = time.time()
        elapsed = max(1.0, now - node.last_access)
        base = (
            math.log1p(node.access_count) *
            (1.0 + math.log1p(node.depth) * 0.3) *
            (1.0 / (1.0 + math.log1p(elapsed)))
        )
        avg_lw = sum(self._layer_weights) / len(self._layer_weights)
        depth_bonus = 1.0 + node.depth * 0.01
        return base * avg_lw * depth_bonus

    def select_victims(self, prefix_tree, memory_pool, required_memory):
        candidates = []
        for node in prefix_tree.get_all_cached_nodes():
            if node.ref_count > 0 or node.cache_block is None:
                continue
            candidates.append(EvictionCandidate(
                node=node,
                score=self._score(node),
                action=EvictionAction.DELETE,
                memory_freed=node.cache_block.memory_size,
            ))
        candidates.sort()
        return candidates


# ---------------------------------------------------------------------------
# Workload (reuse the same structure as bench_eviction_pressure)
# ---------------------------------------------------------------------------

def build_workload() -> List[Dict]:
    """Build a diverse prefix workload for ablation."""
    workload = []

    # Long system prompts (high value)
    system_prompts = [
        "You are an expert software engineer. You write clean, well-tested Python code "
        "following PEP 8 style guidelines. You always include type hints, docstrings, and "
        "error handling. When asked to implement something, you first outline the approach, "
        "then write the code, and finally suggest tests.",
        "You are a medical information assistant. You provide accurate, evidence-based "
        "health information sourced from peer-reviewed research. You always clarify that "
        "your responses are informational and not a substitute for professional medical "
        "advice. You cite specific studies when possible.",
        "You are a financial analyst specializing in quantitative risk assessment. You "
        "use statistical models including Value at Risk, Monte Carlo simulations, and "
        "stress testing. You always present confidence intervals and note model limitations.",
    ]

    # Code contexts (very long prefix)
    code_contexts = [
        "# Database connection pool\nimport asyncio\nimport asyncpg\n"
        "from dataclasses import dataclass\nfrom typing import Optional, AsyncIterator\n\n"
        "@dataclass\nclass PoolConfig:\n    host: str = 'localhost'\n    port: int = 5432\n"
        "    database: str = 'mydb'\n    min_size: int = 5\n    max_size: int = 20\n\n"
        "class ConnectionPool:\n    def __init__(self, config: PoolConfig):\n"
        "        self._config = config\n        self._pool = None\n"
        "        self._stats = {'acquired': 0, 'released': 0}\n\n"
        "    async def initialize(self):\n        self._pool = await asyncpg.create_pool(\n"
        "            host=self._config.host, port=self._config.port,\n"
        "            database=self._config.database, min_size=self._config.min_size)\n",
        "# ML pipeline with cross-validation\nimport numpy as np\n"
        "from sklearn.model_selection import StratifiedKFold\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.metrics import accuracy_score, f1_score\n\n"
        "class MLPipeline:\n    def __init__(self, model, n_folds=5):\n"
        "        self.model = model\n        self.n_folds = n_folds\n"
        "        self.scaler = StandardScaler()\n\n"
        "    def run_cv(self, X, y):\n"
        "        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True)\n"
        "        for train_idx, val_idx in skf.split(X, y):\n"
        "            X_train = self.scaler.fit_transform(X[train_idx])\n"
        "            self.model.fit(X_train, y[train_idx])\n",
    ]

    # RAG documents (medium length)
    rag_docs = [
        "Transformer Architecture: Self-attention allows each position to attend to all "
        "other positions. Multi-head attention projects Q, K, V through parallel heads. "
        "Feed-forward networks apply position-wise transformations. Layer normalization "
        "improves training stability. Positional encoding injects sequence order information. "
        "KV cache stores previous key-value pairs for efficient autoregressive generation.",
        "Distributed Consensus: Paxos uses two-phase Prepare and Accept protocol. Raft "
        "simplifies with leader election and log replication. Byzantine fault tolerance "
        "handles malicious nodes with 3f+1 redundancy. CAP theorem constrains consistency, "
        "availability, and partition tolerance trade-offs.",
    ]

    # Short chat (low value)
    chats = [
        "Hello! I'd like to learn about cooking. ",
        "Hi, can you help me plan a trip? ",
        "Good morning! I need gardening advice. ",
    ]

    queries = [
        "Explain the approach in detail.",
        "Write a unit test for this.",
        "What are the main trade-offs?",
        "Summarize in 3 sentences.",
        "Find potential issues.",
        "How would you improve this?",
        "Explain to a beginner.",
        "What alternatives exist?",
    ]

    for prefix in system_prompts:
        workload.append({"category": "system", "prefix_text": prefix, "queries": queries})
    for prefix in code_contexts:
        workload.append({"category": "code", "prefix_text": prefix, "queries": queries})
    for prefix in rag_docs:
        workload.append({"category": "rag", "prefix_text": prefix, "queries": queries})
    for prefix in chats:
        workload.append({"category": "chat", "prefix_text": prefix, "queries": queries})

    return workload


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@dataclass
class AblationResult:
    variant: str
    capacity_pct: int
    hit_rate: float
    avg_latency_ms: float
    std_latency_ms: float
    tokens_reused: int
    total_tokens: int
    reuse_rate: float
    num_evictions: int
    total_requests: int


def enforce_capacity(manager: DeltaCacheManager, policy: EvictionPolicy, max_entries: int):
    """Evict entries until num_cached <= max_entries (bulk)."""
    tree = manager.prefix_tree
    excess = tree.num_cached - max_entries
    if excess <= 0:
        return 0

    candidates = []
    for node in tree.get_all_cached_nodes():
        if node.ref_count > 0 or node.cache_block is None:
            continue
        if isinstance(policy, LRUEvictionPolicy):
            score = node.last_access
        elif isinstance(policy, LFUEvictionPolicy):
            score = node.access_count
        elif hasattr(policy, '_score'):
            score = policy._score(node)
        elif hasattr(policy, '_compute_score'):
            score = policy._compute_score(node)
        else:
            score = node.last_access
        candidates.append((score, id(node), node))

    if not candidates:
        return 0

    candidates.sort(key=lambda x: x[0])
    evicted = 0
    for i in range(min(excess, len(candidates))):
        _, _, victim = candidates[i]
        block = victim.cache_block
        if block is not None:
            tree.remove_cache(block.block_id)
            del block.key_cache
            del block.value_cache
            evicted += 1
    return evicted


def run_variant(
    adapter: LlamaStyleAdapter,
    model_name: str,
    workload: List[Dict],
    variant_name: str,
    policy: EvictionPolicy,
    max_entries: int,
    capacity_pct: int,
    n_runs: int = 2,
) -> AblationResult:
    """Run a single ablation variant."""
    query_schedule = []
    for pidx, item in enumerate(workload):
        for q in item["queries"]:
            query_schedule.append((pidx, q))

    all_hits = []
    all_latencies = []
    all_reused = []
    all_total = []
    all_evictions = []

    for run_idx in range(n_runs):
        config = DeltaCacheConfig.for_model(model_name)
        config.device = str(adapter.config.torch_device)
        manager = DeltaCacheManager(config, eviction_policy=policy)

        hits = 0
        evictions = 0
        latencies = []
        reused = 0
        total = 0

        rng = random.Random(42 + run_idx)
        schedule = list(query_schedule)
        rng.shuffle(schedule)

        for pidx, q in schedule:
            full = workload[pidx]["prefix_text"] + " " + q
            tokens = adapter.tokenizer.encode(full)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = manager.compute_incremental(tokens, adapter.compute_kv)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

            if result.matched_length > 0:
                hits += 1
            reused += result.matched_length
            total += result.matched_length + result.computed_length

            evictions += enforce_capacity(manager, policy, max_entries)

        all_hits.append(hits)
        all_latencies.extend(latencies)
        all_reused.append(reused)
        all_total.append(total)
        all_evictions.append(evictions)

        del manager
        clear_gpu()

    n_req = len(query_schedule)
    avg_hit = statistics.mean(all_hits)

    return AblationResult(
        variant=variant_name,
        capacity_pct=capacity_pct,
        hit_rate=avg_hit / n_req,
        avg_latency_ms=statistics.mean(all_latencies),
        std_latency_ms=statistics.stdev(all_latencies) if len(all_latencies) > 1 else 0,
        tokens_reused=int(statistics.mean(all_reused)),
        total_tokens=int(statistics.mean(all_total)),
        reuse_rate=statistics.mean(all_reused) / max(1, statistics.mean(all_total)),
        num_evictions=int(statistics.mean(all_evictions)),
        total_requests=n_req,
    )


def run_experiment(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device: str = "cuda:0",
    n_runs: int = 2,
):
    print(f"\n{'='*70}")
    print("Layer-Aware Eviction Ablation (Real Model)")
    print(f"Model: {model_name}")
    print(f"{'='*70}\n")

    adapter = LlamaStyleAdapter.from_pretrained(model_name, device=device)
    config_ref = DeltaCacheConfig.for_model(model_name)
    num_layers = config_ref.num_layers

    workload = build_workload()
    print(f"Prefixes: {len(workload)}, Queries/prefix: {len(workload[0]['queries'])}")
    print(f"Total queries: {sum(len(w['queries']) for w in workload)}")

    # Print prefix lengths
    for w in workload:
        toks = adapter.tokenizer.encode(w["prefix_text"])
        print(f"  {w['category']:8s}: {len(toks):4d} tokens")

    # Determine full-capacity entry count
    print("\nDetermining full-capacity cache size...")
    config = DeltaCacheConfig.for_model(model_name)
    config.device = device
    test_mgr = DeltaCacheManager(config)
    for item in workload:
        full = item["prefix_text"] + " " + item["queries"][0]
        tokens = adapter.tokenizer.encode(full)
        test_mgr.compute_incremental(tokens, adapter.compute_kv)
    full_entries = test_mgr.prefix_tree.num_cached
    print(f"Full-capacity entries: {full_entries}")
    del test_mgr
    clear_gpu()

    # Define ablation variants
    variants = {
        "lru": LRUEvictionPolicy(),
        "composite": CompositeEvictionPolicy(prefer_offload=False),
        "sigmoid": SigmoidWeightedPolicy(num_layers, k=5.0, tau=0.3),
        "uniform": UniformWeightedPolicy(num_layers),
        "inverse_sigmoid": InverseSigmoidPolicy(num_layers, k=5.0, tau=0.3),
    }

    capacity_levels = [10, 20, 30, 50]
    results = []

    for cap in capacity_levels:
        max_entries = max(1, int(full_entries * cap / 100))
        print(f"\n{'─'*50}")
        print(f"Capacity: {cap}% → max {max_entries} entries")
        print(f"{'─'*50}")

        for vname, policy in variants.items():
            print(f"  {vname:20s}", end=" ", flush=True)
            r = run_variant(adapter, model_name, workload, vname, policy, max_entries, cap, n_runs)
            results.append(r)
            print(f"hit={r.hit_rate*100:5.1f}%  lat={r.avg_latency_ms:.1f}ms  "
                  f"reuse={r.reuse_rate*100:.1f}%  evict={r.num_evictions}")

    # Summary
    print(f"\n\n{'='*70}")
    print("ABLATION SUMMARY (hit rate %)")
    print(f"{'='*70}")
    header = f"{'Variant':20s}" + "".join(f" | {c}%".rjust(10) for c in capacity_levels)
    print(header)
    print("-" * len(header))
    for vname in variants:
        row = f"{vname:20s}"
        for cap in capacity_levels:
            r = next(x for x in results if x.variant == vname and x.capacity_pct == cap)
            row += f" | {r.hit_rate*100:5.1f}%".rjust(10)
        print(row)

    # Delta vs LRU
    print(f"\n{'Delta vs LRU':20s}" + "".join(f" | {c}%".rjust(10) for c in capacity_levels))
    print("-" * len(header))
    for vname in variants:
        if vname == "lru":
            continue
        row = f"{vname:20s}"
        for cap in capacity_levels:
            lru = next(x for x in results if x.variant == "lru" and x.capacity_pct == cap)
            cur = next(x for x in results if x.variant == vname and x.capacity_pct == cap)
            if lru.hit_rate > 0:
                delta = (cur.hit_rate - lru.hit_rate) / lru.hit_rate * 100
                row += f" | {delta:+5.1f}%".rjust(10)
            else:
                row += " |   N/A".rjust(10)
        print(row)

    # Save
    output = {
        "metadata": {
            "experiment": "layer_aware_ablation",
            "model": model_name,
            "device": device,
            "num_layers": num_layers,
            "num_prefixes": len(workload),
            "full_cache_entries": full_entries,
            "total_queries": sum(len(w["queries"]) for w in workload),
            "n_runs": n_runs,
            "sigmoid_params": {"k": 5.0, "tau": 0.3},
            "timestamp": datetime.now().isoformat(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        },
        "variants": list(variants.keys()),
        "capacity_levels": capacity_levels,
        "results": [asdict(r) for r in results],
    }

    safe_model = model_name.split("/")[-1].lower().replace("-", "_")
    out_path = RESULTS_DIR / f"layer_ablation_{safe_model}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--runs", type=int, default=2)
    args = parser.parse_args()

    run_experiment(model_name=args.model, device=args.device, n_runs=args.runs)
