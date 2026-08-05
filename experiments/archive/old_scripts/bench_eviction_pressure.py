#!/usr/bin/env python3
"""Real memory-pressure eviction policy comparison.

Loads TinyLlama-1.1B and measures how different eviction policies perform
when cache capacity is constrained to 10-50% of the unique prefixes.

Design:
  - 12 diverse shared prefixes (system, code, RAG, chat)
  - Each prefix queried 8 times with different suffixes → 96 queries/run
  - Cache capacity limited to N unique prefixes (10/20/30/50/100% of 12)
  - After each insert, if cached > max_entries, evict via policy
  - Policies: LRU, LFU, Composite (depth+freq+recency)
  - Metrics: hit rate, latency, token reuse, eviction count

Capacity is enforced at the prefix-tree level (number of cached entries)
because the MemoryPool accounting path is bypassed by the incremental
engine — see note in experiments/README.md.
"""

import gc
import json
import math
import os
import random
import sys
import time
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from deltacache import DeltaCacheConfig, DeltaCacheManager
from deltacache.core.prefix_tree import PrefixTree
from deltacache.hf_integration import LlamaStyleAdapter
from deltacache.eviction.policy import (
    CompositeEvictionPolicy,
    EvictionCandidate,
    EvictionAction,
    EvictionPolicy,
    LFUEvictionPolicy,
    LRUEvictionPolicy,
)

RESULTS_DIR = Path(__file__).parent / "results" / "paper"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Workload definition
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = [
    (
        "You are an expert software engineer. You write clean, well-tested Python code "
        "following PEP 8 style guidelines. You always include type hints, docstrings, and "
        "error handling. When asked to implement something, you first outline the approach, "
        "then write the code, and finally suggest tests. You prefer composition over "
        "inheritance and favor immutable data structures when possible. You are familiar "
        "with modern Python features including dataclasses, pattern matching, and async/await."
    ),
    (
        "You are a medical information assistant. You provide accurate, evidence-based "
        "health information sourced from peer-reviewed research. You always clarify that "
        "your responses are informational and not a substitute for professional medical "
        "advice. You cite specific studies when possible and note the quality of evidence. "
        "You are careful to distinguish between established medical consensus and emerging "
        "research findings that require further validation."
    ),
    (
        "You are a financial analyst specializing in quantitative risk assessment. You "
        "use statistical models including Value at Risk, Monte Carlo simulations, and "
        "stress testing to evaluate portfolio risk. You always present confidence intervals "
        "and note model limitations. You follow regulatory frameworks including Basel III "
        "and Dodd-Frank requirements. You communicate complex quantitative findings in "
        "clear language suitable for board-level presentations."
    ),
]

CODE_CONTEXTS = [
    (
        "# Database connection pool implementation\n"
        "import asyncio\nimport asyncpg\nfrom contextlib import asynccontextmanager\n"
        "from dataclasses import dataclass, field\nfrom typing import Optional, AsyncIterator\n\n"
        "@dataclass\nclass PoolConfig:\n    host: str = 'localhost'\n    port: int = 5432\n"
        "    database: str = 'mydb'\n    min_size: int = 5\n    max_size: int = 20\n"
        "    command_timeout: float = 60.0\n    max_inactive_connection_lifetime: float = 300.0\n\n"
        "class ConnectionPool:\n    def __init__(self, config: PoolConfig):\n"
        "        self._config = config\n        self._pool: Optional[asyncpg.Pool] = None\n"
        "        self._lock = asyncio.Lock()\n        self._stats = {'acquired': 0, 'released': 0, 'errors': 0}\n\n"
        "    async def initialize(self) -> None:\n        async with self._lock:\n"
        "            if self._pool is not None:\n                return\n"
        "            self._pool = await asyncpg.create_pool(\n"
        "                host=self._config.host, port=self._config.port,\n"
        "                database=self._config.database,\n"
        "                min_size=self._config.min_size, max_size=self._config.max_size,\n"
        "                command_timeout=self._config.command_timeout,\n"
        "            )\n\n"
        "    @asynccontextmanager\n    async def acquire(self) -> AsyncIterator[asyncpg.Connection]:\n"
        "        if self._pool is None:\n            await self.initialize()\n"
        "        try:\n            async with self._pool.acquire() as conn:\n"
        "                self._stats['acquired'] += 1\n                yield conn\n"
        "        except Exception as e:\n            self._stats['errors'] += 1\n            raise\n"
        "        finally:\n            self._stats['released'] += 1\n"
    ),
    (
        "# Machine learning pipeline with cross-validation\n"
        "import numpy as np\nfrom sklearn.model_selection import StratifiedKFold\n"
        "from sklearn.preprocessing import StandardScaler\nfrom sklearn.metrics import (\n"
        "    accuracy_score, precision_recall_fscore_support, roc_auc_score\n)\n"
        "from typing import Dict, List, Tuple, Any\nimport logging\n\n"
        "logger = logging.getLogger(__name__)\n\n"
        "class MLPipeline:\n    def __init__(self, model, n_folds: int = 5, random_state: int = 42):\n"
        "        self.model = model\n        self.n_folds = n_folds\n"
        "        self.random_state = random_state\n        self.scaler = StandardScaler()\n"
        "        self.results: List[Dict[str, float]] = []\n\n"
        "    def run_cv(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:\n"
        "        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True,\n"
        "                             random_state=self.random_state)\n"
        "        fold_results = []\n"
        "        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):\n"
        "            X_train, X_val = X[train_idx], X[val_idx]\n"
        "            y_train, y_val = y[train_idx], y[val_idx]\n"
        "            X_train_scaled = self.scaler.fit_transform(X_train)\n"
        "            X_val_scaled = self.scaler.transform(X_val)\n"
        "            self.model.fit(X_train_scaled, y_train)\n"
        "            y_pred = self.model.predict(X_val_scaled)\n"
        "            y_prob = self.model.predict_proba(X_val_scaled)[:, 1]\n"
        "            acc = accuracy_score(y_val, y_pred)\n"
        "            prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='binary')\n"
        "            auc = roc_auc_score(y_val, y_prob)\n"
        "            fold_results.append({'accuracy': acc, 'precision': prec,\n"
        "                                'recall': rec, 'f1': f1, 'auc': auc})\n"
        "            logger.info(f'Fold {fold_idx}: acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}')\n"
    ),
    (
        "# REST API with rate limiting and caching\n"
        "from fastapi import FastAPI, HTTPException, Depends, Request\n"
        "from fastapi.middleware.cors import CORSMiddleware\n"
        "from pydantic import BaseModel, Field\nfrom datetime import datetime, timedelta\n"
        "from typing import Optional, Dict, Any\nimport hashlib\nimport time\n\n"
        "app = FastAPI(title='Data API', version='2.0')\n"
        "app.add_middleware(CORSMiddleware, allow_origins=['*'],\n"
        "                  allow_methods=['*'], allow_headers=['*'])\n\n"
        "class RateLimiter:\n    def __init__(self, max_requests: int = 100, window_seconds: int = 60):\n"
        "        self.max_requests = max_requests\n        self.window = window_seconds\n"
        "        self._requests: Dict[str, list] = {}\n\n"
        "    def check(self, client_id: str) -> bool:\n"
        "        now = time.time()\n        if client_id not in self._requests:\n"
        "            self._requests[client_id] = []\n"
        "        self._requests[client_id] = [\n"
        "            t for t in self._requests[client_id] if t > now - self.window\n        ]\n"
        "        if len(self._requests[client_id]) >= self.max_requests:\n"
        "            return False\n        self._requests[client_id].append(now)\n"
        "        return True\n\n"
        "rate_limiter = RateLimiter()\n\n"
        "@app.get('/api/v2/items/{item_id}')\n"
        "async def get_item(item_id: int, request: Request):\n"
        "    client_ip = request.client.host\n"
        "    if not rate_limiter.check(client_ip):\n"
        "        raise HTTPException(status_code=429, detail='Rate limit exceeded')\n"
    ),
]

RAG_DOCUMENTS = [
    (
        "Transformer Architecture Overview\n\n"
        "The Transformer architecture, introduced by Vaswani et al. (2017), is based on "
        "self-attention mechanisms that allow each position in a sequence to attend to all "
        "other positions. The key components are:\n\n"
        "1. Multi-Head Attention: Projects queries, keys, and values through multiple "
        "parallel attention heads, each computing scaled dot-product attention: "
        "Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V. Multiple heads allow the model "
        "to jointly attend to information from different representation subspaces.\n\n"
        "2. Feed-Forward Networks: Each layer contains a position-wise FFN with two linear "
        "transformations and a ReLU activation: FFN(x) = max(0, xW1+b1)W2+b2. Modern "
        "variants use SwiGLU or GeGLU activation functions.\n\n"
        "3. Layer Normalization: Applied before each sub-layer (pre-norm) or after "
        "(post-norm). Pre-norm has become standard in recent LLMs as it improves training "
        "stability.\n\n"
        "4. Positional Encoding: Since attention is permutation-invariant, position "
        "information must be injected. Options include sinusoidal encoding, learned "
        "embeddings, RoPE (Rotary Position Embedding), and ALiBi.\n\n"
        "5. KV Cache: During autoregressive generation, key and value projections from "
        "previous tokens are cached to avoid redundant computation. The cache grows "
        "linearly with sequence length and number of layers.\n"
    ),
    (
        "Distributed Systems: Consensus Protocols\n\n"
        "Consensus in distributed systems ensures that multiple nodes agree on a single "
        "value despite failures. The key protocols are:\n\n"
        "Paxos (Lamport, 1998): A two-phase protocol with Prepare and Accept phases. "
        "A proposer sends a Prepare request with proposal number n. Acceptors respond "
        "with a promise not to accept proposals numbered less than n. If a majority "
        "responds, the proposer sends an Accept request. Multi-Paxos optimizes for "
        "repeated consensus by electing a stable leader.\n\n"
        "Raft (Ongaro & Ousterhout, 2014): Designed for understandability. Uses leader "
        "election, log replication, and safety mechanisms. Leaders are elected via "
        "randomized timeouts. Log entries are replicated to a majority before committing. "
        "The protocol guarantees that committed entries are durable and consistent.\n\n"
        "Byzantine Fault Tolerance (BFT): Handles arbitrary (malicious) failures. PBFT "
        "requires 3f+1 nodes to tolerate f Byzantine failures. Modern variants like "
        "HotStuff use threshold signatures and pipelining for better performance.\n\n"
        "Practical considerations: Network partitions (CAP theorem), clock synchronization "
        "(logical vs physical clocks), and leader failover latency.\n"
    ),
]

CHAT_PREFIXES = [
    "Hello! I'd like to learn about cooking. ",
    "Hi there, can you help me plan a trip? ",
    "Good morning! I need advice on gardening. ",
    "Hey, I'm interested in learning photography. ",
]


def build_prefix_workload() -> List[Dict]:
    """Build diverse prefix workload.

    Returns list of dicts: {category, prefix_text, queries}
    """
    workload = []
    queries = [
        "Write a function to parse CSV files with custom delimiters.",
        "Explain the difference between threads and coroutines.",
        "How would you implement a retry mechanism with exponential backoff?",
        "Design a simple key-value store with TTL support.",
        "What are the best practices for logging in production systems?",
        "Implement a binary search tree with insertion and deletion.",
        "How do you handle database migrations in a microservices architecture?",
        "Write unit tests for a user authentication module.",
    ]

    for prompt in SYSTEM_PROMPTS:
        workload.append({"category": "system_prompt", "prefix_text": prompt, "queries": queries})
    for code in CODE_CONTEXTS:
        workload.append({"category": "code", "prefix_text": code, "queries": queries})
    for doc in RAG_DOCUMENTS:
        workload.append({"category": "rag", "prefix_text": doc, "queries": queries})
    for chat in CHAT_PREFIXES:
        workload.append({"category": "chat", "prefix_text": chat, "queries": queries})

    return workload


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_gpu_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**2)
    return 0.0


# ---------------------------------------------------------------------------
# Manual capacity enforcement via prefix tree eviction
# ---------------------------------------------------------------------------

def enforce_capacity(manager: DeltaCacheManager, policy: EvictionPolicy, max_entries: int):
    """Evict cache entries until num_cached <= max_entries.

    Scores all candidates once, then evicts the required number in bulk.
    """
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
        elif hasattr(policy, '_compute_score'):
            score = policy._compute_score(node)
        else:
            score = node.last_access
        candidates.append((score, id(node), node))

    if not candidates:
        return 0

    # Sort ascending (lowest score = evict first) and evict the bottom N
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


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

@dataclass
class PolicyResult:
    policy_name: str
    capacity_pct: int
    max_entries: int
    total_requests: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    avg_latency_ms: float
    std_latency_ms: float
    p50_latency_ms: float
    p99_latency_ms: float
    tokens_reused: int
    total_tokens: int
    reuse_rate: float
    num_evictions: int


def run_single(
    adapter: LlamaStyleAdapter,
    model_name: str,
    workload: List[Dict],
    policy_name: str,
    policy: EvictionPolicy,
    max_entries: int,
    capacity_pct: int,
    n_runs: int = 3,
) -> PolicyResult:
    """Run experiment for one (policy, capacity) pair."""
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

            # Enforce capacity limit
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
    sorted_lat = sorted(all_latencies)

    return PolicyResult(
        policy_name=policy_name,
        capacity_pct=capacity_pct,
        max_entries=max_entries,
        total_requests=n_req,
        cache_hits=int(round(avg_hit)),
        cache_misses=n_req - int(round(avg_hit)),
        hit_rate=avg_hit / n_req,
        avg_latency_ms=statistics.mean(all_latencies),
        std_latency_ms=statistics.stdev(all_latencies) if len(all_latencies) > 1 else 0,
        p50_latency_ms=sorted_lat[len(sorted_lat) // 2],
        p99_latency_ms=sorted_lat[int(len(sorted_lat) * 0.99)],
        tokens_reused=int(statistics.mean(all_reused)),
        total_tokens=int(statistics.mean(all_total)),
        reuse_rate=statistics.mean(all_reused) / max(1, statistics.mean(all_total)),
        num_evictions=int(statistics.mean(all_evictions)),
    )


def run_experiment(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device: str = "cuda:0",
    n_runs: int = 3,
):
    print(f"\n{'='*70}")
    print(f"Memory Pressure Eviction Experiment (Real Model)")
    print(f"Model: {model_name}   Device: {device}   Runs: {n_runs}")
    print(f"{'='*70}\n")

    adapter = LlamaStyleAdapter.from_pretrained(model_name, device=device)
    print(f"Model loaded. GPU: {get_gpu_mb():.0f} MB")

    workload = build_prefix_workload()
    num_prefixes = len(workload)
    total_queries = sum(len(w["queries"]) for w in workload)
    print(f"Prefixes: {num_prefixes}, Queries/prefix: {len(workload[0]['queries'])}, Total: {total_queries}")

    for w in workload:
        toks = adapter.tokenizer.encode(w["prefix_text"])
        print(f"  {w['category']:15s}: {len(toks):4d} tokens")

    # Baseline latency (no cache)
    print("\nMeasuring baseline latency (no cache)...")
    baseline_lats = []
    for pidx in range(min(6, num_prefixes)):
        full = workload[pidx]["prefix_text"] + " " + workload[pidx]["queries"][0]
        tokens = adapter.tokenizer.encode(full)
        input_ids = torch.tensor([tokens], device=device)
        pos_ids = adapter.get_position_ids(len(tokens))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        adapter.compute_kv(input_ids, pos_ids)
        torch.cuda.synchronize()
        baseline_lats.append((time.perf_counter() - t0) * 1000)
        clear_gpu()
    baseline_ms = statistics.mean(baseline_lats)
    print(f"Baseline: {baseline_ms:.1f} ms")

    # Policies
    policies = {
        "lru": LRUEvictionPolicy(),
        "lfu": LFUEvictionPolicy(),
        "composite": CompositeEvictionPolicy(prefer_offload=False),
    }

    # Capacity levels: fraction of unique prefixes
    # With prefix tree's internal prefix sharing, the actual num_cached
    # will be larger than num_prefixes. We measure 100% first to learn
    # the real entry count, then scale down.

    # First: determine the actual number of cached nodes at full capacity
    print("\nDetermining full-capacity cache size...")
    config = DeltaCacheConfig.for_model(model_name)
    config.device = device
    test_mgr = DeltaCacheManager(config)
    for pidx, item in enumerate(workload):
        for q in item["queries"][:1]:  # just one query per prefix
            full = item["prefix_text"] + " " + q
            tokens = adapter.tokenizer.encode(full)
            test_mgr.compute_incremental(tokens, adapter.compute_kv)
    full_entries = test_mgr.prefix_tree.num_cached
    print(f"Full-capacity entries: {full_entries} (from {num_prefixes} prefixes)")
    del test_mgr
    clear_gpu()

    capacity_levels = [10, 20, 30, 50, 100]
    results = []

    for cap_pct in capacity_levels:
        max_entries = max(1, int(full_entries * cap_pct / 100))
        print(f"\n{'─'*60}")
        print(f"Capacity: {cap_pct}% → max {max_entries} cached entries")
        print(f"{'─'*60}")

        for pname, policy in policies.items():
            print(f"  {pname.upper():15s}", end=" ", flush=True)
            r = run_single(
                adapter, model_name, workload,
                pname, policy, max_entries, cap_pct, n_runs,
            )
            results.append(r)
            speedup = baseline_ms / r.avg_latency_ms if r.avg_latency_ms > 0 else 0
            print(f"hit={r.hit_rate*100:5.1f}%  lat={r.avg_latency_ms:.1f}ms  "
                  f"speedup={speedup:.2f}x  reuse={r.reuse_rate*100:.1f}%  "
                  f"evict={r.num_evictions}")

    # Summary table
    print(f"\n\n{'='*70}")
    print("SUMMARY: Hit Rate by Policy × Capacity")
    print(f"{'='*70}")
    header = f"{'Policy':12s}" + "".join(f"  {c:>4d}%" for c in capacity_levels)
    print(header)
    print("-" * len(header))
    for pname in policies:
        row = f"{pname:12s}"
        for cap in capacity_levels:
            r = next(x for x in results if x.policy_name == pname and x.capacity_pct == cap)
            row += f"  {r.hit_rate*100:5.1f}%"
        print(row)

    print(f"\n{'Delta vs LRU':12s}" + "".join(f"  {c:>4d}%" for c in capacity_levels))
    print("-" * len(header))
    for pname in ["lfu", "composite"]:
        row = f"{pname:12s}"
        for cap in capacity_levels:
            lru = next(x for x in results if x.policy_name == "lru" and x.capacity_pct == cap)
            cur = next(x for x in results if x.policy_name == pname and x.capacity_pct == cap)
            if lru.hit_rate > 0:
                delta = (cur.hit_rate - lru.hit_rate) / lru.hit_rate * 100
                row += f"  {delta:+5.1f}%"
            else:
                row += "    N/A"
        print(row)

    # Speedup table
    print(f"\n{'Speedup vs no-cache':30s}")
    print("-" * len(header))
    for pname in policies:
        row = f"{pname:12s}"
        for cap in capacity_levels:
            r = next(x for x in results if x.policy_name == pname and x.capacity_pct == cap)
            spd = baseline_ms / r.avg_latency_ms if r.avg_latency_ms > 0 else 0
            row += f"  {spd:5.2f}x"
        print(row)

    # Save
    output = {
        "metadata": {
            "experiment": "eviction_pressure_real_model",
            "model": model_name,
            "device": device,
            "num_prefixes": num_prefixes,
            "full_cache_entries": full_entries,
            "queries_per_prefix": len(workload[0]["queries"]),
            "total_queries": total_queries,
            "n_runs": n_runs,
            "baseline_latency_ms": round(baseline_ms, 2),
            "timestamp": datetime.now().isoformat(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "prefix_token_lengths": {
                f"{w['category']}_{i}": len(adapter.tokenizer.encode(w["prefix_text"]))
                for i, w in enumerate(workload)
            },
        },
        "capacity_levels": capacity_levels,
        "policies": list(policies.keys()),
        "results": [asdict(r) for r in results],
    }

    safe_model = model_name.split("/")[-1].lower().replace("-", "_")
    out_path = RESULTS_DIR / f"eviction_pressure_{safe_model}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    run_experiment(model_name=args.model, device=args.device, n_runs=args.runs)
