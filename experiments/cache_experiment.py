"""Cache hit rate and performance experiments for DeltaCache."""

import json
import time
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
from tqdm import tqdm

import torch
from torch import Tensor
from transformers import AutoTokenizer

from deltacache import DeltaCacheManager, DeltaCacheConfig
from deltacache.core.prefix_tree import PrefixTree


DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"


@dataclass
class ExperimentResult:
    """Result of a cache experiment."""
    name: str
    num_requests: int
    total_tokens: int
    cached_tokens: int
    computed_tokens: int
    cache_hit_rate: float
    token_reuse_rate: float
    total_time_ms: float
    avg_latency_ms: float
    throughput_tokens_per_sec: float


class MockKVCompute:
    """Mock KV computation with configurable latency."""

    def __init__(
        self,
        num_layers: int = 32,
        num_heads: int = 32,
        head_dim: int = 128,
        latency_per_token_ms: float = 0.5,
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.latency_per_token = latency_per_token_ms / 1000

    def __call__(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        past_key_values: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor]:
        batch_size, seq_len = input_ids.shape
        time.sleep(self.latency_per_token * seq_len)

        key = torch.randn(self.num_layers, seq_len, self.num_heads, self.head_dim, dtype=torch.float16)
        value = torch.randn(self.num_layers, seq_len, self.num_heads, self.head_dim, dtype=torch.float16)
        return key, value


def load_tokenized_conversations(tokenizer, max_samples: int = 1000) -> List[List[int]]:
    """Load and tokenize UltraChat conversations."""
    with open(DATA_DIR / "ultrachat.json") as f:
        data = json.load(f)

    tokenized = []
    for item in data[:max_samples]:
        if "messages" not in item:
            continue

        text = ""
        for msg in item["messages"]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            text += f"<|{role}|>\n{content}\n"

        tokens = tokenizer.encode(text, add_special_tokens=True)
        # Limit to reasonable length
        tokenized.append(tokens[:2048])

    return tokenized


def experiment_system_prompt_sharing(
    tokenizer,
    conversations: List[List[int]],
    system_prompt_lengths: List[int] = [50, 100, 200, 500],
) -> List[ExperimentResult]:
    """Experiment: varying system prompt lengths."""
    results = []

    for prompt_len in system_prompt_lengths:
        print(f"\n  Testing system prompt length: {prompt_len} tokens")

        # Create system prompt
        system_prompt = list(range(1, prompt_len + 1))  # Simple token sequence

        # Create config and manager
        config = DeltaCacheConfig(
            num_layers=32,
            num_heads=32,
            head_dim=128,
            gpu_memory_limit=4 * 1024 * 1024 * 1024,
            device="cpu",
        )
        manager = DeltaCacheManager(config)
        compute_fn = MockKVCompute(latency_per_token_ms=0.1)

        # Prepare requests: system_prompt + conversation
        requests = []
        for conv in conversations[:100]:
            # Take first 200 tokens of each conversation
            user_tokens = conv[:200]
            requests.append(system_prompt + user_tokens)

        # Process requests
        total_time = 0
        latencies = []

        for i, req in enumerate(tqdm(requests, desc=f"    Processing", leave=False)):
            start = time.time()
            result = manager.compute_incremental(req, compute_fn)
            elapsed = (time.time() - start) * 1000
            total_time += elapsed
            latencies.append(elapsed)

        stats = manager.get_stats()
        total_tokens = stats["total_tokens"]

        exp_result = ExperimentResult(
            name=f"system_prompt_{prompt_len}",
            num_requests=len(requests),
            total_tokens=total_tokens,
            cached_tokens=stats["cached_tokens"],
            computed_tokens=stats["computed_tokens"],
            cache_hit_rate=stats["hit_rate"],
            token_reuse_rate=stats["token_reuse_rate"],
            total_time_ms=total_time,
            avg_latency_ms=np.mean(latencies),
            throughput_tokens_per_sec=total_tokens / (total_time / 1000),
        )
        results.append(exp_result)

        print(f"    Hit rate: {exp_result.cache_hit_rate:.1%}, Reuse: {exp_result.token_reuse_rate:.1%}")

    return results


def experiment_rag_scenario(
    n_documents: int = 10,
    doc_lengths: List[int] = [200, 500, 1000],
    queries_per_doc: int = 5,
) -> List[ExperimentResult]:
    """Experiment: RAG document caching."""
    results = []

    for doc_len in doc_lengths:
        print(f"\n  Testing document length: {doc_len} tokens")

        config = DeltaCacheConfig(
            num_layers=32,
            num_heads=32,
            head_dim=128,
            gpu_memory_limit=4 * 1024 * 1024 * 1024,
            device="cpu",
        )
        manager = DeltaCacheManager(config)
        compute_fn = MockKVCompute(latency_per_token_ms=0.1)

        # Create documents
        documents = [list(range(i * 10000, i * 10000 + doc_len)) for i in range(n_documents)]

        # Create queries
        queries = [list(range(50000 + i * 100, 50000 + i * 100 + 50)) for i in range(queries_per_doc)]

        # Create requests: each document with multiple queries
        requests = []
        for doc in documents:
            for query in queries:
                requests.append(doc + query)

        # Shuffle to simulate real traffic
        random.shuffle(requests)

        # Process
        total_time = 0
        latencies = []

        for req in tqdm(requests, desc=f"    Processing", leave=False):
            start = time.time()
            result = manager.compute_incremental(req, compute_fn)
            elapsed = (time.time() - start) * 1000
            total_time += elapsed
            latencies.append(elapsed)

        stats = manager.get_stats()
        total_tokens = stats["total_tokens"]

        exp_result = ExperimentResult(
            name=f"rag_doc_{doc_len}",
            num_requests=len(requests),
            total_tokens=total_tokens,
            cached_tokens=stats["cached_tokens"],
            computed_tokens=stats["computed_tokens"],
            cache_hit_rate=stats["hit_rate"],
            token_reuse_rate=stats["token_reuse_rate"],
            total_time_ms=total_time,
            avg_latency_ms=np.mean(latencies),
            throughput_tokens_per_sec=total_tokens / (total_time / 1000),
        )
        results.append(exp_result)

        print(f"    Hit rate: {exp_result.cache_hit_rate:.1%}, Reuse: {exp_result.token_reuse_rate:.1%}")

    return results


def experiment_few_shot_learning(
    n_examples: List[int] = [1, 3, 5, 10],
    example_length: int = 100,
    n_queries: int = 50,
) -> List[ExperimentResult]:
    """Experiment: Few-shot learning with shared examples."""
    results = []

    for n_ex in n_examples:
        print(f"\n  Testing {n_ex}-shot learning")

        config = DeltaCacheConfig(
            num_layers=32,
            num_heads=32,
            head_dim=128,
            gpu_memory_limit=4 * 1024 * 1024 * 1024,
            device="cpu",
        )
        manager = DeltaCacheManager(config)
        compute_fn = MockKVCompute(latency_per_token_ms=0.1)

        # Create few-shot examples (shared prefix)
        few_shot_prefix = list(range(1, n_ex * example_length + 1))

        # Create queries
        requests = []
        for i in range(n_queries):
            query = list(range(100000 + i * 100, 100000 + i * 100 + 50))
            requests.append(few_shot_prefix + query)

        # Process
        total_time = 0
        latencies = []

        for req in tqdm(requests, desc=f"    Processing", leave=False):
            start = time.time()
            result = manager.compute_incremental(req, compute_fn)
            elapsed = (time.time() - start) * 1000
            total_time += elapsed
            latencies.append(elapsed)

        stats = manager.get_stats()
        total_tokens = stats["total_tokens"]

        exp_result = ExperimentResult(
            name=f"few_shot_{n_ex}",
            num_requests=len(requests),
            total_tokens=total_tokens,
            cached_tokens=stats["cached_tokens"],
            computed_tokens=stats["computed_tokens"],
            cache_hit_rate=stats["hit_rate"],
            token_reuse_rate=stats["token_reuse_rate"],
            total_time_ms=total_time,
            avg_latency_ms=np.mean(latencies),
            throughput_tokens_per_sec=total_tokens / (total_time / 1000),
        )
        results.append(exp_result)

        print(f"    Hit rate: {exp_result.cache_hit_rate:.1%}, Reuse: {exp_result.token_reuse_rate:.1%}")

    return results


def experiment_baseline_comparison(
    conversations: List[List[int]],
    system_prompt_len: int = 200,
) -> Dict:
    """Compare DeltaCache vs baseline (no caching)."""
    print("\n  Running baseline comparison...")

    # System prompt
    system_prompt = list(range(1, system_prompt_len + 1))

    # Prepare requests
    requests = []
    for conv in conversations[:50]:
        requests.append(system_prompt + conv[:200])

    compute_fn = MockKVCompute(latency_per_token_ms=0.1)

    # Baseline: compute everything from scratch
    print("    Running baseline (no cache)...")
    baseline_time = 0
    baseline_tokens = 0
    for req in tqdm(requests, desc="    Baseline", leave=False):
        input_ids = torch.tensor([req])
        position_ids = torch.arange(len(req)).unsqueeze(0)
        start = time.time()
        compute_fn(input_ids, position_ids)
        baseline_time += (time.time() - start) * 1000
        baseline_tokens += len(req)

    # DeltaCache
    print("    Running with DeltaCache...")
    config = DeltaCacheConfig(
        num_layers=32,
        num_heads=32,
        head_dim=128,
        gpu_memory_limit=4 * 1024 * 1024 * 1024,
        device="cpu",
    )
    manager = DeltaCacheManager(config)

    delta_time = 0
    for req in tqdm(requests, desc="    DeltaCache", leave=False):
        start = time.time()
        manager.compute_incremental(req, compute_fn)
        delta_time += (time.time() - start) * 1000

    stats = manager.get_stats()

    return {
        "baseline_time_ms": baseline_time,
        "baseline_tokens": baseline_tokens,
        "deltacache_time_ms": delta_time,
        "deltacache_tokens_computed": stats["computed_tokens"],
        "speedup": baseline_time / delta_time,
        "token_savings": 1 - stats["computed_tokens"] / baseline_tokens,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("DeltaCache Comprehensive Experiments")
    print("="*70)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Load data
    print("Loading conversation data...")
    conversations = load_tokenized_conversations(tokenizer, max_samples=500)
    print(f"Loaded {len(conversations)} conversations")

    all_results = {}

    # Experiment 1: System Prompt Sharing
    print("\n" + "="*70)
    print("Experiment 1: System Prompt Length Impact")
    print("="*70)
    sys_results = experiment_system_prompt_sharing(tokenizer, conversations)
    all_results["system_prompt"] = [asdict(r) for r in sys_results]

    # Experiment 2: RAG Document Caching
    print("\n" + "="*70)
    print("Experiment 2: RAG Document Caching")
    print("="*70)
    rag_results = experiment_rag_scenario()
    all_results["rag"] = [asdict(r) for r in rag_results]

    # Experiment 3: Few-Shot Learning
    print("\n" + "="*70)
    print("Experiment 3: Few-Shot Learning")
    print("="*70)
    few_shot_results = experiment_few_shot_learning()
    all_results["few_shot"] = [asdict(r) for r in few_shot_results]

    # Experiment 4: Baseline Comparison
    print("\n" + "="*70)
    print("Experiment 4: Baseline Comparison")
    print("="*70)
    baseline_results = experiment_baseline_comparison(conversations)
    all_results["baseline_comparison"] = baseline_results

    # Save results
    with open(RESULTS_DIR / "cache_experiments.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)

    print("\n1. System Prompt Impact:")
    print(f"   {'Prompt Len':<12} {'Hit Rate':<12} {'Token Reuse':<12} {'Avg Latency':<12}")
    print("-" * 50)
    for r in sys_results:
        prompt_len = r.name.split("_")[-1]
        print(f"   {prompt_len:<12} {r.cache_hit_rate*100:>8.1f}%   {r.token_reuse_rate*100:>8.1f}%   {r.avg_latency_ms:>8.1f}ms")

    print("\n2. RAG Document Caching:")
    print(f"   {'Doc Length':<12} {'Hit Rate':<12} {'Token Reuse':<12} {'Avg Latency':<12}")
    print("-" * 50)
    for r in rag_results:
        doc_len = r.name.split("_")[-1]
        print(f"   {doc_len:<12} {r.cache_hit_rate*100:>8.1f}%   {r.token_reuse_rate*100:>8.1f}%   {r.avg_latency_ms:>8.1f}ms")

    print("\n3. Few-Shot Learning:")
    print(f"   {'N-Shot':<12} {'Hit Rate':<12} {'Token Reuse':<12} {'Avg Latency':<12}")
    print("-" * 50)
    for r in few_shot_results:
        n_shot = r.name.split("_")[-1]
        print(f"   {n_shot:<12} {r.cache_hit_rate*100:>8.1f}%   {r.token_reuse_rate*100:>8.1f}%   {r.avg_latency_ms:>8.1f}ms")

    print("\n4. Baseline Comparison:")
    print(f"   Baseline time: {baseline_results['baseline_time_ms']:.0f}ms")
    print(f"   DeltaCache time: {baseline_results['deltacache_time_ms']:.0f}ms")
    print(f"   Speedup: {baseline_results['speedup']:.2f}x")
    print(f"   Token savings: {baseline_results['token_savings']*100:.1f}%")

    print(f"\n\nResults saved to {RESULTS_DIR / 'cache_experiments.json'}")


if __name__ == "__main__":
    main()
