"""Performance benchmark for DeltaCache."""

import time
import torch
from typing import Tuple, Optional, List
from torch import Tensor
from dataclasses import dataclass

from deltacache import DeltaCacheManager, DeltaCacheConfig


@dataclass
class BenchmarkResult:
    """Benchmark result."""
    name: str
    total_tokens: int
    cached_tokens: int
    computed_tokens: int
    time_ms: float
    tokens_per_sec: float


class MockModel:
    """Mock transformer model for benchmarking."""

    def __init__(self, num_layers: int = 32, num_heads: int = 32, head_dim: int = 128):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.compute_time_per_token = 0.0005  # 0.5ms per token

    def compute_kv(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        past_key_values: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Compute KV cache."""
        batch_size, seq_len = input_ids.shape

        # Simulate computation time
        time.sleep(self.compute_time_per_token * seq_len)

        # Generate KV cache
        key = torch.randn(
            self.num_layers, seq_len, self.num_heads, self.head_dim,
            dtype=torch.float16
        )
        value = torch.randn(
            self.num_layers, seq_len, self.num_heads, self.head_dim,
            dtype=torch.float16
        )
        return key, value


def benchmark_prefix_reuse(
    manager: DeltaCacheManager,
    model: MockModel,
    system_prompt_len: int,
    user_query_len: int,
    num_requests: int,
) -> List[BenchmarkResult]:
    """Benchmark prefix reuse scenario."""
    results = []

    # System prompt tokens
    system_prompt = list(range(system_prompt_len))

    # First request - cold start
    request1 = system_prompt + list(range(1000, 1000 + user_query_len))
    start = time.time()
    result = manager.compute_incremental(request1, model.compute_kv)
    elapsed = (time.time() - start) * 1000

    results.append(BenchmarkResult(
        name="Cold start",
        total_tokens=len(request1),
        cached_tokens=result.matched_length,
        computed_tokens=result.computed_length,
        time_ms=elapsed,
        tokens_per_sec=len(request1) / (elapsed / 1000),
    ))

    # Subsequent requests - warm cache
    for i in range(num_requests - 1):
        request = system_prompt + list(range(2000 + i * 100, 2000 + i * 100 + user_query_len))
        start = time.time()
        result = manager.compute_incremental(request, model.compute_kv)
        elapsed = (time.time() - start) * 1000

        results.append(BenchmarkResult(
            name=f"Request {i + 2}",
            total_tokens=len(request),
            cached_tokens=result.matched_length,
            computed_tokens=result.computed_length,
            time_ms=elapsed,
            tokens_per_sec=len(request) / (elapsed / 1000),
        ))

    return results


def benchmark_no_cache(
    model: MockModel,
    system_prompt_len: int,
    user_query_len: int,
    num_requests: int,
) -> List[BenchmarkResult]:
    """Benchmark without caching (baseline)."""
    results = []

    for i in range(num_requests):
        total_len = system_prompt_len + user_query_len
        input_ids = torch.randint(0, 50000, (1, total_len))
        position_ids = torch.arange(total_len).unsqueeze(0)

        start = time.time()
        model.compute_kv(input_ids, position_ids)
        elapsed = (time.time() - start) * 1000

        results.append(BenchmarkResult(
            name=f"Request {i + 1}",
            total_tokens=total_len,
            cached_tokens=0,
            computed_tokens=total_len,
            time_ms=elapsed,
            tokens_per_sec=total_len / (elapsed / 1000),
        ))

    return results


def print_results(title: str, results: List[BenchmarkResult]):
    """Print benchmark results."""
    print(f"\n{title}")
    print("-" * 80)
    print(f"{'Name':<15} {'Total':>8} {'Cached':>8} {'Computed':>10} {'Time(ms)':>10} {'Tok/s':>10}")
    print("-" * 80)

    total_time = 0
    total_computed = 0

    for r in results:
        print(f"{r.name:<15} {r.total_tokens:>8} {r.cached_tokens:>8} {r.computed_tokens:>10} {r.time_ms:>10.1f} {r.tokens_per_sec:>10.0f}")
        total_time += r.time_ms
        total_computed += r.computed_tokens

    print("-" * 80)
    print(f"{'Total':<15} {'':<8} {'':<8} {total_computed:>10} {total_time:>10.1f}")

    return total_time, total_computed


def main():
    print("="*80)
    print("DeltaCache Performance Benchmark")
    print("="*80)

    # Configuration
    system_prompt_len = 500   # 500 token system prompt
    user_query_len = 50       # 50 token user query
    num_requests = 10

    print(f"\nConfiguration:")
    print(f"  System prompt length: {system_prompt_len} tokens")
    print(f"  User query length: {user_query_len} tokens")
    print(f"  Number of requests: {num_requests}")

    # Create model
    model = MockModel(num_layers=32, num_heads=32, head_dim=128)

    # Benchmark without cache (baseline)
    print("\n" + "="*80)
    print("Baseline: No Caching")
    print("="*80)
    baseline_results = benchmark_no_cache(model, system_prompt_len, user_query_len, num_requests)
    baseline_time, baseline_computed = print_results("Baseline Results", baseline_results)

    # Benchmark with DeltaCache
    print("\n" + "="*80)
    print("DeltaCache: Incremental Computation")
    print("="*80)

    config = DeltaCacheConfig(
        num_layers=32,
        num_heads=32,
        head_dim=128,
        gpu_memory_limit=2 * 1024 * 1024 * 1024,  # 2GB
        device="cpu",
    )
    manager = DeltaCacheManager(config)

    deltacache_results = benchmark_prefix_reuse(
        manager, model, system_prompt_len, user_query_len, num_requests
    )
    deltacache_time, deltacache_computed = print_results("DeltaCache Results", deltacache_results)

    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    speedup = baseline_time / deltacache_time
    compute_saved = (baseline_computed - deltacache_computed) / baseline_computed * 100

    print(f"\n  Baseline total time:    {baseline_time:>10.1f} ms")
    print(f"  DeltaCache total time:  {deltacache_time:>10.1f} ms")
    print(f"  Speedup:                {speedup:>10.2f}x")
    print(f"\n  Baseline tokens computed:   {baseline_computed:>8}")
    print(f"  DeltaCache tokens computed: {deltacache_computed:>8}")
    print(f"  Computation saved:          {compute_saved:>7.1f}%")

    # Cache stats
    stats = manager.get_stats()
    print(f"\n  Cache hit rate:         {stats['hit_rate']*100:>10.1f}%")
    print(f"  Token reuse rate:       {stats['token_reuse_rate']*100:>10.1f}%")


if __name__ == "__main__":
    main()
