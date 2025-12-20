"""DeltaCache demonstration script."""

import time
import torch
from typing import Tuple, Optional
from torch import Tensor

from deltacache import DeltaCacheManager, DeltaCacheConfig
from deltacache.core.prefix_tree import PrefixTree
from deltacache.core.cache_block import CacheBlock


def mock_kv_compute(
    input_ids: Tensor,
    position_ids: Tensor,
    past_key_values: Optional[Tuple[Tensor, Tensor]] = None,
) -> Tuple[Tensor, Tensor]:
    """Mock KV computation simulating transformer forward pass."""
    batch_size, seq_len = input_ids.shape
    # Simulate computation time
    time.sleep(0.001 * seq_len)  # 1ms per token

    # Generate mock KV cache
    num_layers, num_heads, head_dim = 32, 32, 128
    key = torch.randn(num_layers, seq_len, num_heads, head_dim, dtype=torch.float16)
    value = torch.randn(num_layers, seq_len, num_heads, head_dim, dtype=torch.float16)
    return key, value


def demo_prefix_tree():
    """Demonstrate prefix tree operations."""
    print("\n" + "="*60)
    print("Demo 1: Prefix Tree Basic Operations")
    print("="*60)

    tree = PrefixTree()

    # Insert sequences
    sequences = [
        [1, 2, 3, 4, 5],        # System prompt
        [1, 2, 3, 6, 7],        # Same prefix, different suffix
        [1, 2, 3, 4, 5, 8, 9],  # Extended sequence
        [10, 11, 12],           # Completely different
    ]

    print("\nInserting sequences:")
    for seq in sequences:
        tree.insert(seq)
        print(f"  Inserted: {seq}")

    print(f"\nTree size: {tree.size} nodes")

    # Lookup operations
    print("\nLookup operations:")
    queries = [
        [1, 2, 3, 4, 5],        # Exact match
        [1, 2, 3, 100, 101],    # Partial match
        [99, 100, 101],         # No match
    ]

    for query in queries:
        result = tree.lookup(query)
        print(f"  Query: {query}")
        print(f"    Matched length: {result.matched_length}")
        print(f"    Has match: {result.has_match}")


def demo_incremental_compute():
    """Demonstrate incremental KV computation."""
    print("\n" + "="*60)
    print("Demo 2: Incremental KV Computation")
    print("="*60)

    config = DeltaCacheConfig(
        num_layers=32,
        num_heads=32,
        head_dim=128,
        gpu_memory_limit=1024 * 1024 * 1024,  # 1GB
        device="cpu",
    )
    manager = DeltaCacheManager(config)

    # Simulate system prompt (shared prefix)
    system_prompt = list(range(1, 101))  # 100 tokens

    print(f"\n1. First request with system prompt ({len(system_prompt)} tokens):")
    start = time.time()
    result1 = manager.compute_incremental(system_prompt, mock_kv_compute)
    time1 = time.time() - start
    print(f"   Matched: {result1.matched_length}, Computed: {result1.computed_length}")
    print(f"   Time: {time1*1000:.1f}ms")

    # Second request with same prefix + new content
    request2 = system_prompt + list(range(101, 121))  # +20 tokens
    print(f"\n2. Second request (same prefix + 20 new tokens):")
    start = time.time()
    result2 = manager.compute_incremental(request2, mock_kv_compute)
    time2 = time.time() - start
    print(f"   Matched: {result2.matched_length}, Computed: {result2.computed_length}")
    print(f"   Time: {time2*1000:.1f}ms")
    print(f"   Speedup: {time1/time2:.1f}x (cached {result2.matched_length} tokens)")

    # Third request with different content but same prefix
    request3 = system_prompt + list(range(201, 231))  # +30 different tokens
    print(f"\n3. Third request (same prefix + 30 different tokens):")
    start = time.time()
    result3 = manager.compute_incremental(request3, mock_kv_compute)
    time3 = time.time() - start
    print(f"   Matched: {result3.matched_length}, Computed: {result3.computed_length}")
    print(f"   Time: {time3*1000:.1f}ms")

    # Statistics
    stats = manager.get_stats()
    print(f"\n4. Cache Statistics:")
    print(f"   Total lookups: {stats['total_lookups']}")
    print(f"   Cache hits: {stats['cache_hits']}")
    print(f"   Hit rate: {stats['hit_rate']:.1%}")
    print(f"   Token reuse rate: {stats['token_reuse_rate']:.1%}")


def demo_eviction():
    """Demonstrate cache eviction."""
    print("\n" + "="*60)
    print("Demo 3: Cache Eviction")
    print("="*60)

    config = DeltaCacheConfig(
        num_layers=2,
        num_heads=4,
        head_dim=8,
        gpu_memory_limit=10000,  # Very small limit to trigger eviction
        cpu_memory_limit=50000,
        eviction_threshold=0.8,
        device="cpu",
    )
    manager = DeltaCacheManager(config)

    print("\nInserting cache blocks until eviction triggers...")

    for i in range(20):
        tokens = list(range(i * 50, (i + 1) * 50))
        key = torch.randn(2, 50, 4, 8, dtype=torch.float16)
        value = torch.randn(2, 50, 4, 8, dtype=torch.float16)
        manager.insert(tokens, key, value)

        mem_stats = manager.get_memory_stats()
        if (i + 1) % 5 == 0:
            print(f"  After {i+1} insertions:")
            print(f"    Cached sequences: {manager.num_cached_sequences}")
            print(f"    GPU blocks: {mem_stats.num_gpu_blocks}")
            print(f"    CPU blocks: {mem_stats.num_cpu_blocks}")

    stats = manager.get_stats()
    print(f"\nEviction statistics:")
    print(f"  Total evictions: {stats['num_evictions']}")
    print(f"  Offloads to CPU: {stats['num_offloads']}")


def demo_shared_prefix_detection():
    """Demonstrate shared prefix detection for batch scheduling."""
    print("\n" + "="*60)
    print("Demo 4: Shared Prefix Detection")
    print("="*60)

    tree = PrefixTree()

    # Simulate common system prompts
    system_prompt = list(range(1, 51))  # 50 tokens

    # Create cache for system prompt
    key = torch.randn(2, 50, 4, 8, dtype=torch.float16)
    value = torch.randn(2, 50, 4, 8, dtype=torch.float16)
    tree.insert_with_kv(system_prompt, key, value)

    # Incoming requests batch
    requests = [
        system_prompt + [100, 101, 102],      # User 1
        system_prompt + [200, 201, 202, 203], # User 2
        system_prompt + [300],                 # User 3
        [500, 501, 502, 503],                  # Different system (no shared prefix)
    ]

    print("\nAnalyzing request batch:")
    for i, req in enumerate(requests):
        result = tree.lookup(req)
        print(f"  Request {i+1}: {len(req)} tokens")
        print(f"    Shared prefix: {result.matched_length} tokens")
        print(f"    New tokens to compute: {len(req) - result.matched_length}")
        print(f"    Has cached KV: {result.kv_cache is not None}")

    # Find common prefix
    common_prefix, _ = tree.find_shared_prefix(requests[:3])
    print(f"\nCommon prefix among requests 1-3: {len(common_prefix)} tokens")


def demo_batch_optimization():
    """Demonstrate batch computation optimization."""
    print("\n" + "="*60)
    print("Demo 5: Batch Computation Optimization")
    print("="*60)

    config = DeltaCacheConfig(
        num_layers=32,
        num_heads=32,
        head_dim=128,
        gpu_memory_limit=1024 * 1024 * 1024,
        device="cpu",
    )
    manager = DeltaCacheManager(config)

    # Pre-cache system prompt
    system_prompt = list(range(1, 101))
    result = manager.compute_incremental(system_prompt, mock_kv_compute)
    print(f"Pre-cached system prompt: {len(system_prompt)} tokens")

    # Batch of requests
    batch = [
        system_prompt + list(range(101, 111)),  # +10 new
        system_prompt + list(range(201, 221)),  # +20 new
        system_prompt + list(range(301, 316)),  # +15 new
    ]

    print(f"\nProcessing batch of {len(batch)} requests...")

    # Sequential processing
    start = time.time()
    for req in batch:
        manager.compute_incremental(req, mock_kv_compute, store_result=False)
    seq_time = time.time() - start

    # Batch processing
    manager.reset_stats()
    start = time.time()
    results = manager.compute_batch(batch, mock_kv_compute, store_results=False)
    batch_time = time.time() - start

    print(f"\nResults:")
    for i, r in enumerate(results):
        print(f"  Request {i+1}: matched={r.matched_length}, computed={r.computed_length}")

    print(f"\nTiming:")
    print(f"  Sequential: {seq_time*1000:.1f}ms")
    print(f"  Batch: {batch_time*1000:.1f}ms")

    stats = manager.get_stats()
    print(f"\nCompute savings: {stats['compute_savings']:.1%}")


if __name__ == "__main__":
    print("DeltaCache Demonstration")
    print("========================")

    demo_prefix_tree()
    demo_incremental_compute()
    demo_eviction()
    demo_shared_prefix_detection()
    demo_batch_optimization()

    print("\n" + "="*60)
    print("All demos completed!")
    print("="*60)
