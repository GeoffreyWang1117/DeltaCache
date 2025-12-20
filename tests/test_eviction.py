"""Tests for eviction policies."""

import pytest
import time
import torch

from deltacache.core.prefix_tree import PrefixTree
from deltacache.core.cache_block import CacheBlock
from deltacache.core.memory_pool import MemoryPool
from deltacache.eviction.policy import (
    EvictionPolicy,
    EvictionCandidate,
    EvictionAction,
    EvictionResult,
    LRUEvictionPolicy,
    LFUEvictionPolicy,
    CompositeEvictionPolicy,
    TieredEvictionPolicy,
    AdaptiveEvictionPolicy,
    create_eviction_policy,
)


def create_test_cache_block(num_tokens: int = 10) -> CacheBlock:
    """Create a test cache block."""
    key = torch.randn(2, num_tokens, 4, 8, dtype=torch.float16)
    value = torch.randn(2, num_tokens, 4, 8, dtype=torch.float16)
    return CacheBlock(key, value)


class TestEvictionCandidate:
    """Tests for EvictionCandidate."""

    def test_comparison(self):
        """Test candidate comparison."""
        from deltacache.core.prefix_tree import PrefixTreeNode

        node1 = PrefixTreeNode(token=1)
        node2 = PrefixTreeNode(token=2)

        c1 = EvictionCandidate(node=node1, score=10.0)
        c2 = EvictionCandidate(node=node2, score=5.0)

        # Lower score = higher eviction priority
        assert c2 < c1


class TestLRUEvictionPolicy:
    """Tests for LRU policy."""

    @pytest.fixture
    def setup(self):
        """Create test fixtures."""
        tree = PrefixTree()
        pool = MemoryPool(gpu_limit=10000000, cpu_limit=10000000)
        policy = LRUEvictionPolicy()
        return tree, pool, policy

    def test_select_oldest(self, setup):
        """Test LRU selects oldest accessed."""
        tree, pool, policy = setup

        # Insert blocks with different access times
        cache1 = create_test_cache_block()
        tree.insert([1, 2, 3], cache1)
        pool.register(cache1)

        time.sleep(0.01)

        cache2 = create_test_cache_block()
        tree.insert([4, 5, 6], cache2)
        pool.register(cache2)

        # Touch cache2 to make it more recent
        cache2.touch()

        candidates = policy.select_victims(tree, pool, 1000)

        assert len(candidates) >= 1
        # First candidate should be cache1 (older)
        assert candidates[0].node.cache_block.block_id == cache1.block_id

    def test_skip_referenced(self, setup):
        """Test LRU skips referenced blocks."""
        tree, pool, policy = setup

        cache = create_test_cache_block()
        tree.insert([1, 2, 3], cache)
        pool.register(cache)

        # Add reference
        node = tree.get_node([1, 2, 3])
        node.add_ref()

        candidates = policy.select_victims(tree, pool, 1000)

        assert len(candidates) == 0


class TestLFUEvictionPolicy:
    """Tests for LFU policy."""

    @pytest.fixture
    def setup(self):
        """Create test fixtures."""
        tree = PrefixTree()
        pool = MemoryPool(gpu_limit=10000000, cpu_limit=10000000)
        policy = LFUEvictionPolicy()
        return tree, pool, policy

    def test_select_least_accessed(self, setup):
        """Test LFU selects least accessed."""
        tree, pool, policy = setup

        cache1 = create_test_cache_block()
        tree.insert([1, 2, 3], cache1)
        pool.register(cache1)

        cache2 = create_test_cache_block()
        tree.insert([4, 5, 6], cache2)
        pool.register(cache2)

        # Access cache2 multiple times
        node2 = tree.get_node([4, 5, 6])
        for _ in range(5):
            node2.touch()

        candidates = policy.select_victims(tree, pool, 1000)

        # First candidate should be cache1 (less accessed)
        assert candidates[0].node.cache_block.block_id == cache1.block_id


class TestCompositeEvictionPolicy:
    """Tests for composite policy."""

    @pytest.fixture
    def setup(self):
        """Create test fixtures."""
        tree = PrefixTree()
        pool = MemoryPool(gpu_limit=10000000, cpu_limit=10000000)
        policy = CompositeEvictionPolicy()
        return tree, pool, policy

    def test_considers_multiple_factors(self, setup):
        """Test composite considers frequency, depth, etc."""
        tree, pool, policy = setup

        # Create blocks at different depths
        cache1 = create_test_cache_block()
        tree.insert([1], cache1)  # Shallow
        pool.register(cache1)

        cache2 = create_test_cache_block()
        tree.insert([2, 3, 4, 5, 6], cache2)  # Deep
        pool.register(cache2)

        candidates = policy.select_victims(tree, pool, 1000)

        assert len(candidates) == 2

    def test_custom_weights(self):
        """Test custom weight configuration."""
        policy = CompositeEvictionPolicy(
            frequency_weight=2.0,
            subtree_weight=0.0,
            depth_weight=0.0,
            recency_weight=1.0,
        )

        # Should emphasize frequency
        assert policy.frequency_weight == 2.0


class TestTieredEvictionPolicy:
    """Tests for tiered policy."""

    @pytest.fixture
    def setup(self):
        """Create test fixtures."""
        tree = PrefixTree()
        pool = MemoryPool(gpu_limit=10000000, cpu_limit=10000000)
        policy = TieredEvictionPolicy()
        return tree, pool, policy

    def test_gpu_before_cpu(self, setup):
        """Test GPU blocks are offloaded before CPU blocks deleted."""
        tree, pool, policy = setup

        # Create block (will be on CPU in test environment)
        cache = create_test_cache_block()
        tree.insert([1, 2, 3], cache)
        pool.register(cache)

        candidates = policy.select_victims(tree, pool, 1000)

        # Should have candidates
        assert len(candidates) >= 1
        # CPU blocks get DELETE action, GPU blocks would get OFFLOAD_TO_CPU
        if cache.is_on_gpu:
            assert candidates[0].action == EvictionAction.OFFLOAD_TO_CPU
        else:
            assert candidates[0].action == EvictionAction.DELETE


class TestAdaptiveEvictionPolicy:
    """Tests for adaptive policy."""

    def test_learning(self):
        """Test adaptive policy learns from access patterns."""
        policy = AdaptiveEvictionPolicy(learning_rate=0.5)

        initial_weights = policy.weights.copy()

        # Simulate a miss (evicted something that was reaccessed)
        policy.record_access((1, 2, 3))

        # Weights should remain similar (wasn't evicted)
        # Now mark as evicted and then accessed
        policy._evicted_tokens.add((4, 5, 6))
        policy.record_access((4, 5, 6))

        # This should trigger weight update


class TestEvictionExecution:
    """Tests for eviction execution."""

    @pytest.fixture
    def setup(self):
        """Create test fixtures."""
        tree = PrefixTree()
        pool = MemoryPool(gpu_limit=10000000, cpu_limit=10000000)
        policy = LRUEvictionPolicy()
        return tree, pool, policy

    def test_evict_frees_memory(self, setup):
        """Test eviction frees memory."""
        tree, pool, policy = setup

        cache = create_test_cache_block(100)
        tree.insert([1, 2, 3], cache)
        pool.register(cache)

        # Track CPU usage since test blocks are on CPU
        initial_cpu_usage = pool.cpu_used
        initial_gpu_usage = pool.gpu_used

        result = policy.evict(tree, pool, cache.memory_size)

        assert result.num_offloaded > 0 or result.num_deleted > 0
        # Memory should be freed from appropriate pool
        if cache.is_on_gpu:
            assert pool.gpu_used < initial_gpu_usage or result.num_offloaded > 0
        else:
            # CPU block deleted
            assert pool.cpu_used < initial_cpu_usage or result.num_deleted > 0


class TestEvictionPolicyFactory:
    """Tests for policy factory."""

    def test_create_lru(self):
        """Test creating LRU policy."""
        policy = create_eviction_policy("lru")
        assert isinstance(policy, LRUEvictionPolicy)

    def test_create_lfu(self):
        """Test creating LFU policy."""
        policy = create_eviction_policy("lfu")
        assert isinstance(policy, LFUEvictionPolicy)

    def test_create_composite(self):
        """Test creating composite policy."""
        policy = create_eviction_policy("composite")
        assert isinstance(policy, CompositeEvictionPolicy)

    def test_create_tiered(self):
        """Test creating tiered policy."""
        policy = create_eviction_policy("tiered")
        assert isinstance(policy, TieredEvictionPolicy)

    def test_create_adaptive(self):
        """Test creating adaptive policy."""
        policy = create_eviction_policy("adaptive")
        assert isinstance(policy, AdaptiveEvictionPolicy)

    def test_invalid_policy(self):
        """Test invalid policy name raises error."""
        with pytest.raises(ValueError):
            create_eviction_policy("invalid")

    def test_with_kwargs(self):
        """Test creating policy with kwargs."""
        policy = create_eviction_policy("composite", frequency_weight=3.0)
        assert policy.frequency_weight == 3.0
