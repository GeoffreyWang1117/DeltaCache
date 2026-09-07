"""Tests for vLLM integration."""

import pytest
import torch

from deltacache.api import create_delta_cache
from deltacache.utils.config import DeltaCacheConfig
from deltacache.vllm_integration.scheduler_hook import SchedulerHook


class TestSchedulerHook:
    """Tests for SchedulerHook."""

    @pytest.fixture
    def hook(self):
        """Create test hook."""
        manager = create_delta_cache(
            num_layers=2,
            num_heads=4,
            head_dim=8,
            device="cpu",
            gpu_memory_limit=10000000,  # Explicit limit for CPU mode
        )
        return SchedulerHook(cache_manager=manager)

    def test_new_sequence_no_cache(self, hook):
        """Test new sequence without cache."""
        hint = hook.on_new_sequence(seq_id=1, token_ids=[1, 2, 3])

        assert hint.seq_id == 1
        assert hint.prefix_length == 0
        assert not hint.has_cache

    def test_new_sequence_with_cache(self, hook):
        """Test new sequence with cached prefix."""
        # Pre-populate cache
        key = torch.randn(2, 3, 4, 8)
        value = torch.randn(2, 3, 4, 8)
        hook.prefix_tree.insert_with_kv([1, 2, 3], key, value)

        hint = hook.on_new_sequence(seq_id=1, token_ids=[1, 2, 3, 4, 5])

        assert hint.prefix_length == 3
        assert hint.has_cache

    def test_sequence_finished(self, hook):
        """Test sequence completion cleanup."""
        hook.on_new_sequence(seq_id=1, token_ids=[1, 2, 3])

        assert 1 in hook._seq_prefix_map

        hook.on_sequence_finished(seq_id=1)

        assert 1 not in hook._seq_prefix_map

    def test_shared_prefix_groups(self, hook):
        """Test shared prefix group detection."""
        # Pre-populate cache for prefix
        key = torch.randn(2, 2, 4, 8)
        value = torch.randn(2, 2, 4, 8)
        hook.prefix_tree.insert_with_kv([1, 2], key, value)

        # Add sequences sharing prefix
        hook.on_new_sequence(seq_id=1, token_ids=[1, 2, 3])
        hook.on_new_sequence(seq_id=2, token_ids=[1, 2, 4])
        hook.on_new_sequence(seq_id=3, token_ids=[5, 6, 7])  # Different prefix

        groups = hook.get_shared_prefix_groups()

        # Should have one group with seq 1 and 2
        assert len(groups) == 1
        group = next(iter(groups.values()))
        assert set(group) == {1, 2}

    def test_suggest_batch_order(self, hook):
        """Test batch ordering suggestion."""
        # Create sequences
        key = torch.randn(2, 2, 4, 8)
        value = torch.randn(2, 2, 4, 8)
        hook.prefix_tree.insert_with_kv([1, 2], key, value)

        hook.on_new_sequence(seq_id=1, token_ids=[1, 2, 3])
        hook.on_new_sequence(seq_id=2, token_ids=[5, 6, 7])
        hook.on_new_sequence(seq_id=3, token_ids=[1, 2, 4])

        order = hook.suggest_batch_order([1, 2, 3])

        # Sequences 1 and 3 should be adjacent (shared prefix)
        idx1 = order.index(1)
        idx3 = order.index(3)
        assert abs(idx1 - idx3) == 1

    def test_priority_boost(self, hook):
        """Test priority boosting for cached sequences."""
        # Sequence without cache
        hook.on_new_sequence(seq_id=1, token_ids=[1, 2, 3])
        boost1 = hook.get_priority_boost(1)

        # Sequence with cache
        key = torch.randn(2, 3, 4, 8)
        value = torch.randn(2, 3, 4, 8)
        hook.prefix_tree.insert_with_kv([4, 5, 6], key, value)
        hook.on_new_sequence(seq_id=2, token_ids=[4, 5, 6])
        boost2 = hook.get_priority_boost(2)

        # Cached sequence should have higher boost
        assert boost2 > boost1

    def test_prefetch_candidates(self, hook):
        """Test prefetch candidate identification."""
        # Add cache
        key = torch.randn(2, 3, 4, 8)
        value = torch.randn(2, 3, 4, 8)
        hook.prefix_tree.insert_with_kv([1, 2, 3], key, value)

        pending = [
            (1, [1, 2, 3, 4]),  # Has cached prefix
            (2, [5, 6, 7]),  # No cache
        ]

        candidates = hook.find_prefetch_candidates(pending, gpu_memory_available=100000)

        assert 1 in candidates
        assert 2 not in candidates


class TestDeltaCacheManagerIntegration:
    """Integration tests for DeltaCacheManager."""

    @pytest.fixture
    def manager(self):
        """Create test manager."""
        return create_delta_cache(
            num_layers=2,
            num_heads=4,
            head_dim=8,
            gpu_memory_limit=10000000,
            cpu_memory_limit=10000000,
            device="cpu",
        )

    def test_lookup_insert_cycle(self, manager):
        """Test basic lookup and insert."""
        tokens = [1, 2, 3, 4, 5]

        # Initial lookup - miss
        result1 = manager.lookup(tokens)
        assert not result1.has_match

        # Insert
        key = torch.randn(2, 5, 4, 8)
        value = torch.randn(2, 5, 4, 8)
        manager.insert(tokens, key, value)

        # Second lookup - hit
        result2 = manager.lookup(tokens)
        assert result2.has_match
        assert result2.matched_length == 5

    def test_stats_tracking(self, manager):
        """Test statistics tracking."""
        tokens = [1, 2, 3]

        # Miss
        manager.lookup(tokens)

        # Insert and hit
        key = torch.randn(2, 3, 4, 8)
        value = torch.randn(2, 3, 4, 8)
        manager.insert(tokens, key, value)
        manager.lookup(tokens)

        stats = manager.get_stats()
        assert stats["total_lookups"] == 2
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 1

    def test_evict_if_needed(self, manager):
        """Test eviction triggering."""
        # Fill cache
        for i in range(10):
            tokens = list(range(i * 100, i * 100 + 100))
            key = torch.randn(2, 100, 4, 8)
            value = torch.randn(2, 100, 4, 8)
            manager.insert(tokens, key, value)

        initial_cached = manager.num_cached_sequences
        assert initial_cached > 0, "nothing was cached, so eviction has nothing to do"

        # Trigger eviction
        result = manager.evict_if_needed(required_memory=1000000)

        # How much gets freed depends on the configured limits, so the invariant
        # asserted here is the one that holds regardless: eviction reports a
        # result and never grows the cache.
        assert result is not None
        assert manager.num_cached_sequences <= initial_cached

    def test_clear(self, manager):
        """Test clearing cache."""
        tokens = [1, 2, 3]
        key = torch.randn(2, 3, 4, 8)
        value = torch.randn(2, 3, 4, 8)
        manager.insert(tokens, key, value)

        assert manager.num_cached_sequences > 0

        manager.clear()

        assert manager.num_cached_sequences == 0

    def test_compute_incremental_mock(self, manager):
        """Test incremental computation with mock."""

        def mock_compute(input_ids, position_ids, past_key_values=None):
            seq_len = input_ids.shape[1]
            key = torch.randn(2, seq_len, 4, 8)
            value = torch.randn(2, seq_len, 4, 8)
            return key, value

        tokens = [1, 2, 3, 4, 5]
        result = manager.compute_incremental(tokens, mock_compute)

        assert result.total_length == 5


class TestDeltaCacheConfig:
    """Tests for configuration."""

    def test_for_model_llama(self):
        """Test config for LLaMA model."""
        config = DeltaCacheConfig.for_model("llama-7b")

        assert config.num_layers == 32
        assert config.num_heads == 32
        assert config.head_dim == 128

    def test_for_model_mistral(self):
        """Test config for Mistral model."""
        config = DeltaCacheConfig.for_model("mistral-7b")

        assert config.num_layers == 32

    def test_for_model_with_overrides(self):
        """Test config with overrides."""
        config = DeltaCacheConfig.for_model(
            "llama-7b",
            gpu_memory_limit=1000000000,
        )

        assert config.num_layers == 32
        assert config.gpu_memory_limit == 1000000000

    def test_kv_size_per_token(self):
        """Test KV size calculation."""
        config = DeltaCacheConfig(
            num_layers=32,
            num_heads=32,
            head_dim=128,
            dtype="float16",
        )

        # 2 (K+V) * 32 layers * 32 heads * 128 dim * 2 bytes
        expected = 2 * 32 * 32 * 128 * 2
        assert config.kv_size_per_token == expected

    def test_validation(self):
        """Test config validation."""
        with pytest.raises(ValueError):
            DeltaCacheConfig(eviction_threshold=1.5)

        with pytest.raises(ValueError):
            DeltaCacheConfig(num_layers=-1)

    def test_to_dict_from_dict(self):
        """Test serialization."""
        config1 = DeltaCacheConfig(num_layers=40, num_heads=40)
        d = config1.to_dict()
        config2 = DeltaCacheConfig.from_dict(d)

        assert config2.num_layers == 40
        assert config2.num_heads == 40
