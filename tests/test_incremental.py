"""Tests for IncrementalEngine."""

import pytest
import torch
from torch import Tensor
from typing import Optional, Tuple

from deltacache.core.prefix_tree import PrefixTree
from deltacache.core.cache_block import CacheBlock
from deltacache.engine.incremental import IncrementalEngine, IncrementalResult
from deltacache.engine.rope_handler import RoPEHandler, create_position_ids


class MockKVCompute:
    """Mock KV computation function for testing."""

    def __init__(
        self,
        num_layers: int = 2,
        num_heads: int = 4,
        head_dim: int = 8,
        dtype: torch.dtype = torch.float16,
        device: str = "cpu",
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.call_count = 0
        self.last_input_length = 0

    def __call__(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        past_key_values: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Compute mock KV cache."""
        self.call_count += 1
        batch_size, seq_len = input_ids.shape
        self.last_input_length = seq_len

        # Create deterministic output based on input
        shape = (self.num_layers, seq_len, self.num_heads, self.head_dim)
        key = torch.randn(shape, dtype=self.dtype, device=self.device)
        value = torch.randn(shape, dtype=self.dtype, device=self.device)

        return key, value


class TestIncrementalEngine:
    """Tests for IncrementalEngine."""

    @pytest.fixture
    def engine(self):
        """Create test engine."""
        tree = PrefixTree()
        return IncrementalEngine(
            prefix_tree=tree,
            num_layers=2,
            num_heads=4,
            head_dim=8,
            device=torch.device("cpu"),
            dtype=torch.float16,
        )

    @pytest.fixture
    def compute_fn(self):
        """Create mock compute function."""
        return MockKVCompute(device="cpu")

    def test_compute_no_cache(self, engine, compute_fn):
        """Test computation with empty cache."""
        tokens = [1, 2, 3, 4, 5]

        result = engine.compute(tokens, compute_fn)

        assert result.matched_length == 0
        assert result.computed_length == 5
        assert not result.cache_hit
        assert result.total_length == 5
        assert compute_fn.call_count == 1
        assert compute_fn.last_input_length == 5

    def test_compute_full_cache_hit(self, engine, compute_fn):
        """Test computation with full cache hit."""
        tokens = [1, 2, 3]

        # First compute to populate cache
        engine.compute(tokens, compute_fn)
        compute_fn.call_count = 0

        # Second compute should hit cache
        result = engine.compute(tokens, compute_fn)

        assert result.matched_length == 3
        assert result.computed_length == 0
        assert result.cache_hit
        assert compute_fn.call_count == 0  # No computation needed

    def test_compute_partial_cache_hit(self, engine, compute_fn):
        """Test computation with partial cache hit."""
        # First compute prefix
        prefix = [1, 2, 3]
        engine.compute(prefix, compute_fn)
        compute_fn.call_count = 0

        # Compute extended sequence
        extended = [1, 2, 3, 4, 5]
        result = engine.compute(extended, compute_fn)

        assert result.matched_length == 3
        assert result.computed_length == 2
        assert result.cache_hit
        assert compute_fn.call_count == 1
        assert compute_fn.last_input_length == 2

    def test_compute_no_store(self, engine, compute_fn):
        """Test computation without storing result."""
        tokens = [1, 2, 3]

        result = engine.compute(tokens, compute_fn, store_result=False)

        assert result.total_length == 3

        # Second lookup should not find cache
        result2 = engine.prefix_tree.lookup(tokens)
        assert result2.kv_cache is None

    def test_cache_hit_rate(self, engine, compute_fn):
        """Test cache hit rate tracking."""
        assert engine.cache_hit_rate == 0.0

        # First compute - miss
        engine.compute([1, 2, 3], compute_fn)

        # Second compute - hit
        engine.compute([1, 2, 3], compute_fn)

        # Should have some cached tokens
        assert engine.cache_hit_rate > 0

    def test_reset_stats(self, engine, compute_fn):
        """Test statistics reset."""
        engine.compute([1, 2, 3], compute_fn)
        engine.compute([1, 2, 3], compute_fn)

        assert engine._total_tokens > 0

        engine.reset_stats()

        assert engine._total_tokens == 0
        assert engine._cached_tokens == 0

    def test_compute_batch_single(self, engine, compute_fn):
        """Test batch computation with single sequence."""
        batch = [[1, 2, 3]]

        results = engine.compute_batch(batch, compute_fn)

        assert len(results) == 1
        assert results[0].total_length == 3

    def test_compute_batch_multiple(self, engine, compute_fn):
        """Test batch computation with multiple sequences."""
        batch = [
            [1, 2, 3],
            [4, 5, 6],
            [1, 2, 4],  # Shares prefix with first
        ]

        results = engine.compute_batch(batch, compute_fn)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.total_length == 3

    def test_compute_batch_shared_prefix(self, engine, compute_fn):
        """Test batch with shared prefix."""
        # Pre-populate cache
        engine.compute([1, 2], compute_fn)
        compute_fn.call_count = 0

        batch = [
            [1, 2, 3],
            [1, 2, 4],
        ]

        results = engine.compute_batch(batch, compute_fn)

        # Both should have partial hits
        assert results[0].matched_length == 2
        assert results[1].matched_length == 2

    def test_prefetch(self, engine, compute_fn):
        """Test prefetch functionality."""
        # Populate cache
        engine.compute([1, 2, 3], compute_fn)

        # Prefetch
        results = engine.prefetch([[1, 2, 3], [4, 5, 6]])

        assert results[0] is True  # Found in cache
        assert results[1] is False  # Not in cache


class TestIncrementalResult:
    """Tests for IncrementalResult."""

    def test_total_length(self):
        """Test total length calculation."""
        result = IncrementalResult(
            key_cache=torch.randn(2, 5, 4, 8),
            value_cache=torch.randn(2, 5, 4, 8),
            matched_length=3,
            computed_length=2,
            cache_hit=True,
        )

        assert result.total_length == 5

    def test_properties(self):
        """Test result properties."""
        result = IncrementalResult(
            key_cache=torch.randn(2, 3, 4, 8),
            value_cache=torch.randn(2, 3, 4, 8),
            matched_length=3,
            computed_length=0,
            cache_hit=True,
        )

        assert result.cache_hit
        assert result.matched_length == 3
        assert result.computed_length == 0


class TestRoPEHandler:
    """Tests for RoPEHandler."""

    @pytest.fixture
    def handler(self):
        """Create test handler."""
        return RoPEHandler(
            head_dim=64,
            max_position=512,
            device=torch.device("cpu"),
        )

    def test_get_cos_sin(self, handler):
        """Test cos/sin computation."""
        positions = torch.tensor([0, 1, 2, 3, 4])

        cos, sin = handler.get_cos_sin(positions)

        assert cos.shape == (5, 64)
        assert sin.shape == (5, 64)

    def test_apply_rotary_pos_emb(self, handler):
        """Test applying rotary embeddings."""
        q = torch.randn(1, 5, 8, 64)
        k = torch.randn(1, 5, 8, 64)
        positions = torch.tensor([0, 1, 2, 3, 4])

        q_rot, k_rot = handler.apply_rotary_pos_emb(q, k, positions)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_remove_rotary_pos_emb(self, handler):
        """Test removing rotary embeddings."""
        k = torch.randn(1, 5, 8, 64)
        positions = torch.tensor([0, 1, 2, 3, 4])

        # Apply then remove should be close to original
        k_rot = handler.apply_rotary_pos_emb_to_k(k, positions)
        k_restored = handler.remove_rotary_pos_emb(k_rot, positions)

        assert torch.allclose(k, k_restored, atol=1e-5)

    def test_reposition_keys(self, handler):
        """Test repositioning keys."""
        k = torch.randn(1, 5, 8, 64)
        old_pos = torch.tensor([0, 1, 2, 3, 4])
        new_pos = torch.tensor([10, 11, 12, 13, 14])

        k_repositioned = handler.reposition_keys(k, old_pos, new_pos)

        assert k_repositioned.shape == k.shape
        # Values should be different after repositioning
        assert not torch.allclose(k, k_repositioned)


class TestPositionIds:
    """Tests for position ID utilities."""

    def test_create_position_ids(self):
        """Test creating position IDs."""
        positions = create_position_ids(5)
        expected = torch.tensor([0, 1, 2, 3, 4])
        assert torch.equal(positions, expected)

    def test_create_position_ids_with_offset(self):
        """Test creating position IDs with offset."""
        positions = create_position_ids(5, start_pos=10)
        expected = torch.tensor([10, 11, 12, 13, 14])
        assert torch.equal(positions, expected)
