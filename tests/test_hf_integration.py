"""Tests for HuggingFace integration module."""

import pytest
import torch
from torch import Tensor

from deltacache.hf_integration.kv_format import (
    hf_to_deltacache,
    deltacache_to_hf,
    KVFormatConverter,
    slice_hf_cache,
    concat_hf_cache,
    get_hf_cache_seq_len,
)
from deltacache.utils.config import DeltaCacheConfig


class TestKVFormatConversion:
    """Tests for KV format conversion functions."""

    @pytest.fixture
    def sample_hf_cache(self):
        """Create sample HuggingFace format KV cache."""
        num_layers = 4
        batch_size = 1
        num_heads = 8
        seq_len = 16
        head_dim = 64

        return tuple(
            (
                torch.randn(batch_size, num_heads, seq_len, head_dim),
                torch.randn(batch_size, num_heads, seq_len, head_dim),
            )
            for _ in range(num_layers)
        )

    @pytest.fixture
    def sample_dc_cache(self):
        """Create sample DeltaCache format KV cache."""
        num_layers = 4
        seq_len = 16
        num_heads = 8
        head_dim = 64

        return (
            torch.randn(num_layers, seq_len, num_heads, head_dim),
            torch.randn(num_layers, seq_len, num_heads, head_dim),
        )

    def test_hf_to_deltacache_shape(self, sample_hf_cache):
        """Test HF to DeltaCache conversion preserves data."""
        key_cache, value_cache = hf_to_deltacache(sample_hf_cache)

        assert key_cache.shape == (4, 16, 8, 64)  # [layers, seq, heads, dim]
        assert value_cache.shape == (4, 16, 8, 64)

    def test_deltacache_to_hf_shape(self, sample_dc_cache):
        """Test DeltaCache to HF conversion."""
        key_cache, value_cache = sample_dc_cache
        hf_cache = deltacache_to_hf(key_cache, value_cache)

        assert len(hf_cache) == 4  # num_layers
        for k, v in hf_cache:
            assert k.shape == (1, 8, 16, 64)  # [batch, heads, seq, dim]
            assert v.shape == (1, 8, 16, 64)

    def test_roundtrip_hf_to_dc_to_hf(self, sample_hf_cache):
        """Test HF -> DC -> HF roundtrip preserves data."""
        # Convert HF to DC
        key_dc, value_dc = hf_to_deltacache(sample_hf_cache)

        # Convert back to HF
        hf_cache_back = deltacache_to_hf(key_dc, value_dc)

        # Verify shapes match
        assert len(hf_cache_back) == len(sample_hf_cache)

        # Verify data matches (approximately due to floating point)
        for (k1, v1), (k2, v2) in zip(sample_hf_cache, hf_cache_back):
            assert torch.allclose(k1, k2, atol=1e-6)
            assert torch.allclose(v1, v2, atol=1e-6)

    def test_roundtrip_dc_to_hf_to_dc(self, sample_dc_cache):
        """Test DC -> HF -> DC roundtrip preserves data."""
        key_cache, value_cache = sample_dc_cache

        # Convert DC to HF
        hf_cache = deltacache_to_hf(key_cache, value_cache)

        # Convert back to DC
        key_back, value_back = hf_to_deltacache(hf_cache)

        # Verify data matches
        assert torch.allclose(key_cache, key_back, atol=1e-6)
        assert torch.allclose(value_cache, value_back, atol=1e-6)


class TestKVFormatConverter:
    """Tests for KVFormatConverter class."""

    @pytest.fixture
    def converter(self):
        """Create converter instance."""
        return KVFormatConverter(
            num_layers=4,
            num_heads=8,
            head_dim=64,
            dtype=torch.float32,
        )

    def test_from_hf_validation(self, converter):
        """Test HF format validation."""
        # Create valid cache
        valid_cache = tuple(
            (
                torch.randn(1, 8, 16, 64),
                torch.randn(1, 8, 16, 64),
            )
            for _ in range(4)
        )

        # Should not raise
        key, value = converter.from_hf(valid_cache, validate=True)
        assert key.shape[0] == 4  # num_layers

    def test_from_hf_validation_wrong_layers(self, converter):
        """Test validation catches wrong number of layers."""
        wrong_cache = tuple(
            (torch.randn(1, 8, 16, 64), torch.randn(1, 8, 16, 64))
            for _ in range(3)  # Wrong: should be 4
        )

        with pytest.raises(ValueError, match="Expected 4 layers"):
            converter.from_hf(wrong_cache, validate=True)

    def test_from_hf_validation_wrong_heads(self, converter):
        """Test validation catches wrong number of heads."""
        wrong_cache = tuple(
            (
                torch.randn(1, 4, 16, 64),  # Wrong: 4 heads instead of 8
                torch.randn(1, 4, 16, 64),
            )
            for _ in range(4)
        )

        with pytest.raises(ValueError, match="expected 8 heads"):
            converter.from_hf(wrong_cache, validate=True)

    def test_to_hf_validation(self, converter):
        """Test DC format validation."""
        valid_key = torch.randn(4, 16, 8, 64)
        valid_value = torch.randn(4, 16, 8, 64)

        # Should not raise
        hf_cache = converter.to_hf(valid_key, valid_value, validate=True)
        assert len(hf_cache) == 4

    def test_to_hf_validation_wrong_dims(self, converter):
        """Test validation catches wrong dimensions."""
        wrong_key = torch.randn(4, 16, 4, 64)  # Wrong: 4 heads
        wrong_value = torch.randn(4, 16, 4, 64)

        with pytest.raises(ValueError, match="Expected 8 heads"):
            converter.to_hf(wrong_key, wrong_value, validate=True)

    def test_create_empty_hf_cache(self, converter):
        """Test creating empty HF cache."""
        cache = converter.create_empty_hf_cache(seq_len=32, batch_size=1)

        assert len(cache) == 4
        for k, v in cache:
            assert k.shape == (1, 8, 32, 64)
            assert v.shape == (1, 8, 32, 64)
            assert torch.all(k == 0)
            assert torch.all(v == 0)

    def test_create_empty_deltacache(self, converter):
        """Test creating empty DC cache."""
        key, value = converter.create_empty_deltacache(seq_len=32)

        assert key.shape == (4, 32, 8, 64)
        assert value.shape == (4, 32, 8, 64)
        assert torch.all(key == 0)
        assert torch.all(value == 0)


class TestHFCacheOperations:
    """Tests for HF cache manipulation functions."""

    @pytest.fixture
    def hf_cache(self):
        """Create HF cache with sequential values for testing."""
        num_layers = 2
        batch_size = 1
        num_heads = 4
        seq_len = 10
        head_dim = 8

        cache = []
        for layer in range(num_layers):
            # Fill with layer index for easy verification
            k = torch.full((batch_size, num_heads, seq_len, head_dim), float(layer))
            v = torch.full((batch_size, num_heads, seq_len, head_dim), float(layer + 0.5))
            cache.append((k, v))

        return tuple(cache)

    def test_get_hf_cache_seq_len(self, hf_cache):
        """Test getting sequence length from HF cache."""
        assert get_hf_cache_seq_len(hf_cache) == 10

    def test_get_hf_cache_seq_len_empty(self):
        """Test getting sequence length from empty cache."""
        assert get_hf_cache_seq_len(()) == 0

    def test_slice_hf_cache(self, hf_cache):
        """Test slicing HF cache."""
        sliced = slice_hf_cache(hf_cache, start=2, end=7)

        assert get_hf_cache_seq_len(sliced) == 5
        for k, v in sliced:
            assert k.shape[2] == 5  # seq dimension

    def test_slice_hf_cache_to_end(self, hf_cache):
        """Test slicing to end of sequence."""
        sliced = slice_hf_cache(hf_cache, start=5)

        assert get_hf_cache_seq_len(sliced) == 5

    def test_concat_hf_cache(self):
        """Test concatenating HF caches."""
        cache1 = tuple(
            (torch.randn(1, 4, 5, 8), torch.randn(1, 4, 5, 8))
            for _ in range(2)
        )
        cache2 = tuple(
            (torch.randn(1, 4, 3, 8), torch.randn(1, 4, 3, 8))
            for _ in range(2)
        )

        combined = concat_hf_cache(cache1, cache2)

        assert get_hf_cache_seq_len(combined) == 8  # 5 + 3
        for k, v in combined:
            assert k.shape[2] == 8

    def test_concat_hf_cache_mismatched_layers(self):
        """Test concatenating caches with different layer counts raises error."""
        cache1 = tuple((torch.randn(1, 4, 5, 8), torch.randn(1, 4, 5, 8)) for _ in range(2))
        cache2 = tuple((torch.randn(1, 4, 5, 8), torch.randn(1, 4, 5, 8)) for _ in range(3))

        with pytest.raises(ValueError, match="same number of layers"):
            concat_hf_cache(cache1, cache2)


class TestGPT2Config:
    """Tests for GPT-2 configuration."""

    def test_gpt2_config_from_model(self):
        """Test GPT-2 config creation."""
        config = DeltaCacheConfig.for_model("gpt2")

        assert config.num_layers == 12
        assert config.num_heads == 12
        assert config.head_dim == 64
        assert config.rope_base == 0  # No RoPE for GPT-2

    def test_gpt2_medium_config(self):
        """Test GPT-2 medium config."""
        config = DeltaCacheConfig.for_model("gpt2-medium")

        assert config.num_layers == 24
        assert config.num_heads == 16

    def test_gpt2_large_config(self):
        """Test GPT-2 large config."""
        config = DeltaCacheConfig.for_model("gpt2-large")

        assert config.num_layers == 36
        assert config.num_heads == 20

    def test_gpt2_xl_config(self):
        """Test GPT-2 XL config."""
        config = DeltaCacheConfig.for_model("gpt2-xl")

        assert config.num_layers == 48
        assert config.num_heads == 25


# Optional: Tests that require transformers library
@pytest.mark.skipif(
    not pytest.importorskip("transformers", reason="transformers not installed"),
    reason="transformers not installed"
)
class TestGPT2AdapterIntegration:
    """Integration tests with actual GPT-2 model (requires transformers)."""

    @pytest.fixture(scope="class")
    def adapter(self):
        """Load GPT-2 adapter (cached for class)."""
        from deltacache.hf_integration import GPT2Adapter
        return GPT2Adapter.from_pretrained("gpt2", device="cpu", dtype=torch.float32)

    def test_adapter_creation(self, adapter):
        """Test adapter is created correctly."""
        assert adapter.config.num_layers == 12
        assert adapter.config.num_heads == 12

    def test_tokenize_decode(self, adapter):
        """Test tokenization roundtrip."""
        text = "Hello, world!"
        tokens = adapter.tokenize(text)
        decoded = adapter.decode(tokens[0])

        assert "Hello" in decoded
        assert "world" in decoded

    def test_compute_kv(self, adapter):
        """Test KV computation produces correct shapes."""
        tokens = adapter.tokenize("Hello, world!")
        position_ids = adapter.get_position_ids(tokens.shape[1])

        key_cache, value_cache = adapter.compute_kv(tokens, position_ids)

        # Shape: [num_layers, seq_len, num_heads, head_dim]
        assert key_cache.shape[0] == 12  # num_layers
        assert key_cache.shape[2] == 12  # num_heads
        assert key_cache.shape[3] == 64  # head_dim
        assert key_cache.shape == value_cache.shape

    def test_compute_kv_incremental(self, adapter):
        """Test incremental KV computation."""
        # First compute full sequence
        text1 = "Hello, "
        tokens1 = adapter.tokenize(text1)
        pos1 = adapter.get_position_ids(tokens1.shape[1])
        kv1 = adapter.compute_kv(tokens1, pos1)

        # Then compute extended sequence
        text2 = "Hello, world!"
        tokens2 = adapter.tokenize(text2)
        pos2 = adapter.get_position_ids(tokens2.shape[1])
        kv2 = adapter.compute_kv(tokens2, pos2)

        # Full sequence should have more tokens
        assert kv2[0].shape[1] > kv1[0].shape[1]

    def test_adapter_callable(self, adapter):
        """Test adapter works as callable."""
        tokens = adapter.tokenize("Test")
        position_ids = adapter.get_position_ids(tokens.shape[1])

        # Should work when called directly
        key, value = adapter(tokens, position_ids)
        assert key.shape[0] == 12
