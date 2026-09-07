"""Tests for LayerKVStore."""

import pytest
import torch

from deltacache.core.layer_kv_store import LayerKVStore


class TestLayerKVStore:
    """Tests for per-layer KV storage."""

    @pytest.fixture
    def store(self):
        """Create a store for a 4-layer model."""
        return LayerKVStore(num_layers=4, num_heads=8, head_dim=64)

    @pytest.fixture
    def sample_kv(self):
        """Sample KV tensors: (1, 32, 8, 64) per layer."""
        torch.manual_seed(42)
        keys = torch.randn(1, 32, 8, 64, dtype=torch.float16)
        values = torch.randn(1, 32, 8, 64, dtype=torch.float16)
        return keys, values

    def test_store_and_retrieve_fp16(self, store, sample_kv):
        """FP16 store/retrieve should be exact."""
        keys, values = sample_kv
        indices = torch.arange(32)

        store.store_layer(0, keys, values, indices, quant_bits=16)

        k_out, v_out, idx_out = store.get_layer(0)
        assert k_out.shape == (1, 32, 8, 64)
        assert torch.equal(k_out, keys)
        assert torch.equal(v_out, values)
        assert torch.equal(idx_out, indices)

    def test_store_and_retrieve_int8(self, store, sample_kv):
        """INT8 store/retrieve should be close to original."""
        keys, values = sample_kv
        indices = torch.arange(32)

        store.store_layer(0, keys, values, indices, quant_bits=8)

        k_out, _v_out, _idx_out = store.get_layer(0)
        assert k_out.shape == (1, 32, 8, 64)

        key_error = (keys.float() - k_out.float()).abs().mean().item()
        assert key_error < 0.05, f"INT8 key error too high: {key_error}"

    def test_store_and_retrieve_int4(self, store, sample_kv):
        """INT4 store/retrieve should be reasonable."""
        keys, values = sample_kv
        indices = torch.arange(32)

        store.store_layer(0, keys, values, indices, quant_bits=4)

        k_out, _v_out, _idx_out = store.get_layer(0)
        assert k_out.shape == (1, 32, 8, 64)

        key_error = (keys.float() - k_out.float()).abs().mean().item()
        assert key_error < 0.2, f"INT4 key error too high: {key_error}"

    def test_token_selection(self, store, sample_kv):
        """Storing subset of tokens should return correct shape."""
        keys, values = sample_kv
        indices = torch.tensor([0, 1, 5, 10, 31])  # 5 tokens

        store.store_layer(0, keys, values, indices, quant_bits=16)

        k_out, _v_out, idx_out = store.get_layer(0)
        assert k_out.shape == (1, 5, 8, 64)
        assert torch.equal(idx_out, indices)

        # Values should match selected positions
        expected_k = keys[:, indices, :, :]
        assert torch.equal(k_out, expected_k)

    def test_multiple_layers(self, store, sample_kv):
        """Store different layers with different configs."""
        keys, values = sample_kv
        full_idx = torch.arange(32)
        partial_idx = torch.arange(16)

        store.store_layer(0, keys, values, full_idx, quant_bits=4)
        store.store_layer(1, keys, values, partial_idx, quant_bits=8)
        store.store_layer(2, keys, values, full_idx, quant_bits=16)

        assert len(store) == 3
        assert 0 in store
        assert 1 in store
        assert 2 in store
        assert 3 not in store

        # Check shapes
        k0, _, _ = store.get_layer(0)
        k1, _, _ = store.get_layer(1)
        k2, _, _ = store.get_layer(2)
        assert k0.shape[1] == 32
        assert k1.shape[1] == 16
        assert k2.shape[1] == 32

    def test_memory_usage(self, store, sample_kv):
        """INT8 should use less memory than FP16."""
        keys, values = sample_kv
        indices = torch.arange(32)

        store.store_layer(0, keys, values, indices, quant_bits=16)
        mem_fp16 = store.memory_usage()

        store.clear()
        store.store_layer(0, keys, values, indices, quant_bits=8)
        mem_int8 = store.memory_usage()

        assert mem_int8 < mem_fp16

    def test_layer_summary(self, store, sample_kv):
        """Summary should reflect stored state."""
        keys, values = sample_kv

        store.store_layer(0, keys, values, torch.arange(32), quant_bits=16)
        store.store_layer(2, keys, values, torch.arange(16), quant_bits=4)

        summary = store.layer_summary()
        assert len(summary) == 4  # All layers, including empty

        assert summary[0]["tokens"] == 32
        assert summary[0]["bits"] == 16
        assert summary[1]["tokens"] == 0  # Not stored
        assert summary[2]["tokens"] == 16
        assert summary[2]["bits"] == 4

    def test_clear(self, store, sample_kv):
        keys, values = sample_kv
        store.store_layer(0, keys, values, torch.arange(32), quant_bits=16)
        assert len(store) == 1

        store.clear()
        assert len(store) == 0
        assert store.memory_usage() == 0

    def test_missing_layer_raises(self, store):
        with pytest.raises(KeyError):
            store.get_layer(99)

    def test_3d_input(self, store):
        """3-D input (without batch dim) should work."""
        keys = torch.randn(32, 8, 64, dtype=torch.float16)
        values = torch.randn(32, 8, 64, dtype=torch.float16)
        indices = torch.arange(32)

        store.store_layer(0, keys, values, indices, quant_bits=16)

        k_out, _v_out, _idx_out = store.get_layer(0)
        assert k_out.shape == (1, 32, 8, 64)

    def test_default_token_selection(self, store):
        """Default H2O-style selection should include sink and recent tokens."""
        keys = torch.randn(1, 100, 8, 64, dtype=torch.float16)
        values = torch.randn(1, 100, 8, 64, dtype=torch.float16)

        indices = store._default_token_selection(keys, values, n_tokens=30, seq_len=100)

        # Should include first 4 (sink) and last 16 (recent)
        assert 0 in indices
        assert 1 in indices
        assert 2 in indices
        assert 3 in indices
        assert 99 in indices
        assert len(indices) <= 30

    def test_default_selection_full_budget(self, store):
        """Full budget should return all tokens."""
        keys = torch.randn(1, 50, 8, 64, dtype=torch.float16)
        values = torch.randn(1, 50, 8, 64, dtype=torch.float16)

        indices = store._default_token_selection(keys, values, n_tokens=50, seq_len=50)
        assert len(indices) == 50
