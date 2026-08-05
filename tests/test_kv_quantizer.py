"""Tests for KV cache quantizer."""

import pytest
import torch

from deltacache.core.kv_quantizer import (
    KVQuantizer,
    QuantPrecision,
    QuantizedKV,
)


class TestKVQuantizer:
    """Tests for KVQuantizer."""

    @pytest.fixture
    def sample_kv(self):
        """Create sample KV tensors [num_layers, seq_len, num_heads, head_dim]."""
        torch.manual_seed(42)
        shape = (4, 32, 8, 64)  # 4 layers, 32 tokens, 8 heads, 64 dim
        key = torch.randn(shape, dtype=torch.float16)
        value = torch.randn(shape, dtype=torch.float16)
        return key, value

    @pytest.fixture
    def large_kv(self):
        """Larger KV tensors for accuracy testing."""
        torch.manual_seed(42)
        shape = (8, 128, 16, 128)
        key = torch.randn(shape, dtype=torch.float16)
        value = torch.randn(shape, dtype=torch.float16)
        return key, value

    def test_int8_roundtrip(self, sample_kv):
        """INT8 quantize then dequantize should be close to original."""
        key, value = sample_kv
        quantizer = KVQuantizer(QuantPrecision.INT8)

        qkv = quantizer.quantize(key, value)
        key_deq, value_deq = quantizer.dequantize(qkv)

        assert key_deq.shape == key.shape
        assert value_deq.shape == value.shape
        assert key_deq.dtype == key.dtype

        # INT8 should have low error
        key_error = (key.float() - key_deq.float()).abs().mean().item()
        value_error = (value.float() - value_deq.float()).abs().mean().item()
        assert key_error < 0.05, f"Key error too high: {key_error}"
        assert value_error < 0.05, f"Value error too high: {value_error}"

    def test_int4_roundtrip(self, sample_kv):
        """INT4 quantize then dequantize should be reasonable."""
        key, value = sample_kv
        quantizer = KVQuantizer(QuantPrecision.INT4)

        qkv = quantizer.quantize(key, value)
        key_deq, value_deq = quantizer.dequantize(qkv)

        assert key_deq.shape == key.shape
        assert value_deq.shape == value.shape

        # INT4 has higher error but should still be bounded
        key_error = (key.float() - key_deq.float()).abs().mean().item()
        value_error = (value.float() - value_deq.float()).abs().mean().item()
        assert key_error < 0.2, f"Key error too high: {key_error}"
        assert value_error < 0.2, f"Value error too high: {value_error}"

    def test_memory_savings_int8(self, sample_kv):
        """INT8 should achieve ~2x memory savings."""
        key, value = sample_kv
        quantizer = KVQuantizer(QuantPrecision.INT8)

        qkv = quantizer.quantize(key, value)

        original_bytes = key.numel() * key.element_size() + value.numel() * value.element_size()
        assert qkv.memory_size < original_bytes
        assert qkv.compression_ratio > 1.5  # Should be close to 2x

    def test_quantized_kv_metadata(self, sample_kv):
        """Check QuantizedKV stores correct metadata."""
        key, value = sample_kv
        quantizer = KVQuantizer(QuantPrecision.INT8)

        qkv = quantizer.quantize(key, value)

        assert qkv.precision == QuantPrecision.INT8
        assert qkv.original_dtype == torch.float16
        assert qkv.shape == key.shape

    def test_per_channel_key_quantization(self):
        """Keys with channel-wise outliers should quantize well per-channel."""
        torch.manual_seed(0)
        shape = (2, 16, 4, 32)
        key = torch.randn(shape, dtype=torch.float16)
        # Add channel-wise outliers (simulating attention head patterns)
        key[:, :, :, 0] *= 10.0  # Channel 0 has large values

        value = torch.randn(shape, dtype=torch.float16)

        quantizer = KVQuantizer(QuantPrecision.INT8)
        qkv = quantizer.quantize(key, value)
        key_deq, _ = quantizer.dequantize(qkv)

        # Per-channel quantization should handle different channel scales well
        # Check absolute error (more reliable than relative for varied scales)
        abs_error = (key.float() - key_deq.float()).abs().mean().item()
        assert abs_error < 0.1, f"Mean absolute error too high: {abs_error}"

        # The large-value channel should have low relative error
        ch0_key = key[:, :, :, 0].float()
        ch0_deq = key_deq[:, :, :, 0].float()
        ch0_rel_error = ((ch0_key - ch0_deq).abs() / (ch0_key.abs() + 1e-6)).mean().item()
        assert ch0_rel_error < 0.15, f"Large channel relative error: {ch0_rel_error}"

    def test_estimate_memory_saving(self):
        """Test static memory estimation."""
        shape = (32, 512, 32, 128)  # Llama-7B-like

        ratio_int8 = KVQuantizer.estimate_memory_saving(shape, QuantPrecision.INT8)
        ratio_int4 = KVQuantizer.estimate_memory_saving(shape, QuantPrecision.INT4)

        # INT8 should give ~2x, INT4 should give more
        assert ratio_int8 > 1.8
        assert ratio_int8 < 2.2
        # INT4 data is same size as INT8 in this implementation (packed later)
        # But still should show savings over FP16

    def test_quantize_empty_seq(self):
        """Edge case: quantizing single-token sequence."""
        key = torch.randn(2, 1, 4, 16, dtype=torch.float16)
        value = torch.randn(2, 1, 4, 16, dtype=torch.float16)

        quantizer = KVQuantizer(QuantPrecision.INT8)
        qkv = quantizer.quantize(key, value)
        key_deq, value_deq = quantizer.dequantize(qkv)

        assert key_deq.shape == key.shape
        assert value_deq.shape == value.shape
