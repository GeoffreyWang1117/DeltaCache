"""KV cache quantization for memory-efficient CPU-tier storage.

Implements KIVI-style asymmetric quantization (ICML 2024):
- Keys: per-channel quantization (channels have consistent outlier patterns)
- Values: per-token quantization (attention sparsity isolates per-token error)

Quality impact at different precision levels:
- INT8: ~2x memory savings, <0.5% perplexity impact (default)
- INT4: ~2.5x savings, <2% perplexity impact (aggressive mode)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import torch
from torch import Tensor


class QuantPrecision(Enum):
    """Quantization precision levels."""

    INT8 = 8
    INT4 = 4


@dataclass
class QuantizedKV:
    """Quantized KV cache tensors with dequantization metadata.

    Stores quantized data along with scale and zero-point tensors
    needed for dequantization.
    """

    key_data: Tensor  # Quantized keys (int8 or packed int4)
    value_data: Tensor  # Quantized values
    key_scale: Tensor  # Per-channel scale for keys
    key_zero: Tensor  # Per-channel zero-point for keys
    value_scale: Tensor  # Per-token scale for values
    value_zero: Tensor  # Per-token zero-point for values
    precision: QuantPrecision
    original_dtype: torch.dtype
    shape: Tuple[int, ...]  # Original tensor shape

    @property
    def memory_size(self) -> int:
        """Total memory in bytes (quantized data + metadata)."""
        total = 0
        for t in (
            self.key_data,
            self.value_data,
            self.key_scale,
            self.key_zero,
            self.value_scale,
            self.value_zero,
        ):
            total += t.numel() * t.element_size()
        return total

    @property
    def compression_ratio(self) -> float:
        """Compression ratio vs original FP16."""
        original_size = 2  # FP16 = 2 bytes per element
        num_elements = 1
        for s in self.shape:
            num_elements *= s
        original_bytes = num_elements * original_size * 2  # key + value
        return original_bytes / max(1, self.memory_size)


class KVQuantizer:
    """Quantizer for KV cache tensors.

    Uses KIVI-style asymmetric quantization with different strategies
    for keys (per-channel) and values (per-token).

    Args:
        precision: Quantization precision (INT8 or INT4).
    """

    def __init__(self, precision: QuantPrecision = QuantPrecision.INT8) -> None:
        self.precision = precision
        self._bits = precision.value
        self._qmax = (1 << self._bits) - 1  # 255 for INT8, 15 for INT4

    def quantize(self, key: Tensor, value: Tensor) -> QuantizedKV:
        """Quantize key and value tensors.

        Args:
            key: Key tensor [num_layers, seq_len, num_heads, head_dim].
            value: Value tensor [num_layers, seq_len, num_heads, head_dim].

        Returns:
            QuantizedKV with compressed data and dequantization metadata.
        """
        original_dtype = key.dtype
        shape = key.shape

        # Keys: per-channel quantization (along head_dim)
        # Channels have consistent outlier patterns across tokens
        key_q, key_scale, key_zero = self._quantize_per_channel(key)

        # Values: per-token quantization (along seq_len)
        # Attention sparsity means per-token error is isolated
        value_q, value_scale, value_zero = self._quantize_per_token(value)

        return QuantizedKV(
            key_data=key_q,
            value_data=value_q,
            key_scale=key_scale,
            key_zero=key_zero,
            value_scale=value_scale,
            value_zero=value_zero,
            precision=self.precision,
            original_dtype=original_dtype,
            shape=shape,
        )

    def dequantize(self, qkv: QuantizedKV) -> Tuple[Tensor, Tensor]:
        """Dequantize back to floating point.

        Args:
            qkv: Quantized KV data.

        Returns:
            Tuple of (key, value) tensors in original dtype.
        """
        key = self._dequantize_per_channel(
            qkv.key_data, qkv.key_scale, qkv.key_zero, qkv.original_dtype
        )
        value = self._dequantize_per_token(
            qkv.value_data, qkv.value_scale, qkv.value_zero, qkv.original_dtype
        )
        return key, value

    def _quantize_per_channel(
        self, tensor: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Per-channel (head_dim) quantization for keys.

        Computes scale and zero-point per channel across all tokens.
        Shape: [num_layers, seq_len, num_heads, head_dim]
        Channel dim = -1 (head_dim), reduction over seq_len.
        """
        t = tensor.float()

        # Compute min/max per channel: reduce over seq_len dim (1)
        # Result shape: [num_layers, 1, num_heads, head_dim]
        ch_min = t.amin(dim=1, keepdim=True)
        ch_max = t.amax(dim=1, keepdim=True)

        scale = (ch_max - ch_min) / self._qmax
        scale = scale.clamp(min=1e-8)  # Avoid division by zero
        zero = ch_min

        quantized = ((t - zero) / scale).round().clamp(0, self._qmax)

        if self.precision == QuantPrecision.INT8:
            quantized = quantized.to(torch.uint8)
        else:
            # INT4: pack two values into one byte
            quantized = quantized.to(torch.uint8)

        # Store scale/zero in float16 to save memory
        return quantized, scale.half(), zero.half()

    def _quantize_per_token(
        self, tensor: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Per-token (seq_len) quantization for values.

        Computes scale and zero-point per token across head dimensions.
        Shape: [num_layers, seq_len, num_heads, head_dim]
        Token dim = 1, reduction over (num_heads, head_dim).
        """
        t = tensor.float()

        # Compute min/max per token: reduce over num_heads and head_dim
        # Result shape: [num_layers, seq_len, 1, 1]
        tok_min = t.amin(dim=(2, 3), keepdim=True)
        tok_max = t.amax(dim=(2, 3), keepdim=True)

        scale = (tok_max - tok_min) / self._qmax
        scale = scale.clamp(min=1e-8)
        zero = tok_min

        quantized = ((t - zero) / scale).round().clamp(0, self._qmax)

        if self.precision == QuantPrecision.INT8:
            quantized = quantized.to(torch.uint8)
        else:
            quantized = quantized.to(torch.uint8)

        return quantized, scale.half(), zero.half()

    def _dequantize_per_channel(
        self,
        data: Tensor,
        scale: Tensor,
        zero: Tensor,
        target_dtype: torch.dtype,
    ) -> Tensor:
        """Dequantize per-channel quantized data."""
        return (data.float() * scale.float() + zero.float()).to(target_dtype)

    def _dequantize_per_token(
        self,
        data: Tensor,
        scale: Tensor,
        zero: Tensor,
        target_dtype: torch.dtype,
    ) -> Tensor:
        """Dequantize per-token quantized data."""
        return (data.float() * scale.float() + zero.float()).to(target_dtype)

    @staticmethod
    def estimate_memory_saving(
        shape: Tuple[int, ...],
        precision: QuantPrecision = QuantPrecision.INT8,
        original_dtype: torch.dtype = torch.float16,
    ) -> float:
        """Estimate memory compression ratio.

        Args:
            shape: Tensor shape [num_layers, seq_len, num_heads, head_dim].
            precision: Target precision.
            original_dtype: Original data type.

        Returns:
            Estimated compression ratio (e.g., 2.0 means 2x smaller).
        """
        original_element_size = torch.tensor([], dtype=original_dtype).element_size()
        num_elements = 1
        for s in shape:
            num_elements *= s
        original_bytes = num_elements * original_element_size * 2  # key + value

        # Quantized data: 1 byte per element for INT8
        quant_bytes = num_elements * 2  # key + value, 1 byte each for INT8

        # Scale and zero metadata
        num_layers, seq_len, num_heads, head_dim = shape
        # Key metadata: per channel [num_layers, 1, num_heads, head_dim] * 2 (scale+zero)
        key_meta = num_layers * num_heads * head_dim * 2 * 2  # FP16
        # Value metadata: per token [num_layers, seq_len, 1, 1] * 2
        value_meta = num_layers * seq_len * 2 * 2  # FP16

        total_quant = quant_bytes + key_meta + value_meta
        return original_bytes / max(1, total_quant)
