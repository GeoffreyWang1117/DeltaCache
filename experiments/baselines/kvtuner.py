"""KVTuner baseline reproduction (ICML 2025, arXiv:2502.04420).

KVTuner performs per-layer quantization precision assignment through:
  1. Layer sensitivity analysis: measure KV reconstruction error at
     different bit-widths (offline calibration).
  2. DBSCAN clustering: group layers with similar sensitivity.
  3. Pareto search: find optimal per-layer precision assignment
     that minimizes total error under memory budget.

This is a simplified reproduction — we implement the core sensitivity
analysis and precision assignment, without the full DBSCAN clustering
(we use a simpler threshold-based grouping instead).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from deltacache.core.kv_quantizer import KVQuantizer, QuantPrecision


@dataclass
class LayerSensitivity:
    """Quantization sensitivity for a single layer."""

    layer_idx: int
    key_error_int8: float  # Mean absolute error of keys at INT8
    value_error_int8: float
    key_error_int4: float
    value_error_int4: float
    sensitivity_score: float  # Combined sensitivity (higher = more sensitive)


@dataclass
class KVTunerAllocation:
    """Per-layer precision assignment from KVTuner."""

    layer_idx: int
    quant_bits: int  # 4, 8, or 16
    estimated_error: float
    memory_bytes: int


class KVTunerBaseline:
    """KVTuner per-layer quantization baseline.

    Assigns per-layer quantization precision based on offline
    sensitivity analysis. More sensitive layers get higher precision.

    Args:
        num_layers: Number of transformer layers.
        num_heads: Number of KV heads.
        head_dim: Head dimension.
        available_bits: Allowed precision levels.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        available_bits: Optional[List[int]] = None,
    ) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.available_bits = sorted(available_bits or [4, 8, 16])
        self._quantizers = {
            4: KVQuantizer(QuantPrecision.INT4),
            8: KVQuantizer(QuantPrecision.INT8),
        }

    def analyze_sensitivity(
        self,
        full_keys: Tensor,
        full_values: Tensor,
    ) -> List[LayerSensitivity]:
        """Measure quantization error per layer at each precision.

        Args:
            full_keys: (num_layers, seq_len, H, D) in FP16.
            full_values: Same shape.

        Returns:
            Per-layer sensitivity profiles.
        """
        profiles = []

        for l in range(self.num_layers):
            layer_k = full_keys[l:l+1]  # (1, S, H, D)
            layer_v = full_values[l:l+1]

            # INT8 error
            q8 = self._quantizers[8]
            qkv8 = q8.quantize(layer_k, layer_v)
            k8, v8 = q8.dequantize(qkv8)
            key_err_8 = (layer_k.float() - k8.float()).abs().mean().item()
            val_err_8 = (layer_v.float() - v8.float()).abs().mean().item()

            # INT4 error
            q4 = self._quantizers[4]
            qkv4 = q4.quantize(layer_k, layer_v)
            k4, v4 = q4.dequantize(qkv4)
            key_err_4 = (layer_k.float() - k4.float()).abs().mean().item()
            val_err_4 = (layer_v.float() - v4.float()).abs().mean().item()

            # Combined sensitivity: weighted sum of key and value errors
            # Value errors matter more for output quality
            sensitivity = (
                0.3 * key_err_4 + 0.7 * val_err_4  # INT4 gap
                + 0.1 * key_err_8 + 0.2 * val_err_8  # INT8 gap
            )

            profiles.append(LayerSensitivity(
                layer_idx=l,
                key_error_int8=key_err_8,
                value_error_int8=val_err_8,
                key_error_int4=key_err_4,
                value_error_int4=val_err_4,
                sensitivity_score=sensitivity,
            ))

        return profiles

    def assign_precision(
        self,
        sensitivities: List[LayerSensitivity],
        budget_bytes: int,
        seq_len: int,
    ) -> List[KVTunerAllocation]:
        """Assign per-layer precision under memory budget.

        Strategy: sort layers by sensitivity, assign highest precision
        to most sensitive layers first, then downgrade less sensitive
        layers until budget is met.

        Args:
            sensitivities: Per-layer sensitivity profiles.
            budget_bytes: Total memory budget.
            seq_len: Sequence length (all tokens retained).

        Returns:
            Per-layer precision assignments.
        """
        # Start: all layers at highest precision
        assignments = {l: max(self.available_bits) for l in range(self.num_layers)}

        # Sort layers by sensitivity (least sensitive first — downgrade these)
        sorted_layers = sorted(
            range(self.num_layers),
            key=lambda l: sensitivities[l].sensitivity_score,
        )

        # Iteratively downgrade least-sensitive layers until under budget
        total_mem = sum(
            self._memory_cost(seq_len, assignments[l])
            for l in range(self.num_layers)
        )

        for l in sorted_layers:
            if total_mem <= budget_bytes:
                break

            # Try downgrading this layer
            current_bits = assignments[l]
            bits_idx = self.available_bits.index(current_bits)

            if bits_idx > 0:
                new_bits = self.available_bits[bits_idx - 1]
                old_mem = self._memory_cost(seq_len, current_bits)
                new_mem = self._memory_cost(seq_len, new_bits)
                total_mem -= (old_mem - new_mem)
                assignments[l] = new_bits

        # If still over budget, do another pass
        for l in sorted_layers:
            if total_mem <= budget_bytes:
                break

            current_bits = assignments[l]
            bits_idx = self.available_bits.index(current_bits)

            if bits_idx > 0:
                new_bits = self.available_bits[bits_idx - 1]
                old_mem = self._memory_cost(seq_len, current_bits)
                new_mem = self._memory_cost(seq_len, new_bits)
                total_mem -= (old_mem - new_mem)
                assignments[l] = new_bits

        # Build allocations
        allocations = []
        for l in range(self.num_layers):
            bits = assignments[l]
            error = self._estimated_error(sensitivities[l], bits)
            allocations.append(KVTunerAllocation(
                layer_idx=l,
                quant_bits=bits,
                estimated_error=error,
                memory_bytes=self._memory_cost(seq_len, bits),
            ))

        return allocations

    def compress(
        self,
        full_keys: Tensor,
        full_values: Tensor,
        compression_ratio: float,
    ) -> List[Tuple[Tensor, Tensor, int]]:
        """Full KVTuner compression pipeline.

        Args:
            full_keys: (num_layers, seq_len, H, D)
            full_values: Same shape.
            compression_ratio: Target compression.

        Returns:
            List of (keys, values, bits) per layer. Keys/values are
            dequantized back to FP16 for fair comparison.
        """
        seq_len = full_keys.shape[1]

        # Step 1: Sensitivity analysis
        sensitivities = self.analyze_sensitivity(full_keys, full_values)

        # Step 2: Compute budget
        full_mem = self._full_memory(seq_len)
        budget = int(full_mem / compression_ratio)

        # Step 3: Assign precision
        allocations = self.assign_precision(sensitivities, budget, seq_len)

        # Step 4: Quantize and dequantize per layer
        results = []
        for alloc in allocations:
            l = alloc.layer_idx
            bits = alloc.quant_bits
            layer_k = full_keys[l:l+1]
            layer_v = full_values[l:l+1]

            if bits < 16 and bits in self._quantizers:
                quantizer = self._quantizers[bits]
                qkv = quantizer.quantize(layer_k, layer_v)
                k_deq, v_deq = quantizer.dequantize(qkv)
            else:
                k_deq, v_deq = layer_k, layer_v

            results.append((k_deq, v_deq, bits))

        return results

    def _memory_cost(self, seq_len: int, bits: int) -> int:
        """Memory for one layer (all tokens) at given precision."""
        base = 2 * seq_len * self.num_heads * self.head_dim * bits // 8
        if bits < 16:
            key_meta = self.num_heads * self.head_dim * 4
            value_meta = seq_len * 4
            base += key_meta + value_meta
        return base

    def _full_memory(self, seq_len: int) -> int:
        """Total FP16 memory across all layers."""
        return self._memory_cost(seq_len, 16) * self.num_layers

    def _estimated_error(self, sensitivity: LayerSensitivity, bits: int) -> float:
        """Estimate reconstruction error at given precision."""
        if bits == 16:
            return 0.0
        elif bits == 8:
            return (sensitivity.key_error_int8 + sensitivity.value_error_int8) / 2
        elif bits == 4:
            return (sensitivity.key_error_int4 + sensitivity.value_error_int4) / 2
        return 0.0
