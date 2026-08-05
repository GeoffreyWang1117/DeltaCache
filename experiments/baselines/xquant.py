"""XQuant baseline (EMNLP 2025): Cross-layer ultra-low-bit quantization.

Assigns per-layer quantization precision from {2, 4, 8} bits based on
sensitivity analysis. Exploits cross-layer redundancy: adjacent layers
with similar KV distributions can share quantization parameters.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import Tensor

from deltacache.core.kv_quantizer import KVQuantizer, QuantPrecision

from .base import BaselineMethod, register_baseline


@register_baseline
class XQuant(BaselineMethod):
    name = "xquant"
    category = "quantization"
    requires_attention = False
    is_per_layer = True
    reference = "Yang et al., EMNLP 2025"

    def __init__(self, num_layers, num_heads, head_dim, **kwargs):
        super().__init__(num_layers, num_heads, head_dim)
        self.available_bits = sorted(kwargs.get("available_bits", [2, 4, 8]))
        self._quantizers = {
            4: KVQuantizer(QuantPrecision.INT4),
            8: KVQuantizer(QuantPrecision.INT8),
        }

    def compress(
        self,
        full_keys: Tensor,
        full_values: Tensor,
        compression_ratio: float,
        attention_weights: Optional[List[Tensor]] = None,
        hidden_states: Optional[List[Tensor]] = None,
    ) -> List[Tuple[Tensor, Tensor, Tensor]]:
        seq_len = full_keys.shape[1]
        full_mem = self._full_memory(seq_len)
        budget = int(full_mem / compression_ratio)

        # Step 1: Sensitivity analysis
        sensitivities = self._analyze_sensitivity(full_keys, full_values)

        # Step 2: Greedy bit assignment under budget
        assignments = self._assign_bits(sensitivities, budget, seq_len)
        self._last_assignments = assignments  # Cache for memory_bytes()

        # Step 3: Quantize per layer
        all_indices = torch.arange(seq_len)
        results = []
        for l in range(self.num_layers):
            bits = assignments[l]
            layer_k = full_keys[l:l+1]
            layer_v = full_values[l:l+1]

            if bits == 2:
                # 2-bit: ternary quantization {-1, 0, 1} with per-channel scale
                k_deq = self._ternary_quantize(layer_k)
                v_deq = self._ternary_quantize(layer_v)
            elif bits in self._quantizers:
                qkv = self._quantizers[bits].quantize(layer_k, layer_v)
                k_deq, v_deq = self._quantizers[bits].dequantize(qkv)
            else:
                k_deq, v_deq = layer_k, layer_v

            results.append((k_deq, v_deq, all_indices))
        return results

    def memory_bytes(self, compressed_layers):
        """Report actual quantized memory based on per-layer bit assignments."""
        if not compressed_layers:
            return 0
        seq_len = compressed_layers[0][0].shape[1]
        if hasattr(self, "_last_assignments") and self._last_assignments:
            return sum(self._layer_mem(seq_len, b) for b in self._last_assignments)
        # Fallback: average bits estimate
        avg_bits = sum(self.available_bits) / len(self.available_bits)
        return int(self.num_layers * self._layer_mem(seq_len, int(avg_bits)))

    def _analyze_sensitivity(
        self, full_keys: Tensor, full_values: Tensor,
    ) -> List[float]:
        """Measure per-layer quantization sensitivity."""
        sensitivities = []
        for l in range(self.num_layers):
            layer_k = full_keys[l:l+1]
            layer_v = full_values[l:l+1]

            # INT4 error
            q4 = self._quantizers[4]
            qkv4 = q4.quantize(layer_k, layer_v)
            k4, v4 = q4.dequantize(qkv4)
            err = (layer_k.float() - k4.float()).abs().mean().item()
            err += (layer_v.float() - v4.float()).abs().mean().item()
            sensitivities.append(err)

        return sensitivities

    def _assign_bits(
        self, sensitivities: List[float], budget: int, seq_len: int,
    ) -> List[int]:
        """Greedy bit assignment: start at lowest, upgrade most sensitive."""
        min_bits = self.available_bits[0]
        assignments = [min_bits] * self.num_layers

        # Sort layers by sensitivity (most sensitive first — upgrade these)
        sorted_layers = sorted(
            range(self.num_layers),
            key=lambda l: sensitivities[l],
            reverse=True,
        )

        total_mem = sum(self._layer_mem(seq_len, min_bits) for _ in range(self.num_layers))

        for l in sorted_layers:
            if total_mem <= budget:
                # Try upgrading remaining layers too
                pass
            current = assignments[l]
            idx = self.available_bits.index(current)
            while idx < len(self.available_bits) - 1:
                next_bits = self.available_bits[idx + 1]
                delta = self._layer_mem(seq_len, next_bits) - self._layer_mem(seq_len, current)
                if total_mem + delta <= budget:
                    total_mem += delta
                    assignments[l] = next_bits
                    current = next_bits
                    idx += 1
                else:
                    break

        return assignments

    def _layer_mem(self, seq_len: int, bits: int) -> int:
        """Memory for one layer at given bits."""
        base = 2 * seq_len * self.num_heads * self.head_dim * bits // 8
        if bits < 16:
            base += self.num_heads * self.head_dim * 4 + seq_len * 4
        return base

    def _ternary_quantize(self, tensor: Tensor) -> Tensor:
        """2-bit ternary quantization: {-1, 0, 1} × scale.

        Per-channel absolute-mean scaling.
        """
        scale = tensor.float().abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
        normalized = tensor.float() / scale
        # Quantize to {-1, 0, 1}
        ternary = normalized.sign() * (normalized.abs() > 0.5).float()
        # Dequantize
        return (ternary * scale).to(tensor.dtype)
