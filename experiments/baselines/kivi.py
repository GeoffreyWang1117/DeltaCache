"""KIVI baseline (ICML 2024): Uniform asymmetric INT4 quantization.

All tokens retained, all layers quantized to the same bit-width (INT4).
Dequantized back to FP16 for quality comparison.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import Tensor

from deltacache.core.kv_quantizer import KVQuantizer, QuantPrecision

from .base import BaselineMethod, register_baseline


@register_baseline
class KIVIUniform(BaselineMethod):
    name = "kivi_uniform"
    category = "quantization"
    requires_attention = False
    is_per_layer = False
    reference = "Liu et al., ICML 2024"

    def __init__(self, num_layers, num_heads, head_dim, **kwargs):
        super().__init__(num_layers, num_heads, head_dim)
        self.quant_bits = kwargs.get("quant_bits", 4)
        prec = QuantPrecision.INT4 if self.quant_bits == 4 else QuantPrecision.INT8
        self._quantizer = KVQuantizer(prec)

    def compress(
        self,
        full_keys: Tensor,
        full_values: Tensor,
        compression_ratio: float,
        attention_weights: Optional[List[Tensor]] = None,
        hidden_states: Optional[List[Tensor]] = None,
    ) -> List[Tuple[Tensor, Tensor, Tensor]]:
        seq_len = full_keys.shape[1]
        all_indices = torch.arange(seq_len)

        results = []
        for l in range(self.num_layers):
            qkv = self._quantizer.quantize(full_keys[l:l+1], full_values[l:l+1])
            k_deq, v_deq = self._quantizer.dequantize(qkv)
            results.append((k_deq, v_deq, all_indices))
        return results

    def memory_bytes(self, compressed_layers):
        """Report actual quantized memory, not dequantized FP16 size."""
        if not compressed_layers:
            return 0
        seq_len = compressed_layers[0][0].shape[1]
        per_layer = 2 * seq_len * self.num_heads * self.head_dim * self.quant_bits // 8
        # Metadata: scales + zeros
        key_meta = self.num_heads * self.head_dim * 4
        value_meta = seq_len * 4
        per_layer += key_meta + value_meta
        return per_layer * self.num_layers
