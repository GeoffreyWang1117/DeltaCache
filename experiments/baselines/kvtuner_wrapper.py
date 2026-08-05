"""KVTuner wrapper conforming to BaselineMethod interface."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import Tensor

from .base import BaselineMethod, register_baseline
from .kvtuner import KVTunerBaseline


@register_baseline
class KVTuner(BaselineMethod):
    name = "kvtuner"
    category = "quantization"
    requires_attention = False
    is_per_layer = True
    reference = "ICML 2025, arXiv:2502.04420"

    def __init__(self, num_layers, num_heads, head_dim, **kwargs):
        super().__init__(num_layers, num_heads, head_dim)
        self._impl = KVTunerBaseline(num_layers, num_heads, head_dim, **kwargs)

    def compress(
        self,
        full_keys: Tensor,
        full_values: Tensor,
        compression_ratio: float,
        attention_weights: Optional[List[Tensor]] = None,
        hidden_states: Optional[List[Tensor]] = None,
    ) -> List[Tuple[Tensor, Tensor, Tensor]]:
        raw = self._impl.compress(full_keys, full_values, compression_ratio)
        seq_len = full_keys.shape[1]
        all_indices = torch.arange(seq_len)
        # Store bits per layer for accurate memory reporting
        self._layer_bits = [bits for _, _, bits in raw]
        return [(k, v, all_indices) for k, v, _ in raw]

    def memory_bytes(self, compressed_layers):
        """Report actual quantized memory based on per-layer bit assignments."""
        if not compressed_layers:
            return 0
        seq_len = compressed_layers[0][0].shape[1]
        if hasattr(self, "_layer_bits") and self._layer_bits:
            total = 0
            for bits in self._layer_bits:
                total += self._impl._memory_cost(seq_len, bits)
            return total
        return self._impl._full_memory(seq_len)
