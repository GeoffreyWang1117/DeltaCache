"""H2O baseline (NeurIPS 2023): Heavy-Hitter Oracle.

Uniform token budget across all layers. Within each layer, retains
sink tokens + recent tokens + heavy-hitter middle tokens selected
by cumulative attention mass or value norm.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import Tensor

from .base import BaselineMethod, h2o_token_selection, register_baseline


@register_baseline
class H2OUniform(BaselineMethod):
    name = "h2o_uniform"
    category = "eviction"
    requires_attention = True
    is_per_layer = False
    reference = "Zhang et al., NeurIPS 2023"

    def __init__(self, num_layers, num_heads, head_dim, **kwargs):
        super().__init__(num_layers, num_heads, head_dim)
        self.sink_tokens = kwargs.get("sink_tokens", 4)
        self.recent_tokens = kwargs.get("recent_tokens", 16)

    def compress(
        self,
        full_keys: Tensor,
        full_values: Tensor,
        compression_ratio: float,
        attention_weights: Optional[List[Tensor]] = None,
        hidden_states: Optional[List[Tensor]] = None,
    ) -> List[Tuple[Tensor, Tensor, Tensor]]:
        seq_len = full_keys.shape[1]
        n_tokens = max(1, int(seq_len / compression_ratio))

        results = []
        for l in range(self.num_layers):
            attn = attention_weights[l] if attention_weights else None
            indices = h2o_token_selection(
                full_keys[l:l+1], full_values[l:l+1],
                n_tokens, seq_len,
                attention_weights=attn,
                sink_tokens=self.sink_tokens,
                recent_tokens=self.recent_tokens,
            )
            results.append((
                full_keys[l:l+1, indices.long()],
                full_values[l:l+1, indices.long()],
                indices,
            ))
        return results
