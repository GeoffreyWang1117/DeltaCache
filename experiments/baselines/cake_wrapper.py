"""CAKE wrapper conforming to BaselineMethod interface."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import Tensor

from .base import BaselineMethod, register_baseline
from .cake import CAKEBaseline


@register_baseline
class CAKE(BaselineMethod):
    name = "cake"
    category = "eviction"
    requires_attention = True
    is_per_layer = True
    reference = "ICLR 2025, arXiv:2503.12491"

    def __init__(self, num_layers, num_heads, head_dim, **kwargs):
        super().__init__(num_layers, num_heads, head_dim)
        self._impl = CAKEBaseline(num_layers, num_heads, head_dim, **kwargs)

    def compress(
        self,
        full_keys: Tensor,
        full_values: Tensor,
        compression_ratio: float,
        attention_weights: Optional[List[Tensor]] = None,
        hidden_states: Optional[List[Tensor]] = None,
    ) -> List[Tuple[Tensor, Tensor, Tensor]]:
        return self._impl.compress(
            full_keys, full_values, attention_weights, compression_ratio,
        )
