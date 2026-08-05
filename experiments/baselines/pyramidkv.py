"""PyramidKV baseline (COLM 2025).

Fixed pyramid-shaped per-layer budget: lower layers keep fewer tokens
(attention is more dispersed → less informative), upper layers keep
more tokens (attention is more focused → more task-specific).

No attention weights needed — allocation is purely positional.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import Tensor

from .base import BaselineMethod, h2o_token_selection, register_baseline


@register_baseline
class PyramidKV(BaselineMethod):
    name = "pyramidkv"
    category = "eviction"
    requires_attention = False
    is_per_layer = True
    reference = "Cai et al., COLM 2025"

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
        total_budget = int(seq_len * self.num_layers / compression_ratio)
        min_tokens = min(self.sink_tokens + self.recent_tokens, seq_len)

        # Pyramid allocation: B_l = B_min + (B_max - B_min) * (l / (L-1))
        # Solve for B_min and B_max such that sum = total_budget
        # sum = L * B_min + (B_max - B_min) * L/2 = total_budget
        # Let B_min = alpha * avg, B_max = (2 - alpha) * avg
        avg_budget = total_budget / self.num_layers
        alpha = 0.3  # Controls pyramid steepness
        b_min = max(min_tokens, int(alpha * avg_budget))
        b_max = max(b_min, int((2 - alpha) * avg_budget))
        b_max = min(b_max, seq_len)

        budgets = []
        for l in range(self.num_layers):
            if self.num_layers > 1:
                frac = l / (self.num_layers - 1)
            else:
                frac = 0.5
            budget = int(b_min + (b_max - b_min) * frac)
            budget = max(min_tokens, min(budget, seq_len))
            budgets.append(budget)

        # Rescale to match total budget
        current_total = sum(budgets)
        if current_total > 0 and abs(current_total - total_budget) > self.num_layers:
            scale = total_budget / current_total
            budgets = [max(min_tokens, min(int(b * scale), seq_len)) for b in budgets]

        results = []
        for l in range(self.num_layers):
            attn = attention_weights[l] if attention_weights and l < len(attention_weights) else None
            indices = h2o_token_selection(
                full_keys[l:l+1], full_values[l:l+1],
                budgets[l], seq_len,
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
