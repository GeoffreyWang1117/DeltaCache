"""D2O baseline (ICLR 2025): Dynamic Discriminative Operations.

Two-level approach:
  Layer level: compute per-layer "diversity score" from attention weight
  distributions. Layers with more diverse attention need more tokens.
  Token level: cumulative attention with density persistence.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from .base import BaselineMethod, h2o_token_selection, register_baseline


@register_baseline
class D2O(BaselineMethod):
    name = "d2o"
    category = "eviction"
    requires_attention = True
    is_per_layer = True
    reference = "Wan et al., ICLR 2025"

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

        # Layer-level: compute diversity scores
        diversity = self._compute_diversity(attention_weights, seq_len)

        # Allocate proportionally to diversity
        total_div = sum(diversity) + 1e-10
        budgets = []
        for l in range(self.num_layers):
            frac = diversity[l] / total_div
            n = max(min_tokens, int(frac * total_budget))
            n = min(n, seq_len)
            budgets.append(n)

        # Rescale to match total budget
        current = sum(budgets)
        if current > 0:
            scale = total_budget / current
            budgets = [max(min_tokens, min(int(b * scale), seq_len)) for b in budgets]

        # Token-level selection per layer
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

    def _compute_diversity(
        self, attention_weights: Optional[List[Tensor]], seq_len: int,
    ) -> List[float]:
        """Compute per-layer attention diversity score.

        Diversity = std of cumulative attention across positions.
        High diversity = layer uses many different tokens = needs more budget.
        """
        if not attention_weights:
            return [1.0] * self.num_layers

        scores = []
        for l in range(self.num_layers):
            if l >= len(attention_weights):
                scores.append(1.0)
                continue

            attn = attention_weights[l]  # (B, H, S, S)
            # Last-token attention row, averaged over heads
            last_row = attn[0, :, -1, :].mean(dim=0)  # (S,)

            # Diversity: standard deviation of attention distribution
            # More spread out = higher diversity = layer needs more tokens
            std = last_row.float().std().item()

            # Also consider attention entropy as a secondary signal
            clamped = last_row.float().clamp(min=1e-10)
            entropy = -(clamped * clamped.log()).sum().item()
            max_entropy = math.log(seq_len) if seq_len > 1 else 1.0
            norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

            # Combined diversity: entropy-weighted std
            scores.append(std * (1 + norm_entropy))

        return scores
