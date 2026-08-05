"""DynamicKV baseline (EMNLP 2025 Findings).

Per-layer token budget based on attention score variance. Layers with
high variance have more dynamic attention patterns and need more tokens
to preserve quality.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import Tensor

from .base import BaselineMethod, h2o_token_selection, register_baseline


@register_baseline
class DynamicKV(BaselineMethod):
    name = "dynamickv"
    category = "eviction"
    requires_attention = True
    is_per_layer = True
    reference = "Jiang et al., EMNLP 2025 Findings"

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

        # Compute per-layer attention variance
        variances = self._compute_variance(attention_weights)

        # Allocate proportional to variance
        total_var = sum(variances) + 1e-10
        budgets = []
        for l in range(self.num_layers):
            frac = variances[l] / total_var
            n = max(min_tokens, int(frac * total_budget))
            n = min(n, seq_len)
            budgets.append(n)

        # Rescale
        current = sum(budgets)
        if current > 0:
            scale = total_budget / current
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

    def _compute_variance(
        self, attention_weights: Optional[List[Tensor]],
    ) -> List[float]:
        """Per-layer attention score variance."""
        if not attention_weights:
            return [1.0] * self.num_layers

        scores = []
        for l in range(self.num_layers):
            if l >= len(attention_weights):
                scores.append(1.0)
                continue

            # Last-token attention row averaged over heads
            last_row = attention_weights[l][0, :, -1, :].mean(dim=0)  # (S,)
            # Variance of attention distribution
            var = last_row.float().var().item()
            scores.append(max(1e-10, var))

        return scores
