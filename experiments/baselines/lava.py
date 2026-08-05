"""LAVa baseline (EMNLP 2025 Findings): Layer-wise KV Cache Eviction.

Uses information change in the residual stream to determine per-layer
importance. Layers where the residual changes significantly are more
important and get larger token budgets.

If hidden states are not available, falls back to attention-entropy-based
importance estimation.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from .base import BaselineMethod, h2o_token_selection, register_baseline


@register_baseline
class LAVa(BaselineMethod):
    name = "lava"
    category = "eviction"
    requires_attention = True
    requires_hidden_states = True
    is_per_layer = True
    reference = "Shen et al., EMNLP 2025 Findings"

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

        # Compute per-layer importance
        importance = self._compute_importance(hidden_states, attention_weights, seq_len)

        # Allocate proportional to importance
        total_imp = sum(importance) + 1e-10
        budgets = []
        for l in range(self.num_layers):
            frac = importance[l] / total_imp
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

    def _compute_importance(
        self,
        hidden_states: Optional[List[Tensor]],
        attention_weights: Optional[List[Tensor]],
        seq_len: int,
    ) -> List[float]:
        """Compute per-layer importance from residual stream changes.

        If hidden_states available: ||h_l - h_{l-1}||_F per layer.
        Otherwise fallback to attention-based importance.
        """
        if hidden_states and len(hidden_states) > 1:
            return self._residual_importance(hidden_states)
        return self._attention_importance(attention_weights, seq_len)

    def _residual_importance(self, hidden_states: List[Tensor]) -> List[float]:
        """Importance from residual stream change magnitude."""
        scores = []
        for l in range(self.num_layers):
            if l + 1 < len(hidden_states):
                # ||h_{l+1} - h_l||_F: how much this layer changes the representation
                h_before = hidden_states[l].float()
                h_after = hidden_states[l + 1].float()
                delta = (h_after - h_before).norm().item()
                scores.append(max(1e-10, delta))
            else:
                scores.append(1.0)
        return scores

    def _attention_importance(
        self, attention_weights: Optional[List[Tensor]], seq_len: int,
    ) -> List[float]:
        """Fallback: use attention concentration as importance proxy."""
        if not attention_weights:
            return [1.0] * self.num_layers

        scores = []
        for l in range(self.num_layers):
            if l >= len(attention_weights):
                scores.append(1.0)
                continue
            last_row = attention_weights[l][0, :, -1, :].mean(dim=0)
            clamped = last_row.float().clamp(min=1e-10)
            entropy = -(clamped * clamped.log()).sum().item()
            max_ent = math.log(seq_len) if seq_len > 1 else 1.0
            # Lower entropy = more focused = more important
            scores.append(max(1e-10, 1.0 - entropy / max_ent))
        return scores
