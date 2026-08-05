"""SqueezeAttention baseline (ICLR 2025).

2D optimization over layers AND sequence positions.
Layer importance measured by cosine similarity of input hidden states
before/after self-attention. Important layers keep full cache;
unimportant layers get aggressively pruned.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from .base import BaselineMethod, h2o_token_selection, register_baseline


@register_baseline
class SqueezeAttention(BaselineMethod):
    name = "squeeze_attention"
    category = "eviction"
    requires_attention = True
    is_per_layer = True
    reference = "Tang et al., ICLR 2025"

    def __init__(self, num_layers, num_heads, head_dim, **kwargs):
        super().__init__(num_layers, num_heads, head_dim)
        self.sink_tokens = kwargs.get("sink_tokens", 4)
        self.recent_tokens = kwargs.get("recent_tokens", 16)
        self.important_ratio = kwargs.get("important_ratio", 0.5)

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

        # Classify layers as important/unimportant
        importance = self._layer_importance(attention_weights, seq_len)

        # Binary classification: top important_ratio layers get full budget
        sorted_layers = sorted(range(self.num_layers), key=lambda l: importance[l], reverse=True)
        n_important = max(1, int(self.num_layers * self.important_ratio))
        important_set = set(sorted_layers[:n_important])

        # Distribute budget: important layers get seq_len, rest share remainder
        budget_for_important = min(n_important * seq_len, total_budget)
        per_important = min(seq_len, budget_for_important // max(1, n_important))
        remaining = total_budget - per_important * n_important
        n_unimportant = self.num_layers - n_important
        per_unimportant = max(min_tokens, remaining // max(1, n_unimportant)) if n_unimportant > 0 else 0

        results = []
        for l in range(self.num_layers):
            if l in important_set:
                budget = min(per_important, seq_len)
            else:
                budget = min(per_unimportant, seq_len)
            budget = max(min_tokens, budget)

            attn = attention_weights[l] if attention_weights and l < len(attention_weights) else None
            indices = h2o_token_selection(
                full_keys[l:l+1], full_values[l:l+1],
                budget, seq_len,
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

    def _layer_importance(
        self, attention_weights: Optional[List[Tensor]], seq_len: int,
    ) -> List[float]:
        """Measure layer importance via attention concentration.

        SqueezeAttention uses cosine similarity of input before/after
        self-attention, but this requires hidden_states. We approximate
        using attention entropy: low entropy = concentrated attention =
        the layer is making strong decisions = more important.
        """
        if not attention_weights:
            return [1.0] * self.num_layers

        scores = []
        for l in range(self.num_layers):
            if l >= len(attention_weights):
                scores.append(0.5)
                continue

            attn = attention_weights[l][0]  # (H, S, S)
            # Compute per-head entropy of last-token attention row
            last_row = attn[:, -1, :]  # (H, S)
            clamped = last_row.float().clamp(min=1e-10)
            per_head_entropy = -(clamped * clamped.log()).sum(dim=-1)  # (H,)
            max_ent = math.log(seq_len) if seq_len > 1 else 1.0

            # Lower entropy = more concentrated = more important
            avg_entropy = per_head_entropy.mean().item() / max_ent
            importance = 1.0 - avg_entropy  # Invert: high importance = low entropy
            scores.append(max(0.0, importance))

        return scores
