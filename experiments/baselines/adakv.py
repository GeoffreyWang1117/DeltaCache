"""Ada-KV baseline (NeurIPS 2025): Head-wise adaptive budget allocation.

Instead of giving each head the same budget, allocates more tokens to
heads with higher entropy (more spread-out attention). Total per-layer
budget is uniform, but per-head budget varies.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from .base import BaselineMethod, register_baseline


@register_baseline
class AdaKV(BaselineMethod):
    name = "adakv"
    category = "eviction"
    requires_attention = True
    is_per_layer = False  # Per-head within each layer, but uniform across layers
    reference = "Feng et al., NeurIPS 2025"

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
            if attention_weights and l < len(attention_weights):
                indices = self._head_adaptive_selection(
                    attention_weights[l], n_tokens, seq_len,
                )
            else:
                indices = torch.arange(min(n_tokens, seq_len))

            indices = indices.long()
            results.append((
                full_keys[l:l+1, indices],
                full_values[l:l+1, indices],
                indices,
            ))
        return results

    def _head_adaptive_selection(
        self, attn: Tensor, n_tokens: int, seq_len: int,
    ) -> Tensor:
        """Per-head adaptive budget, then union of selections.

        Each head gets a budget proportional to its attention entropy.
        Higher entropy = attention more spread out = needs more tokens.
        """
        if n_tokens >= seq_len:
            return torch.arange(seq_len)

        n_heads = attn.shape[1]
        last_row = attn[0, :, -1, :]  # (H, S)

        # Compute per-head entropy
        clamped = last_row.float().clamp(min=1e-10)
        per_head_entropy = -(clamped * clamped.log()).sum(dim=-1)  # (H,)

        # Allocate budget proportional to entropy
        total_entropy = per_head_entropy.sum().item() + 1e-10
        head_budgets = []
        sink = min(self.sink_tokens, seq_len)
        for h in range(n_heads):
            frac = per_head_entropy[h].item() / total_entropy
            hb = max(sink + 1, int(frac * n_tokens * n_heads / n_heads))
            hb = min(hb, seq_len)
            head_budgets.append(hb)

        # Per-head token selection (cumulative attention)
        all_selected = set()
        # Always include sink tokens
        for i in range(sink):
            all_selected.add(i)

        for h in range(n_heads):
            head_attn = last_row[h].cpu()  # (S,)
            _, top_idx = head_attn.topk(min(head_budgets[h], seq_len))
            all_selected.update(top_idx.tolist())

        # If we selected too many, trim by global importance
        selected = sorted(all_selected)
        if len(selected) > n_tokens:
            # Keep the n_tokens most important by average attention
            avg_attn = last_row.mean(dim=0).cpu()
            scored = [(idx, avg_attn[idx].item()) for idx in selected]
            scored.sort(key=lambda x: x[1], reverse=True)
            # Always keep sink tokens
            kept = set(range(sink))
            for idx, _ in scored:
                if len(kept) >= n_tokens:
                    break
                kept.add(idx)
            selected = sorted(kept)

        return torch.tensor(selected[:n_tokens], dtype=torch.long)
