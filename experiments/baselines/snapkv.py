"""SnapKV baseline (NeurIPS 2024).

Uses an observation window of the last W query positions' attention
patterns to vote on which tokens to keep. Tokens that consistently
receive high attention across the observation window are retained.
Uniform budget across layers.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import Tensor

from .base import BaselineMethod, register_baseline


@register_baseline
class SnapKV(BaselineMethod):
    name = "snapkv"
    category = "eviction"
    requires_attention = True
    is_per_layer = False
    reference = "Li et al., NeurIPS 2024"

    def __init__(self, num_layers, num_heads, head_dim, **kwargs):
        super().__init__(num_layers, num_heads, head_dim)
        self.observation_window = kwargs.get("observation_window", 32)
        self.sink_tokens = kwargs.get("sink_tokens", 4)
        self.kernel_size = kwargs.get("kernel_size", 7)

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
                indices = self._select_tokens(
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

    def _select_tokens(
        self, attn: Tensor, n_tokens: int, seq_len: int,
    ) -> Tensor:
        """SnapKV observation-window voting.

        Args:
            attn: (batch, heads, seq_len, seq_len)
            n_tokens: Budget.
            seq_len: Full sequence length.
        """
        if n_tokens >= seq_len:
            return torch.arange(seq_len)

        obs_window = min(self.observation_window, seq_len)
        sink = min(self.sink_tokens, seq_len)

        # Observation window: attention from the last W query positions
        # to all key positions. Shape: (heads, W, seq_len)
        obs_attn = attn[0, :, -obs_window:, :]  # (H, W, S)

        # Pool across observation window: sum importance votes
        importance = obs_attn.sum(dim=1)  # (H, S)

        # Optional: apply average pooling kernel for smoothing
        if self.kernel_size > 1 and importance.shape[-1] > self.kernel_size:
            pad = self.kernel_size // 2
            importance = torch.nn.functional.avg_pool1d(
                importance.unsqueeze(0),  # (1, H, S)
                kernel_size=self.kernel_size,
                padding=pad,
                stride=1,
            ).squeeze(0)  # (H, S)

        # Average over heads
        importance = importance.mean(dim=0).cpu()  # (S,)

        # Always keep sink tokens
        importance[:sink] = float("inf")

        # Select top-k
        k = min(n_tokens, seq_len)
        _, top_idx = importance.topk(k)
        top_idx, _ = top_idx.sort()
        return top_idx
