"""StreamingLLM baseline (Xiao et al., 2023 / 2024).

Retains only attention sink tokens (initial) + a sliding recent window.
Middle tokens are discarded entirely.

This is the simplest possible eviction policy and serves as the
"lower bound" for token eviction quality. It demonstrates why
position-unaware eviction fails: critical early tokens are sinks,
and the model becomes incoherent without them (the "lost in the middle"
effect is acute without sink pinning).

Reference:
    Xiao et al., "Efficient Streaming Language Models with Attention Sinks",
    ICLR 2024. https://arxiv.org/abs/2309.17453
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import Tensor

from .base import BaselineMethod, register_baseline


@register_baseline
class StreamingLLM(BaselineMethod):
    """Attention sink + sliding window eviction.

    Budget allocation:
      - sink_tokens: always retained (attention sinks at positions 0..K-1)
      - recent_tokens: remaining budget = max(0, target_tokens - sink_tokens)
      - Middle tokens: discarded

    The target_tokens count is derived from compression_ratio:
      target_tokens = round(seq_len / compression_ratio)

    This means at 4× compression on a 1024-token sequence, only 256 tokens
    are kept (e.g., 4 sinks + 252 most-recent tokens).
    """

    name = "streaming_llm"
    category = "eviction"
    requires_attention = False
    is_per_layer = False  # Same window applied to all layers
    reference = "Xiao et al., ICLR 2024"

    def __init__(self, num_layers: int, num_heads: int, head_dim: int, **kwargs):
        super().__init__(num_layers, num_heads, head_dim)
        self.sink_tokens = kwargs.get("sink_tokens", 4)

    def compress(
        self,
        full_keys: Tensor,
        full_values: Tensor,
        compression_ratio: float,
        attention_weights: Optional[List[Tensor]] = None,
        hidden_states: Optional[List[Tensor]] = None,
    ) -> List[Tuple[Tensor, Tensor, Tensor]]:
        seq_len = full_keys.shape[1]

        # How many tokens to keep in total
        target_tokens = max(
            self.sink_tokens + 1,
            round(seq_len / compression_ratio),
        )
        target_tokens = min(target_tokens, seq_len)

        # Sink indices: first sink_tokens positions
        n_sink = min(self.sink_tokens, target_tokens)
        # Recent window: fills remaining budget from the end
        n_recent = max(0, target_tokens - n_sink)

        sink_idx = torch.arange(n_sink, dtype=torch.long)

        if n_recent > 0:
            recent_start = max(n_sink, seq_len - n_recent)
            recent_idx = torch.arange(recent_start, seq_len, dtype=torch.long)
        else:
            recent_idx = torch.tensor([], dtype=torch.long)

        # Merge, deduplicate, sort (handles edge case where sink overlaps recent)
        indices = torch.cat([sink_idx, recent_idx]).unique()
        indices, _ = indices.sort()
        # Trim to target in case overlap added extras
        indices = indices[:target_tokens]

        results = []
        for l in range(self.num_layers):
            k_sel = full_keys[l:l+1, indices.long()]
            v_sel = full_values[l:l+1, indices.long()]
            results.append((k_sel, v_sel, indices))

        return results
