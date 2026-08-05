"""DuoAttention baseline (Xiao et al., ICLR 2025).

Faithful implementation of the DuoAttention method from:
    "DuoAttention: Efficient Long-Context LLM Inference with
    Retrieval and Streaming Heads"
    Xiao, Tang, Zhang, Cai, Han, Lu, Han. ICLR 2025.
    https://arxiv.org/abs/2410.10819

Key idea
────────
Each attention head is classified into one of two types via offline
gradient-based optimization on a calibration set:

  • Retrieval head:  needs full KV cache (long-range information)
  • Streaming head:  only needs sink + recent tokens (local information)

The classification uses a learnable per-head parameter α_h ∈ [0, 1]
optimized via:

    L = ‖f_α(x) − f_full(x)‖²  +  λ · Σ_h |α_h|

where f_α blends full attention (weight α) and streaming attention
(weight 1−α) per head, and λ pushes α toward 0 (more streaming heads).
After optimization, α is thresholded to {0, 1}.

Profile generation
──────────────────
Profiles are produced offline by:
    python -m experiments.scripts.profile_duo_attention <model_key>
and cached at:
    ~/.cache/duoattention_profiles/<short_name>.pt

A profile is a dict {layer_idx: Tensor(num_heads,) ∈ {0, 1}}.

Interface adaptation note
─────────────────────────
The original DuoAttention applies per-head retention at inference time
(retrieval heads keep all tokens, streaming heads keep sink+recent).
The unified BaselineMethod interface in this benchmark requires a single
per-layer index set, so per-head retention is reduced to a per-layer
budget proportional to the fraction of retrieval heads in that layer:

    n_kept_l = max(n_streaming, round(r_l · seq_len + (1−r_l) · n_streaming))

where r_l = mean over heads of α_h for layer l.

This preserves DuoAttention's core insight (some layers need more long-
range tokens than others, driven by per-head specialization) while
fitting the per-layer interface used by all 16 other baselines.  The
α profile itself is computed faithfully via the original optimization.
The reduction is documented and is the only adaptation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from .base import BaselineMethod, register_baseline


PROFILE_CACHE_DIR = Path.home() / ".cache" / "duoattention_profiles"

# In-memory cache so we don't re-read the profile for every compress() call
_PROFILE_CACHE: Dict[str, Dict[int, Tensor]] = {}


def profile_path(model_short_name: str) -> Path:
    """Path where the profile for a given model is cached."""
    safe = model_short_name.replace("/", "_")
    return PROFILE_CACHE_DIR / f"{safe}.pt"


def load_profile(model_short_name: str) -> Dict[int, Tensor]:
    """Load a saved DuoAttention profile.

    Returns a dict {layer_idx: Tensor(num_heads,)} with α ∈ [0, 1].
    Raises FileNotFoundError with a clear message if missing.
    """
    if model_short_name in _PROFILE_CACHE:
        return _PROFILE_CACHE[model_short_name]
    p = profile_path(model_short_name)
    if not p.exists():
        raise FileNotFoundError(
            f"DuoAttention profile not found at {p}\n"
            f"Run offline profiling first:\n"
            f"  python -m experiments.scripts.profile_duo_attention "
            f"--model {model_short_name}"
        )
    profile = torch.load(p, map_location="cpu", weights_only=False)
    _PROFILE_CACHE[model_short_name] = profile
    return profile


@register_baseline
class DuoAttention(BaselineMethod):
    """DuoAttention with offline-profiled per-head retrieval/streaming policy.

    Requires a profile file at ~/.cache/duoattention_profiles/<short_name>.pt
    produced by experiments/scripts/profile_duo_attention.py.
    """

    name = "duo_attention"
    category = "eviction"
    requires_attention = False  # Profile is offline; no attention at runtime
    is_per_layer = True
    reference = "Xiao et al., ICLR 2025"

    def __init__(self, num_layers: int, num_heads: int, head_dim: int, **kwargs):
        super().__init__(num_layers, num_heads, head_dim)
        self.sink_tokens = kwargs.get("sink_tokens", 16)
        self.recent_tokens = kwargs.get("recent_tokens", 64)
        # Model identifier needed to locate the profile
        self.model_short_name = kwargs.get("model_short_name", None)
        self._profile: Optional[Dict[int, Tensor]] = None
        if self.model_short_name is not None:
            self._profile = load_profile(self.model_short_name)
            self._validate_profile()

    def _validate_profile(self) -> None:
        if self._profile is None:
            return
        if len(self._profile) != self.num_layers:
            raise ValueError(
                f"DuoAttention profile has {len(self._profile)} layers, "
                f"model has {self.num_layers}"
            )
        for l, alpha in self._profile.items():
            if alpha.numel() != self.num_heads:
                raise ValueError(
                    f"Profile layer {l}: α has {alpha.numel()} entries, "
                    f"model has {self.num_heads} KV heads"
                )

    def _layer_retrieval_fraction(self, layer_idx: int) -> float:
        """Mean retrieval-ness (α) over heads in this layer."""
        if self._profile is None:
            return 0.5  # uniform fallback (only used in unit tests)
        return float(self._profile[layer_idx].mean().clamp(0, 1).item())

    def compress(
        self,
        full_keys: Tensor,
        full_values: Tensor,
        compression_ratio: float,
        attention_weights: Optional[List[Tensor]] = None,
        hidden_states: Optional[List[Tensor]] = None,
    ) -> List[Tuple[Tensor, Tensor, Tensor]]:
        seq_len = full_keys.shape[1]

        # Streaming-only floor: sink + recent at the target compression ratio.
        # When the layer is "all streaming" this is what we keep.
        target_total = max(1, round(seq_len / compression_ratio))
        n_sink = min(self.sink_tokens, target_total)
        n_recent = max(0, min(self.recent_tokens, target_total - n_sink))
        n_streaming = n_sink + n_recent

        results = []
        for l in range(self.num_layers):
            r_l = self._layer_retrieval_fraction(l)
            # Layers with more retrieval heads keep more tokens.
            # Linearly interpolate between streaming-only and full retention.
            n_kept_l = round(r_l * seq_len + (1 - r_l) * n_streaming)
            n_kept_l = max(n_streaming, min(n_kept_l, seq_len))

            indices = self._select_indices(seq_len, n_kept_l, n_sink, n_recent)

            k_sel = full_keys[l:l+1, indices]
            v_sel = full_values[l:l+1, indices]
            results.append((k_sel, v_sel, indices))

        return results

    @staticmethod
    def _select_indices(
        seq_len: int, n_kept: int, n_sink: int, n_recent: int,
    ) -> Tensor:
        """Sink + recent + uniformly subsampled middle."""
        if n_kept >= seq_len:
            return torch.arange(seq_len, dtype=torch.long)

        sink_idx = torch.arange(n_sink, dtype=torch.long)
        recent_start = max(n_sink, seq_len - n_recent)
        recent_idx = torch.arange(recent_start, seq_len, dtype=torch.long)

        n_middle_budget = max(0, n_kept - n_sink - n_recent)
        if n_middle_budget > 0 and recent_start > n_sink:
            middle_pool = torch.arange(n_sink, recent_start, dtype=torch.long)
            if n_middle_budget >= len(middle_pool):
                middle_idx = middle_pool
            else:
                # Uniformly spaced subsample (deterministic, no attention needed)
                step = len(middle_pool) / n_middle_budget
                pick = torch.tensor(
                    [int(i * step) for i in range(n_middle_budget)],
                    dtype=torch.long,
                )
                middle_idx = middle_pool[pick]
        else:
            middle_idx = torch.tensor([], dtype=torch.long)

        all_idx = torch.cat([sink_idx, middle_idx, recent_idx]).unique()
        all_idx, _ = all_idx.sort()
        return all_idx[:n_kept]
