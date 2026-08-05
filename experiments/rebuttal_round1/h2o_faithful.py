"""Faithful H2O baseline: per-head heavy-hitter selection + larger recent window.

Differences from our existing h2o_uniform:
  1. Per-head attention scoring (official: tmp_sum = softmax(attn).sum(dim=-2) per head)
     We take the per-head top-K and UNION across heads, then truncate to budget.
  2. Recent window scaled with seq_len (official: 0.1×seq_len), not fixed 16.
  3. No sink tokens (official has none).

This isolates whether our 36.5% MMLU on Llama-2-7B at CR=4× is due to a faithful
H2O algorithm or a methodological mismatch with the official artifact.

Run with the same MMLU pipeline as the suite. Compares against existing h2o_uniform
checkpoint (416/1140 = 36.49%) and KIVI (544/1140 = 47.72%).
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

DELTACACHE_ROOT = Path("/home/coder-gw/Projects/DeltaCache")
sys.path.insert(0, str(DELTACACHE_ROOT))
sys.path.insert(0, str(DELTACACHE_ROOT / "experiments"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import Tensor  # noqa: E402

from baselines.base import BaselineMethod, register_baseline  # noqa: E402

OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def h2o_perhead_selection(
    full_keys: Tensor,
    full_values: Tensor,
    n_tokens: int,
    seq_len: int,
    attention_weights: Tensor,  # (1, heads, seq_len, seq_len)
    recent_ratio: float = 0.1,
) -> Tensor:
    """Faithful per-head H2O selection: union of per-head top-K heavy + recent window.

    Args:
        full_keys: (1, seq_len, H, D) — unused, kept for signature parity
        full_values: (1, seq_len, H, D) — unused
        n_tokens: total token budget for this layer
        seq_len: sequence length
        attention_weights: (1, heads, seq_len, seq_len) attention probs
        recent_ratio: fraction of seq_len to retain at the recent end (default 0.1)

    Returns:
        Sorted long tensor of selected token indices.
    """
    if n_tokens >= seq_len:
        return torch.arange(seq_len, dtype=torch.long)

    # Recent window
    recent = max(1, int(recent_ratio * seq_len))
    recent = min(recent, seq_len)
    heavy_budget = max(0, n_tokens - recent)

    # Per-head accumulated attention scores
    # attn shape: (1, heads, seq_len, seq_len)
    attn = attention_weights[0]  # (H, S, S)
    # Sum across queries (dim=1), gives (H, S) — per-head per-key total attention received
    per_head_scores = attn.sum(dim=1)  # (H, S)

    # Per-head top-K
    if heavy_budget > 0:
        _, per_head_top = per_head_scores.topk(k=min(heavy_budget, seq_len - recent), dim=-1)
        # Union across heads
        heavy_idx = per_head_top.unique()
    else:
        heavy_idx = torch.tensor([], dtype=torch.long, device=attn.device)

    # Recent window
    recent_idx = torch.arange(seq_len - recent, seq_len, device=attn.device)

    # Combine and dedupe
    all_idx = torch.cat([heavy_idx, recent_idx]).unique()

    # Truncate / pad to exactly n_tokens
    if all_idx.numel() > n_tokens:
        # Drop earliest non-recent heavy-hitters first
        all_idx, _ = all_idx.sort()
        # Keep last n_tokens (which will include recent + late heavy)
        all_idx = all_idx[-n_tokens:]
    elif all_idx.numel() < n_tokens:
        # Pad with positions not yet selected, prioritizing high mean-per-head score
        mean_score = per_head_scores.mean(dim=0)  # (S,)
        used = torch.zeros(seq_len, dtype=torch.bool, device=attn.device)
        used[all_idx] = True
        free_scores = mean_score.clone()
        free_scores[used] = float("-inf")
        n_extra = n_tokens - all_idx.numel()
        _, extra_idx = free_scores.topk(n_extra)
        all_idx = torch.cat([all_idx, extra_idx]).unique()

    all_idx, _ = all_idx.sort()
    return all_idx[:n_tokens].cpu().long()


@register_baseline
class H2OFaithful(BaselineMethod):
    """Faithful H2O matching the official algorithm: per-head heavy + recent window."""
    name = "h2o_faithful"
    category = "eviction"
    requires_attention = True
    is_per_layer = False
    reference = "Zhang et al., NeurIPS 2023 (per-head, recent_ratio=0.1)"

    def __init__(self, num_layers, num_heads, head_dim, **kwargs):
        super().__init__(num_layers, num_heads, head_dim)
        self.recent_ratio = kwargs.get("recent_ratio", 0.1)

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
            attn = attention_weights[l] if attention_weights else None
            if attn is None:
                # Without attention, fall back to value-norm
                v = full_values[l, :, :, :].float().norm(dim=-1).mean(dim=-1).cpu()  # (S,)
                _, top = v.topk(min(n_tokens, seq_len))
                indices, _ = top.sort()
            else:
                indices = h2o_perhead_selection(
                    full_keys[l:l+1], full_values[l:l+1],
                    n_tokens, seq_len,
                    attention_weights=attn,
                    recent_ratio=self.recent_ratio,
                )
            results.append((
                full_keys[l:l+1, indices.long()],
                full_values[l:l+1, indices.long()],
                indices,
            ))
        return results


def main() -> None:
    """Run MMLU on Llama-2-7B with h2o_uniform vs h2o_faithful, compare to KIVI."""
    # Use the existing suite runner
    sys.path.insert(0, str(DELTACACHE_ROOT / "experiments" / "suite"))
    # Ensure baselines/__init__.py picks up our new class
    import baselines  # noqa: F401  triggers loading
    # Force-import faithful so it registers
    from baselines.base import REGISTRY
    if "h2o_faithful" not in REGISTRY:
        REGISTRY["h2o_faithful"] = H2OFaithful

    print(f"Available baselines: {sorted(REGISTRY.keys())}")
    print(f"\nh2o_faithful registered: {'h2o_faithful' in REGISTRY}")

    # Build a fake CLI invocation to trigger MMLU eval
    # Easiest path: invoke run_all.py with --task mmlu --method h2o_faithful --model llama2-7b
    # but that requires modifying compress_kv dispatch. Let me check.
    from suite.eval_utils import compress_kv  # noqa: E402
    print(f"compress_kv known methods: trying h2o_faithful...")
    # Quick smoke: build a fake K/V/attn and see compress_kv routes correctly
    nl, nh, hd, S = 4, 8, 64, 64
    K = torch.randn(nl, S, nh, hd, dtype=torch.float16)
    V = torch.randn_like(K)
    attns = [torch.softmax(torch.randn(1, nh, S, S), dim=-1) for _ in range(nl)]
    try:
        out = compress_kv("h2o_faithful", K, V, 4.0, nl, nh, hd, attention_weights=attns)
        print(f"  smoke test: returned {len(out[0]) if out else None} layers")
    except Exception as e:  # noqa: BLE001
        print(f"  smoke test FAILED: {e}")
        import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
