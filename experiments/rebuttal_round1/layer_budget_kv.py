"""LayerBudget-KV: experimental (n_l, b_K_l, b_V_l) extension.

Generalizes LayerBudget to independent K/V bit-widths, matching MoE-nD's
degrees of freedom while retaining inverted importance + closed-form coverage.
Goal: close the 21pp gap on Mistral-7B at CR=6× while preserving moderate-CR wins.

Differences from production layer_budget:
  - State: (n_l, b_K_l, b_V_l) per layer instead of (n_l, b_l)
  - Memory cost: K_bytes(n, b_K) + V_bytes(n, b_V) instead of 2*KV_bytes(n, b)
  - Greedy actions: add_tokens, upgrade_K, upgrade_V instead of
    add_tokens, upgrade_bits
  - Quality model: C(l, n) * F(b_K) * F(b_V) * importance(l)

Same as LayerBudget for everything else (sink, recent, token_step, importance signal).

Registers method "layer_budget_kv".
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DELTACACHE_ROOT = Path("/home/coder-gw/Projects/DeltaCache")
sys.path.insert(0, str(DELTACACHE_ROOT))
sys.path.insert(0, str(DELTACACHE_ROOT / "experiments"))

import torch  # noqa: E402
from torch import Tensor  # noqa: E402

from baselines.base import BaselineMethod, h2o_token_selection, register_baseline  # noqa: E402

# Per-bit fidelity from F(b) calibration (Mistral-7B; same as moend_baseline.py)
F_BITS = {16: 1.000, 8: 0.9998, 4: 0.9794}


def coverage(n_tokens: int, seq_len: int, gini: float) -> float:
    """Coverage law with squared penalty (matches production LayerBudget)."""
    if n_tokens <= 0:
        return 0.0
    if n_tokens >= seq_len:
        return 1.0
    attention_mass = (n_tokens / max(seq_len, 1)) ** max(1.0 - gini, 0.01)
    return attention_mass ** 2  # Squared penalty


def sigmoid_importance(l: int, L: int, k: float = 5.0, tau: float = 0.3, invert: bool = True) -> float:
    import math
    x = l / max(L - 1, 1)
    z = k * ((1.0 - x - tau) if invert else (x - tau))
    return 1.0 / (1.0 + math.exp(-z))


def kv_bytes(n: int, b_K: int, b_V: int, num_kv_heads: int, head_dim: int) -> int:
    """Bytes for one layer's K + V at retention n with separate K/V bit-widths.

    Same metadata model as production LayerBudget: per-channel K scale/zero,
    per-token V scale/zero (only at <16 bit).
    """
    k_base = n * num_kv_heads * head_dim * b_K // 8
    v_base = n * num_kv_heads * head_dim * b_V // 8
    if b_K < 16:
        k_base += num_kv_heads * head_dim * 4  # FP16 scale + zero per channel
    if b_V < 16:
        v_base += n * 4  # FP16 scale + zero per token
    return k_base + v_base


@register_baseline
class LayerBudgetKVBaseline(BaselineMethod):
    """LayerBudget with independent K/V bit-widths."""

    name = "layer_budget_kv"
    category = "joint"
    requires_attention = True
    is_per_layer = True
    reference = "LayerBudget extension to (n_l, b_K_l, b_V_l)"

    BIT_OPTIONS = (4, 8, 16)
    SINK = 4
    RECENT = 16
    DELTA_N = 8

    def __init__(self, num_layers: int, num_heads: int, head_dim: int, **kwargs):
        super().__init__(num_layers, num_heads, head_dim)

    def _allocate(
        self,
        sparsities: List[float],
        importance: List[float],
        seq_len: int,
        budget_bytes: float,
    ) -> List[Tuple[int, int, int]]:
        """Greedy joint (n_l, b_K_l, b_V_l) allocation."""
        L = self.num_layers
        H = self.num_heads
        D = self.head_dim

        n = [self.SINK + self.RECENT] * L
        bK = [self.BIT_OPTIONS[0]] * L
        bV = [self.BIT_OPTIONS[0]] * L

        def used() -> float:
            return sum(kv_bytes(n[l], bK[l], bV[l], H, D) for l in range(L))

        def quality(l: int) -> float:
            return (
                coverage(n[l], seq_len, sparsities[l])
                * F_BITS[bK[l]]
                * F_BITS[bV[l]]
                * importance[l]
            )

        max_iters = L * (seq_len // self.DELTA_N + 6)
        for _ in range(max_iters):
            current = used()
            if current >= budget_bytes:
                break
            best_dq_db = 0.0
            best_action = None
            for l in range(L):
                q_now = quality(l)
                # Action: add tokens
                if n[l] + self.DELTA_N <= seq_len:
                    db = (
                        kv_bytes(n[l] + self.DELTA_N, bK[l], bV[l], H, D)
                        - kv_bytes(n[l], bK[l], bV[l], H, D)
                    )
                    if db > 0 and current + db <= budget_bytes:
                        n[l] += self.DELTA_N
                        dq = quality(l) - q_now
                        n[l] -= self.DELTA_N
                        ratio = dq / db
                        if ratio > best_dq_db:
                            best_dq_db = ratio
                            best_action = ("add", l)
                # Action: upgrade K
                idx = self.BIT_OPTIONS.index(bK[l])
                if idx < len(self.BIT_OPTIONS) - 1:
                    new_bK = self.BIT_OPTIONS[idx + 1]
                    db = (
                        kv_bytes(n[l], new_bK, bV[l], H, D)
                        - kv_bytes(n[l], bK[l], bV[l], H, D)
                    )
                    if db > 0 and current + db <= budget_bytes:
                        old = bK[l]; bK[l] = new_bK
                        dq = quality(l) - q_now
                        bK[l] = old
                        ratio = dq / db
                        if ratio > best_dq_db:
                            best_dq_db = ratio
                            best_action = ("upK", l)
                # Action: upgrade V
                idx = self.BIT_OPTIONS.index(bV[l])
                if idx < len(self.BIT_OPTIONS) - 1:
                    new_bV = self.BIT_OPTIONS[idx + 1]
                    db = (
                        kv_bytes(n[l], bK[l], new_bV, H, D)
                        - kv_bytes(n[l], bK[l], bV[l], H, D)
                    )
                    if db > 0 and current + db <= budget_bytes:
                        old = bV[l]; bV[l] = new_bV
                        dq = quality(l) - q_now
                        bV[l] = old
                        ratio = dq / db
                        if ratio > best_dq_db:
                            best_dq_db = ratio
                            best_action = ("upV", l)
            if best_action is None:
                break
            kind, l = best_action
            if kind == "add":
                n[l] += self.DELTA_N
            elif kind == "upK":
                bK[l] = self.BIT_OPTIONS[self.BIT_OPTIONS.index(bK[l]) + 1]
            elif kind == "upV":
                bV[l] = self.BIT_OPTIONS[self.BIT_OPTIONS.index(bV[l]) + 1]

        return [(min(n[l], seq_len), bK[l], bV[l]) for l in range(L)]

    def compress(
        self,
        full_keys: Tensor,
        full_values: Tensor,
        compression_ratio: float,
        attention_weights: Optional[List[Tensor]] = None,
        hidden_states: Optional[List[Tensor]] = None,
    ) -> List[Tuple[Tensor, Tensor, Tensor]]:
        L, sl, H, D = full_keys.shape  # (L, sl, nh, d)

        # Sparsity from attention (Gini per layer)
        sparsities: List[float] = []
        for l in range(L):
            if attention_weights and attention_weights[l] is not None:
                a = attention_weights[l][0]  # (heads, seq, seq)
                last_row = a[:, -1, :].mean(dim=0).abs().cpu().numpy()
                last_row = last_row / max(last_row.sum(), 1e-9)
                sorted_vals = sorted(last_row)
                S = len(sorted_vals)
                cum = sum((i + 1) * v for i, v in enumerate(sorted_vals))
                gini = (
                    (2 * cum) / (S * sum(sorted_vals)) - (S + 1) / S
                    if sum(sorted_vals) > 0 else 0.5
                )
                sparsities.append(min(max(gini, 0.0), 0.99))
            else:
                sparsities.append(0.5)

        # Inverted importance (early layers get higher weight)
        importance = [sigmoid_importance(l, L, k=5.0, tau=0.3, invert=True) for l in range(L)]

        # Total memory budget in bytes (FP16 reference)
        full_bytes = 2 * L * H * D * sl * 2
        budget = full_bytes / max(compression_ratio, 1.001)

        plan = self._allocate(sparsities, importance, sl, budget)

        # Apply: token selection by H2O-style + FP16 emit (quantization is in the cost model only)
        results = []
        for l in range(L):
            n_l, b_K, b_V = plan[l]
            attn = attention_weights[l] if attention_weights else None
            indices = h2o_token_selection(
                full_keys[l:l + 1], full_values[l:l + 1], n_l, sl,
                attention_weights=attn,
                sink_tokens=self.SINK, recent_tokens=self.RECENT,
            )
            k_sub = full_keys[l:l + 1, indices.long()]
            v_sub = full_values[l:l + 1, indices.long()]
            results.append((k_sub, v_sub, indices))
        return results
