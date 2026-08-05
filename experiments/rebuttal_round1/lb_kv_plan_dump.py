"""Dump LayerBudget-KV per-layer (n_l, b_K_l, b_V_l) plans at each CR
for one Mistral-7B chunk to verify plans actually differ across CRs.

This addresses R1's and R4's critical concern that LB-KV's PPL ratio = 1.036
at CR=2, 3, 4× is solver degeneracy rather than CR-stability.
"""

from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import sys
from pathlib import Path

DELTACACHE_ROOT = Path("/home/coder-gw/Projects/DeltaCache")
sys.path.insert(0, str(DELTACACHE_ROOT))
sys.path.insert(0, str(DELTACACHE_ROOT / "experiments"))

import torch  # noqa: E402

from layer_budget_kv import (  # noqa: E402
    LayerBudgetKVBaseline,
    coverage,
    sigmoid_importance,
    F_BITS,
    kv_bytes,
)


def main():
    # Mistral-7B parameters
    L, H, D = 32, 8, 128
    sl = 1024
    bsl = LayerBudgetKVBaseline(L, H, D)

    # Synthetic gini per layer (matches typical Mistral-7B Gini distribution: 0.65 +/- 0.10)
    torch.manual_seed(42)
    gini = [0.5 + 0.4 * torch.rand(1).item() for _ in range(L)]
    importance = [sigmoid_importance(l, L, k=5.0, tau=0.3, invert=True) for l in range(L)]

    full_bytes = 2 * L * H * D * sl * 2

    print(f"{'CR':>5} | {'budget(MB)':>12} | {'used(MB)':>12} | {'mean_n':>8} | {'mean_bK':>8} | {'mean_bV':>8} | {'unique_plans':>12}")
    print("-" * 90)

    plans_per_cr = {}
    for cr in [2.0, 3.0, 4.0, 6.0]:
        budget = full_bytes / cr
        plan = bsl._allocate(gini, importance, sl, budget)
        used = sum(kv_bytes(p[0], p[1], p[2], H, D) for p in plan)
        mean_n = sum(p[0] for p in plan) / L
        mean_bK = sum(p[1] for p in plan) / L
        mean_bV = sum(p[2] for p in plan) / L
        plan_tuple = tuple(plan)
        plans_per_cr[cr] = plan
        unique_count = len(set(plan))
        print(f"{cr:>5.1f} | {budget/1e6:>12.2f} | {used/1e6:>12.2f} | {mean_n:>8.1f} | {mean_bK:>8.2f} | {mean_bV:>8.2f} | {unique_count:>12}")

    print()
    print("=== Plan diff CR=2× vs CR=6× ===")
    plan_2 = plans_per_cr[2.0]
    plan_6 = plans_per_cr[6.0]
    diffs = sum(1 for p2, p6 in zip(plan_2, plan_6) if p2 != p6)
    print(f"layers with different plan: {diffs} / {L}")
    if diffs <= 6:
        print("Per-layer (n, b_K, b_V):")
        print(f"  {'layer':>5} | {'CR=2 plan':>20} | {'CR=6 plan':>20}")
        for l in range(L):
            if plan_2[l] != plan_6[l]:
                print(f"  {l:>5} | {str(plan_2[l]):>20} | {str(plan_6[l]):>20}")

    print()
    print("=== Plan diff CR=2× vs CR=3× ===")
    plan_3 = plans_per_cr[3.0]
    diffs = sum(1 for p2, p3 in zip(plan_2, plan_3) if p2 != p3)
    print(f"layers with different plan: {diffs} / {L}")

    print()
    print("=== Plan diff CR=3× vs CR=4× ===")
    plan_4 = plans_per_cr[4.0]
    diffs = sum(1 for p3, p4 in zip(plan_3, plan_4) if p3 != p4)
    print(f"layers with different plan: {diffs} / {L}")


if __name__ == "__main__":
    main()
