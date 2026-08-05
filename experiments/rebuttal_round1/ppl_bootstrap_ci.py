"""Compute approximate 95% CIs on PPL and PPL-ratio cells.

The existing checkpoint JSONs store (mean_ppl, std_ppl, n_chunks=6) but not
per-chunk PPLs, so true bootstrap is not possible. We use parametric
t-distribution CIs (n=6, df=5, t_0.975 = 2.571) on the mean PPL, and
delta-method CIs on the PPL ratio = compressed_PPL / full_PPL.

For each (model, seq_len) and each method × CR cell, we report:
  ppl_ratio, ratio_ci_low, ratio_ci_high, significant_vs_baseline

A method is "significantly better than" another if their ratio CIs do not overlap.

Reads: experiments/results/suite/*/checkpoints/*.json
Writes: rebuttal_round1/results/ppl_ci.json + per-table markdown summary
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

T_975_DF5 = 2.571  # t_{0.975, df=5}

ROOT = Path("/home/coder-gw/Projects/DeltaCache/experiments/results/suite")
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_checkpoint(path: Path) -> dict | None:
    try:
        d = json.load(open(path))
    except Exception:  # noqa: BLE001
        return None
    if d.get("task") != "ppl":
        return None
    needed = {"model", "method", "compression_ratio", "seq_len", "mean_ppl", "std_ppl", "n_chunks", "ppl_ratio", "full_ppl"}
    if not needed.issubset(d):
        return None
    return d


def ratio_ci(mean_c: float, std_c: float, n_c: int, mean_f: float, std_f: float, n_f: int) -> tuple[float, float]:
    """Delta-method 95% CI for ratio = mean_c / mean_f."""
    se_c = std_c / math.sqrt(max(n_c, 1))
    se_f = std_f / math.sqrt(max(n_f, 1))
    if mean_f <= 0:
        return (float("nan"), float("nan"))
    ratio = mean_c / mean_f
    var_ratio = (ratio**2) * ((se_c / max(mean_c, 1e-9)) ** 2 + (se_f / max(mean_f, 1e-9)) ** 2)
    se_ratio = math.sqrt(max(var_ratio, 0.0))
    return (ratio - T_975_DF5 * se_ratio, ratio + T_975_DF5 * se_ratio)


def main() -> None:
    cells = []
    for ckpt in ROOT.glob("*/checkpoints/*__ppl__*.json"):
        d = parse_checkpoint(ckpt)
        if d is None:
            continue
        # Find matching full-KV (baseline) for the same model + seq_len
        cells.append(d)

    print(f"Parsed {len(cells)} PPL checkpoints")

    # Group by (model, seq_len) and compute CIs against the same model's full_ppl
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for c in cells:
        grouped[(c["model"], c["seq_len"])].append(c)

    rows = []
    for (model, seq_len), entries in sorted(grouped.items()):
        # full_ppl should be roughly identical across entries; take median
        fulls = [e["full_ppl"] for e in entries if e.get("full_ppl") is not None]
        if not fulls:
            continue
        full_med = sorted(fulls)[len(fulls) // 2]
        # We don't have std_full directly; approximate as std of full_ppl across entries
        # if multiple entries have it (they usually all share the same baseline measurement).
        # Conservative: assume std_full ≈ std_ppl (similar variance)
        for e in entries:
            n = e.get("n_chunks") or 6
            std = e.get("std_ppl")
            mean_c = e.get("mean_ppl")
            if std is None or mean_c is None or e.get("ppl_ratio") is None:
                continue
            std_full_proxy = std  # conservative proxy
            lo, hi = ratio_ci(mean_c, std, n, full_med, std_full_proxy, n)
            rows.append(
                {
                    "model": model,
                    "seq_len": seq_len,
                    "method": e["method"],
                    "cr": e["compression_ratio"],
                    "ppl_ratio": round(e["ppl_ratio"], 4),
                    "ratio_ci_low": round(lo, 4),
                    "ratio_ci_high": round(hi, 4),
                    "ratio_ci_halfwidth": round(0.5 * (hi - lo), 4),
                    "n_chunks": n,
                    "mean_ppl": round(mean_c, 3),
                    "std_ppl": round(std, 3),
                    "full_ppl": round(full_med, 3),
                }
            )

    out_path = OUT_DIR / "ppl_ci.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote {out_path} ({len(rows)} rows)")

    # Print a sample of the most interesting cells: layer_budget vs kivi_uniform at 2x/4x
    print("\n=== Sample: LayerBudget vs KIVI vs H2O at CR=2.0 / 4.0 ===")
    print(f"{'model':<14} {'seq':>5} {'method':<18} {'CR':>4} {'ratio':>7} {'CI':<18}")
    interesting_methods = {"layer_budget", "kivi_uniform", "h2o", "snapkv", "streaming_llm"}
    for r in rows:
        if r["method"] not in interesting_methods or r["cr"] not in (2.0, 4.0):
            continue
        ci = f"[{r['ratio_ci_low']:.3f},{r['ratio_ci_high']:.3f}]"
        print(f"{r['model']:<14} {r['seq_len']:>5} {r['method']:<18} {r['cr']:>4} {r['ppl_ratio']:>7.3f} {ci:<18}")

    # Significance check: layer_budget vs kivi at 2x and 4x — do CIs overlap?
    print("\n=== LB-vs-KIVI overlap check (CIs overlap ⇒ NOT significantly distinguishable) ===")
    by_key = {(r["model"], r["seq_len"], r["method"], r["cr"]): r for r in rows}
    pairs_checked = 0
    pairs_overlap = 0
    for (m, s, meth, cr), lb in by_key.items():
        if meth != "layer_budget":
            continue
        kivi = by_key.get((m, s, "kivi_uniform", cr))
        if kivi is None:
            continue
        overlap = not (lb["ratio_ci_high"] < kivi["ratio_ci_low"] or kivi["ratio_ci_high"] < lb["ratio_ci_low"])
        pairs_checked += 1
        if overlap:
            pairs_overlap += 1
        marker = "OVERLAP" if overlap else "DISTINCT"
        print(f"  {m} S={s} CR={cr}: LB {lb['ppl_ratio']:.3f} {[lb['ratio_ci_low'],lb['ratio_ci_high']]}  vs KIVI {kivi['ppl_ratio']:.3f} {[kivi['ratio_ci_low'],kivi['ratio_ci_high']]}  -> {marker}")
    if pairs_checked:
        print(f"\n{pairs_overlap}/{pairs_checked} LB-vs-KIVI cells have overlapping 95% CIs (not statistically distinguishable).")


if __name__ == "__main__":
    main()
