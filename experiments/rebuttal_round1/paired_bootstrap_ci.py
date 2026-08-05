"""Paired-bootstrap CIs on per-chunk PPL ratio.

Operates on the per-chunk JSONs produced by moend_vs_lb*.py.
For each (model, CR) cell we resample chunk indices with replacement and
compute the per-chunk paired difference between method A and method B.
Reports a 95% CI for ppl_ratio[A] - ppl_ratio[B].

This unblocks the "matches KIVI" claim by showing whether small gaps (e.g.
0.9pp at CR=4×) are within sampling noise.

Outputs results/paired_bootstrap_ci.json + results/paired_bootstrap_ci.md.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
OUT_JSON = RESULTS_DIR / "paired_bootstrap_ci.json"
OUT_MD = RESULTS_DIR / "paired_bootstrap_ci.md"

INPUT_FILES = [
    "moend_vs_lb_20260503_020247.json",  # Mistral-7B (Llama-2-7B was OOM)
]
# Picks up llama2_7b re-run + xKV head-to-head
LATE_GLOBS = ["moend_vs_lb_llama2_*.json", "xkv_vs_lb_*.json"]

N_BOOT = 10000
SEED = 42


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


def paired_bootstrap(a: list[float], b: list[float], n_boot: int, seed: int) -> dict:
    assert len(a) == len(b) and len(a) >= 2
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    a_means = []
    b_means = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        a_resamp = [a[i] for i in idx]
        b_resamp = [b[i] for i in idx]
        a_mean = sum(a_resamp) / n
        b_mean = sum(b_resamp) / n
        a_means.append(a_mean)
        b_means.append(b_mean)
        diffs.append(a_mean - b_mean)
    return {
        "diff_mean": round(sum(diffs) / n_boot, 5),
        "diff_ci_low": round(percentile(diffs, 0.025), 5),
        "diff_ci_high": round(percentile(diffs, 0.975), 5),
        "frac_diff_negative": round(sum(1 for d in diffs if d < 0) / n_boot, 4),
        "a_ci_low": round(percentile(a_means, 0.025), 5),
        "a_ci_high": round(percentile(a_means, 0.975), 5),
        "b_ci_low": round(percentile(b_means, 0.025), 5),
        "b_ci_high": round(percentile(b_means, 0.975), 5),
    }


def load_files() -> list:
    out = []
    for name in INPUT_FILES:
        p = RESULTS_DIR / name
        if p.exists():
            out.append(p)
    for g in LATE_GLOBS:
        out.extend(sorted(RESULTS_DIR.glob(g)))
    return out


PAIRS = [
    ("layer_budget", "kivi_uniform"),
    ("layer_budget", "moend_perlayer"),
    ("layer_budget", "full_kv"),
    ("layer_budget", "xkv_single"),
    ("layer_budget", "xkv_g2"),
    ("moend_perlayer", "kivi_uniform"),
    ("xkv_single", "kivi_uniform"),
    ("xkv_g2", "xkv_single"),
]


def main() -> None:
    files = load_files()
    print(f"Inputs: {[str(f.name) for f in files]}")
    overall = {"n_boot": N_BOOT, "seed": SEED, "rows": []}
    for fp in files:
        with open(fp) as f:
            data = json.load(f)
        for r in data["results"]:
            if "error" in r or "cells" not in r:
                continue
            model = r["model"]
            for cell in r["cells"]:
                cr = cell["cr"]
                methods = cell["methods"]
                for a, b in PAIRS:
                    if a not in methods or b not in methods:
                        continue
                    a_chunks = methods[a]["ppl_ratio_per_chunk"]
                    b_chunks = methods[b]["ppl_ratio_per_chunk"]
                    if len(a_chunks) != len(b_chunks):
                        continue
                    boot = paired_bootstrap(a_chunks, b_chunks, N_BOOT, SEED)
                    row = {
                        "model": model,
                        "cr": cr,
                        "a": a,
                        "b": b,
                        "n_chunks": len(a_chunks),
                        "a_mean": round(sum(a_chunks) / len(a_chunks), 5),
                        "b_mean": round(sum(b_chunks) / len(b_chunks), 5),
                        **boot,
                        "significant_a_better_at_95": bool(boot["diff_ci_high"] < 0),
                        "significant_a_worse_at_95": bool(boot["diff_ci_low"] > 0),
                    }
                    overall["rows"].append(row)

    with open(OUT_JSON, "w") as f:
        json.dump(overall, f, indent=2)

    md = ["# Paired-bootstrap 95% CI on PPL-ratio differences", "",
          f"`n_boot={N_BOOT}`, `seed={SEED}`. CI is for `mean(ppl_ratio[A] - ppl_ratio[B])`",
          "across paired chunks. Negative = A better than B.", ""]
    by_model = {}
    for row in overall["rows"]:
        by_model.setdefault(row["model"], []).append(row)
    for model, rows in by_model.items():
        md.append(f"## {model}")
        md.append("")
        md.append("| CR | A | B | mean(A) | mean(B) | Δ (A−B) | 95% CI | Sig A better? |")
        md.append("|---|---|---|---|---|---|---|---|")
        for r in rows:
            ci = f"[{r['diff_ci_low']:+.4f}, {r['diff_ci_high']:+.4f}]"
            sig = "Yes" if r["significant_a_better_at_95"] else (
                "No (A worse)" if r["significant_a_worse_at_95"] else "No (overlap)"
            )
            md.append(
                f"| {r['cr']:.0f}× | {r['a']} | {r['b']} | {r['a_mean']:.4f} | {r['b_mean']:.4f} | "
                f"{r['diff_mean']:+.4f} | {ci} | {sig} |"
            )
        md.append("")

    OUT_MD.write_text("\n".join(md))
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
