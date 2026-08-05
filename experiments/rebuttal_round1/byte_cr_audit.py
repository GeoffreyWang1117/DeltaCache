"""Byte-CR audit: recompute effective CR (= Full_KV_bytes / method_KV_bytes) per cell.

Reads existing PPL checkpoints which store mean_memory_bytes, and divides the
full-KV byte estimate by the method's actual byte usage. Produces a table
showing nominal_CR vs effective_CR for every method × cell.

Outputs:
  results/byte_cr_audit.json — flat table
  results/byte_cr_audit_summary.md — human-readable diff per method × CR
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/coder-gw/Projects/DeltaCache/experiments/results/suite")
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# FP16 KV bytes per token at full precision, per model: 2 * L * H_kv * d_h * 2 bytes
MODEL_FP16_BYTES_PER_TOKEN = {
    "Llama-2-7B": 2 * 32 * 32 * 128 * 2,
    "Llama-2-13B": 2 * 40 * 40 * 128 * 2,
    "Llama-3.1-8B": 2 * 32 * 8 * 128 * 2,
    "Mistral-7B": 2 * 32 * 8 * 128 * 2,
    "Qwen3-0.6B": 2 * 28 * 8 * 64 * 2,
    "Qwen3-1.7B": 2 * 28 * 8 * 128 * 2,
    "Qwen3-8B": 2 * 36 * 8 * 128 * 2,
    "Qwen2.5-3B": 2 * 36 * 2 * 128 * 2,
    "Qwen2.5-14B": 2 * 48 * 4 * 128 * 2,
    "Qwen2.5-72B": 2 * 80 * 8 * 128 * 2,
}


def main() -> None:
    rows = []
    for ckpt in ROOT.glob("*/checkpoints/*__ppl__*.json"):
        try:
            d = json.load(open(ckpt))
        except Exception:  # noqa: BLE001
            continue
        if d.get("task") != "ppl":
            continue
        model = d.get("model")
        if model not in MODEL_FP16_BYTES_PER_TOKEN:
            continue
        seq_len = d.get("seq_len")
        method = d.get("method")
        nominal_cr = d.get("compression_ratio")
        method_bytes = d.get("mean_memory_bytes")
        if not (seq_len and method_bytes and nominal_cr):
            continue
        full_bytes = MODEL_FP16_BYTES_PER_TOKEN[model] * seq_len
        effective_cr = full_bytes / method_bytes if method_bytes > 0 else None
        rows.append(
            {
                "model": model,
                "seq_len": seq_len,
                "method": method,
                "nominal_cr": nominal_cr,
                "effective_cr": round(effective_cr, 3) if effective_cr else None,
                "delta_cr": round(effective_cr - nominal_cr, 3) if effective_cr else None,
                "full_bytes": full_bytes,
                "method_bytes": method_bytes,
            }
        )

    out_path = OUT_DIR / "byte_cr_audit.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote {out_path} ({len(rows)} rows)")

    # Summary: median delta_cr per method × nominal_cr
    grouped: dict[tuple[str, float], list[float]] = defaultdict(list)
    for r in rows:
        if r["delta_cr"] is None:
            continue
        grouped[(r["method"], r["nominal_cr"])].append(r["delta_cr"])

    print("\n=== Median (effective_CR - nominal_CR) per method × nominal CR ===")
    print("Negative = method uses MORE memory than nominal CR suggests (LB at 6x is here).")
    print(f"{'method':<22} {'nominal_CR':>10} {'median_Δ':>10} {'min_Δ':>8} {'max_Δ':>8} {'n':>4}")
    summary_rows = []
    for (method, nominal), deltas in sorted(grouped.items()):
        deltas_sorted = sorted(deltas)
        median = deltas_sorted[len(deltas_sorted) // 2]
        summary_rows.append({"method": method, "nominal_cr": nominal, "median_delta_cr": median, "min_delta_cr": min(deltas), "max_delta_cr": max(deltas), "n": len(deltas)})
        print(f"{method:<22} {nominal:>10} {median:>+10.2f} {min(deltas):>+8.2f} {max(deltas):>+8.2f} {len(deltas):>4}")

    # Markdown summary
    md = ["# Byte-CR Audit Summary", "",
          "**Definition:** `effective_CR = Full_FP16_KV_bytes / method_actual_KV_bytes`. Δ = effective − nominal.",
          "",
          "**Reading:** negative Δ means the method uses MORE bytes than nominal CR suggests (e.g., LB at nominal 6× actually delivers ~3× memory reduction because INT4-quantized retained tokens still cost bytes).",
          "",
          "| Method | Nominal CR | Median Δ | Min Δ | Max Δ | n cells |",
          "|---|---|---|---|---|---|"]
    for r in summary_rows:
        md.append(f"| {r['method']} | {r['nominal_cr']} | {r['median_delta_cr']:+.2f} | {r['min_delta_cr']:+.2f} | {r['max_delta_cr']:+.2f} | {r['n']} |")
    (OUT_DIR / "byte_cr_audit_summary.md").write_text("\n".join(md))
    print(f"\nMarkdown summary: {OUT_DIR / 'byte_cr_audit_summary.md'}")


if __name__ == "__main__":
    main()
