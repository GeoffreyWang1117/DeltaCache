"""Generalize the 96% KL-reduction-from-mean-fill claim across models.

Existing claim is Mistral-7B only. Run exp_attention_distortion (already present in
run_layerwise_and_distortion.py) on Llama-2-7B and Llama-3.1-8B at CR=6× to check
whether the KL reduction from zero-fill → mean-fill generalizes.

Outputs results/mean_fill_kl_<timestamp>.json with per-model logits-KL for both fills.

GPU est: ~30-45 min per model on RTX 3090.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(EXP_DIR.parent))

from run_layerwise_and_distortion import exp_attention_distortion, load_model_4bit  # noqa: E402

OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("meta-llama/Llama-2-7b-chat-hf", "llama2_7b"),
    ("meta-llama/Llama-3.1-8B-Instruct", "llama3.1_8b"),
]


def main() -> None:
    overall = {"date": datetime.now().isoformat(), "cr": 6.0, "seq_len": 512, "n_texts": 2, "results": []}
    for hf_name, short in MODELS:
        print(f"\n{'#' * 70}\n# {short}\n{'#' * 70}")
        try:
            model, tokenizer = load_model_4bit(hf_name)
            t0 = time.time()
            res = exp_attention_distortion(model, tokenizer, short, seq_len=512, cr=6.0, n_texts=2)
            elapsed = time.time() - t0
            # exp_attention_distortion returns {"logits": {...}, "per_layer": [...]}
            zero_kl = res["logits"]["zero"]["kl_div"]
            mean_kl = res["logits"]["mean"]["kl_div"]
            reduction = 100 * (1 - mean_kl / zero_kl) if zero_kl else None
            overall["results"].append({
                "model": short, "hf_name": hf_name,
                "elapsed_s": round(elapsed, 1),
                "zero_logits_kl": zero_kl, "mean_logits_kl": mean_kl,
                "reduction_pct": reduction,
                "raw": res,
            })
            del model, tokenizer
            import torch
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            overall["results"].append({"model": short, "error": str(e)})

    out_path = OUT_DIR / f"mean_fill_kl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(overall, f, indent=2, default=str)
    print(f"\nWrote {out_path}")

    # Compare to Mistral-7B paper claim of 96% reduction
    md_lines = ["# Mean-Fill KL Generalization", "",
                "Mistral-7B paper claim: zero-fill logits-KL = 0.249, mean-fill = 0.009 (96% reduction).",
                "This script checks whether the reduction generalizes.",
                "",
                "| Model | Zero-fill KL | Mean-fill KL | Reduction |",
                "|---|---|---|---|"]
    for r in overall["results"]:
        if "error" in r:
            md_lines.append(f"| {r['model']} | FAILED | | {r['error'][:60]} |")
            continue
        z = r["zero_logits_kl"]; m = r["mean_logits_kl"]; red = r["reduction_pct"]
        z_s = f"{z:.4f}" if z else "—"
        m_s = f"{m:.4f}" if m else "—"
        red_s = f"{red:.1f}%" if red else "—"
        md_lines.append(f"| {r['model']} | {z_s} | {m_s} | **{red_s}** |")
    md_lines += ["", "**Outcome interpretation:** if reduction ≥80% on both models, the mean-fill claim is generalizable — abstract can stay at '96% on Mistral, ≥80% across other tested models'. If reduction <50% on either, scope down."]
    (OUT_DIR / "mean_fill_kl_summary.md").write_text("\n".join(md_lines))


if __name__ == "__main__":
    main()
