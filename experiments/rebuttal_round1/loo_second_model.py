"""Leave-one-out importance on a SECOND model family.

The paper's "early layers are the bottleneck" claim is from Mistral-7B (32L GQA).
This script reuses experiments/run_layerwise_and_distortion.py:exp_layerwise_loo
and runs it on Llama-2-13B (40L MHA) and Qwen2.5-14B (48L GQA) at CR=6×.

Outcome decides whether the inverted-importance principle generalizes.

Outputs results/loo_<model>_<timestamp>.json with per-layer LOO improvement.
A summary md ranks layers by LOO improvement and bins them into early / mid / late thirds.

GPU est: ~1.5h per model on A100. Llama-2-13B is ~25 min/model on A100; 14B ~40 min.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Pull in the existing implementation
EXP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(EXP_DIR.parent))

from run_layerwise_and_distortion import exp_layerwise_loo, load_model_4bit  # noqa: E402

OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("meta-llama/Llama-2-13b-chat-hf", "llama2_13b", 40, "MHA"),
    ("Qwen/Qwen2.5-14B-Instruct", "qwen2.5_14b", 48, "GQA"),
]


def bin_layers(layer_summary: list[dict], n_layers: int) -> dict:
    """Split into 3 bins (early / mid / late thirds) and report mean LOO improvement."""
    third = n_layers // 3
    early = [l for l in layer_summary if l["layer"] < third]
    mid = [l for l in layer_summary if third <= l["layer"] < 2 * third]
    late = [l for l in layer_summary if l["layer"] >= 2 * third]
    return {
        "early_layers": (0, third - 1),
        "mid_layers": (third, 2 * third - 1),
        "late_layers": (2 * third, n_layers - 1),
        "early_loo_mean": round(sum(l["loo_improvement"] for l in early) / max(len(early), 1), 4),
        "mid_loo_mean": round(sum(l["loo_improvement"] for l in mid) / max(len(mid), 1), 4),
        "late_loo_mean": round(sum(l["loo_improvement"] for l in late) / max(len(late), 1), 4),
        "early_loo_max": round(max(l["loo_improvement"] for l in early), 4),
        "late_loo_max": round(max(l["loo_improvement"] for l in late), 4),
    }


def main() -> None:
    overall = {"date": datetime.now().isoformat(), "results": []}
    for hf_name, short, n_layers, arch in MODELS:
        print(f"\n{'#' * 70}\n# {short} ({arch}, {n_layers} layers)\n{'#' * 70}")
        try:
            model, tokenizer = load_model_4bit(hf_name)
            t0 = time.time()
            layer_summary = exp_layerwise_loo(model, tokenizer, short, seq_len=512, cr=6.0, n_texts=2)
            elapsed = time.time() - t0
            bins = bin_layers(layer_summary, n_layers)
            verdict = "early-bottleneck" if bins["early_loo_mean"] > bins["late_loo_mean"] * 1.5 else (
                "late-bottleneck" if bins["late_loo_mean"] > bins["early_loo_mean"] * 1.5 else "indeterminate"
            )
            print(f"\n  early_LOO={bins['early_loo_mean']:+.4f}  late_LOO={bins['late_loo_mean']:+.4f}  -> {verdict}")
            overall["results"].append({
                "model": short, "hf_name": hf_name, "n_layers": n_layers, "arch": arch,
                "elapsed_s": round(elapsed, 1), "verdict": verdict, "bins": bins,
                "layers": layer_summary,
            })
            del model, tokenizer
            import torch
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            overall["results"].append({"model": short, "error": str(e)})

    out_path = OUT_DIR / f"loo_second_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(overall, f, indent=2)
    print(f"\nWrote {out_path}")

    # Markdown summary
    md_lines = ["# LOO Second-Model Findings", "",
                "Goal: confirm or reject the 'early layers are the eviction bottleneck' principle on architectures other than Mistral-7B.",
                "",
                "| Model | Arch | L | Early-LOO mean | Late-LOO mean | Verdict |",
                "|---|---|---|---|---|---|"]
    for r in overall["results"]:
        if "bins" not in r:
            md_lines.append(f"| {r['model']} | — | — | FAILED | — | {r.get('error','?')[:60]} |")
            continue
        md_lines.append(f"| {r['model']} | {r['arch']} | {r['n_layers']} | {r['bins']['early_loo_mean']:+.4f} | {r['bins']['late_loo_mean']:+.4f} | **{r['verdict']}** |")
    md_lines += ["",
                 "**Reading:** higher LOO improvement at early-bin layers ⇒ restoring an early layer recovers more PPL ⇒ early layers are the eviction bottleneck (matches Mistral-7B finding).",
                 "If both models show 'early-bottleneck', the inverted-importance principle generalizes; if one or both show 'late-bottleneck' or 'indeterminate', the principle is Mistral-specific and the paper should scope the claim down."]
    (OUT_DIR / "loo_second_model_summary.md").write_text("\n".join(md_lines))


if __name__ == "__main__":
    main()
