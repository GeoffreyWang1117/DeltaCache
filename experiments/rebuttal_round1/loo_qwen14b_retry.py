"""Retry LOO on Qwen2.5-14B alone after the chained run OOM'd."""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(EXP_DIR.parent))

import torch  # noqa: E402

# Set fragmentation-friendly allocator
import os  # noqa: E402
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from run_layerwise_and_distortion import exp_layerwise_loo, load_model_4bit  # noqa: E402

OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def bin_layers(layer_summary: list[dict], n_layers: int) -> dict:
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
    hf_name, short, n_layers, arch = "Qwen/Qwen2.5-14B-Instruct", "qwen2.5_14b", 48, "GQA"
    print(f"\n{'#' * 70}\n# {short} ({arch}, {n_layers} layers) — retry\n{'#' * 70}")
    free_b, total_b = torch.cuda.mem_get_info()
    print(f"GPU free at start: {free_b/1e9:.1f} GB / {total_b/1e9:.1f} GB total")

    model, tokenizer = load_model_4bit(hf_name)
    free_b, total_b = torch.cuda.mem_get_info()
    print(f"GPU free after model load: {free_b/1e9:.1f} GB")

    t0 = time.time()
    layer_summary = exp_layerwise_loo(model, tokenizer, short, seq_len=512, cr=6.0, n_texts=2)
    elapsed = time.time() - t0
    bins = bin_layers(layer_summary, n_layers)
    verdict = "early-bottleneck" if bins["early_loo_mean"] > bins["late_loo_mean"] * 1.5 else (
        "late-bottleneck" if bins["late_loo_mean"] > bins["early_loo_mean"] * 1.5 else "indeterminate"
    )
    print(f"\n  early_LOO={bins['early_loo_mean']:+.4f}  late_LOO={bins['late_loo_mean']:+.4f}  -> {verdict}")

    out = {"model": short, "hf_name": hf_name, "n_layers": n_layers, "arch": arch,
           "elapsed_s": round(elapsed, 1), "verdict": verdict, "bins": bins, "layers": layer_summary}
    out_path = OUT_DIR / f"loo_qwen14b_retry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
