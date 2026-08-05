"""Head-to-head: MoE-nD-style baseline vs LayerBudget on Mistral-7B / Llama-2-7B.

Evaluates 4 methods on WikiText-2 PPL across CR ∈ {2, 3, 4, 6}:
  - full_kv (sanity)
  - kivi_uniform (quant-only floor)
  - moend_perlayer (concurrent competitor — our faithful reimpl)
  - layer_budget (ours)

Goal: defensible head-to-head against MoE-nD's per-layer joint framing.

Outputs results/moend_vs_lb_<timestamp>.json.

GPU est: ~25 min on RTX 3090 (4 methods × 4 CRs × 2 models × 6 chunks).
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

DELTACACHE_ROOT = Path("/home/coder-gw/Projects/DeltaCache")
sys.path.insert(0, str(DELTACACHE_ROOT))
sys.path.insert(0, str(DELTACACHE_ROOT / "experiments"))

import math  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# Register the MoE-nD baseline before importing eval_utils
import moend_baseline  # noqa: F401, E402
from baselines.base import REGISTRY  # noqa: E402

print(f"Methods registered: {sorted(REGISTRY.keys())}")
assert "moend_perlayer" in REGISTRY, "MoE-nD baseline not registered"

from suite.eval_utils import compress_kv, extract_kv_and_attention, build_hf_cache  # noqa: E402
from deltacache.core.layer_profiler import LayerAttentionProfiler  # noqa: E402
from deltacache.core.layer_budget_allocator import LayerBudgetAllocator  # noqa: E402

OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("mistralai/Mistral-7B-Instruct-v0.2", "mistral_7b", 32, 8, 128),
    ("meta-llama/Llama-2-7b-chat-hf", "llama2_7b", 32, 32, 128),
]
CRS = [2.0, 3.0, 4.0, 6.0]
METHODS = ["full_kv", "kivi_uniform", "moend_perlayer", "layer_budget"]
SEQ_LEN = 1024
N_CHUNKS = 6
PREFIX_RATIO = 0.6


def get_wikitext_chunks(tokenizer, target_len: int, n: int) -> list:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    all_text = " ".join([t for t in ds["text"] if len(t.strip()) > 100])
    all_ids = tokenizer.encode(all_text)
    return [all_ids[i:i + target_len] for i in range(0, len(all_ids) - target_len, target_len)][:n]


def compute_ppl(model, input_ids: torch.Tensor, past_kv, prefix_len: int, device) -> float:
    suffix = input_ids[:, prefix_len:]
    if suffix.shape[1] <= 1:
        return 1.0
    with torch.no_grad():
        pos = torch.arange(prefix_len, prefix_len + suffix.shape[1], device=device).unsqueeze(0)
        out = model(input_ids=suffix, past_key_values=past_kv, position_ids=pos, return_dict=True)
    logits = out.logits[:, :-1, :].contiguous()
    labels = suffix[:, 1:].contiguous()
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction="mean")
    return math.exp(min(loss.item(), 20))


def load_model(hf_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    print(f"Loading {hf_name} ...")
    tok = AutoTokenizer.from_pretrained(hf_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        hf_name, quantization_config=bnb, device_map={"": "cuda:0"},
        attn_implementation="eager", trust_remote_code=True,
    )
    model.eval()
    return model, tok


def run_one_model(hf_name: str, short: str, nl: int, nh: int, hd: int) -> dict:
    print(f"\n{'#' * 70}\n# {short}\n{'#' * 70}")
    model, tok = load_model(hf_name)
    device = next(model.parameters()).device

    chunks = get_wikitext_chunks(tok, SEQ_LEN, N_CHUNKS)
    prefix_len = int(SEQ_LEN * PREFIX_RATIO)

    results: dict = {"model": short, "seq_len": SEQ_LEN, "n_chunks": len(chunks), "prefix_len": prefix_len, "cells": []}

    # Pre-compute full KV + attention for each chunk (heavy)
    chunk_data = []
    for idx, tokens in enumerate(chunks):
        input_ids = torch.tensor([tokens], device=device)
        with torch.no_grad():
            out = model(input_ids=input_ids[:, :prefix_len], output_attentions=True, return_dict=True)
        full_k = torch.stack([kv[0].squeeze(0).transpose(0, 1) for kv in out.past_key_values])
        full_v = torch.stack([kv[1].squeeze(0).transpose(0, 1) for kv in out.past_key_values])
        attns = list(out.attentions)
        # Reference PPL with full KV
        full_kv_hf = []
        for l in range(nl):
            k = full_k[l].unsqueeze(0).transpose(1, 2)
            v = full_v[l].unsqueeze(0).transpose(1, 2)
            full_kv_hf.append((k, v))
        from transformers.cache_utils import DynamicCache
        cache = DynamicCache()
        for l, (k, v) in enumerate(full_kv_hf):
            cache.update(k, v, l)
        ref_ppl = compute_ppl(model, input_ids, cache, prefix_len, device)
        chunk_data.append((input_ids, full_k, full_v, attns, ref_ppl))
        print(f"  chunk {idx+1}/{len(chunks)}: ref_ppl={ref_ppl:.2f}", flush=True)
        del out, cache
        torch.cuda.empty_cache()

    # Per-method per-CR PPL
    for cr in CRS:
        cr_row = {"cr": cr, "methods": {}}
        for method in METHODS:
            if method == "full_kv" and cr != 2.0:
                continue  # only run full_kv once
            ppls = []
            for (input_ids, full_k, full_v, attns, ref_ppl) in chunk_data:
                try:
                    if method == "full_kv":
                        ratio = 1.0
                    else:
                        # LayerBudget needs gini_scores + importance_weights
                        gini = importance = None
                        if method == "layer_budget":
                            profiler = LayerAttentionProfiler()
                            pr = profiler.profile_from_attention_weights(attns)
                            gini = pr.gini_scores()  # Dict[int, float]
                            importance = LayerBudgetAllocator.compute_importance_weights(nl)
                        layers, _, _ = compress_kv(
                            method, full_k, full_v, cr, nl, nh, hd,
                            attention_weights=attns,
                            gini_scores=gini,
                            importance_weights=importance,
                        )
                        cache = build_hf_cache(layers, prefix_len, device, fill="mean")
                        ppl = compute_ppl(model, input_ids, cache, prefix_len, device)
                        ratio = ppl / max(ref_ppl, 1e-6)
                        del cache
                        torch.cuda.empty_cache()
                    ppls.append(ratio)
                except Exception as e:  # noqa: BLE001
                    print(f"  [{method} CR={cr}] FAILED: {type(e).__name__}: {e}")
            if ppls:
                mean = sum(ppls) / len(ppls)
                cr_row["methods"][method] = {
                    "ppl_ratio_mean": round(mean, 4),
                    "ppl_ratio_per_chunk": [round(r, 4) for r in ppls],
                }
                print(f"  CR={cr} {method}: ppl_ratio={mean:.4f}", flush=True)
        results["cells"].append(cr_row)

    del model, tok
    torch.cuda.empty_cache()
    return results


def main() -> None:
    overall = {"date": datetime.now().isoformat(), "results": []}
    for hf_name, short, nl, nh, hd in MODELS:
        try:
            overall["results"].append(run_one_model(hf_name, short, nl, nh, hd))
        except Exception as e:  # noqa: BLE001
            print(f"FAILED {short}: {e}")
            import traceback; traceback.print_exc()
            overall["results"].append({"model": short, "error": str(e)})
    out_path = OUT_DIR / f"moend_vs_lb_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(overall, f, indent=2)
    print(f"\nWrote {out_path}")

    # Markdown summary
    md = ["# MoE-nD vs LayerBudget — head-to-head", "",
          f"WikiText-2 PPL ratio (compressed/full), seq_len={SEQ_LEN}, prefix={PREFIX_RATIO*100:.0f}%, n={N_CHUNKS} chunks", ""]
    for r in overall["results"]:
        if "error" in r:
            md.append(f"## {r['model']}: FAILED — {r['error']}"); continue
        md.append(f"## {r['model']}")
        md.append("")
        md.append("| CR | full_kv | kivi_uniform | moend_perlayer | layer_budget |")
        md.append("|---|---|---|---|---|")
        full_ppl = None
        for cell in r["cells"]:
            if "full_kv" in cell["methods"]:
                full_ppl = cell["methods"]["full_kv"]["ppl_ratio_mean"]
        for cell in r["cells"]:
            row = f"| {cell['cr']:.0f}× | {full_ppl if full_ppl else '—'} "
            for m in ["kivi_uniform", "moend_perlayer", "layer_budget"]:
                v = cell["methods"].get(m, {}).get("ppl_ratio_mean", "—")
                row += f"| {v} "
            row += "|"
            md.append(row)
        md.append("")
    md.append("## Interpretation")
    md.append("- **moend_perlayer** is our faithful re-implementation of MoE-nD's per-layer (n_l, b_K_l, b_V_l) joint allocation framing (Sun et al. 2026).")
    md.append("- **layer_budget** uses shared b_l with inverted importance + closed-form coverage law.")
    md.append("- A close result on Mistral-7B suggests both joint framings are essentially equivalent at moderate CR; a divergence indicates the importance signal or the K/V split matters at the scale tested.")
    (OUT_DIR / "moend_vs_lb_summary.md").write_text("\n".join(md))
    print(f"Wrote summary {OUT_DIR / 'moend_vs_lb_summary.md'}")


if __name__ == "__main__":
    main()
