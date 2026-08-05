"""LayerBudget-KV (extension) vs LayerBudget vs MoE-nD vs KIVI head-to-head.

Tests whether the (n_l, b_K_l, b_V_l) extension closes the 21pp gap at CR=6×
on Mistral-7B without sacrificing the moderate-CR wins.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

DELTACACHE_ROOT = Path("/home/coder-gw/Projects/DeltaCache")
sys.path.insert(0, str(DELTACACHE_ROOT))
sys.path.insert(0, str(DELTACACHE_ROOT / "experiments"))

import moend_baseline  # noqa: F401, E402
import layer_budget_kv  # noqa: F401, E402
from baselines.base import REGISTRY  # noqa: E402

assert "moend_perlayer" in REGISTRY
assert "layer_budget_kv" in REGISTRY

from suite.eval_utils import compress_kv, build_hf_cache  # noqa: E402
from deltacache.core.layer_profiler import LayerAttentionProfiler  # noqa: E402
from deltacache.core.layer_budget_allocator import LayerBudgetAllocator  # noqa: E402

OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("mistralai/Mistral-7B-Instruct-v0.2", "mistral_7b", 32, 8, 128),
    ("meta-llama/Llama-2-7b-chat-hf", "llama2_7b", 32, 32, 128),
]
CRS = [2.0, 3.0, 4.0, 6.0]
METHODS = ["full_kv", "kivi_uniform", "moend_perlayer", "layer_budget", "layer_budget_kv"]
SEQ_LEN = 1024
N_CHUNKS = 6
PREFIX_RATIO = 0.6


def get_wikitext_chunks(tokenizer, target_len, n):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = " ".join([t for t in ds["text"] if len(t.strip()) > 100])
    ids = tokenizer.encode(text)
    return [ids[i:i + target_len] for i in range(0, len(ids) - target_len, target_len)][:n]


def compute_ppl(model, input_ids, past_kv, prefix_len, device):
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


def run_one(hf_name, short, NL, NH, HD):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from transformers.cache_utils import DynamicCache

    print(f"\n=== {short} ===", flush=True)
    tok = AutoTokenizer.from_pretrained(hf_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        hf_name, quantization_config=bnb, device_map={"": "cuda:0"},
        attn_implementation="eager", trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device
    chunks = get_wikitext_chunks(tok, SEQ_LEN, N_CHUNKS)
    prefix_len = int(SEQ_LEN * PREFIX_RATIO)

    chunk_data = []
    for idx, tokens in enumerate(chunks):
        input_ids = torch.tensor([tokens], device=device)
        with torch.no_grad():
            out = model(input_ids=input_ids[:, :prefix_len], output_attentions=True, return_dict=True)
        full_k = torch.stack([kv[0].squeeze(0).transpose(0, 1) for kv in out.past_key_values])
        full_v = torch.stack([kv[1].squeeze(0).transpose(0, 1) for kv in out.past_key_values])
        attns = list(out.attentions)
        cache = DynamicCache()
        for l in range(NL):
            cache.update(
                full_k[l].unsqueeze(0).transpose(1, 2),
                full_v[l].unsqueeze(0).transpose(1, 2),
                l,
            )
        ref_ppl = compute_ppl(model, input_ids, cache, prefix_len, device)
        chunk_data.append((input_ids, full_k, full_v, attns, ref_ppl))
        print(f"  chunk {idx+1}/{len(chunks)}: ref={ref_ppl:.2f}", flush=True)
        del out, cache
        torch.cuda.empty_cache()

    results = {"model": short, "seq_len": SEQ_LEN, "n_chunks": len(chunks), "prefix_len": prefix_len, "cells": []}
    for cr in CRS:
        cr_row = {"cr": cr, "methods": {}}
        for method in METHODS:
            if method == "full_kv" and cr != 2.0:
                continue
            ppls = []
            for (input_ids, full_k, full_v, attns, ref_ppl) in chunk_data:
                try:
                    if method == "full_kv":
                        ratio = 1.0
                    else:
                        gini = importance = None
                        if method == "layer_budget":
                            profiler = LayerAttentionProfiler()
                            pr = profiler.profile_from_attention_weights(attns)
                            gini = pr.gini_scores()
                            importance = LayerBudgetAllocator.compute_importance_weights(NL)
                        layers, _, _ = compress_kv(
                            method, full_k, full_v, cr, NL, NH, HD,
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
                except Exception as e:
                    print(f"  [{method} CR={cr}] FAILED: {type(e).__name__}: {e}", flush=True)
            if ppls:
                m = sum(ppls) / len(ppls)
                cr_row["methods"][method] = {
                    "ppl_ratio_mean": round(m, 4),
                    "ppl_ratio_per_chunk": [round(r, 4) for r in ppls],
                }
                print(f"  CR={cr} {method}: ppl={m:.4f}", flush=True)
        results["cells"].append(cr_row)

    del model, tok
    torch.cuda.empty_cache()
    return results


def main():
    overall = {"date": datetime.now().isoformat(), "results": []}
    for hf_name, short, nl, nh, hd in MODELS:
        try:
            overall["results"].append(run_one(hf_name, short, nl, nh, hd))
        except Exception as e:
            print(f"FAILED {short}: {e}")
            import traceback; traceback.print_exc()
            overall["results"].append({"model": short, "error": str(e)})
    out_path = OUT_DIR / f"lb_kv_vs_lb_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(overall, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
