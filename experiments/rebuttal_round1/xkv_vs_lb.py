"""xKV head-to-head: per-layer SVD (xkv_single) and grouped SVD (xkv_g2) vs LayerBudget.

Based on Chang et al. xKV 2025. Their compression operates on a different axis
(low-rank SVD rather than token-level eviction or quantization), so we expect
xKV and LayerBudget to be largely composable. This script measures both as
standalone baselines on Mistral-7B WikiText-2 PPL.
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

import xkv_baseline  # noqa: F401, E402
from baselines.base import REGISTRY  # noqa: E402

assert "xkv_single" in REGISTRY, "xkv_single not registered"
assert "xkv_g2" in REGISTRY, "xkv_g2 not registered"

from suite.eval_utils import compress_kv, build_hf_cache  # noqa: E402
from deltacache.core.layer_profiler import LayerAttentionProfiler  # noqa: E402
from deltacache.core.layer_budget_allocator import LayerBudgetAllocator  # noqa: E402

OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HF_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
SHORT = "mistral_7b"
NL, NH, HD = 32, 8, 128
CRS = [2.0, 3.0, 4.0, 6.0]
METHODS = ["full_kv", "kivi_uniform", "xkv_single", "xkv_g2", "layer_budget"]
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


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from transformers.cache_utils import DynamicCache

    print(f"Loading {HF_NAME} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(HF_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        HF_NAME, quantization_config=bnb, device_map={"": "cuda:0"},
        attn_implementation="eager", trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device

    chunks = get_wikitext_chunks(tok, SEQ_LEN, N_CHUNKS)
    prefix_len = int(SEQ_LEN * PREFIX_RATIO)
    print(f"chunks={len(chunks)} prefix_len={prefix_len}", flush=True)

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
            k = full_k[l].unsqueeze(0).transpose(1, 2)
            v = full_v[l].unsqueeze(0).transpose(1, 2)
            cache.update(k, v, l)
        ref_ppl = compute_ppl(model, input_ids, cache, prefix_len, device)
        chunk_data.append((input_ids, full_k, full_v, attns, ref_ppl))
        print(f"  chunk {idx+1}/{len(chunks)}: ref_ppl={ref_ppl:.2f}", flush=True)
        del out, cache
        torch.cuda.empty_cache()

    results = {"model": SHORT, "seq_len": SEQ_LEN, "n_chunks": len(chunks), "prefix_len": prefix_len, "cells": []}
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
                print(f"  CR={cr} {method}: ppl_ratio={m:.4f}", flush=True)
        results["cells"].append(cr_row)

    out_path = OUT_DIR / f"xkv_vs_lb_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump({"date": datetime.now().isoformat(), "results": [results]}, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
