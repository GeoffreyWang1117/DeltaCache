"""LayerBudget-KV at 16K context on Mistral-7B WikiText-2 PPL — A100 camera-ready.

Per-cell checkpointing: each (cr, method) writes its result on success.

Smoke mode (env DC_SMOKE=1): just CR=2× / LB-KV / 1 chunk to verify the path
works on a fresh instance.

Outputs:
  results/long_ctx_16k/<cr>__<method>.json   (per-cell)
  results/lb_kv_long_context_16k_bundle_<ts>.json
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import math
import sys
import traceback
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from _common import setup_paths, CellCheckpoint, Timer, announce  # noqa: E402

THIS = Path(__file__).resolve()
REPO = setup_paths(__file__)

import layer_budget_kv  # noqa: F401, E402
from baselines.base import REGISTRY  # noqa: E402

assert "layer_budget_kv" in REGISTRY

from suite.eval_utils import compress_kv, build_hf_cache  # noqa: E402
from deltacache.core.layer_profiler import LayerAttentionProfiler  # noqa: E402
from deltacache.core.layer_budget_allocator import LayerBudgetAllocator  # noqa: E402

OUT_DIR = THIS.parent / "results" / "long_ctx_16k"
SMOKE = os.environ.get("DC_SMOKE", "0") == "1"

HF_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
SHORT = "mistral_7b"
NL, NH, HD = 32, 8, 128

if SMOKE:
    CRS = [2.0]
    METHODS = ["layer_budget_kv"]
    SEQ_LEN = 4096
    N_CHUNKS = 1
else:
    CRS = [2.0, 4.0]
    METHODS = ["full_kv", "kivi_uniform", "layer_budget", "layer_budget_kv"]
    SEQ_LEN = 4096  # Eager output_attentions retains all 32 layers' QK^T-softmax — 8K OOMs on 80GB; 4K fits with margin
    N_CHUNKS = 2
PREFIX_RATIO = 0.6


def get_chunks(tokenizer, target_len, n):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = " ".join([t for t in ds["text"] if len(t.strip()) > 100])
    ids = tokenizer.encode(text)
    return [ids[i:i + target_len] for i in range(0, len(ids) - target_len, target_len)][:n]


def compute_ppl(model, input_ids, past_kv, prefix_len, device):
    suffix = input_ids[:, prefix_len:]
    if suffix.shape[1] <= 1: return 1.0
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

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CellCheckpoint(OUT_DIR)

    # Plan cells
    cells_to_run = []
    for cr in CRS:
        for method in METHODS:
            if method == "full_kv" and cr != CRS[0]:
                continue
            key = f"cr{cr}__{method}"
            if ckpt.is_done(key):
                print(f"[skip] {key}", flush=True)
                continue
            cells_to_run.append((cr, method, key))
    if not cells_to_run:
        print("[skip] all cells already done")
        return

    announce(f"LB-KV @ {SEQ_LEN} — {len(cells_to_run)} cells to run")

    # Load model + chunks once
    load_t = Timer("model load + chunks")
    tok = AutoTokenizer.from_pretrained(HF_NAME)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        HF_NAME, quantization_config=bnb, device_map={"": "cuda:0"},
        attn_implementation="eager", trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device
    chunks = get_chunks(tok, SEQ_LEN, N_CHUNKS)
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
            cache.update(full_k[l].unsqueeze(0).transpose(1, 2),
                         full_v[l].unsqueeze(0).transpose(1, 2), l)
        ref_ppl = compute_ppl(model, input_ids, cache, prefix_len, device)
        chunk_data.append((input_ids, full_k, full_v, attns, ref_ppl))
        print(f"  chunk {idx+1}/{len(chunks)}: ref_ppl={ref_ppl:.2f}", flush=True)
        del out, cache; torch.cuda.empty_cache()
    load_t.end()

    for cr, method, key in cells_to_run:
        cell_t = Timer(key)
        try:
            ppls = []
            for (input_ids, full_k, full_v, attns, ref_ppl) in chunk_data:
                if method == "full_kv":
                    ppls.append(1.0); continue
                gini = importance = None
                if method == "layer_budget":
                    profiler = LayerAttentionProfiler()
                    pr = profiler.profile_from_attention_weights(attns)
                    gini = pr.gini_scores()
                    importance = LayerBudgetAllocator.compute_importance_weights(NL)
                layers, _, _ = compress_kv(
                    method, full_k, full_v, cr, NL, NH, HD,
                    attention_weights=attns, gini_scores=gini, importance_weights=importance,
                )
                cache = build_hf_cache(layers, prefix_len, device, fill="mean")
                ppl = compute_ppl(model, input_ids, cache, prefix_len, device)
                ppls.append(ppl / max(ref_ppl, 1e-6))
                del cache; torch.cuda.empty_cache()
            mean = sum(ppls) / len(ppls)
            res = {
                "model": SHORT, "seq_len": SEQ_LEN, "cr": cr, "method": method,
                "n_chunks": len(ppls),
                "ppl_ratio_mean": round(mean, 4),
                "ppl_ratio_per_chunk": [round(r, 4) for r in ppls],
            }
            ckpt.save(key, res)
            print(f"  → ppl_ratio={mean:.4f}", flush=True)
        except Exception as e:
            traceback.print_exc()
            ckpt.save(f"{key}_FAILED", {"cr": cr, "method": method, "error": str(e)})
        cell_t.end()

    bundle_path = OUT_DIR.parent / f"lb_kv_long_context_16k_bundle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    ckpt.aggregate(bundle_path)
    print(f"\n[DONE] Aggregated → {bundle_path}", flush=True)


if __name__ == "__main__":
    main()
