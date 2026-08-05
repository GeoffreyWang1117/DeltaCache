#!/usr/bin/env python3
"""Corrected ablation v2: re-run A1/A3 with verified pipeline.

Uses the same infrastructure as diagnose_a2.py but focuses on:
  A1: Component analysis (eviction-only, quant-only, joint)
  A3: Precision levels ({4,16} vs {4,8,16})

Run on both 7B models at 3x AND 6x to show the compression-regime effect.
"""

import argparse
import gc
import json
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from deltacache.core.layer_profiler import LayerAttentionProfiler
from deltacache.core.layer_budget_allocator import LayerBudgetAllocator
from deltacache.core.layer_kv_store import LayerKVStore
from deltacache.hf_integration.kv_format import hf_to_deltacache, deltacache_to_hf

RESULTS_DIR = Path(__file__).parent / "results" / "ablation_v2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def build_cache_zerofill(layers_data, full_seq_len, device):
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    for li, (k, v, idx) in enumerate(layers_data):
        k, v = k.to(device), v.to(device)
        if k.dim() == 3:
            k, v = k.unsqueeze(0), v.unsqueeze(0)
        nt, nh, hd = k.shape[1], k.shape[2], k.shape[3]
        if nt == full_seq_len:
            cache.update(k.transpose(1, 2), v.transpose(1, 2), li)
        else:
            kf = torch.zeros(1, full_seq_len, nh, hd, dtype=k.dtype, device=device)
            vf = torch.zeros(1, full_seq_len, nh, hd, dtype=v.dtype, device=device)
            ix = idx.long().to(device)
            valid = ix[ix < full_seq_len]
            if valid.numel() > 0:
                kf[0, valid] = k[0, :valid.numel()]
                vf[0, valid] = v[0, :valid.numel()]
            cache.update(kf.transpose(1, 2), vf.transpose(1, 2), li)
    return cache


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


def get_wikitext_chunks(tokenizer, target_len, n=6):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    all_text = " ".join([t for t in ds["text"] if len(t.strip()) > 100])
    all_ids = tokenizer.encode(all_text)
    return [all_ids[i:i + target_len] for i in range(0, len(all_ids) - target_len, target_len)][:n]


def load_model_4bit(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    print(f"\nLoading {model_name} (4-bit)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16),
        device_map={"": "cuda:0"}, attn_implementation="eager", trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def run_ablation(model, tokenizer, model_short, seq_len=512, n_texts=4):
    print(f"\n{'='*70}")
    print(f"  CORRECTED ABLATION v2: {model_short} @ {seq_len}tok")
    print(f"{'='*70}")

    device = next(model.parameters()).device
    nl = model.config.num_hidden_layers
    nh = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    hd = model.config.hidden_size // model.config.num_attention_heads

    chunks = get_wikitext_chunks(tokenizer, seq_len, n_texts)
    CRS = [3.0, 4.0, 6.0]

    # Variant definitions: (name, kwargs for allocator)
    # evict_only: force FP16, allocator must evict tokens
    # quant_only: force all tokens retained, allocator only quantizes
    # joint: default (quantize first, evict last)
    # bits_4_16: only INT4 and FP16 available
    # bits_4_8_16: default (INT4, INT8, FP16)

    all_results = {}

    for tidx, tokens in enumerate(chunks):
        input_ids = torch.tensor([tokens], device=device)
        prefix_len = int(seq_len * 0.6)
        print(f"  [{tidx+1}/{len(chunks)}]", end=" ", flush=True)

        with torch.no_grad():
            out = model(input_ids=input_ids[:, :prefix_len], output_attentions=True, return_dict=True)
        full_k, full_v = hf_to_deltacache(out.past_key_values)
        attn = list(out.attentions)
        del out; clear_gpu()

        ref_kv = deltacache_to_hf(full_k, full_v, add_batch_dim=True)
        ref_ppl = compute_ppl(model, input_ids, ref_kv, prefix_len, device)
        del ref_kv; clear_gpu()

        profiler = LayerAttentionProfiler()
        pr = profiler.profile_from_attention_weights(attn)
        sparsity = pr.gini_scores()
        importance = LayerBudgetAllocator.compute_importance_weights(nl)

        for cr in CRS:
            variants = {}

            # A1: Component analysis
            for vname, evict_only, quant_only in [
                ("eviction_only", True, False),
                ("quant_only", False, True),
                ("joint", False, False),
            ]:
                allocator = LayerBudgetAllocator(nl, nh, hd)
                store = LayerKVStore(nl, nh, hd)
                fm = allocator.full_memory(prefix_len)
                budget = int(fm / cr)
                alloc = allocator.allocate(sparsity, importance, budget, prefix_len)

                if evict_only:
                    for a in alloc.allocations:
                        a.quant_bits = 16
                elif quant_only:
                    for a in alloc.allocations:
                        a.token_budget = prefix_len

                store.store_from_full_cache(full_k, full_v, alloc.allocations)
                layers = store.get_all_layers()
                pkv = build_cache_zerofill(layers, prefix_len, device)
                ppl = compute_ppl(model, input_ids, pkv, prefix_len, device)

                # Record per-layer details
                layer_details = []
                for a in alloc.allocations:
                    layer_details.append({
                        "tokens": a.token_budget,
                        "bits": a.quant_bits,
                        "retention": round(a.token_budget / prefix_len, 3),
                    })

                variants[vname] = {
                    "ppl": ppl,
                    "ratio": ppl / max(ref_ppl, 1e-6),
                    "mem_kb": store.memory_usage() / 1024,
                    "layers": layer_details,
                }
                del pkv, layers, store; clear_gpu()

            # A3: Precision levels
            for vname, bits_set in [
                ("bits_4_16", [4, 16]),
                ("bits_4_8_16", [4, 8, 16]),
            ]:
                allocator = LayerBudgetAllocator(nl, nh, hd, available_bits=bits_set)
                store = LayerKVStore(nl, nh, hd)
                fm = allocator.full_memory(prefix_len)
                budget = int(fm / cr)
                alloc = allocator.allocate(sparsity, importance, budget, prefix_len)
                store.store_from_full_cache(full_k, full_v, alloc.allocations)
                layers = store.get_all_layers()
                pkv = build_cache_zerofill(layers, prefix_len, device)
                ppl = compute_ppl(model, input_ids, pkv, prefix_len, device)

                variants[vname] = {
                    "ppl": ppl,
                    "ratio": ppl / max(ref_ppl, 1e-6),
                    "mem_kb": store.memory_usage() / 1024,
                }
                del pkv, layers, store; clear_gpu()

            key = f"cr_{cr}"
            if key not in all_results:
                all_results[key] = {v: {"ppls": [], "ratios": [], "mems": []} for v in variants}
            for v, data in variants.items():
                if v not in all_results[key]:
                    all_results[key][v] = {"ppls": [], "ratios": [], "mems": []}
                all_results[key][v]["ppls"].append(data["ppl"])
                all_results[key][v]["ratios"].append(data["ratio"])
                all_results[key][v]["mems"].append(data["mem_kb"])

        del full_k, full_v, attn; clear_gpu()
        print(f"ref={ref_ppl:.2f}", flush=True)

    # Print summary
    summary = []
    for cr in CRS:
        key = f"cr_{cr}"
        print(f"\n  --- {cr}x ---")
        print(f"  {'Variant':20s} {'Ratio':>8s} {'Mem(KB)':>10s}")
        for vname in ["eviction_only", "quant_only", "joint", "bits_4_16", "bits_4_8_16"]:
            v = all_results[key].get(vname, {})
            if not v.get("ratios"):
                continue
            r = statistics.mean(v["ratios"])
            m = statistics.mean(v["mems"])
            print(f"  {vname:20s} {r:>8.4f} {m:>9.0f}KB")
            summary.append({"cr": cr, "variant": vname, "mean_ratio": round(r, 4),
                            "mean_mem_kb": round(m, 1), "n": len(v["ratios"])})

    path = RESULTS_DIR / f"ablation_v2_{model_short.lower().replace('-', '_')}_{datetime.now():%Y%m%d_%H%M}.json"
    with open(path, "w") as f:
        json.dump({"metadata": {"model": model_short, "seq_len": seq_len,
                                "timestamp": datetime.now().isoformat()},
                   "summary": summary}, f, indent=2)
    print(f"\n  Saved: {path}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--model-short", default=None)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--n-texts", type=int, default=4)
    args = parser.parse_args()

    model_short = args.model_short or args.model.split("/")[-1]
    model, tokenizer = load_model_4bit(args.model)
    run_ablation(model, tokenizer, model_short, args.seq_len, args.n_texts)


if __name__ == "__main__":
    main()
