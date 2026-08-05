#!/usr/bin/env python3
"""Fill-value ablation (#11) and Llama-2-7B budget sweep (#13).

#11: Compare zero-fill, mean-fill, random-fill, median-fill for evicted positions
#13: Fine-grained budget sweep on Llama-2-7B with combined vs importance-only + gini_capped
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

RESULTS_DIR = Path(__file__).parent / "results" / "fill_and_sweep"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


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


def get_wikitext_chunks(tokenizer, target_len, n=4):
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


def build_cache_with_fill(layers_data, full_seq_len, device, fill="zero"):
    """Build HF DynamicCache with configurable fill for evicted positions."""
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()

    for li, (k, v, idx) in enumerate(layers_data):
        k, v = k.to(device), v.to(device)
        if k.dim() == 3:
            k, v = k.unsqueeze(0), v.unsqueeze(0)
        nt, nh, hd = k.shape[1], k.shape[2], k.shape[3]

        if nt == full_seq_len:
            cache.update(k.transpose(1, 2), v.transpose(1, 2), li)
            continue

        if fill == "mean":
            k_mean = k[0].mean(dim=0, keepdim=True)
            v_mean = v[0].mean(dim=0, keepdim=True)
            kf = k_mean.expand(full_seq_len, -1, -1).clone().unsqueeze(0)
            vf = v_mean.expand(full_seq_len, -1, -1).clone().unsqueeze(0)
        elif fill == "median":
            k_med = k[0].median(dim=0, keepdim=True).values
            v_med = v[0].median(dim=0, keepdim=True).values
            kf = k_med.expand(full_seq_len, -1, -1).clone().unsqueeze(0)
            vf = v_med.expand(full_seq_len, -1, -1).clone().unsqueeze(0)
        elif fill == "random":
            k_mean = k[0].float().mean(dim=0)
            k_std = k[0].float().std(dim=0).clamp(min=1e-6)
            v_mean = v[0].float().mean(dim=0)
            v_std = v[0].float().std(dim=0).clamp(min=1e-6)
            kf = (torch.randn(1, full_seq_len, nh, hd, device=device) * k_std + k_mean).to(k.dtype)
            vf = (torch.randn(1, full_seq_len, nh, hd, device=device) * v_std + v_mean).to(v.dtype)
        else:  # zero
            kf = torch.zeros(1, full_seq_len, nh, hd, dtype=k.dtype, device=device)
            vf = torch.zeros(1, full_seq_len, nh, hd, dtype=v.dtype, device=device)

        ix = idx.long().to(device)
        valid = ix[ix < full_seq_len]
        if valid.numel() > 0:
            kf[0, valid] = k[0, :valid.numel()]
            vf[0, valid] = v[0, :valid.numel()]

        cache.update(kf.transpose(1, 2), vf.transpose(1, 2), li)

    return cache


# ═══════════════════════════════════════════════════════════════════
#  #11: Fill-value ablation
# ═══════════════════════════════════════════════════════════════════

def exp_fill_ablation(model, tokenizer, model_short, seq_len=512, n_texts=4):
    """Compare fill strategies at 4x and 6x (where eviction actually occurs)."""
    print(f"\n{'='*70}")
    print(f"  #11: Fill-Value Ablation ({model_short})")
    print(f"{'='*70}")

    device = next(model.parameters()).device
    nl = model.config.num_hidden_layers
    nh = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    hd = model.config.hidden_size // model.config.num_attention_heads

    chunks = get_wikitext_chunks(tokenizer, seq_len, n_texts)
    CRS = [4.0, 6.0]
    FILLS = ["zero", "mean", "median", "random"]
    results = {}

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
            # Same allocation for all fills (control variable)
            allocator = LayerBudgetAllocator(nl, nh, hd)
            store = LayerKVStore(nl, nh, hd)
            fm = allocator.full_memory(prefix_len)
            alloc = allocator.allocate(sparsity, importance, int(fm / cr), prefix_len)
            store.store_from_full_cache(full_k, full_v, alloc.allocations)
            layers = store.get_all_layers()

            for fill in FILLS:
                key = f"{fill}@{cr}"
                if key not in results:
                    results[key] = {"fill": fill, "cr": cr, "ratios": []}
                try:
                    pkv = build_cache_with_fill(layers, prefix_len, device, fill=fill)
                    ppl = compute_ppl(model, input_ids, pkv, prefix_len, device)
                    results[key]["ratios"].append(ppl / max(ref_ppl, 1e-6))
                    del pkv
                except Exception as e:
                    print(f"ERR:{fill}@{cr}:{e}", end=" ")
                clear_gpu()

            del layers, store; clear_gpu()

        del full_k, full_v, attn; clear_gpu()
        print(f"ref={ref_ppl:.2f}", flush=True)

    # Summary
    print(f"\n  {'Fill':>10s} {'4x Ratio':>10s} {'6x Ratio':>10s}")
    print(f"  {'-'*35}")
    summary = []
    for cr in CRS:
        for fill in FILLS:
            key = f"{fill}@{cr}"
            v = results.get(key, {})
            if v.get("ratios"):
                avg = statistics.mean(v["ratios"])
                summary.append({"fill": fill, "cr": cr, "mean_ratio": round(avg, 4), "n": len(v["ratios"])})

    for fill in FILLS:
        r4 = next((s["mean_ratio"] for s in summary if s["fill"] == fill and s["cr"] == 4.0), "---")
        r6 = next((s["mean_ratio"] for s in summary if s["fill"] == fill and s["cr"] == 6.0), "---")
        print(f"  {fill:>10s} {r4:>10.4f} {r6:>10.4f}")

    return summary


# ═══════════════════════════════════════════════════════════════════
#  #13: Llama-2-7B budget sweep with gini_capped variant
# ═══════════════════════════════════════════════════════════════════

def exp_llama_budget_sweep(model, tokenizer, model_short, seq_len=512, n_texts=4):
    """Fine-grained budget sweep on Llama-2-7B: combined vs importance-only vs gini_capped."""
    print(f"\n{'='*70}")
    print(f"  #13: Budget Sweep ({model_short})")
    print(f"{'='*70}")

    device = next(model.parameters()).device
    nl = model.config.num_hidden_layers
    nh = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    hd = model.config.hidden_size // model.config.num_attention_heads

    chunks = get_wikitext_chunks(tokenizer, seq_len, n_texts)
    CRS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0]
    importance = LayerBudgetAllocator.compute_importance_weights(nl)

    results = {cr: {} for cr in CRS}

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
        real_sp = pr.gini_scores()
        uniform_sp = {l: 0.5 for l in range(nl)}
        capped_sp = {l: min(real_sp[l], 0.7) for l in range(nl)}

        variants = {
            "combined": real_sp,
            "importance_only": uniform_sp,
            "gini_capped_0.7": capped_sp,
        }

        for cr in CRS:
            for vname, sp in variants.items():
                if vname not in results[cr]:
                    results[cr][vname] = []
                allocator = LayerBudgetAllocator(nl, nh, hd)
                store = LayerKVStore(nl, nh, hd)
                fm = allocator.full_memory(prefix_len)
                alloc = allocator.allocate(sp, importance, int(fm / cr), prefix_len)
                store.store_from_full_cache(full_k, full_v, alloc.allocations)
                layers = store.get_all_layers()
                pkv = build_cache_with_fill(layers, prefix_len, device, fill="zero")
                ppl = compute_ppl(model, input_ids, pkv, prefix_len, device)
                results[cr][vname].append(ppl / max(ref_ppl, 1e-6))
                del pkv, layers, store; clear_gpu()

        del full_k, full_v, attn; clear_gpu()
        print(f"ref={ref_ppl:.2f}", flush=True)

    # Summary
    print(f"\n  {'CR':>5s} {'Combined':>10s} {'Imp-Only':>10s} {'Capped0.7':>10s} {'Winner':>12s}")
    print(f"  {'-'*52}")
    summary = []
    for cr in CRS:
        row = {"cr": cr}
        for vname in ["combined", "importance_only", "gini_capped_0.7"]:
            vals = results[cr].get(vname, [])
            row[vname] = round(statistics.mean(vals), 4) if vals else 999
        best = min(row["combined"], row["importance_only"], row["gini_capped_0.7"])
        winner = [k for k in ["combined", "importance_only", "gini_capped_0.7"] if row[k] == best][0]
        row["winner"] = winner
        summary.append(row)
        print(f"  {cr:>5.1f}x {row['combined']:>10.4f} {row['importance_only']:>10.4f} "
              f"{row['gini_capped_0.7']:>10.4f} {winner:>12s}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--model-short", default=None)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--n-texts", type=int, default=4)
    parser.add_argument("--exps", nargs="+", default=["11", "13"], choices=["11", "13"])
    args = parser.parse_args()

    model_short = args.model_short or args.model.split("/")[-1]
    model, tokenizer = load_model_4bit(args.model)

    output = {"metadata": {"model": args.model, "model_short": model_short,
                           "timestamp": datetime.now().isoformat()}}

    if "11" in args.exps:
        output["fill_ablation"] = exp_fill_ablation(model, tokenizer, model_short, args.seq_len, args.n_texts)

    if "13" in args.exps:
        output["budget_sweep"] = exp_llama_budget_sweep(model, tokenizer, model_short, args.seq_len, args.n_texts)

    safe = model_short.replace("/", "_").replace("-", "_").lower()
    path = RESULTS_DIR / f"fill_sweep_{safe}_{datetime.now():%Y%m%d_%H%M}.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
