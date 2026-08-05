#!/usr/bin/env python3
"""Diagnose why combined (Gini + importance) underperforms importance-only.

Runs systematically on Mistral-7B (fast) then Llama-2-7B (hard case).

Experiments:
  A. Signal combination variants (8 ways to combine signals)
  B. Budget sweep (1.5x to 8x, fine-grained)
  C. Layer-wise error analysis (per-layer PPL contribution)
  D. Token floor variants (constrain minimum retention)
"""

import gc
import json
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from copy import deepcopy

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from deltacache.core.layer_profiler import LayerAttentionProfiler
from deltacache.core.layer_budget_allocator import LayerBudgetAllocator, AllocationResult
from deltacache.core.layer_kv_store import LayerKVStore
from deltacache.hf_integration.kv_format import hf_to_deltacache, deltacache_to_hf


RESULTS_DIR = Path(__file__).parent / "results" / "diagnose"
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


def run_variant(full_k, full_v, attn, nl, nh, hd, prefix_len, cr,
                sparsity_override=None, importance_override=None,
                bits_set=None, token_floor_frac=None,
                evict_only=False, quant_only=False):
    """Run a single allocation variant and return layers + allocation details."""
    profiler = LayerAttentionProfiler()
    pr = profiler.profile_from_attention_weights(attn)
    real_sparsity = pr.gini_scores()
    real_importance = LayerBudgetAllocator.compute_importance_weights(nl)

    sp = sparsity_override if sparsity_override is not None else real_sparsity
    imp = importance_override if importance_override is not None else real_importance

    allocator = LayerBudgetAllocator(nl, nh, hd, available_bits=bits_set or [4, 8, 16])

    # Apply token floor if specified
    if token_floor_frac is not None:
        allocator.sink_tokens = max(allocator.sink_tokens, int(prefix_len * token_floor_frac * 0.3))
        allocator.recent_tokens = max(allocator.recent_tokens, int(prefix_len * token_floor_frac * 0.7))

    store = LayerKVStore(nl, nh, hd)
    fm = allocator.full_memory(prefix_len)
    budget = int(fm / cr)
    alloc = allocator.allocate(sp, imp, budget, prefix_len)

    if evict_only:
        for a in alloc.allocations:
            a.quant_bits = 16
    elif quant_only:
        for a in alloc.allocations:
            a.token_budget = prefix_len

    store.store_from_full_cache(full_k, full_v, alloc.allocations)
    layers = store.get_all_layers()
    return layers, alloc, store.memory_usage()


# ═══════════════════════════════════════════════════════════════════
#  EXPERIMENT A: Signal Combination Variants
# ═══════════════════════════════════════════════════════════════════

def exp_a_signal_combinations(model, tokenizer, model_short, seq_len=512, cr=3.0, n_texts=4):
    """8 signal combination variants."""
    print(f"\n{'='*70}")
    print(f"  EXP A: Signal Combinations ({model_short}, {cr}x)")
    print(f"{'='*70}")

    device = next(model.parameters()).device
    nl = model.config.num_hidden_layers
    nh = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    hd = model.config.hidden_size // model.config.num_attention_heads

    chunks = get_wikitext_chunks(tokenizer, seq_len, n_texts)
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
        real_sp = pr.gini_scores()
        real_imp = LayerBudgetAllocator.compute_importance_weights(nl)
        uniform_sp = {l: 0.5 for l in range(nl)}
        uniform_imp = {l: 0.5 for l in range(nl)}

        # Variant definitions
        variants = {
            # Original paper variants
            "gini_only":        (real_sp,    uniform_imp),
            "importance_only":  (uniform_sp, real_imp),
            "combined":         (real_sp,    real_imp),
            # New: weighted combinations
            "gini_scaled_0.3":  ({l: 0.5 + 0.3*(real_sp[l]-0.5) for l in range(nl)}, real_imp),
            "gini_scaled_0.5":  ({l: 0.5 + 0.5*(real_sp[l]-0.5) for l in range(nl)}, real_imp),
            # New: two-stage (importance for bits, gini for tokens)
            "imp_bits_gini_tok": (real_sp, real_imp),  # default combined
            # New: reversed roles (gini for bits, importance for tokens)
            "gini_bits_imp_tok": (uniform_sp, real_imp),  # importance drives tokens; will override bits below
            # New: capped gini (prevent extreme sparsity signals)
            "gini_capped_0.7":  ({l: min(real_sp[l], 0.7) for l in range(nl)}, real_imp),
            "gini_capped_0.6":  ({l: min(real_sp[l], 0.6) for l in range(nl)}, real_imp),
        }

        for vname, (sp, imp) in variants.items():
            if vname not in results:
                results[vname] = {"ppls": [], "ratios": [], "allocs": []}
            try:
                layers, alloc, mem = run_variant(
                    full_k, full_v, attn, nl, nh, hd, prefix_len, cr,
                    sparsity_override=sp, importance_override=imp,
                )
                pkv = build_cache_zerofill(layers, prefix_len, device)
                ppl = compute_ppl(model, input_ids, pkv, prefix_len, device)
                results[vname]["ppls"].append(ppl)
                results[vname]["ratios"].append(ppl / max(ref_ppl, 1e-6))
                # Record per-layer allocation
                results[vname]["allocs"].append([
                    {"layer": a.layer_idx, "tokens": a.token_budget, "bits": a.quant_bits}
                    for a in alloc.allocations
                ])
                del pkv, layers
            except Exception as e:
                print(f"ERR:{vname}:{e}", end=" ", flush=True)
            clear_gpu()

        del full_k, full_v, attn; clear_gpu()
        print(f"ref={ref_ppl:.2f}", flush=True)

    # Print summary
    print(f"\n  {'Variant':25s} {'Ratio':>8s} {'StdDev':>8s}")
    print(f"  {'-'*45}")
    summary = []
    for vname in sorted(results.keys(), key=lambda x: statistics.mean(results[x]["ratios"]) if results[x]["ratios"] else 999):
        v = results[vname]
        if not v["ratios"]:
            continue
        avg = statistics.mean(v["ratios"])
        std = statistics.stdev(v["ratios"]) if len(v["ratios"]) > 1 else 0
        marker = " <-- BEST" if avg == min(statistics.mean(results[k]["ratios"]) for k in results if results[k]["ratios"]) else ""
        print(f"  {vname:25s} {avg:>8.4f} {std:>8.4f}{marker}")
        summary.append({"variant": vname, "mean_ratio": round(avg, 4), "std": round(std, 4),
                         "n": len(v["ratios"])})

    return results, summary


# ═══════════════════════════════════════════════════════════════════
#  EXPERIMENT B: Budget Sweep
# ═══════════════════════════════════════════════════════════════════

def exp_b_budget_sweep(model, tokenizer, model_short, seq_len=512, n_texts=4):
    """Fine-grained budget sweep: 1.5x to 8x."""
    print(f"\n{'='*70}")
    print(f"  EXP B: Budget Sweep ({model_short})")
    print(f"{'='*70}")

    device = next(model.parameters()).device
    nl = model.config.num_hidden_layers
    nh = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    hd = model.config.hidden_size // model.config.num_attention_heads

    chunks = get_wikitext_chunks(tokenizer, seq_len, n_texts)
    CRS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0]

    # Track combined vs importance_only at each CR
    results = {cr: {"combined": [], "importance_only": []} for cr in CRS}

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

        uniform_sp = {l: 0.5 for l in range(nl)}

        for cr in CRS:
            for variant_name, sp_override in [("combined", None), ("importance_only", uniform_sp)]:
                try:
                    layers, alloc, mem = run_variant(
                        full_k, full_v, attn, nl, nh, hd, prefix_len, cr,
                        sparsity_override=sp_override,
                    )
                    pkv = build_cache_zerofill(layers, prefix_len, device)
                    ppl = compute_ppl(model, input_ids, pkv, prefix_len, device)
                    results[cr][variant_name].append(ppl / max(ref_ppl, 1e-6))
                    del pkv, layers
                except:
                    pass
                clear_gpu()

        del full_k, full_v, attn; clear_gpu()
        print(f"ref={ref_ppl:.2f}", flush=True)

    print(f"\n  {'CR':>5s} {'Combined':>10s} {'Imp-Only':>10s} {'Winner':>10s}")
    print(f"  {'-'*40}")
    summary = []
    for cr in CRS:
        c = statistics.mean(results[cr]["combined"]) if results[cr]["combined"] else 999
        i = statistics.mean(results[cr]["importance_only"]) if results[cr]["importance_only"] else 999
        winner = "combined" if c < i else "imp-only"
        print(f"  {cr:>5.1f}x {c:>10.4f} {i:>10.4f} {winner:>10s}")
        summary.append({"cr": cr, "combined": round(c, 4), "importance_only": round(i, 4), "winner": winner})

    return summary


# ═══════════════════════════════════════════════════════════════════
#  EXPERIMENT C: Layer-wise Error Analysis
# ═══════════════════════════════════════════════════════════════════

def exp_c_layerwise_error(model, tokenizer, model_short, seq_len=512, cr=3.0, n_texts=2):
    """Per-layer analysis: what does the allocator do, and which layers hurt most?"""
    print(f"\n{'='*70}")
    print(f"  EXP C: Layer-wise Error ({model_short}, {cr}x)")
    print(f"{'='*70}")

    device = next(model.parameters()).device
    nl = model.config.num_hidden_layers
    nh = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    hd = model.config.hidden_size // model.config.num_attention_heads

    chunks = get_wikitext_chunks(tokenizer, seq_len, n_texts)
    layer_data = []

    for tidx, tokens in enumerate(chunks):
        input_ids = torch.tensor([tokens], device=device)
        prefix_len = int(seq_len * 0.6)

        with torch.no_grad():
            out = model(input_ids=input_ids[:, :prefix_len], output_attentions=True, return_dict=True)
        full_k, full_v = hf_to_deltacache(out.past_key_values)
        attn = list(out.attentions)
        del out; clear_gpu()

        profiler = LayerAttentionProfiler()
        pr = profiler.profile_from_attention_weights(attn)
        gini = pr.gini_scores()
        importance = LayerBudgetAllocator.compute_importance_weights(nl)

        # Run combined allocation
        layers_c, alloc_c, _ = run_variant(full_k, full_v, attn, nl, nh, hd, prefix_len, cr)

        # Run importance-only allocation
        uniform_sp = {l: 0.5 for l in range(nl)}
        layers_i, alloc_i, _ = run_variant(
            full_k, full_v, attn, nl, nh, hd, prefix_len, cr,
            sparsity_override=uniform_sp,
        )

        for l in range(nl):
            ac = alloc_c.allocations[l]
            ai = alloc_i.allocations[l]

            # Value norm for this layer (proxy for information content)
            v_norm = full_v[l].float().norm().item()

            layer_data.append({
                "text_idx": tidx,
                "layer": l,
                "gini": round(gini[l], 4),
                "importance": round(importance[l], 4),
                "value_norm": round(v_norm, 2),
                "combined_tokens": ac.token_budget,
                "combined_bits": ac.quant_bits,
                "combined_retention": round(ac.token_budget / prefix_len, 4),
                "imponly_tokens": ai.token_budget,
                "imponly_bits": ai.quant_bits,
                "imponly_retention": round(ai.token_budget / prefix_len, 4),
            })

        del full_k, full_v, attn, layers_c, layers_i; clear_gpu()
        print(f"  [{tidx+1}/{len(chunks)}] done", flush=True)

    # Aggregate per layer
    print(f"\n  {'L':>3s} {'Gini':>6s} {'Imp':>6s} {'C-Ret':>7s} {'C-Bits':>7s} {'I-Ret':>7s} {'I-Bits':>7s} {'Diff':>7s}")
    print(f"  {'-'*55}")
    for l in range(nl):
        ld = [d for d in layer_data if d["layer"] == l]
        g = statistics.mean([d["gini"] for d in ld])
        imp = statistics.mean([d["importance"] for d in ld])
        cr_ = statistics.mean([d["combined_retention"] for d in ld])
        cb = statistics.mean([d["combined_bits"] for d in ld])
        ir = statistics.mean([d["imponly_retention"] for d in ld])
        ib = statistics.mean([d["imponly_bits"] for d in ld])
        diff = cr_ - ir  # positive means combined retains MORE tokens
        flag = " <<< over-evicts" if diff < -0.1 else ""
        print(f"  {l:>3d} {g:>6.3f} {imp:>6.3f} {cr_:>7.1%} {cb:>6.1f}b {ir:>7.1%} {ib:>6.1f}b {diff:>+7.1%}{flag}")

    return layer_data


# ═══════════════════════════════════════════════════════════════════
#  EXPERIMENT D: Token Floor Variants
# ═══════════════════════════════════════════════════════════════════

def exp_d_token_floor(model, tokenizer, model_short, seq_len=512, cr=3.0, n_texts=4):
    """Test whether a minimum token retention floor fixes combined."""
    print(f"\n{'='*70}")
    print(f"  EXP D: Token Floor ({model_short}, {cr}x)")
    print(f"{'='*70}")

    device = next(model.parameters()).device
    nl = model.config.num_hidden_layers
    nh = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    hd = model.config.hidden_size // model.config.num_attention_heads

    chunks = get_wikitext_chunks(tokenizer, seq_len, n_texts)
    results = {}

    floors = [None, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

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

        for floor in floors:
            vname = f"floor_{floor}" if floor else "no_floor"
            if vname not in results:
                results[vname] = {"ppls": [], "ratios": []}
            try:
                layers, alloc, mem = run_variant(
                    full_k, full_v, attn, nl, nh, hd, prefix_len, cr,
                    token_floor_frac=floor,
                )
                pkv = build_cache_zerofill(layers, prefix_len, device)
                ppl = compute_ppl(model, input_ids, pkv, prefix_len, device)
                results[vname]["ppls"].append(ppl)
                results[vname]["ratios"].append(ppl / max(ref_ppl, 1e-6))
                del pkv, layers
            except:
                pass
            clear_gpu()

        del full_k, full_v, attn; clear_gpu()
        print(f"ref={ref_ppl:.2f}", flush=True)

    print(f"\n  {'Floor':>15s} {'Ratio':>8s}")
    print(f"  {'-'*28}")
    summary = []
    for floor in floors:
        vname = f"floor_{floor}" if floor else "no_floor"
        v = results.get(vname, {})
        if not v.get("ratios"):
            continue
        avg = statistics.mean(v["ratios"])
        print(f"  {vname:>15s} {avg:>8.4f}")
        summary.append({"floor": floor, "mean_ratio": round(avg, 4)})

    return summary


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--model-short", default=None)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--n-texts", type=int, default=4)
    parser.add_argument("--cr", type=float, default=3.0)
    parser.add_argument("--exps", nargs="+", default=["a", "b", "c", "d"],
                        choices=["a", "b", "c", "d"])
    args = parser.parse_args()

    model_short = args.model_short or args.model.split("/")[-1]
    model, tokenizer = load_model_4bit(args.model)

    all_output = {
        "metadata": {
            "model": args.model, "model_short": model_short,
            "seq_len": args.seq_len, "cr": args.cr,
            "timestamp": datetime.now().isoformat(),
        },
    }

    if "a" in args.exps:
        results_a, summary_a = exp_a_signal_combinations(
            model, tokenizer, model_short, args.seq_len, args.cr, args.n_texts)
        all_output["exp_a_signals"] = summary_a

    if "b" in args.exps:
        summary_b = exp_b_budget_sweep(
            model, tokenizer, model_short, args.seq_len, args.n_texts)
        all_output["exp_b_budget_sweep"] = summary_b

    if "c" in args.exps:
        layer_data_c = exp_c_layerwise_error(
            model, tokenizer, model_short, args.seq_len, args.cr, min(args.n_texts, 2))
        all_output["exp_c_layerwise"] = layer_data_c

    if "d" in args.exps:
        summary_d = exp_d_token_floor(
            model, tokenizer, model_short, args.seq_len, args.cr, args.n_texts)
        all_output["exp_d_token_floor"] = summary_d

    safe = model_short.replace("/", "_").replace("-", "_").lower()
    path = RESULTS_DIR / f"diagnose_a2_{safe}_{datetime.now():%Y%m%d_%H%M}.json"
    with open(path, "w") as f:
        json.dump(all_output, f, indent=2, default=str)
    print(f"\nAll results saved: {path}")


if __name__ == "__main__":
    main()
