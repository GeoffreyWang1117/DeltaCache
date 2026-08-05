#!/usr/bin/env python3
"""Signal analysis experiments: sparsity indicators, Gini stability, coverage validation.

Covers tasks #10, #12, #17:
  #10: Compare Gini vs entropy vs top-k mass as allocation signals
  #12: Gini stability across context lengths (256-4096)
  #17: Coverage model C(l,n) = (n/S)^(1-g) validation against ground truth

All use the same prefill infrastructure — run once, analyze three ways.
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
from typing import Dict, List

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from deltacache.core.layer_profiler import LayerAttentionProfiler, compute_gini, compute_entropy
from deltacache.core.layer_budget_allocator import LayerBudgetAllocator
from deltacache.core.layer_kv_store import LayerKVStore
from deltacache.hf_integration.kv_format import hf_to_deltacache, deltacache_to_hf

RESULTS_DIR = Path(__file__).parent / "results" / "signal_analysis"
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


def get_wikitext_chunks(tokenizer, target_len, n=4):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    all_text = " ".join([t for t in ds["text"] if len(t.strip()) > 100])
    all_ids = tokenizer.encode(all_text)
    return [all_ids[i:i + target_len] for i in range(0, len(all_ids) - target_len, target_len)][:n]


def load_model(model_name, load_in_4bit=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\nLoading {model_name} ({'4-bit' if load_in_4bit else 'fp16'})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {"trust_remote_code": True, "attn_implementation": "eager"}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        kwargs["device_map"] = {"": "cuda:0"}
    else:
        kwargs["dtype"] = torch.float16
        kwargs["device_map"] = {"": "cuda:0"}
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model, tokenizer


def extract_sparsity_indicators(attn_weights, nl):
    """Extract multiple sparsity indicators from attention weights.

    Returns dict: {indicator_name: {layer_idx: score}}
    """
    indicators = {
        "gini": {},
        "entropy": {},
        "top10_mass": {},
        "top20_mass": {},
        "effective_support": {},  # 1/sum(p^2), normalized
    }
    for l in range(nl):
        # Last-token attention row, averaged over heads → (seq_len,)
        last_row = attn_weights[l][0, :, -1, :].mean(dim=0).detach().cpu()
        last_row = last_row / (last_row.sum() + 1e-10)
        seq_len = last_row.numel()

        indicators["gini"][l] = compute_gini(last_row)
        indicators["entropy"][l] = compute_entropy(last_row)

        k10 = max(1, seq_len // 10)
        k20 = max(1, seq_len // 5)
        top10_vals, _ = last_row.topk(min(k10, seq_len))
        top20_vals, _ = last_row.topk(min(k20, seq_len))
        indicators["top10_mass"][l] = top10_vals.sum().item()
        indicators["top20_mass"][l] = top20_vals.sum().item()

        # Effective support size (inverse participation ratio), normalized to [0,1]
        p2 = (last_row ** 2).sum().item()
        eff_support = 1.0 / (p2 * seq_len + 1e-10)  # 1/seq_len = uniform, 1 = delta
        indicators["effective_support"][l] = min(1.0, eff_support)

    return indicators


# ═══════════════════════════════════════════════════════════════════
#  #10: Alternative sparsity indicators
# ═══════════════════════════════════════════════════════════════════

def exp_sparsity_indicators(model, tokenizer, model_short, seq_len=512, n_texts=4):
    """Compare allocation quality using different sparsity signals."""
    print(f"\n{'='*70}")
    print(f"  #10: Sparsity Indicators ({model_short}, {seq_len}tok)")
    print(f"{'='*70}")

    device = next(model.parameters()).device
    nl = model.config.num_hidden_layers
    nh = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    hd = model.config.hidden_size // model.config.num_attention_heads

    chunks = get_wikitext_chunks(tokenizer, seq_len, n_texts)
    CRS = [4.0, 6.0]  # Only test where eviction occurs
    importance = LayerBudgetAllocator.compute_importance_weights(nl)

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

        indicators = extract_sparsity_indicators(attn, nl)

        for cr in CRS:
            for ind_name, ind_scores in indicators.items():
                key = f"{ind_name}@{cr}"
                if key not in results:
                    results[key] = {"indicator": ind_name, "cr": cr, "ratios": []}

                # Use this indicator as the sparsity signal
                allocator = LayerBudgetAllocator(nl, nh, hd)
                store = LayerKVStore(nl, nh, hd)
                fm = allocator.full_memory(prefix_len)
                alloc = allocator.allocate(ind_scores, importance, int(fm / cr), prefix_len)
                store.store_from_full_cache(full_k, full_v, alloc.allocations)
                layers = store.get_all_layers()
                pkv = build_cache_zerofill(layers, prefix_len, device)
                ppl = compute_ppl(model, input_ids, pkv, prefix_len, device)
                results[key]["ratios"].append(ppl / max(ref_ppl, 1e-6))
                del pkv, layers, store; clear_gpu()

            # Also test uniform (no sparsity signal) as baseline
            key_u = f"uniform@{cr}"
            if key_u not in results:
                results[key_u] = {"indicator": "uniform", "cr": cr, "ratios": []}
            allocator = LayerBudgetAllocator(nl, nh, hd)
            store = LayerKVStore(nl, nh, hd)
            fm = allocator.full_memory(prefix_len)
            alloc = allocator.allocate({l: 0.5 for l in range(nl)}, importance, int(fm / cr), prefix_len)
            store.store_from_full_cache(full_k, full_v, alloc.allocations)
            layers = store.get_all_layers()
            pkv = build_cache_zerofill(layers, prefix_len, device)
            ppl = compute_ppl(model, input_ids, pkv, prefix_len, device)
            results[key_u]["ratios"].append(ppl / max(ref_ppl, 1e-6))
            del pkv, layers, store; clear_gpu()

        del full_k, full_v, attn; clear_gpu()
        print(f"ref={ref_ppl:.2f}", flush=True)

    # Summary
    print(f"\n  {'Indicator':20s} {'4x Ratio':>10s} {'6x Ratio':>10s}")
    print(f"  {'-'*45}")
    summary = []
    for cr in CRS:
        for ind in ["gini", "entropy", "top10_mass", "top20_mass", "effective_support", "uniform"]:
            key = f"{ind}@{cr}"
            v = results.get(key, {})
            if not v.get("ratios"):
                continue
            avg = statistics.mean(v["ratios"])
            summary.append({"indicator": ind, "cr": cr, "mean_ratio": round(avg, 4), "n": len(v["ratios"])})

    # Print grouped by CR
    for cr in CRS:
        entries = [(s["indicator"], s["mean_ratio"]) for s in summary if s["cr"] == cr]
        entries.sort(key=lambda x: x[1])
        print(f"\n  --- {cr}x ---")
        for ind, ratio in entries:
            best = " <-- BEST" if ratio == entries[0][1] else ""
            print(f"  {ind:20s} {ratio:>10.4f}{best}")

    return summary


# ═══════════════════════════════════════════════════════════════════
#  #12: Gini stability across context lengths
# ═══════════════════════════════════════════════════════════════════

def exp_gini_stability(model, tokenizer, model_short):
    """Measure Gini coefficient stability across context lengths."""
    print(f"\n{'='*70}")
    print(f"  #12: Gini Stability ({model_short})")
    print(f"{'='*70}")

    device = next(model.parameters()).device
    nl = model.config.num_hidden_layers
    max_pos = min(getattr(model.config, "max_position_embeddings", 4096), 4096)

    # 4096 OOMs with output_attentions on 7B models (24GB GPU)
    seq_lens = [s for s in [256, 512, 1024, 2048] if s <= max_pos]
    n_texts = 3

    # Collect Gini per layer per seq_len
    gini_data = {sl: {l: [] for l in range(nl)} for sl in seq_lens}

    for sl in seq_lens:
        chunks = get_wikitext_chunks(tokenizer, sl, n_texts)
        print(f"  seq_len={sl}: {len(chunks)} chunks", end=" ", flush=True)

        for tokens in chunks:
            input_ids = torch.tensor([tokens], device=device)
            with torch.no_grad():
                out = model(input_ids=input_ids, output_attentions=True, return_dict=True)
            attn = list(out.attentions)
            del out; clear_gpu()

            for l in range(nl):
                last_row = attn[l][0, :, -1, :].mean(dim=0).detach().cpu()
                last_row = last_row / (last_row.sum() + 1e-10)
                gini_data[sl][l].append(compute_gini(last_row))

            del attn; clear_gpu()

        print("done", flush=True)

    # Compute per-layer statistics
    print(f"\n  {'Layer':>5s}", end="")
    for sl in seq_lens:
        print(f" {'G@'+str(sl):>8s}", end="")
    print(f" {'Std':>8s} {'MaxDev':>8s}")
    print(f"  {'-'*(5 + 8*len(seq_lens) + 16)}")

    stability_data = []
    for l in range(nl):
        means = []
        row = {"layer": l}
        for sl in seq_lens:
            vals = gini_data[sl][l]
            m = statistics.mean(vals) if vals else 0
            means.append(m)
            row[f"gini_{sl}"] = round(m, 4)

        std_across = statistics.stdev(means) if len(means) > 1 else 0
        max_dev = max(means) - min(means) if means else 0
        row["std_across_lengths"] = round(std_across, 4)
        row["max_deviation"] = round(max_dev, 4)
        stability_data.append(row)

        print(f"  {l:>5d}", end="")
        for m in means:
            print(f" {m:>8.3f}", end="")
        print(f" {std_across:>8.4f} {max_dev:>8.4f}")

    # Overall stability summary
    all_stds = [d["std_across_lengths"] for d in stability_data]
    print(f"\n  Overall: mean_std={statistics.mean(all_stds):.4f}, "
          f"max_std={max(all_stds):.4f}, "
          f"median_std={statistics.median(all_stds):.4f}")

    return stability_data


# ═══════════════════════════════════════════════════════════════════
#  #17: Coverage model validation
# ═══════════════════════════════════════════════════════════════════

def exp_coverage_validation(model, tokenizer, model_short, seq_len=512, n_texts=3):
    """Validate C(l,n) = (n/S)^(1-g) against actual attention mass captured."""
    print(f"\n{'='*70}")
    print(f"  #17: Coverage Model Validation ({model_short}, {seq_len}tok)")
    print(f"{'='*70}")

    device = next(model.parameters()).device
    nl = model.config.num_hidden_layers

    chunks = get_wikitext_chunks(tokenizer, seq_len, n_texts)
    fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    all_errors = []  # (layer, frac, predicted, actual, error)

    for tidx, tokens in enumerate(chunks):
        input_ids = torch.tensor([tokens], device=device)
        print(f"  [{tidx+1}/{len(chunks)}]", end=" ", flush=True)

        with torch.no_grad():
            out = model(input_ids=input_ids, output_attentions=True, return_dict=True)
        attn = list(out.attentions)
        del out; clear_gpu()

        for l in range(nl):
            last_row = attn[l][0, :, -1, :].mean(dim=0).detach().cpu()
            last_row = last_row / (last_row.sum() + 1e-10)
            s = last_row.numel()
            gini = compute_gini(last_row)

            # Sort by attention weight (descending) for top-k selection
            sorted_vals, sorted_idx = last_row.sort(descending=True)
            cumsum = sorted_vals.cumsum(0)

            for frac in fractions:
                n = max(1, int(s * frac))

                # Predicted coverage: (n/S)^(1-g)
                predicted = (n / s) ** max(0.01, 1.0 - gini)

                # Actual: sum of top-n attention weights
                actual = cumsum[min(n - 1, s - 1)].item()

                error = predicted - actual
                all_errors.append({
                    "text": tidx, "layer": l, "frac": frac,
                    "gini": round(gini, 4),
                    "predicted": round(predicted, 4),
                    "actual": round(actual, 4),
                    "error": round(error, 4),
                    "abs_error": round(abs(error), 4),
                })

        del attn; clear_gpu()
        print("done", flush=True)

    # Summary statistics
    abs_errors = [e["abs_error"] for e in all_errors]
    signed_errors = [e["error"] for e in all_errors]

    print(f"\n  Overall MAE: {statistics.mean(abs_errors):.4f}")
    print(f"  Overall bias (mean signed error): {statistics.mean(signed_errors):+.4f}")
    print(f"  Max absolute error: {max(abs_errors):.4f}")

    # Per-fraction breakdown
    print(f"\n  {'Frac':>6s} {'MAE':>8s} {'Bias':>8s}")
    print(f"  {'-'*25}")
    per_frac = []
    for frac in fractions:
        frac_errs = [e for e in all_errors if e["frac"] == frac]
        mae = statistics.mean([e["abs_error"] for e in frac_errs])
        bias = statistics.mean([e["error"] for e in frac_errs])
        print(f"  {frac:>6.1f} {mae:>8.4f} {bias:>+8.4f}")
        per_frac.append({"frac": frac, "mae": round(mae, 4), "bias": round(bias, 4)})

    # Per-layer breakdown (averaged across fractions)
    print(f"\n  {'Layer':>5s} {'MAE':>8s} {'Gini':>8s}")
    print(f"  {'-'*25}")
    per_layer = []
    for l in range(nl):
        layer_errs = [e for e in all_errors if e["layer"] == l]
        mae = statistics.mean([e["abs_error"] for e in layer_errs])
        gini = statistics.mean([e["gini"] for e in layer_errs])
        per_layer.append({"layer": l, "mae": round(mae, 4), "mean_gini": round(gini, 4)})
        if l % 4 == 0 or l == nl - 1:
            print(f"  {l:>5d} {mae:>8.4f} {gini:>8.3f}")

    return {
        "overall_mae": round(statistics.mean(abs_errors), 4),
        "overall_bias": round(statistics.mean(signed_errors), 4),
        "per_fraction": per_frac,
        "per_layer": per_layer,
    }


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--model-short", default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--n-texts", type=int, default=4)
    parser.add_argument("--exps", nargs="+", default=["10", "12", "17"],
                        choices=["10", "12", "17"])
    args = parser.parse_args()

    model_short = args.model_short or args.model.split("/")[-1]
    model, tokenizer = load_model(args.model, args.load_in_4bit)

    output = {"metadata": {"model": args.model, "model_short": model_short,
                           "timestamp": datetime.now().isoformat()}}

    if "10" in args.exps:
        s10 = exp_sparsity_indicators(model, tokenizer, model_short, args.seq_len, args.n_texts)
        output["sparsity_indicators"] = s10

    if "12" in args.exps:
        s12 = exp_gini_stability(model, tokenizer, model_short)
        output["gini_stability"] = s12

    if "17" in args.exps:
        s17 = exp_coverage_validation(model, tokenizer, model_short, args.seq_len, min(args.n_texts, 3))
        output["coverage_validation"] = s17

    safe = model_short.replace("/", "_").replace("-", "_").lower()
    path = RESULTS_DIR / f"signal_analysis_{safe}_{datetime.now():%Y%m%d_%H%M}.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nAll saved: {path}")


if __name__ == "__main__":
    main()
