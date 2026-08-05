#!/usr/bin/env python3
"""Layer-wise error analysis (#14) and attention distortion metrics (#16).

#14: Leave-one-out per-layer analysis at 6x — restore one layer to full KV,
     measure PPL improvement to identify which layers are most damaged.
#16: Attention distortion: softmax entropy shift, KL divergence between
     full and compressed attention, for zero-fill vs mean-fill.
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

from deltacache.core.layer_profiler import LayerAttentionProfiler, compute_gini
from deltacache.core.layer_budget_allocator import LayerBudgetAllocator
from deltacache.core.layer_kv_store import LayerKVStore
from deltacache.hf_integration.kv_format import hf_to_deltacache, deltacache_to_hf

RESULTS_DIR = Path(__file__).parent / "results" / "layerwise"
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


def build_cache_leave_one_out(layers_data, full_k, full_v, restore_layer, full_seq_len, device):
    """Build compressed cache but restore one layer to full KV."""
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    for li, (k, v, idx) in enumerate(layers_data):
        if li == restore_layer:
            # Restore this layer to full KV
            fk = full_k[li:li+1].to(device)  # (1, seq, H, D)
            fv = full_v[li:li+1].to(device)
            cache.update(fk.transpose(1, 2), fv.transpose(1, 2), li)
        else:
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


# ═══════════════════════════════════════════════════════════════════
#  #14: Layer-wise leave-one-out analysis
# ═══════════════════════════════════════════════════════════════════

def exp_layerwise_loo(model, tokenizer, model_short, seq_len=512, cr=6.0, n_texts=2):
    """Leave-one-out: restore each layer to full KV, measure PPL improvement."""
    print(f"\n{'='*70}")
    print(f"  #14: Layer-wise Leave-One-Out ({model_short}, {cr}x)")
    print(f"{'='*70}")

    device = next(model.parameters()).device
    nl = model.config.num_hidden_layers
    nh = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    hd = model.config.hidden_size // model.config.num_attention_heads

    chunks = get_wikitext_chunks(tokenizer, seq_len, n_texts)

    # Per-layer: gini, importance, retention, bits, ppl_improvement
    layer_stats = {l: {"gini": [], "imp": [], "retention": [], "bits": [],
                       "loo_improvement": []} for l in range(nl)}

    for tidx, tokens in enumerate(chunks):
        input_ids = torch.tensor([tokens], device=device)
        prefix_len = int(seq_len * 0.6)
        print(f"  [{tidx+1}/{len(chunks)}]", end=" ", flush=True)

        with torch.no_grad():
            out = model(input_ids=input_ids[:, :prefix_len], output_attentions=True, return_dict=True)
        full_k, full_v = hf_to_deltacache(out.past_key_values)
        attn = list(out.attentions)
        del out; clear_gpu()

        # Reference PPL
        ref_kv = deltacache_to_hf(full_k, full_v, add_batch_dim=True)
        ref_ppl = compute_ppl(model, input_ids, ref_kv, prefix_len, device)
        del ref_kv; clear_gpu()

        # Compressed allocation
        profiler = LayerAttentionProfiler()
        pr = profiler.profile_from_attention_weights(attn)
        sparsity = pr.gini_scores()
        importance = LayerBudgetAllocator.compute_importance_weights(nl)

        allocator = LayerBudgetAllocator(nl, nh, hd)
        store = LayerKVStore(nl, nh, hd)
        fm = allocator.full_memory(prefix_len)
        alloc = allocator.allocate(sparsity, importance, int(fm / cr), prefix_len)
        store.store_from_full_cache(full_k, full_v, alloc.allocations)
        layers = store.get_all_layers()

        # Baseline compressed PPL
        pkv_base = build_cache_with_fill(layers, prefix_len, device, fill="zero")
        base_ppl = compute_ppl(model, input_ids, pkv_base, prefix_len, device)
        base_ratio = base_ppl / max(ref_ppl, 1e-6)
        del pkv_base; clear_gpu()

        print(f"base_ratio={base_ratio:.2f}", end=" ", flush=True)

        # Leave-one-out for each layer
        for l in range(nl):
            a = alloc.allocations[l]
            layer_stats[l]["gini"].append(sparsity[l])
            layer_stats[l]["imp"].append(importance[l])
            layer_stats[l]["retention"].append(a.token_budget / prefix_len)
            layer_stats[l]["bits"].append(a.quant_bits)

            pkv_loo = build_cache_leave_one_out(layers, full_k, full_v, l, prefix_len, device)
            loo_ppl = compute_ppl(model, input_ids, pkv_loo, prefix_len, device)
            loo_ratio = loo_ppl / max(ref_ppl, 1e-6)
            improvement = base_ratio - loo_ratio  # positive = restoring this layer helps
            layer_stats[l]["loo_improvement"].append(improvement)
            del pkv_loo; clear_gpu()

        del full_k, full_v, attn, layers, store; clear_gpu()
        print("done", flush=True)

    # Summary
    print(f"\n  {'L':>3s} {'Gini':>6s} {'Imp':>6s} {'Ret%':>6s} {'Bits':>5s} {'LOO-Impr':>9s} {'Verdict':>12s}")
    print(f"  {'-'*55}")
    summary = []
    for l in range(nl):
        s = layer_stats[l]
        g = statistics.mean(s["gini"])
        imp = statistics.mean(s["imp"])
        ret = statistics.mean(s["retention"])
        bits = statistics.mean(s["bits"])
        loo = statistics.mean(s["loo_improvement"])
        verdict = "CRITICAL" if loo > 0.5 else ("sensitive" if loo > 0.1 else "ok")
        print(f"  {l:>3d} {g:>6.3f} {imp:>6.3f} {ret:>5.0%} {bits:>5.1f} {loo:>+9.4f} {verdict:>12s}")
        summary.append({"layer": l, "gini": round(g, 4), "importance": round(imp, 4),
                         "retention": round(ret, 4), "bits": round(bits, 1),
                         "loo_improvement": round(loo, 4)})

    return summary


# ═══════════════════════════════════════════════════════════════════
#  #16: Attention distortion metrics
# ═══════════════════════════════════════════════════════════════════

def exp_attention_distortion(model, tokenizer, model_short, seq_len=512, cr=6.0, n_texts=2):
    """Measure attention distortion: entropy shift and KL divergence."""
    print(f"\n{'='*70}")
    print(f"  #16: Attention Distortion ({model_short}, {cr}x)")
    print(f"{'='*70}")

    device = next(model.parameters()).device
    nl = model.config.num_hidden_layers
    nh = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    hd = model.config.hidden_size // model.config.num_attention_heads

    chunks = get_wikitext_chunks(tokenizer, seq_len, n_texts)

    # Per-layer per-fill metrics
    metrics = {fill: {l: {"entropy_shift": [], "kl_div": []} for l in range(nl)}
               for fill in ["zero", "mean"]}
    logits_metrics = {fill: {"kl_div": [], "top1_match": []} for fill in ["zero", "mean"]}

    for tidx, tokens in enumerate(chunks):
        input_ids = torch.tensor([tokens], device=device)
        prefix_len = int(seq_len * 0.6)
        print(f"  [{tidx+1}/{len(chunks)}]", end=" ", flush=True)

        # Full KV forward with attention output
        with torch.no_grad():
            out_full = model(input_ids=input_ids[:, :prefix_len], output_attentions=True, return_dict=True)
        full_k, full_v = hf_to_deltacache(out_full.past_key_values)
        full_attn = list(out_full.attentions)  # per-layer attention matrices
        del out_full; clear_gpu()

        # Get full KV logits on suffix
        ref_kv = deltacache_to_hf(full_k, full_v, add_batch_dim=True)
        suffix = input_ids[:, prefix_len:]
        with torch.no_grad():
            pos = torch.arange(prefix_len, prefix_len + suffix.shape[1], device=device).unsqueeze(0)
            out_ref = model(input_ids=suffix, past_key_values=ref_kv, position_ids=pos,
                           return_dict=True, output_attentions=True)
        ref_logits = out_ref.logits[:, -1, :].float()  # last token logits
        ref_suffix_attn = list(out_ref.attentions)
        del ref_kv, out_ref; clear_gpu()

        # Compress
        profiler = LayerAttentionProfiler()
        pr = profiler.profile_from_attention_weights(full_attn)
        sparsity = pr.gini_scores()
        importance = LayerBudgetAllocator.compute_importance_weights(nl)
        allocator = LayerBudgetAllocator(nl, nh, hd)
        store = LayerKVStore(nl, nh, hd)
        fm = allocator.full_memory(prefix_len)
        alloc = allocator.allocate(sparsity, importance, int(fm / cr), prefix_len)
        store.store_from_full_cache(full_k, full_v, alloc.allocations)
        layers = store.get_all_layers()

        for fill in ["zero", "mean"]:
            pkv = build_cache_with_fill(layers, prefix_len, device, fill=fill)

            # Get compressed logits + attention on suffix
            with torch.no_grad():
                out_comp = model(input_ids=suffix, past_key_values=pkv, position_ids=pos,
                                return_dict=True, output_attentions=True)
            comp_logits = out_comp.logits[:, -1, :].float()
            comp_suffix_attn = list(out_comp.attentions)

            # Logits-level metrics
            ref_probs = F.softmax(ref_logits, dim=-1).clamp(min=1e-10)
            comp_probs = F.softmax(comp_logits, dim=-1).clamp(min=1e-10)
            kl = F.kl_div(comp_probs.log(), ref_probs, reduction="batchmean").item()
            top1_match = (ref_logits.argmax(-1) == comp_logits.argmax(-1)).float().mean().item()
            logits_metrics[fill]["kl_div"].append(kl)
            logits_metrics[fill]["top1_match"].append(top1_match)

            # Per-layer attention metrics (on suffix's last token)
            for l in range(min(nl, len(ref_suffix_attn), len(comp_suffix_attn))):
                ref_a = ref_suffix_attn[l][0, :, -1, :].mean(dim=0).float()
                comp_a = comp_suffix_attn[l][0, :, -1, :].mean(dim=0).float()

                # Entropy shift
                ref_ent = -(ref_a.clamp(min=1e-10) * ref_a.clamp(min=1e-10).log()).sum().item()
                comp_ent = -(comp_a.clamp(min=1e-10) * comp_a.clamp(min=1e-10).log()).sum().item()
                metrics[fill][l]["entropy_shift"].append(comp_ent - ref_ent)

                # KL divergence
                ref_a_norm = ref_a.clamp(min=1e-10) / ref_a.sum()
                comp_a_norm = comp_a.clamp(min=1e-10) / comp_a.sum()
                layer_kl = F.kl_div(comp_a_norm.log(), ref_a_norm, reduction="sum").item()
                metrics[fill][l]["kl_div"].append(layer_kl)

            del pkv, out_comp, comp_suffix_attn; clear_gpu()

        del full_k, full_v, full_attn, ref_suffix_attn, layers, store; clear_gpu()
        print("done", flush=True)

    # Summary: per-fill logits-level
    print(f"\n  Logits-level distortion:")
    print(f"  {'Fill':>8s} {'KL-div':>10s} {'Top1-Match':>12s}")
    print(f"  {'-'*35}")
    logits_summary = {}
    for fill in ["zero", "mean"]:
        kl = statistics.mean(logits_metrics[fill]["kl_div"])
        t1 = statistics.mean(logits_metrics[fill]["top1_match"])
        print(f"  {fill:>8s} {kl:>10.4f} {t1:>11.1%}")
        logits_summary[fill] = {"kl_div": round(kl, 4), "top1_match": round(t1, 4)}

    # Summary: per-layer attention distortion
    print(f"\n  Per-layer attention distortion (mean across texts):")
    print(f"  {'L':>3s} {'Zero-Ent':>9s} {'Zero-KL':>9s} {'Mean-Ent':>9s} {'Mean-KL':>9s} {'KL-Reduction':>13s}")
    print(f"  {'-'*55}")
    layer_summary = []
    for l in range(nl):
        row = {"layer": l}
        for fill in ["zero", "mean"]:
            ent = statistics.mean(metrics[fill][l]["entropy_shift"]) if metrics[fill][l]["entropy_shift"] else 0
            kl = statistics.mean(metrics[fill][l]["kl_div"]) if metrics[fill][l]["kl_div"] else 0
            row[f"{fill}_entropy_shift"] = round(ent, 4)
            row[f"{fill}_kl_div"] = round(kl, 4)
        z_kl = row["zero_kl_div"]
        m_kl = row["mean_kl_div"]
        reduction = 1.0 - m_kl / max(z_kl, 1e-10) if z_kl > 0.001 else 0
        row["kl_reduction"] = round(reduction, 4)
        layer_summary.append(row)
        if l % 4 == 0 or l == nl - 1:
            print(f"  {l:>3d} {row['zero_entropy_shift']:>+9.4f} {z_kl:>9.4f} "
                  f"{row['mean_entropy_shift']:>+9.4f} {m_kl:>9.4f} {reduction:>12.1%}")

    return {"logits": logits_summary, "per_layer": layer_summary}


# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--model-short", default=None)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--cr", type=float, default=6.0)
    parser.add_argument("--n-texts", type=int, default=2)
    parser.add_argument("--exps", nargs="+", default=["14", "16"], choices=["14", "16"])
    args = parser.parse_args()

    model_short = args.model_short or args.model.split("/")[-1]
    model, tokenizer = load_model_4bit(args.model)

    output = {"metadata": {"model": args.model, "model_short": model_short,
                           "cr": args.cr, "seq_len": args.seq_len,
                           "timestamp": datetime.now().isoformat()}}

    if "14" in args.exps:
        output["layerwise_loo"] = exp_layerwise_loo(
            model, tokenizer, model_short, args.seq_len, args.cr, args.n_texts)

    if "16" in args.exps:
        output["attention_distortion"] = exp_attention_distortion(
            model, tokenizer, model_short, args.seq_len, args.cr, args.n_texts)

    safe = model_short.replace("/", "_").replace("-", "_").lower()
    path = RESULTS_DIR / f"layerwise_{safe}_{datetime.now():%Y%m%d_%H%M}.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
