#!/usr/bin/env python3
"""#22: Fixed vs online profile comparison.
#23: Greedy allocator optimality vs random search.

Both run on Mistral-7B at 4x/6x with inverted importance (the new best config).
"""

import argparse, gc, json, math, random, statistics, sys, time
from datetime import datetime
from pathlib import Path
import torch, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from deltacache.core.layer_profiler import LayerAttentionProfiler
from deltacache.core.layer_budget_allocator import LayerBudgetAllocator
from deltacache.core.layer_kv_store import LayerKVStore
from deltacache.hf_integration.kv_format import hf_to_deltacache, deltacache_to_hf

RESULTS_DIR = Path(__file__).parent / "results" / "profile_optimality"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()

def compute_ppl(model, input_ids, past_kv, prefix_len, device):
    suffix = input_ids[:, prefix_len:]
    if suffix.shape[1] <= 1: return 1.0
    with torch.no_grad():
        pos = torch.arange(prefix_len, prefix_len + suffix.shape[1], device=device).unsqueeze(0)
        out = model(input_ids=suffix, past_key_values=past_kv, position_ids=pos, return_dict=True)
    logits = out.logits[:, :-1, :].contiguous()
    labels = suffix[:, 1:].contiguous()
    return math.exp(min(F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1)).item(), 20))

def get_wikitext_chunks(tokenizer, target_len, n=4):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    all_text = " ".join([t for t in ds["text"] if len(t.strip()) > 100])
    all_ids = tokenizer.encode(all_text)
    return [all_ids[i:i+target_len] for i in range(0, len(all_ids)-target_len, target_len)][:n]

def load_model_4bit(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    print(f"\nLoading {model_name} (4-bit)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16),
        device_map={"": "cuda:0"}, attn_implementation="eager", trust_remote_code=True)
    model.eval()
    return model, tokenizer

def build_cache_zerofill(layers_data, full_seq_len, device):
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    for li, (k, v, idx) in enumerate(layers_data):
        k, v = k.to(device), v.to(device)
        if k.dim() == 3: k, v = k.unsqueeze(0), v.unsqueeze(0)
        nt, nh, hd = k.shape[1], k.shape[2], k.shape[3]
        if nt == full_seq_len:
            cache.update(k.transpose(1,2), v.transpose(1,2), li)
        else:
            kf = torch.zeros(1, full_seq_len, nh, hd, dtype=k.dtype, device=device)
            vf = torch.zeros(1, full_seq_len, nh, hd, dtype=v.dtype, device=device)
            ix = idx.long().to(device); valid = ix[ix < full_seq_len]
            if valid.numel() > 0: kf[0, valid] = k[0, :valid.numel()]; vf[0, valid] = v[0, :valid.numel()]
            cache.update(kf.transpose(1,2), vf.transpose(1,2), li)
    return cache

def inverted_importance(nl):
    """Inverted sigmoid: early layers get higher weight."""
    sigmoid = LayerBudgetAllocator.compute_importance_weights(nl)
    return {l: sigmoid[nl - 1 - l] for l in range(nl)}


def run_allocation_and_eval(full_k, full_v, sparsity, importance, nl, nh, hd,
                            prefix_len, cr, model, input_ids, device):
    """Run one allocation variant and return PPL ratio."""
    allocator = LayerBudgetAllocator(nl, nh, hd)
    store = LayerKVStore(nl, nh, hd)
    fm = allocator.full_memory(prefix_len)
    alloc = allocator.allocate(sparsity, importance, int(fm / cr), prefix_len)
    store.store_from_full_cache(full_k, full_v, alloc.allocations)
    layers = store.get_all_layers()
    pkv = build_cache_zerofill(layers, prefix_len, device)
    ref_kv = deltacache_to_hf(full_k, full_v, add_batch_dim=True)
    ref_ppl = compute_ppl(model, input_ids, ref_kv, prefix_len, device)
    ppl = compute_ppl(model, input_ids, pkv, prefix_len, device)
    del pkv, layers, store, ref_kv; clear_gpu()
    return ppl / max(ref_ppl, 1e-6)


# ═══════════════════════════════════════════════════════════════════
#  #22: Fixed vs online profile
# ═══════════════════════════════════════════════════════════════════

def exp_fixed_vs_online(model, tokenizer, model_short, seq_len=512, n_texts=4):
    """Compare online per-input profiling vs fixed profile."""
    print(f"\n{'='*70}")
    print(f"  #22: Fixed vs Online Profile ({model_short})")
    print(f"{'='*70}")

    device = next(model.parameters()).device
    nl = model.config.num_hidden_layers
    nh = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    hd = model.config.hidden_size // model.config.num_attention_heads

    chunks = get_wikitext_chunks(tokenizer, seq_len, n_texts + 2)  # extra for calibration
    CRS = [4.0, 6.0]
    imp = inverted_importance(nl)

    # Build fixed profile from first 2 chunks (calibration set)
    cal_gini = {l: [] for l in range(nl)}
    for tokens in chunks[:2]:
        input_ids = torch.tensor([tokens], device=device)
        with torch.no_grad():
            out = model(input_ids=input_ids, output_attentions=True, return_dict=True)
        attn = list(out.attentions)
        del out; clear_gpu()
        pr = LayerAttentionProfiler().profile_from_attention_weights(attn)
        for l, g in pr.gini_scores().items():
            cal_gini[l].append(g)
        del attn; clear_gpu()

    fixed_sparsity = {l: statistics.mean(cal_gini[l]) for l in range(nl)}
    # Model-default: uniform 0.5 (no profiling at all)
    default_sparsity = {l: 0.5 for l in range(nl)}

    test_chunks = chunks[2:2+n_texts]  # separate test set
    results = {f"{mode}@{cr}": [] for mode in ["online", "fixed", "default"] for cr in CRS}

    for tidx, tokens in enumerate(test_chunks):
        input_ids = torch.tensor([tokens], device=device)
        prefix_len = int(seq_len * 0.6)
        print(f"  [{tidx+1}/{len(test_chunks)}]", end=" ", flush=True)

        with torch.no_grad():
            out = model(input_ids=input_ids[:, :prefix_len], output_attentions=True, return_dict=True)
        full_k, full_v = hf_to_deltacache(out.past_key_values)
        attn = list(out.attentions)
        del out; clear_gpu()

        online_sp = LayerAttentionProfiler().profile_from_attention_weights(attn).gini_scores()

        for cr in CRS:
            for mode, sp in [("online", online_sp), ("fixed", fixed_sparsity), ("default", default_sparsity)]:
                ratio = run_allocation_and_eval(full_k, full_v, sp, imp, nl, nh, hd,
                                                prefix_len, cr, model, input_ids, device)
                results[f"{mode}@{cr}"].append(ratio)

        del full_k, full_v, attn; clear_gpu()
        print("done", flush=True)

    print(f"\n  {'Mode':>10s} {'4x':>8s} {'6x':>8s}")
    print(f"  {'-'*30}")
    summary = []
    for mode in ["online", "fixed", "default"]:
        r4 = statistics.mean(results[f"{mode}@4.0"])
        r6 = statistics.mean(results[f"{mode}@6.0"])
        print(f"  {mode:>10s} {r4:>8.4f} {r6:>8.4f}")
        summary.append({"mode": mode, "4x": round(r4, 4), "6x": round(r6, 4)})

    return summary


# ═══════════════════════════════════════════════════════════════════
#  #23: Greedy optimality vs random search
# ═══════════════════════════════════════════════════════════════════

def exp_greedy_vs_random(model, tokenizer, model_short, seq_len=512, n_texts=2):
    """Compare greedy allocator quality score vs random search."""
    print(f"\n{'='*70}")
    print(f"  #23: Greedy vs Random Search ({model_short})")
    print(f"{'='*70}")

    device = next(model.parameters()).device
    nl = model.config.num_hidden_layers
    nh = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    hd = model.config.hidden_size // model.config.num_attention_heads

    chunks = get_wikitext_chunks(tokenizer, seq_len, n_texts)
    CRS = [4.0, 6.0]
    imp = inverted_importance(nl)
    N_RANDOM = 500

    results = []

    for tidx, tokens in enumerate(chunks):
        input_ids = torch.tensor([tokens], device=device)
        prefix_len = int(seq_len * 0.6)
        print(f"  [{tidx+1}/{len(chunks)}]", end=" ", flush=True)

        with torch.no_grad():
            out = model(input_ids=input_ids[:, :prefix_len], output_attentions=True, return_dict=True)
        full_k, full_v = hf_to_deltacache(out.past_key_values)
        attn = list(out.attentions)
        del out; clear_gpu()

        sparsity = LayerAttentionProfiler().profile_from_attention_weights(attn).gini_scores()
        allocator = LayerBudgetAllocator(nl, nh, hd)

        for cr in CRS:
            fm = allocator.full_memory(prefix_len)
            budget = int(fm / cr)

            # Greedy allocation quality score
            greedy_alloc = allocator.allocate(sparsity, imp, budget, prefix_len)
            greedy_score = sum(
                allocator._quality_score(a.layer_idx, a.token_budget, a.quant_bits,
                                         prefix_len, sparsity, imp)
                for a in greedy_alloc.allocations
            )

            # Random search: sample N_RANDOM random allocations within budget
            best_random_score = 0
            bits_options = [4, 8, 16]
            for _ in range(N_RANDOM):
                # Random per-layer (tokens, bits) within budget
                trial_tokens = []
                trial_bits = []
                remaining = budget
                for l in range(nl):
                    b = random.choice(bits_options)
                    max_n = min(prefix_len, remaining * 8 // (2 * nh * hd * b))
                    n = random.randint(20, max(20, max_n)) if max_n >= 20 else 20
                    cost = allocator.memory_cost(n, b)
                    if remaining >= cost:
                        trial_tokens.append(n)
                        trial_bits.append(b)
                        remaining -= cost
                    else:
                        trial_tokens.append(20)
                        trial_bits.append(4)
                        remaining -= allocator.memory_cost(20, 4)

                score = sum(
                    allocator._quality_score(l, trial_tokens[l], trial_bits[l],
                                             prefix_len, sparsity, imp)
                    for l in range(nl)
                )
                best_random_score = max(best_random_score, score)

            ratio = greedy_score / max(best_random_score, 1e-10)
            print(f"{cr}x:g={greedy_score:.3f}/r={best_random_score:.3f}={ratio:.3f}", end=" ")
            results.append({"text": tidx, "cr": cr,
                           "greedy_score": round(greedy_score, 4),
                           "best_random_score": round(best_random_score, 4),
                           "greedy_vs_random": round(ratio, 4)})

        del full_k, full_v, attn; clear_gpu()
        print("done", flush=True)

    # Summary
    print(f"\n  {'CR':>5s} {'Greedy':>10s} {'BestRand':>10s} {'G/R Ratio':>10s}")
    print(f"  {'-'*40}")
    for cr in CRS:
        entries = [r for r in results if r["cr"] == cr]
        g = statistics.mean([r["greedy_score"] for r in entries])
        r = statistics.mean([r["best_random_score"] for r in entries])
        ratio = g / max(r, 1e-10)
        print(f"  {cr:>5.0f}x {g:>10.4f} {r:>10.4f} {ratio:>9.1%}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--model-short", default=None)
    parser.add_argument("--n-texts", type=int, default=4)
    parser.add_argument("--exps", nargs="+", default=["22", "23"], choices=["22", "23"])
    args = parser.parse_args()

    model_short = args.model_short or args.model.split("/")[-1]
    model, tokenizer = load_model_4bit(args.model)

    output = {"metadata": {"model": args.model, "model_short": model_short,
                           "importance": "inverted",
                           "timestamp": datetime.now().isoformat()}}

    if "22" in args.exps:
        output["fixed_vs_online"] = exp_fixed_vs_online(model, tokenizer, model_short, n_texts=args.n_texts)

    if "23" in args.exps:
        output["greedy_vs_random"] = exp_greedy_vs_random(model, tokenizer, model_short, n_texts=min(args.n_texts, 2))

    safe = model_short.replace("/", "_").replace("-", "_").lower()
    path = RESULTS_DIR / f"prof_opt_{safe}_{datetime.now():%Y%m%d_%H%M}.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
