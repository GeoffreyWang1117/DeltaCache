#!/usr/bin/env python3
"""#15: Test allocator fixes for the early-layer under-budget problem.

Diagnosis from #14: layers 0-10 are CRITICAL but get lowest budget because
importance sigmoid prioritizes late layers. Fixes tested:

  A. Token floor: minimum retention per layer (10%-30%)
  B. Flat importance: w_l = 0.5 for all layers (no monotone bias)
  C. U-shaped importance: high for early + late, low for middle
  D. Inverted importance: earlier layers get higher weight (reverses sigmoid)
  E. Capped coverage exponent: limit (1-gini) to prevent aggressive eviction

Run on Mistral-7B at 4x/5x/6x and Llama-2-7B at 4x/5x/6x.
"""

import argparse, gc, json, math, statistics, sys, time
from datetime import datetime
from pathlib import Path
import torch, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from deltacache.core.layer_profiler import LayerAttentionProfiler
from deltacache.core.layer_budget_allocator import LayerBudgetAllocator
from deltacache.core.layer_kv_store import LayerKVStore
from deltacache.hf_integration.kv_format import hf_to_deltacache, deltacache_to_hf

RESULTS_DIR = Path(__file__).parent / "results" / "allocator_fixes"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

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


def make_importance_variants(nl):
    """Generate different importance weight schemes."""
    sigmoid = LayerBudgetAllocator.compute_importance_weights(nl)  # default

    # Flat: equal for all layers
    flat = {l: 0.5 for l in range(nl)}

    # U-shaped: high for early + late, low for middle
    u_shaped = {}
    for l in range(nl):
        pos = l / max(1, nl - 1)
        u_shaped[l] = 0.3 + 0.7 * (4 * (pos - 0.5) ** 2)  # min at center

    # Inverted sigmoid: early layers get higher weight
    inverted = {l: sigmoid[nl - 1 - l] for l in range(nl)}

    return {
        "sigmoid (default)": sigmoid,
        "flat": flat,
        "u_shaped": u_shaped,
        "inverted": inverted,
    }


def run_experiment(model, tokenizer, model_short, seq_len=512, n_texts=4):
    print(f"\n{'='*70}")
    print(f"  #15: Allocator Fixes ({model_short})")
    print(f"{'='*70}")

    device = next(model.parameters()).device
    nl = model.config.num_hidden_layers
    nh = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    hd = model.config.hidden_size // model.config.num_attention_heads

    chunks = get_wikitext_chunks(tokenizer, seq_len, n_texts)
    CRS = [4.0, 5.0, 6.0]
    importance_variants = make_importance_variants(nl)
    token_floors = [None, 0.15, 0.25]

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

        for cr in CRS:
            for imp_name, imp_weights in importance_variants.items():
                for floor in token_floors:
                    floor_str = f"_floor{floor}" if floor else ""
                    key = f"{imp_name}{floor_str}@{cr}"
                    if key not in results:
                        results[key] = {"imp": imp_name, "floor": floor, "cr": cr, "ratios": []}

                    sink = 4
                    recent = 16
                    if floor is not None:
                        sink = max(sink, int(prefix_len * floor * 0.3))
                        recent = max(recent, int(prefix_len * floor * 0.7))

                    allocator = LayerBudgetAllocator(nl, nh, hd,
                                                     sink_tokens=sink, recent_tokens=recent)
                    store = LayerKVStore(nl, nh, hd)
                    fm = allocator.full_memory(prefix_len)
                    alloc = allocator.allocate(sparsity, imp_weights, int(fm / cr), prefix_len)
                    store.store_from_full_cache(full_k, full_v, alloc.allocations)
                    layers = store.get_all_layers()
                    pkv = build_cache_zerofill(layers, prefix_len, device)
                    ppl = compute_ppl(model, input_ids, pkv, prefix_len, device)
                    results[key]["ratios"].append(ppl / max(ref_ppl, 1e-6))
                    del pkv, layers, store; clear_gpu()

        del full_k, full_v, attn; clear_gpu()
        print(f"ref={ref_ppl:.2f}", flush=True)

    # Summary by CR
    summary = []
    for cr in CRS:
        print(f"\n  --- {cr}x ---")
        print(f"  {'Variant':35s} {'Ratio':>8s}")
        print(f"  {'-'*48}")
        entries = [(k, v) for k, v in results.items() if v["cr"] == cr and v["ratios"]]
        entries.sort(key=lambda x: statistics.mean(x[1]["ratios"]))
        for key, v in entries:
            avg = statistics.mean(v["ratios"])
            label = f"{v['imp']}" + (f" +floor{v['floor']}" if v['floor'] else "")
            best = " *** BEST" if avg == statistics.mean(entries[0][1]["ratios"]) else ""
            print(f"  {label:35s} {avg:>8.4f}{best}")
            summary.append({"variant": label, "cr": cr, "mean_ratio": round(avg, 4)})

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--model-short", default=None)
    parser.add_argument("--n-texts", type=int, default=4)
    args = parser.parse_args()

    model_short = args.model_short or args.model.split("/")[-1]
    model, tokenizer = load_model_4bit(args.model)
    summary = run_experiment(model, tokenizer, model_short, n_texts=args.n_texts)

    safe = model_short.replace("/", "_").replace("-", "_").lower()
    path = RESULTS_DIR / f"fixes_{safe}_{datetime.now():%Y%m%d_%H%M}.json"
    with open(path, "w") as f:
        json.dump({"metadata": {"model": args.model, "model_short": model_short,
                                "timestamp": datetime.now().isoformat()},
                   "summary": summary}, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
