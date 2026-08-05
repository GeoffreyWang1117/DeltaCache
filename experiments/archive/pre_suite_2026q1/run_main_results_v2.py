#!/usr/bin/env python3
"""Re-run main results table with inverted importance + both fill modes.

Produces the data for an updated Table 1 with:
  - LayerBudget (inverted importance) vs 12 baselines
  - Both zero-fill and mean-fill evaluation
  - Mistral-7B and Llama-2-7B at 512 tokens
"""

import gc, json, math, statistics, sys, time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from baselines import REGISTRY
from deltacache.core.layer_profiler import LayerAttentionProfiler
from deltacache.core.layer_budget_allocator import LayerBudgetAllocator
from deltacache.core.layer_kv_store import LayerKVStore
from deltacache.hf_integration.kv_format import hf_to_deltacache, deltacache_to_hf

RESULTS_DIR = Path(__file__).parent / "results" / "main_v2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

METHODS = [
    "full_kv", "h2o_uniform", "snapkv", "pyramidkv", "d2o",
    "squeeze_attention", "adakv", "dynamickv", "lava", "cake",
    "kvtuner", "kivi_uniform",
    "layer_budget_inverted",  # our method with inverted importance
]
CRS = [2.0, 3.0, 4.0, 6.0]


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()


def get_wikitext_chunks(tokenizer, target_len, n=6):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    all_text = " ".join([t for t in ds["text"] if len(t.strip()) > 100])
    all_ids = tokenizer.encode(all_text)
    return [all_ids[i:i+target_len] for i in range(0, len(all_ids)-target_len, target_len)][:n]


def build_cache(layers_data, full_seq_len, device, fill="zero"):
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    for li, (k, v, idx) in enumerate(layers_data):
        k, v = k.to(device), v.to(device)
        if k.dim() == 3: k, v = k.unsqueeze(0), v.unsqueeze(0)
        nt, nh, hd = k.shape[1], k.shape[2], k.shape[3]
        if nt == full_seq_len:
            cache.update(k.transpose(1,2), v.transpose(1,2), li)
            continue
        if fill == "mean":
            km = k[0].mean(dim=0, keepdim=True)
            vm = v[0].mean(dim=0, keepdim=True)
            kf = km.expand(full_seq_len, -1, -1).clone().unsqueeze(0)
            vf = vm.expand(full_seq_len, -1, -1).clone().unsqueeze(0)
        else:
            kf = torch.zeros(1, full_seq_len, nh, hd, dtype=k.dtype, device=device)
            vf = torch.zeros(1, full_seq_len, nh, hd, dtype=v.dtype, device=device)
        ix = idx.long().to(device); valid = ix[ix < full_seq_len]
        if valid.numel() > 0:
            kf[0, valid] = k[0, :valid.numel()]
            vf[0, valid] = v[0, :valid.numel()]
        cache.update(kf.transpose(1,2), vf.transpose(1,2), li)
    return cache


def compute_ppl(model, input_ids, past_kv, prefix_len, device):
    suffix = input_ids[:, prefix_len:]
    if suffix.shape[1] <= 1: return 1.0
    with torch.no_grad():
        pos = torch.arange(prefix_len, prefix_len + suffix.shape[1], device=device).unsqueeze(0)
        out = model(input_ids=suffix, past_key_values=past_kv, position_ids=pos, return_dict=True)
    logits = out.logits[:, :-1, :].contiguous()
    labels = suffix[:, 1:].contiguous()
    return math.exp(min(F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1)).item(), 20))


def inverted_importance(nl):
    sigmoid = LayerBudgetAllocator.compute_importance_weights(nl)
    return {l: sigmoid[nl - 1 - l] for l in range(nl)}


def attn_token_selector(attn_weights):
    from baselines.base import h2o_token_selection
    def selector(layer_idx, keys, values, n_tokens):
        seq_len = keys.shape[1]
        layer_attn = attn_weights[layer_idx] if layer_idx < len(attn_weights) else None
        return h2o_token_selection(keys, values, n_tokens, seq_len, attention_weights=layer_attn)
    return selector


def do_compress(method, full_k, full_v, attn, prefix_len, cr, nl, nh, hd):
    if method == "full_kv":
        layers = [(full_k[l:l+1], full_v[l:l+1], torch.arange(prefix_len)) for l in range(nl)]
        return layers, full_k.numel() * 2 * 2
    elif method == "layer_budget_inverted":
        profiler = LayerAttentionProfiler()
        allocator = LayerBudgetAllocator(nl, nh, hd)
        store = LayerKVStore(nl, nh, hd)
        pr = profiler.profile_from_attention_weights(attn)
        sp = pr.gini_scores()
        imp = inverted_importance(nl)  # KEY CHANGE: inverted
        fm = allocator.full_memory(prefix_len)
        alloc = allocator.allocate(sp, imp, int(fm / cr), prefix_len)
        store.store_from_full_cache(full_k, full_v, alloc.allocations,
                                     token_selector=attn_token_selector(attn))
        return store.get_all_layers(), store.memory_usage()
    elif method in REGISTRY:
        cls = REGISTRY[method]
        bl = cls(nl, nh, hd)
        kw = {}
        if cls.requires_attention:
            kw["attention_weights"] = attn
        if getattr(cls, "requires_hidden_states", False):
            kw["hidden_states"] = None
        layers = bl.compress(full_k, full_v, cr, **kw)
        return layers, bl.memory_bytes(layers)
    return None, 0


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


def run_config(model, tokenizer, model_short, target_len, n_texts=6):
    device = next(model.parameters()).device
    nl = model.config.num_hidden_layers
    nh = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    hd = model.config.hidden_size // model.config.num_attention_heads

    print(f"\n{'='*80}")
    print(f"  MAIN RESULTS v2: {model_short} | {target_len}tok | inverted importance")
    print(f"{'='*80}")

    chunks = get_wikitext_chunks(tokenizer, target_len, n_texts)
    print(f"  {len(chunks)} sequences x {target_len} tokens")

    # Track results for both fill modes
    results = {"zero": {}, "mean": {}}

    for tidx, tokens in enumerate(chunks):
        input_ids = torch.tensor([tokens], device=device)
        seq_len = len(tokens)
        prefix_len = int(seq_len * 0.6)
        print(f"  [{tidx+1}/{len(chunks)}] prefix={prefix_len}", end=" ", flush=True)

        try:
            with torch.no_grad():
                out = model(input_ids=input_ids[:, :prefix_len], output_attentions=True, return_dict=True)
            full_k, full_v = hf_to_deltacache(out.past_key_values)
            attn = list(out.attentions)
            del out; clear_gpu()
        except Exception as e:
            print(f"PREFILL ERR: {e}"); clear_gpu(); continue

        # Reference PPL
        ref_kv = deltacache_to_hf(full_k, full_v, add_batch_dim=True)
        ref_ppl = compute_ppl(model, input_ids, ref_kv, prefix_len, device)
        del ref_kv; clear_gpu()
        print(f"ref={ref_ppl:.2f}", end=" ", flush=True)

        for cr in CRS:
            for m in METHODS:
                key = f"{m}@{cr}"
                actual_cr = 1.0 if m == "full_kv" else cr
                try:
                    layers, mem = do_compress(m, full_k, full_v, attn, prefix_len, actual_cr, nl, nh, hd)
                    if layers is None: continue

                    for fill in ["zero", "mean"]:
                        fkey = f"{m}@{cr}"
                        if fkey not in results[fill]:
                            results[fill][fkey] = {"m": m, "cr": cr, "ppls": [], "ratios": [], "mems": []}
                        pkv = build_cache(layers, prefix_len, device, fill=fill)
                        p = compute_ppl(model, input_ids, pkv, prefix_len, device)
                        results[fill][fkey]["ppls"].append(p)
                        results[fill][fkey]["ratios"].append(p / max(ref_ppl, 1e-6))
                        results[fill][fkey]["mems"].append(mem)
                        del pkv

                    del layers
                except Exception as e:
                    pass
                clear_gpu()

        del full_k, full_v, attn; clear_gpu()
        print("done", flush=True)

    # Print results for both fill modes
    all_summary = {}
    for fill in ["zero", "mean"]:
        print(f"\n  === {fill.upper()}-FILL ===")
        print(f"  {'Method':25s} {'CR':>4s} {'PPL':>8s} {'Ratio':>8s} {'Mem(KB)':>8s}")
        print(f"  {'-'*58}")
        summary_rows = []
        for cr in CRS:
            entries = [(k, v) for k, v in results[fill].items() if v["cr"] == cr and v["ppls"]]
            entries.sort(key=lambda x: statistics.mean(x[1]["ratios"]))
            for key, v in entries:
                avg_ppl = statistics.mean(v["ppls"])
                avg_r = statistics.mean(v["ratios"])
                avg_mem = statistics.mean(v["mems"]) / 1024
                marker = " ***" if "layer_budget" in v["m"] else ""
                print(f"  {v['m']:25s} {cr:>4.0f}x {avg_ppl:>8.2f} {avg_r:>8.4f} {avg_mem:>7.0f}KB{marker}")
                summary_rows.append({
                    "method": v["m"], "cr": cr, "fill": fill,
                    "mean_ppl": round(avg_ppl, 4),
                    "std_ppl": round(statistics.stdev(v["ppls"]), 4) if len(v["ppls"]) > 1 else 0,
                    "mean_ratio": round(avg_r, 4),
                    "mean_mem_kb": round(avg_mem, 1),
                    "n": len(v["ppls"]),
                })
            print()
        all_summary[fill] = summary_rows

    # Save
    output = {
        "metadata": {
            "experiment": "main_results_v2_inverted_importance",
            "model": model_short,
            "seq_len": target_len,
            "importance": "inverted",
            "n_texts": len(chunks),
            "num_layers": nl, "num_heads": nh, "head_dim": hd,
            "timestamp": datetime.now().isoformat(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        },
        "summary_zero_fill": all_summary["zero"],
        "summary_mean_fill": all_summary["mean"],
    }
    safe = model_short.replace("/", "_").replace("-", "_").lower()
    path = RESULTS_DIR / f"main_v2_{target_len}tok_{safe}.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {path.name}")
    return output


if __name__ == "__main__":
    configs = [
        ("mistralai/Mistral-7B-Instruct-v0.2", "Mistral-7B", [512]),
        ("meta-llama/Llama-2-7b-chat-hf", "Llama-2-7B", [512]),
    ]
    for model_name, model_short, seq_lens in configs:
        model, tokenizer = load_model_4bit(model_name)
        for sl in seq_lens:
            try:
                run_config(model, tokenizer, model_short, sl, n_texts=6)
            except Exception as e:
                print(f"\n  FATAL: {model_short}@{sl}: {e}")
                import traceback; traceback.print_exc()
            clear_gpu()
        del model, tokenizer; clear_gpu()
    print("\n\nAll main results v2 complete!")
