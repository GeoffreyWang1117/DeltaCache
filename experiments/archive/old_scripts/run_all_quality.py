#!/usr/bin/env python3
"""Batch runner: all model × seq_len quality experiments.

Runs unified quality benchmark across all configurations sequentially,
reusing model weights when possible to minimize load time.
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

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from baselines import REGISTRY
from deltacache.core.layer_profiler import LayerAttentionProfiler
from deltacache.core.layer_budget_allocator import LayerBudgetAllocator
from deltacache.core.layer_kv_store import LayerKVStore
from deltacache.hf_integration.kv_format import hf_to_deltacache, deltacache_to_hf

RESULTS_DIR = Path(__file__).parent / "results" / "paper"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

METHODS = [
    "full_kv", "h2o_uniform", "snapkv", "pyramidkv", "d2o",
    "squeeze_attention", "adakv", "dynamickv", "lava", "cake",
    "kvtuner", "kivi_uniform", "layer_budget",
]
CRS = [2.0, 3.0, 4.0, 6.0]


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_wikitext_chunks(tokenizer, target_len: int, n_chunks: int = 6):
    """Get n_chunks sequences of target_len tokens from WikiText-2."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    all_text = " ".join([t for t in ds["text"] if len(t.strip()) > 100])
    all_ids = tokenizer.encode(all_text)
    chunks = []
    for i in range(0, len(all_ids) - target_len, target_len):
        chunks.append(all_ids[i : i + target_len])
        if len(chunks) >= n_chunks:
            break
    return chunks


def compute_ppl(model, input_ids, past_kv, prefix_len, device):
    suffix = input_ids[:, prefix_len:]
    if suffix.shape[1] <= 1:
        return 1.0
    with torch.no_grad():
        pos_ids = torch.arange(
            prefix_len, prefix_len + suffix.shape[1], device=device
        ).unsqueeze(0)
        out = model(
            input_ids=suffix,
            past_key_values=past_kv,
            position_ids=pos_ids,
            return_dict=True,
        )
    logits = out.logits[:, :-1, :].contiguous()
    labels = suffix[:, 1:].contiguous()
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.view(-1), reduction="mean"
    )
    return math.exp(min(loss.item(), 20))


def build_cache(layers_data, full_seq_len, device, full_k=None, full_v=None):
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
            if full_k is not None:
                kf[0] = full_k[li].to(device)
                vf[0] = full_v[li].to(device)
            ix = idx.long().to(device)
            valid = ix[ix < full_seq_len]
            vc = valid.shape[0]
            if vc > 0:
                kf[0, valid] = k[0, :vc]
                vf[0, valid] = v[0, :vc]
            cache.update(kf.transpose(1, 2), vf.transpose(1, 2), li)
    return cache


def do_compress(
    method, full_k, full_v, attn, hs, prefix_len, cr, num_layers, num_heads, head_dim
):
    if method == "full_kv":
        layers = [
            (full_k[l : l + 1], full_v[l : l + 1], torch.arange(prefix_len))
            for l in range(num_layers)
        ]
        return layers, full_k.numel() * 2 * 2
    elif method == "layer_budget":
        profiler = LayerAttentionProfiler()
        allocator = LayerBudgetAllocator(num_layers, num_heads, head_dim)
        store = LayerKVStore(num_layers, num_heads, head_dim)
        pr = profiler.profile_from_attention_weights(attn)
        sp = pr.gini_scores()
        imp = allocator.compute_importance_weights(num_layers)
        fm = allocator.full_memory(prefix_len)
        alloc = allocator.allocate(sp, imp, int(fm / cr), prefix_len)
        store.store_from_full_cache(full_k, full_v, alloc.allocations)
        return store.get_all_layers(), store.memory_usage()
    elif method in REGISTRY:
        cls = REGISTRY[method]
        bl = cls(num_layers, num_heads, head_dim)
        kw = {}
        if cls.requires_attention:
            kw["attention_weights"] = attn
        if getattr(cls, "requires_hidden_states", False):
            kw["hidden_states"] = hs
        layers = bl.compress(full_k, full_v, cr, **kw)
        return layers, bl.memory_bytes(layers)
    return None, 0


def run_single_config(
    model, tokenizer, model_name_short, target_len, n_texts=6, prefix_frac=0.6
):
    """Run one (model, seq_len) configuration."""
    device = next(model.parameters()).device
    num_layers = model.config.num_hidden_layers
    num_heads = getattr(
        model.config, "num_key_value_heads", model.config.num_attention_heads
    )
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    print(f"\n{'='*80}")
    print(f"  {model_name_short} | {target_len} tokens | {num_layers}L {num_heads}H {head_dim}D")
    print(f"{'='*80}")

    chunks = get_wikitext_chunks(tokenizer, target_len, n_texts)
    print(f"  {len(chunks)} sequences x {target_len} tokens")

    results: Dict[str, dict] = {}

    for tidx, tokens in enumerate(chunks):
        input_ids = torch.tensor([tokens], device=device)
        seq_len = len(tokens)
        prefix_len = int(seq_len * prefix_frac)
        print(f"  [{tidx+1}/{len(chunks)}] prefix={prefix_len} suffix={seq_len-prefix_len}", end="", flush=True)

        try:
            with torch.no_grad():
                out = model(
                    input_ids=input_ids[:, :prefix_len],
                    output_attentions=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
            full_k, full_v = hf_to_deltacache(out.past_key_values)
            attn = list(out.attentions) if out.attentions else None
            hs = list(out.hidden_states) if out.hidden_states else None
        except Exception as e:
            print(f" PREFILL ERR: {e}")
            clear_gpu()
            continue

        ref_kv = deltacache_to_hf(full_k, full_v, add_batch_dim=True)
        ref_ppl = compute_ppl(model, input_ids, ref_kv, prefix_len, device)
        del ref_kv
        print(f" ref_ppl={ref_ppl:.2f}", end="", flush=True)

        for cr in CRS:
            for m in METHODS:
                key = f"{m}@{cr}"
                if key not in results:
                    results[key] = {"m": m, "cr": cr, "ppls": [], "ratios": [], "mems": []}
                try:
                    actual_cr = 1.0 if m == "full_kv" else cr
                    layers, mem = do_compress(
                        m, full_k, full_v, attn, hs, prefix_len, actual_cr,
                        num_layers, num_heads, head_dim,
                    )
                    if layers is None:
                        continue
                    pkv = build_cache(layers, prefix_len, device, full_k, full_v)
                    ppl = compute_ppl(model, input_ids, pkv, prefix_len, device)
                    results[key]["ppls"].append(ppl)
                    results[key]["ratios"].append(ppl / max(ref_ppl, 1e-6))
                    results[key]["mems"].append(mem)
                    del pkv, layers
                except Exception as e:
                    print(f" ERR:{m}@{cr}", end="", flush=True)

        del out, full_k, full_v, attn, hs
        clear_gpu()
        print(" done", flush=True)

    # Print table
    print(f"\n  {'Method':20s} {'CR':>4s} {'PPL':>8s} {'Ratio':>8s} {'Mem(KB)':>8s}")
    print(f"  {'-'*55}")
    summary_rows = []
    for cr in CRS:
        entries = [(k, v) for k, v in results.items() if v["cr"] == cr and v["ppls"]]
        entries.sort(key=lambda x: statistics.mean(x[1]["ratios"]))
        for rank, (key, v) in enumerate(entries, 1):
            avg_ppl = statistics.mean(v["ppls"])
            avg_r = statistics.mean(v["ratios"])
            avg_mem = statistics.mean(v["mems"]) / 1024
            marker = " ***" if v["m"] == "layer_budget" else ""
            print(f"  {v['m']:20s} {cr:>4.0f}x {avg_ppl:>8.2f} {avg_r:>8.4f} {avg_mem:>7.0f}KB{marker}")
            summary_rows.append({
                "method": v["m"], "cr": cr,
                "mean_ppl": round(avg_ppl, 4),
                "std_ppl": round(statistics.stdev(v["ppls"]), 4) if len(v["ppls"]) > 1 else 0,
                "mean_ratio": round(avg_r, 4),
                "mean_mem_kb": round(avg_mem, 1),
                "n": len(v["ppls"]),
            })
        print()

    # Save
    output = {
        "metadata": {
            "model": model_name_short,
            "seq_len": target_len,
            "prefix_frac": prefix_frac,
            "crs": CRS,
            "methods": METHODS,
            "n_texts": len(chunks),
            "num_layers": num_layers,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "source": "wikitext-2-raw-v1",
            "timestamp": datetime.now().isoformat(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        },
        "summary": summary_rows,
    }
    safe = model_name_short.replace("/", "_").replace("-", "_").lower()
    path = RESULTS_DIR / f"unified_quality_{target_len}tok_{safe}.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {path.name}")
    return output


def load_model_4bit(model_name, device="cuda:0"):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"\nLoading {model_name} (4-bit)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        ),
        device_map="auto",
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="all",
                        help="Comma-separated model:seqlen pairs, or 'all' for full matrix")
    parser.add_argument("--texts", type=int, default=6)
    args = parser.parse_args()

    if args.configs == "all":
        configs = [
            ("meta-llama/Llama-2-7b-chat-hf", [1024]),          # Missing from matrix
            ("mistralai/Mistral-7B-Instruct-v0.2", [512, 1024, 2048]),  # All missing
        ]
    else:
        configs = []
        for c in args.configs.split(","):
            model, seqlen = c.rsplit(":", 1)
            configs.append((model, [int(seqlen)]))

    for model_name, seq_lens in configs:
        model, tokenizer = load_model_4bit(model_name)
        short_name = model_name.split("/")[-1]

        for sl in seq_lens:
            try:
                run_single_config(model, tokenizer, short_name, sl, n_texts=args.texts)
            except Exception as e:
                print(f"\n  FATAL ERROR {short_name}@{sl}: {e}")
                import traceback
                traceback.print_exc()
            clear_gpu()

        # Free model before loading next
        del model, tokenizer
        clear_gpu()

    print("\n\nAll experiments complete!")


if __name__ == "__main__":
    main()
