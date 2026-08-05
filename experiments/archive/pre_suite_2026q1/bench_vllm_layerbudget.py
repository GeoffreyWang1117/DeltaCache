#!/usr/bin/env python3
"""Benchmark: LayerBudgetBlockManager with vLLM-style block cache.

Simulates vLLM's PagedAttention block structure using a real HF model,
then applies LayerBudget block-level compression and measures PPL impact.

Usage:
  python bench_vllm_layerbudget.py
  python bench_vllm_layerbudget.py --model mistralai/Mistral-7B-Instruct-v0.2 --load-in-4bit
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

from bench_e2e_real import load_model, get_wikitext_chunks, compute_ppl, clear_gpu, gpu_mem_mb


def kv_to_vllm_blocks(full_k, full_v, block_size: int):
    """Convert DeltaCache-format KV to vLLM block format.

    Input:  full_k/full_v shape: (num_layers, seq_len, num_kv_heads, head_dim)
    Output: list of per-layer tensors, each (2, num_blocks, block_size, H, D)
    """
    nl, seq_len, nh, hd = full_k.shape
    num_blocks = (seq_len + block_size - 1) // block_size
    # Pad to full blocks
    pad = num_blocks * block_size - seq_len

    cache = []
    for l in range(nl):
        k = full_k[l]  # (seq_len, H, D)
        v = full_v[l]
        if pad > 0:
            k = F.pad(k, (0, 0, 0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, 0, 0, pad))
        # Reshape to blocks
        k = k.reshape(num_blocks, block_size, nh, hd)
        v = v.reshape(num_blocks, block_size, nh, hd)
        # Stack K,V: (2, num_blocks, block_size, H, D)
        layer_cache = torch.stack([k, v], dim=0)
        cache.append(layer_cache)
    return cache


def vllm_blocks_to_hf(block_cache, seq_len: int):
    """Convert vLLM block cache back to HF DynamicCache for PPL evaluation."""
    from transformers.cache_utils import DynamicCache
    hf_cache = DynamicCache()

    for layer_idx, layer_cache in enumerate(block_cache):
        # layer_cache: (2, num_blocks, block_size, H, D)
        k_blocks = layer_cache[0]  # (num_blocks, block_size, H, D)
        v_blocks = layer_cache[1]

        # Flatten blocks -> (total_tokens, H, D) then trim to seq_len
        k = k_blocks.reshape(-1, k_blocks.shape[2], k_blocks.shape[3])[:seq_len]
        v = v_blocks.reshape(-1, v_blocks.shape[2], v_blocks.shape[3])[:seq_len]

        # HF format: (batch, heads, seq, dim)
        k = k.unsqueeze(0).transpose(1, 2)
        v = v.unsqueeze(0).transpose(1, 2)
        hf_cache.update(k, v, layer_idx)

    return hf_cache


def run_benchmark(model, tokenizer, seq_len: int = 512, n_texts: int = 4,
                  block_size: int = 16):
    """Run block-level LayerBudget benchmark."""
    from deltacache.vllm_integration.layer_budget_block_manager import (
        LayerBudgetBlockManager,
    )
    from deltacache.hf_integration.kv_format import hf_to_deltacache

    device = next(model.parameters()).device
    nl = model.config.num_hidden_layers
    nh = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    hd = model.config.hidden_size // model.config.num_attention_heads

    mgr = LayerBudgetBlockManager(
        num_layers=nl, num_kv_heads=nh, head_dim=hd,
        block_size=block_size, sink_blocks=1, recent_blocks=2,
    )

    print(f"\n{'='*70}")
    print(f"  vLLM Block-Level LayerBudget: {nl}L {nh}H {hd}D | block_size={block_size}")
    print(f"  seq_len={seq_len} → {(seq_len + block_size - 1) // block_size} blocks/layer")
    print(f"{'='*70}")

    chunks = get_wikitext_chunks(tokenizer, seq_len, n_texts)
    CRS = [2.0, 3.0, 4.0, 6.0]

    all_results: Dict[str, Dict] = {}

    for tidx, tokens in enumerate(chunks):
        input_ids = torch.tensor([tokens], device=device)
        prefix_len = int(seq_len * 0.6)
        print(f"\n  [{tidx+1}/{len(chunks)}] prefix={prefix_len}", end=" ", flush=True)

        # Prefill
        try:
            clear_gpu()
            with torch.no_grad():
                out = model(
                    input_ids=input_ids[:, :prefix_len],
                    output_attentions=True,
                    return_dict=True,
                )
            full_k, full_v = hf_to_deltacache(out.past_key_values)
            attn = list(out.attentions)
            del out
            clear_gpu()
        except Exception as e:
            print(f"PREFILL ERR: {e}")
            continue

        # Reference PPL (full KV, no blocks)
        from deltacache.hf_integration.kv_format import deltacache_to_hf
        ref_kv = deltacache_to_hf(full_k, full_v, add_batch_dim=True)
        ref_ppl = compute_ppl(model, input_ids, ref_kv, prefix_len, device)
        del ref_kv
        clear_gpu()
        print(f"ref={ref_ppl:.2f}", end=" ", flush=True)

        # Convert to vLLM blocks
        block_cache_orig = kv_to_vllm_blocks(full_k, full_v, block_size)
        num_blocks = block_cache_orig[0].shape[1]
        full_mem = mgr.full_memory(num_blocks)

        for cr in CRS:
            key = f"block_lb@{cr}"
            if key not in all_results:
                all_results[key] = {"cr": cr, "ppls": [], "ratios": [],
                                    "compress_ms": [], "freed_pct": []}

            budget = int(full_mem / cr)

            # Clone and compress
            block_cache = [t.clone() for t in block_cache_orig]

            t0 = time.perf_counter()
            alloc = mgr.profile_and_allocate(attn, prefix_len, budget)
            mgr.apply_compression(block_cache, alloc)
            compress_ms = (time.perf_counter() - t0) * 1000

            # Evaluate PPL
            hf_kv = vllm_blocks_to_hf(block_cache, prefix_len)
            ppl = compute_ppl(model, input_ids, hf_kv, prefix_len, device)

            total_freed = sum(len(v) for v in alloc.freed_block_indices.values())
            total_blocks = num_blocks * nl
            freed_pct = total_freed / total_blocks if total_blocks > 0 else 0

            all_results[key]["ppls"].append(ppl)
            all_results[key]["ratios"].append(ppl / max(ref_ppl, 1e-6))
            all_results[key]["compress_ms"].append(compress_ms)
            all_results[key]["freed_pct"].append(freed_pct)

            del block_cache, hf_kv
            clear_gpu()

        del full_k, full_v, attn, block_cache_orig
        clear_gpu()
        print("OK", flush=True)

    # Print summary
    print(f"\n  {'CR':>4s} {'PPL':>8s} {'Ratio':>8s} {'Comp(ms)':>9s} {'Freed%':>7s}")
    print(f"  {'-'*42}")
    summary = []
    for cr in CRS:
        key = f"block_lb@{cr}"
        v = all_results.get(key, {})
        if not v.get("ppls"):
            continue
        avg_ppl = statistics.mean(v["ppls"])
        avg_r = statistics.mean(v["ratios"])
        avg_t = statistics.mean(v["compress_ms"])
        avg_f = statistics.mean(v["freed_pct"])
        print(f"  {cr:>4.0f}x {avg_ppl:>8.2f} {avg_r:>8.4f} {avg_t:>8.1f}ms {avg_f:>6.1%}")
        summary.append({
            "cr": cr, "mean_ppl": round(avg_ppl, 4),
            "mean_ratio": round(avg_r, 4),
            "mean_compress_ms": round(avg_t, 2),
            "mean_freed_pct": round(avg_f, 4),
            "n": len(v["ppls"]),
        })

    return summary


def main():
    parser = argparse.ArgumentParser(description="vLLM block-level LayerBudget benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2-0.5B")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--n-texts", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=16)
    args = parser.parse_args()

    RESULTS_DIR = Path(__file__).parent / "results" / "e2e"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(args.model, args.load_in_4bit)

    summary = run_benchmark(model, tokenizer, seq_len=args.seq_len,
                            n_texts=args.n_texts, block_size=args.block_size)

    output = {
        "metadata": {
            "model": args.model,
            "seq_len": args.seq_len,
            "block_size": args.block_size,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "timestamp": datetime.now().isoformat(),
        },
        "block_layerbudget_summary": summary,
    }

    safe = args.model.replace("/", "_").replace("-", "_").lower()
    path = RESULTS_DIR / f"vllm_block_lb_{safe}_{datetime.now():%Y%m%d_%H%M}.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
