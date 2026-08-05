#!/usr/bin/env python3
"""Measure LayerBudget profiling overhead vs standard prefill.

Measures:
  1. Standard prefill time (no profiling hooks)
  2. Prefill + Gini profiling time (with hooks)
  3. Allocation time (greedy solver)
  4. Total LayerBudget pipeline time

Across sequence lengths: 128, 256, 512, 1024, 2048.
"""

import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from deltacache.core.layer_profiler import LayerAttentionProfiler
from deltacache.core.layer_budget_allocator import LayerBudgetAllocator
from deltacache.core.layer_kv_store import LayerKVStore

RESULTS_DIR = Path(__file__).parent / "results" / "paper"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def measure_overhead(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device="cuda:0",
    seq_lengths=None,
    n_warmup=2,
    n_trials=5,
):
    if seq_lengths is None:
        seq_lengths = [128, 256, 512, 1024]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    num_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    print(f"Model: {num_layers}L, {num_heads}H, {head_dim}D")

    results = []

    for seq_len in seq_lengths:
        print(f"\n  seq_len={seq_len}")

        # Generate random input of target length
        input_ids = torch.randint(100, 30000, (1, seq_len), device=device)

        # --- 1. Standard prefill (no hooks, no output_attentions) ---
        times_standard = []
        for i in range(n_warmup + n_trials):
            clear_gpu()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(input_ids=input_ids, return_dict=True)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            if i >= n_warmup:
                times_standard.append((t1 - t0) * 1000)
        avg_standard = sum(times_standard) / len(times_standard)

        # --- 2. Prefill with output_attentions (needed for profiling) ---
        times_with_attn = []
        for i in range(n_warmup + n_trials):
            clear_gpu()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(input_ids=input_ids, output_attentions=True, return_dict=True)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            if i >= n_warmup:
                times_with_attn.append((t1 - t0) * 1000)
        avg_with_attn = sum(times_with_attn) / len(times_with_attn)

        # --- 3. Gini computation from attention weights ---
        attention_weights = list(out.attentions)
        profiler = LayerAttentionProfiler()

        times_gini = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            profile_result = profiler.profile_from_attention_weights(attention_weights)
            gini = profile_result.gini_scores()
            t1 = time.perf_counter()
            times_gini.append((t1 - t0) * 1000)
        avg_gini = sum(times_gini) / len(times_gini)

        # --- 4. Allocator ---
        allocator = LayerBudgetAllocator(num_layers, num_heads, head_dim)
        importance = allocator.compute_importance_weights(num_layers)
        full_mem = allocator.full_memory(seq_len)
        budget = int(full_mem / 3.0)

        times_alloc = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            allocation = allocator.allocate(gini, importance, budget, seq_len)
            t1 = time.perf_counter()
            times_alloc.append((t1 - t0) * 1000)
        avg_alloc = sum(times_alloc) / len(times_alloc)

        # --- 5. Token selection + storage ---
        store = LayerKVStore(num_layers, num_heads, head_dim)
        full_keys = torch.randn(num_layers, seq_len, num_heads, head_dim,
                                dtype=torch.float16, device=device)
        full_values = torch.randn_like(full_keys)

        times_store = []
        for _ in range(n_trials):
            store.clear()
            t0 = time.perf_counter()
            store.store_from_full_cache(full_keys, full_values, allocation.allocations)
            t1 = time.perf_counter()
            times_store.append((t1 - t0) * 1000)
        avg_store = sum(times_store) / len(times_store)

        del full_keys, full_values, out, attention_weights
        clear_gpu()

        # Compute overheads
        attn_overhead_pct = ((avg_with_attn - avg_standard) / avg_standard) * 100
        total_pipeline = avg_gini + avg_alloc + avg_store
        total_overhead_pct = ((avg_with_attn - avg_standard + total_pipeline) / avg_standard) * 100

        entry = {
            "seq_len": seq_len,
            "prefill_ms": round(avg_standard, 2),
            "prefill_with_attn_ms": round(avg_with_attn, 2),
            "attn_overhead_pct": round(attn_overhead_pct, 2),
            "gini_ms": round(avg_gini, 2),
            "allocator_ms": round(avg_alloc, 2),
            "store_ms": round(avg_store, 2),
            "pipeline_ms": round(total_pipeline, 2),
            "total_overhead_pct": round(total_overhead_pct, 2),
        }
        results.append(entry)

        print(f"    Prefill: {avg_standard:.1f}ms | +attn: {avg_with_attn:.1f}ms "
              f"({attn_overhead_pct:+.1f}%) | Gini: {avg_gini:.2f}ms | "
              f"Alloc: {avg_alloc:.2f}ms | Store: {avg_store:.1f}ms | "
              f"Total overhead: {total_overhead_pct:.1f}%")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"Profiling Overhead Summary ({model_name})")
    print(f"{'='*70}")
    print(f"{'SeqLen':>8s} {'Prefill':>10s} {'Attn OH%':>10s} {'Gini':>8s} "
          f"{'Alloc':>8s} {'Store':>8s} {'Total OH%':>10s}")
    print(f"{'-'*62}")
    for r in results:
        print(f"{r['seq_len']:>8d} {r['prefill_ms']:>9.1f}ms {r['attn_overhead_pct']:>9.1f}% "
              f"{r['gini_ms']:>7.2f}ms {r['allocator_ms']:>7.2f}ms {r['store_ms']:>7.1f}ms "
              f"{r['total_overhead_pct']:>9.1f}%")

    output = {
        "metadata": {
            "experiment": "profiling_overhead",
            "model": model_name,
            "num_layers": num_layers,
            "n_warmup": n_warmup,
            "n_trials": n_trials,
            "timestamp": datetime.now().isoformat(),
            "gpu": torch.cuda.get_device_name(0),
        },
        "results": results,
    }

    safe_model = model_name.split("/")[-1].lower().replace("-", "_")
    out_path = RESULTS_DIR / f"profiling_overhead_{safe_model}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    measure_overhead(model_name=args.model, device=args.device)
