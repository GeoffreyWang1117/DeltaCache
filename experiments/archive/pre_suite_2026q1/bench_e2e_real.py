#!/usr/bin/env python3
"""End-to-end benchmark: LayerBudget on a real LLM.

Tests two integration paths:
  1. LayerBudgetCache (drop-in DynamicCache subclass) with model.generate()
  2. Manual pipeline: profile → allocate → compress → evaluate PPL

Supports any HuggingFace model. Auto-computes budgets from model dimensions.

Usage:
  python bench_e2e_real.py                                          # Qwen2-0.5B (default)
  python bench_e2e_real.py --model mistralai/Mistral-7B-Instruct-v0.2 --load-in-4bit
  python bench_e2e_real.py --model meta-llama/Llama-2-7b-chat-hf --load-in-4bit --seq-len 1024
"""

import argparse
import gc
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# ── Helpers ──────────────────────────────────────────────────────────────


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def gpu_mem_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1024 ** 2


@dataclass
class ModelBudgets:
    """Auto-computed budgets based on model KV cache dimensions."""
    full_kv_per_token_bytes: int   # bytes per token for full KV cache
    full_kv_1024_mb: float         # full KV at 1024 tokens (MB)
    dropin_budget_mb: float        # budget for drop-in generate test
    generation_budgets_mb: List[float]  # budgets for generation quality test
    scaling_budget_mb: float       # budget for scaling test


def auto_budget(num_layers: int, num_kv_heads: int, head_dim: int) -> ModelBudgets:
    """Compute appropriate budgets from model KV dimensions."""
    # Full KV per token: 2 (K+V) * num_layers * num_kv_heads * head_dim * 2 (fp16 bytes)
    kv_per_token = 2 * num_layers * num_kv_heads * head_dim * 2
    full_kv_1024 = kv_per_token * 1024 / (1024 ** 2)  # MB

    # Drop-in budget: ~25% of full KV at 1024 tokens
    dropin = max(0.25, round(full_kv_1024 * 0.25, 1))

    # Generation quality: 4 budgets from aggressive to no-compression
    gen_budgets = [
        max(0.1, round(full_kv_1024 * 0.1, 2)),   # aggressive (~10%)
        max(0.25, round(full_kv_1024 * 0.25, 2)),  # moderate (~25%)
        max(0.5, round(full_kv_1024 * 0.5, 2)),    # light (~50%)
        round(full_kv_1024 * 2, 1),                 # no compression
    ]

    return ModelBudgets(
        full_kv_per_token_bytes=kv_per_token,
        full_kv_1024_mb=round(full_kv_1024, 2),
        dropin_budget_mb=dropin,
        generation_budgets_mb=gen_budgets,
        scaling_budget_mb=dropin,
    )


def load_model(model_name: str, load_in_4bit: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_name} ({'4-bit' if load_in_4bit else 'fp16'})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"trust_remote_code": True, "attn_implementation": "eager"}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        )
        kwargs["device_map"] = {"": "cuda:0"}
    else:
        kwargs["dtype"] = torch.float16
        kwargs["device_map"] = {"": "cuda:0"}

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    print(f"  Loaded. GPU mem: {gpu_mem_mb():.0f} MB")
    return model, tokenizer


def get_wikitext_chunks(tokenizer, target_len: int, n: int = 8):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    all_text = " ".join([t for t in ds["text"] if len(t.strip()) > 100])
    all_ids = tokenizer.encode(all_text)
    chunks = []
    for i in range(0, len(all_ids) - target_len, target_len):
        chunks.append(all_ids[i : i + target_len])
        if len(chunks) >= n:
            break
    return chunks


def compute_ppl(model, input_ids, past_kv, prefix_len, device):
    """Compute perplexity on suffix tokens given prefix KV cache."""
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


# ── Test 1: LayerBudgetCache drop-in with model.generate() ──────────────

def test_generate_dropin(model, tokenizer, prompts: List[str],
                         budget_mb: float, max_new_tokens: int = 64):
    """Test LayerBudgetCache as a drop-in for model.generate()."""
    from deltacache.integrations.hf_cache import LayerBudgetCache

    device = next(model.parameters()).device
    results = {"baseline": [], "layerbudget": []}

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        prompt_len = input_ids.shape[1]

        attn_mask = torch.ones_like(input_ids)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens, do_sample=False,
            attention_mask=attn_mask,
            pad_token_id=tokenizer.eos_token_id,
        )

        # ── Baseline: standard generate (DynamicCache) ──
        clear_gpu()
        mem_before = gpu_mem_mb()
        t0 = time.perf_counter()
        with torch.no_grad():
            out_base = model.generate(input_ids, **gen_kwargs)
        t_base = (time.perf_counter() - t0) * 1000
        mem_base = gpu_mem_mb() - mem_before
        tokens_base = out_base[0, prompt_len:].tolist()

        # ── LayerBudgetCache: drop-in ──
        clear_gpu()
        mem_before = gpu_mem_mb()
        t0 = time.perf_counter()
        with torch.no_grad():
            cache = LayerBudgetCache(
                max_memory_mb=budget_mb, compression_ratio=0.25,
                sink_tokens=4, recent_tokens=16,
            )
            out_lb = model.generate(
                input_ids, past_key_values=cache, **gen_kwargs,
            )
        t_lb = (time.perf_counter() - t0) * 1000
        mem_lb = gpu_mem_mb() - mem_before
        tokens_lb = out_lb[0, prompt_len:].tolist()

        # Token match rate
        min_len = min(len(tokens_base), len(tokens_lb))
        matches = sum(a == b for a, b in zip(tokens_base[:min_len], tokens_lb[:min_len]))
        match_rate = matches / max(min_len, 1)

        results["baseline"].append({
            "prompt_len": prompt_len,
            "gen_tokens": len(tokens_base),
            "latency_ms": round(t_base, 1),
            "mem_delta_mb": round(mem_base, 1),
        })
        results["layerbudget"].append({
            "prompt_len": prompt_len,
            "gen_tokens": len(tokens_lb),
            "latency_ms": round(t_lb, 1),
            "mem_delta_mb": round(mem_lb, 1),
            "compressed": cache._compressed,
            "budget_mb": budget_mb,
            "token_match_rate": round(match_rate, 4),
        })

        text_base = tokenizer.decode(tokens_base, skip_special_tokens=True)[:80]
        text_lb = tokenizer.decode(tokens_lb, skip_special_tokens=True)[:80]
        print(f"  prompt={prompt_len}tok | base={t_base:.0f}ms | LB={t_lb:.0f}ms | "
              f"match={match_rate:.2%} | compressed={cache._compressed}")
        print(f"    base: {text_base}")
        print(f"    LB:   {text_lb}")

    return results


# ── Test 2: Manual pipeline — PPL evaluation at multiple CRs ────────────

def test_ppl_pipeline(model, tokenizer, seq_len: int = 512, n_texts: int = 6):
    """Manual LayerBudget pipeline: profile → allocate → compress → PPL."""
    from deltacache.core.layer_profiler import LayerAttentionProfiler
    from deltacache.core.layer_budget_allocator import LayerBudgetAllocator
    from deltacache.core.layer_kv_store import LayerKVStore
    from deltacache.hf_integration.kv_format import hf_to_deltacache, deltacache_to_hf

    device = next(model.parameters()).device
    nl = model.config.num_hidden_layers
    nh = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    hd = model.config.hidden_size // model.config.num_attention_heads

    print(f"\n{'='*70}")
    print(f"  PPL Pipeline: {nl}L {nh}H {hd}D | seq_len={seq_len}")
    print(f"{'='*70}")

    chunks = get_wikitext_chunks(tokenizer, seq_len, n_texts)
    print(f"  {len(chunks)} sequences x {seq_len} tokens")

    CRS = [2.0, 3.0, 4.0, 6.0]
    METHODS = ["full_kv", "layer_budget"]
    try:
        from baselines import REGISTRY
        for m in ["h2o_uniform", "snapkv", "kivi_uniform", "adakv"]:
            if m in REGISTRY:
                METHODS.append(m)
    except ImportError:
        pass

    all_results: Dict[str, Dict] = {}

    for tidx, tokens in enumerate(chunks):
        input_ids = torch.tensor([tokens], device=device)
        prefix_len = int(seq_len * 0.6)
        print(f"\n  [{tidx+1}/{len(chunks)}] prefix={prefix_len}", end=" ", flush=True)

        try:
            clear_gpu()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(
                    input_ids=input_ids[:, :prefix_len],
                    output_attentions=True,
                    return_dict=True,
                )
            prefill_ms = (time.perf_counter() - t0) * 1000
            full_k, full_v = hf_to_deltacache(out.past_key_values)
            attn = list(out.attentions)
            del out
            clear_gpu()
        except Exception as e:
            print(f"PREFILL ERR: {e}")
            continue

        ref_kv = deltacache_to_hf(full_k, full_v, add_batch_dim=True)
        ref_ppl = compute_ppl(model, input_ids, ref_kv, prefix_len, device)
        del ref_kv
        clear_gpu()
        print(f"ref_ppl={ref_ppl:.2f} prefill={prefill_ms:.0f}ms", end=" ", flush=True)

        for cr in CRS:
            for method in METHODS:
                key = f"{method}@{cr}"
                if key not in all_results:
                    all_results[key] = {"method": method, "cr": cr, "ppls": [],
                                        "ratios": [], "times_ms": [], "mems_kb": []}
                try:
                    t0 = time.perf_counter()

                    if method == "full_kv":
                        layers = [(full_k[l:l+1], full_v[l:l+1], torch.arange(prefix_len))
                                  for l in range(nl)]
                        mem = full_k.numel() * 2 * 2
                    elif method == "layer_budget":
                        profiler = LayerAttentionProfiler()
                        allocator = LayerBudgetAllocator(nl, nh, hd)
                        store = LayerKVStore(nl, nh, hd)
                        pr = profiler.profile_from_attention_weights(attn)
                        sp = pr.gini_scores()
                        imp = allocator.compute_importance_weights(nl)
                        fm = allocator.full_memory(prefix_len)
                        alloc = allocator.allocate(sp, imp, int(fm / cr), prefix_len)
                        store.store_from_full_cache(full_k, full_v, alloc.allocations)
                        layers = store.get_all_layers()
                        mem = store.memory_usage()
                    elif method in REGISTRY:
                        cls = REGISTRY[method]
                        bl = cls(nl, nh, hd)
                        kw = {}
                        if cls.requires_attention:
                            kw["attention_weights"] = attn
                        layers = bl.compress(full_k, full_v, cr, **kw)
                        mem = bl.memory_bytes(layers)
                    else:
                        continue

                    compress_ms = (time.perf_counter() - t0) * 1000

                    pkv = _build_cache_zerofill(layers, prefix_len, device)
                    ppl = compute_ppl(model, input_ids, pkv, prefix_len, device)

                    all_results[key]["ppls"].append(ppl)
                    all_results[key]["ratios"].append(ppl / max(ref_ppl, 1e-6))
                    all_results[key]["times_ms"].append(compress_ms)
                    all_results[key]["mems_kb"].append(mem / 1024)
                    del pkv, layers
                except Exception as e:
                    print(f"\n    WARN {method}@{cr}: {e}", end="", flush=True)
                clear_gpu()

        del full_k, full_v, attn
        clear_gpu()
        print("OK", flush=True)

    # ── Print summary table ──
    print(f"\n  {'Method':<22s} {'CR':>4s} {'PPL':>8s} {'Ratio':>8s} {'Comp(ms)':>9s} {'Mem(KB)':>8s}")
    print(f"  {'-'*62}")
    summary_rows = []
    for cr in CRS:
        entries = [(k, v) for k, v in all_results.items() if v["cr"] == cr and v["ppls"]]
        entries.sort(key=lambda x: statistics.mean(x[1]["ratios"]))
        for key, v in entries:
            avg_ppl = statistics.mean(v["ppls"])
            avg_r = statistics.mean(v["ratios"])
            avg_t = statistics.mean(v["times_ms"])
            avg_m = statistics.mean(v["mems_kb"])
            marker = " ***" if v["method"] == "layer_budget" else ""
            print(f"  {v['method']:<22s} {cr:>4.0f}x {avg_ppl:>8.2f} {avg_r:>8.4f} "
                  f"{avg_t:>8.1f}ms {avg_m:>7.0f}KB{marker}")
            summary_rows.append({
                "method": v["method"], "cr": cr,
                "mean_ppl": round(avg_ppl, 4),
                "std_ppl": round(statistics.stdev(v["ppls"]), 4) if len(v["ppls"]) > 1 else 0,
                "mean_ratio": round(avg_r, 4),
                "mean_compress_ms": round(avg_t, 2),
                "mean_mem_kb": round(avg_m, 1),
                "n": len(v["ppls"]),
            })
        if entries:
            print()

    return summary_rows


def _build_cache_zerofill(layers_data, full_seq_len, device):
    """Build HF DynamicCache, zero-filling evicted positions."""
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()

    for layer_idx, (keys, values, indices) in enumerate(layers_data):
        k = keys.to(device)
        v = values.to(device)
        if k.dim() == 3:
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)

        n_tokens, num_heads, head_dim = k.shape[1], k.shape[2], k.shape[3]

        if n_tokens == full_seq_len:
            k_out = k.transpose(1, 2)
            v_out = v.transpose(1, 2)
        else:
            k_full = torch.zeros(1, full_seq_len, num_heads, head_dim, dtype=k.dtype, device=device)
            v_full = torch.zeros(1, full_seq_len, num_heads, head_dim, dtype=v.dtype, device=device)
            idx = indices.long().to(device)
            valid = idx[idx < full_seq_len]
            if valid.numel() > 0:
                k_full[0, valid] = k[0, :valid.numel()]
                v_full[0, valid] = v[0, :valid.numel()]
            k_out = k_full.transpose(1, 2)
            v_out = v_full.transpose(1, 2)

        cache.update(k_out, v_out, layer_idx)

    return cache


# ── Test 3: Generation quality (side-by-side) ───────────────────────────

def test_generation_quality(model, tokenizer, budgets_mb: List[float],
                            max_new_tokens: int = 100):
    """Compare generation quality: full KV vs LayerBudgetCache at various budgets."""
    from deltacache.integrations.hf_cache import LayerBudgetCache

    device = next(model.parameters()).device
    prompts = [
        "The theory of general relativity states that",
        "In a recent study published in Nature, researchers found that",
        "The capital of France is Paris, which is known for",
        "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
    ]

    print(f"\n{'='*70}")
    print(f"  Generation Quality Comparison (budgets: {budgets_mb})")
    print(f"{'='*70}")

    results = []
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        clear_gpu()
        attn_mask = torch.ones_like(input_ids)
        gen_kw = dict(max_new_tokens=max_new_tokens, do_sample=False,
                      attention_mask=attn_mask, pad_token_id=tokenizer.eos_token_id)
        with torch.no_grad():
            out_full = model.generate(input_ids, **gen_kw)
        text_full = tokenizer.decode(out_full[0], skip_special_tokens=True)

        for budget_mb in budgets_mb:
            clear_gpu()
            cache = LayerBudgetCache(max_memory_mb=budget_mb, compression_ratio=0.25)
            with torch.no_grad():
                out_lb = model.generate(
                    input_ids, past_key_values=cache, **gen_kw,
                )
            text_lb = tokenizer.decode(out_lb[0], skip_special_tokens=True)

            toks_full = out_full[0, input_ids.shape[1]:].tolist()
            toks_lb = out_lb[0, input_ids.shape[1]:].tolist()
            min_len = min(len(toks_full), len(toks_lb))
            match = sum(a == b for a, b in zip(toks_full[:min_len], toks_lb[:min_len]))
            match_rate = match / max(min_len, 1)

            results.append({
                "prompt": prompt[:50],
                "budget_mb": budget_mb,
                "compressed": cache._compressed,
                "token_match": round(match_rate, 4),
                "len_full": len(toks_full),
                "len_lb": len(toks_lb),
            })

        print(f"\n  Prompt: \"{prompt[:60]}...\"")
        print(f"    Full:  {text_full[len(prompt):len(prompt)+120]}")
        for r in results[-len(budgets_mb):]:
            tag = "C" if r["compressed"] else "F"
            print(f"    {r['budget_mb']:6.1f}MB [{tag}]: match={r['token_match']:.2%}")

    return results


# ── Test 4: Memory & latency scaling ────────────────────────────────────

def test_scaling(model, tokenizer, budget_mb: float):
    """Measure how LayerBudgetCache scales with sequence length."""
    from deltacache.integrations.hf_cache import LayerBudgetCache

    device = next(model.parameters()).device
    max_pos = getattr(model.config, "max_position_embeddings", 2048)
    seq_lens = [128, 256, 512, 1024]
    seq_lens = [s for s in seq_lens if s <= max_pos]

    print(f"\n{'='*70}")
    print(f"  Scaling: LayerBudgetCache (budget={budget_mb}MB) vs DynamicCache")
    print(f"{'='*70}")
    print(f"  {'SeqLen':>7s} {'Base(ms)':>9s} {'LB(ms)':>9s} {'Speedup':>8s} "
          f"{'BaseMem':>8s} {'LBMem':>8s} {'MemSave':>8s} {'Compressed':>11s}")

    chunks = get_wikitext_chunks(tokenizer, max(seq_lens), n=2)
    results = []

    for sl in seq_lens:
        tokens = chunks[0][:sl]
        input_ids = tokenizer.encode(
            tokenizer.decode(tokens), return_tensors="pt",
        ).to(device)
        actual_len = input_ids.shape[1]

        gen_kw = dict(max_new_tokens=32, do_sample=False,
                      pad_token_id=tokenizer.eos_token_id)

        # Baseline
        clear_gpu()
        torch.cuda.reset_peak_memory_stats()
        mem0 = gpu_mem_mb()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(input_ids, **gen_kw)
        t_base = (time.perf_counter() - t0) * 1000
        mem_base = torch.cuda.max_memory_allocated() / 1024 ** 2 - mem0

        # LayerBudgetCache
        clear_gpu()
        torch.cuda.reset_peak_memory_stats()
        mem0 = gpu_mem_mb()
        try:
            t0 = time.perf_counter()
            cache = LayerBudgetCache(max_memory_mb=budget_mb, compression_ratio=0.25)
            with torch.no_grad():
                out_lb = model.generate(
                    input_ids, past_key_values=cache, **gen_kw,
                )
            t_lb = (time.perf_counter() - t0) * 1000
            mem_lb = torch.cuda.max_memory_allocated() / 1024 ** 2 - mem0
            compressed = cache._compressed
        except RuntimeError as e:
            t_lb = float("nan")
            mem_lb = float("nan")
            compressed = "ERROR: " + str(e)[:80]

        speedup = t_base / max(t_lb, 0.1) if not math.isnan(t_lb) else float("nan")
        mem_save = (1.0 - mem_lb / max(mem_base, 0.1)) if not math.isnan(mem_lb) else float("nan")

        comp_str = str(compressed)[:20]
        if math.isnan(t_lb):
            print(f"  {actual_len:>7d} {t_base:>8.1f}ms {'ERR':>9s} {'N/A':>8s} "
                  f"{mem_base:>7.1f}MB {'ERR':>8s} {'N/A':>8s} {comp_str}")
        else:
            print(f"  {actual_len:>7d} {t_base:>8.1f}ms {t_lb:>8.1f}ms {speedup:>7.2f}x "
                  f"{mem_base:>7.1f}MB {mem_lb:>7.1f}MB {mem_save:>7.1%} {comp_str}")

        results.append({
            "seq_len": actual_len,
            "base_ms": round(t_base, 1),
            "lb_ms": round(t_lb, 1) if not math.isnan(t_lb) else "error",
            "speedup": round(speedup, 3) if not math.isnan(speedup) else "error",
            "base_mem_mb": round(mem_base, 1),
            "lb_mem_mb": round(mem_lb, 1) if not math.isnan(mem_lb) else "error",
            "mem_savings": round(mem_save, 4) if not math.isnan(mem_save) else "error",
            "compressed": str(compressed),
        })

    return results


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LayerBudget end-to-end benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2-0.5B",
                        help="HuggingFace model name or path")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load model in 4-bit quantization (recommended for 7B+)")
    parser.add_argument("--budget-mb", type=float, default=None,
                        help="Override auto-computed budget (MB)")
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Sequence length for PPL evaluation")
    parser.add_argument("--n-texts", type=int, default=6,
                        help="Number of WikiText sequences for PPL test")
    parser.add_argument("--skip-ppl", action="store_true",
                        help="Skip PPL pipeline test (slow for 7B)")
    args = parser.parse_args()

    RESULTS_DIR = Path(__file__).parent / "results" / "e2e"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(args.model, args.load_in_4bit)
    device = next(model.parameters()).device
    nl = model.config.num_hidden_layers
    nh = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    hd = model.config.hidden_size // model.config.num_attention_heads

    # Auto-compute budgets
    budgets = auto_budget(nl, nh, hd)
    if args.budget_mb is not None:
        budgets.dropin_budget_mb = args.budget_mb
        budgets.scaling_budget_mb = args.budget_mb

    print(f"\nModel: {args.model}")
    print(f"  Layers={nl}, KV-heads={nh}, head_dim={hd}")
    print(f"  Full KV @1024tok = {budgets.full_kv_1024_mb:.1f} MB")
    print(f"  Auto budgets: dropin={budgets.dropin_budget_mb:.1f}MB, "
          f"gen={budgets.generation_budgets_mb}, scaling={budgets.scaling_budget_mb:.1f}MB")
    print(f"  Device={device}, GPU={torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {gpu_mem_mb():.0f} MB used / "
          f"{torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB total")

    all_output = {
        "metadata": {
            "model": args.model,
            "load_in_4bit": args.load_in_4bit,
            "num_layers": nl, "num_kv_heads": nh, "head_dim": hd,
            "full_kv_1024_mb": budgets.full_kv_1024_mb,
            "budgets": {
                "dropin": budgets.dropin_budget_mb,
                "generation": budgets.generation_budgets_mb,
                "scaling": budgets.scaling_budget_mb,
            },
            "gpu": torch.cuda.get_device_name(0),
            "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1),
            "timestamp": datetime.now().isoformat(),
        },
    }

    # ── Test 1: Drop-in generate() ──
    print(f"\n{'#'*70}")
    print(f"  TEST 1: LayerBudgetCache drop-in (budget={budgets.dropin_budget_mb:.1f}MB)")
    print(f"{'#'*70}")
    prompts = [
        "The history of machine learning begins with",
        "In quantum computing, a qubit is fundamentally different from a classical bit because",
        "The following Python function computes the greatest common divisor:\ndef gcd(a, b):\n",
    ]
    gen_results = test_generate_dropin(model, tokenizer, prompts,
                                       budget_mb=budgets.dropin_budget_mb)
    all_output["test1_generate_dropin"] = gen_results

    # ── Test 2: PPL pipeline ──
    if not args.skip_ppl:
        print(f"\n{'#'*70}")
        print(f"  TEST 2: PPL Pipeline (WikiText-2, seq_len={args.seq_len})")
        print(f"{'#'*70}")
        ppl_results = test_ppl_pipeline(model, tokenizer,
                                         seq_len=args.seq_len, n_texts=args.n_texts)
        all_output["test2_ppl_pipeline"] = ppl_results
    else:
        print(f"\n  Skipping PPL pipeline (--skip-ppl)")

    # ── Test 3: Generation quality ──
    print(f"\n{'#'*70}")
    print(f"  TEST 3: Generation Quality Comparison")
    print(f"{'#'*70}")
    qual_results = test_generation_quality(model, tokenizer,
                                           budgets_mb=budgets.generation_budgets_mb)
    all_output["test3_generation_quality"] = qual_results

    # ── Test 4: Scaling ──
    print(f"\n{'#'*70}")
    print(f"  TEST 4: Scaling with Sequence Length")
    print(f"{'#'*70}")
    scale_results = test_scaling(model, tokenizer,
                                  budget_mb=budgets.scaling_budget_mb)
    all_output["test4_scaling"] = scale_results

    # ── Save ──
    safe_name = args.model.replace("/", "_").replace("-", "_").lower()
    out_path = RESULTS_DIR / f"e2e_{safe_name}_{datetime.now():%Y%m%d_%H%M}.json"
    with open(out_path, "w") as f:
        json.dump(all_output, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")
    print("\nAll tests complete!")


if __name__ == "__main__":
    main()
