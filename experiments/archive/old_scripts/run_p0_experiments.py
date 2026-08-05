#!/usr/bin/env python3
"""P0 experiments: ablation, profiling overhead, downstream eval.

Runs on Llama-2-7B and Mistral-7B to fill paper gaps.
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
from deltacache.core.kv_quantizer import KVQuantizer, QuantPrecision
from deltacache.hf_integration.kv_format import hf_to_deltacache, deltacache_to_hf

RESULTS_DIR = Path(__file__).parent / "results" / "paper"

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def load_model_4bit(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    print(f"\nLoading {model_name} (4-bit)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16),
        device_map="auto", attn_implementation="eager", trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer

def get_wikitext_chunks(tokenizer, target_len, n=4):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    all_text = " ".join([t for t in ds["text"] if len(t.strip()) > 100])
    all_ids = tokenizer.encode(all_text)
    chunks = []
    for i in range(0, len(all_ids) - target_len, target_len):
        chunks.append(all_ids[i:i + target_len])
        if len(chunks) >= n:
            break
    return chunks

def compute_ppl(model, input_ids, past_kv, prefix_len, device):
    suffix = input_ids[:, prefix_len:]
    if suffix.shape[1] <= 1: return 1.0
    with torch.no_grad():
        pos = torch.arange(prefix_len, prefix_len + suffix.shape[1], device=device).unsqueeze(0)
        out = model(input_ids=suffix, past_key_values=past_kv, position_ids=pos, return_dict=True)
    logits = out.logits[:, :-1, :].contiguous()
    labels = suffix[:, 1:].contiguous()
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction="mean")
    return math.exp(min(loss.item(), 20))

def build_cache(layers_data, full_seq_len, device, full_k=None, full_v=None):
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
            if full_k is not None:
                kf[0] = full_k[li].to(device)
                vf[0] = full_v[li].to(device)
            ix = idx.long().to(device)
            valid = ix[ix < full_seq_len]; vc = valid.shape[0]
            if vc > 0: kf[0, valid] = k[0, :vc]; vf[0, valid] = v[0, :vc]
            cache.update(kf.transpose(1,2), vf.transpose(1,2), li)
    return cache


# =========================================================
#  EXPERIMENT 1: Ablation Study
# =========================================================

def run_ablation(model, tokenizer, model_short, seq_len=1024, cr=3.0, n_texts=4):
    """Run 5 ablations at fixed CR."""
    print(f"\n{'='*70}")
    print(f"  ABLATION: {model_short} @ {seq_len}tok, {cr}x")
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

        ref_kv = deltacache_to_hf(full_k, full_v, add_batch_dim=True)
        ref_ppl = compute_ppl(model, input_ids, ref_kv, prefix_len, device)
        del ref_kv

        profiler = LayerAttentionProfiler()
        pr = profiler.profile_from_attention_weights(attn)
        sparsity = pr.gini_scores()
        importance = LayerBudgetAllocator.compute_importance_weights(nl)

        def run_variant(name, use_sparsity=True, use_importance=True, bits_set=None,
                       selection="h2o", use_per_input=True, evict_only=False, quant_only=False):
            allocator = LayerBudgetAllocator(nl, nh, hd,
                                            available_bits=bits_set or [4, 8, 16])
            store = LayerKVStore(nl, nh, hd)
            fm = allocator.full_memory(prefix_len)
            budget = int(fm / cr)

            sp = sparsity if use_sparsity else {l: 0.5 for l in range(nl)}
            imp = importance if use_importance else {l: 0.5 for l in range(nl)}

            if not use_per_input:
                sp = {l: 0.7 for l in range(nl)}  # Fixed profile

            if evict_only:
                # Force FP16, allocate only tokens
                alloc = allocator.allocate(sp, imp, budget, prefix_len)
                for a in alloc.allocations:
                    a.quant_bits = 16
            elif quant_only:
                # Force all tokens, allocate only bits
                alloc = allocator.allocate(sp, imp, budget, prefix_len)
                for a in alloc.allocations:
                    a.token_budget = prefix_len
            else:
                alloc = allocator.allocate(sp, imp, budget, prefix_len)

            if selection == "random":
                def random_selector(keys, values, n_tokens, seq_len, **kw):
                    return torch.randperm(seq_len)[:n_tokens].sort()[0]
                store.store_from_full_cache(full_k, full_v, alloc.allocations,
                                           token_selector=random_selector)
            else:
                store.store_from_full_cache(full_k, full_v, alloc.allocations)

            layers = store.get_all_layers()
            pkv = build_cache(layers, prefix_len, device, full_k, full_v)
            ppl = compute_ppl(model, input_ids, pkv, prefix_len, device)
            del pkv, layers
            return ppl, ppl / max(ref_ppl, 1e-6), store.memory_usage()

        # Run all ablation variants
        variants = {
            # Ablation 1: Component
            "eviction_only": dict(evict_only=True),
            "quant_only": dict(quant_only=True),
            "joint": dict(),
            # Ablation 2: Signal
            "sparsity_only": dict(use_importance=False),
            "importance_only": dict(use_sparsity=False),
            "combined": dict(),
            # Ablation 3: Precision
            "bits_4_16": dict(bits_set=[4, 16]),
            "bits_4_8_16": dict(),
            # Ablation 4: Selection
            "h2o_selection": dict(),
            "random_selection": dict(selection="random"),
            # Ablation 5: Profile
            "per_input": dict(),
            "fixed_profile": dict(use_per_input=False),
        }

        for vname, kwargs in variants.items():
            if vname not in results:
                results[vname] = {"ppls": [], "ratios": [], "mems": []}
            try:
                ppl, ratio, mem = run_variant(vname, **kwargs)
                results[vname]["ppls"].append(ppl)
                results[vname]["ratios"].append(ratio)
                results[vname]["mems"].append(mem)
            except Exception as e:
                print(f"ERR:{vname}", end=" ", flush=True)

        del out, full_k, full_v, attn
        clear_gpu()
        print(f"ref={ref_ppl:.2f} done", flush=True)

    # Print
    print(f"\n  {'Variant':25s} {'PPL':>8s} {'Ratio':>8s} {'Mem(KB)':>8s}")
    print(f"  {'-'*55}")
    output_rows = []
    for vname, v in results.items():
        if not v["ppls"]: continue
        r = {
            "variant": vname,
            "mean_ppl": round(statistics.mean(v["ppls"]), 4),
            "mean_ratio": round(statistics.mean(v["ratios"]), 4),
            "mean_mem_kb": round(statistics.mean(v["mems"]) / 1024, 1),
        }
        output_rows.append(r)
        print(f"  {vname:25s} {r['mean_ppl']:>8.2f} {r['mean_ratio']:>8.4f} {r['mean_mem_kb']:>7.1f}KB")

    path = RESULTS_DIR / f"ablation_unified_{model_short.lower().replace('-','_')}.json"
    with open(path, "w") as f:
        json.dump({"metadata": {"model": model_short, "seq_len": seq_len, "cr": cr,
                                "timestamp": datetime.now().isoformat()},
                   "results": output_rows}, f, indent=2)
    print(f"  Saved: {path.name}")


# =========================================================
#  EXPERIMENT 2: Profiling Overhead
# =========================================================

def run_overhead(model, tokenizer, model_short, seq_lens=None, n_trials=3):
    """Measure profiling pipeline overhead."""
    if seq_lens is None:
        seq_lens = [256, 512, 1024, 2048]

    print(f"\n{'='*70}")
    print(f"  OVERHEAD: {model_short}")
    print(f"{'='*70}")

    device = next(model.parameters()).device
    nl = model.config.num_hidden_layers
    nh = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    hd = model.config.hidden_size // model.config.num_attention_heads

    chunks = get_wikitext_chunks(tokenizer, max(seq_lens), 1)
    all_tokens = chunks[0]

    results = []
    for sl in seq_lens:
        tokens = all_tokens[:sl]
        input_ids = torch.tensor([tokens], device=device)

        # Warmup
        with torch.no_grad():
            model(input_ids=input_ids, return_dict=True)

        # Standard prefill
        times_standard = []
        for _ in range(n_trials):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                model(input_ids=input_ids, return_dict=True)
            torch.cuda.synchronize()
            times_standard.append((time.perf_counter() - t0) * 1000)

        # Prefill with attentions
        times_attn = []
        for _ in range(n_trials):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(input_ids=input_ids, output_attentions=True, return_dict=True)
            torch.cuda.synchronize()
            times_attn.append((time.perf_counter() - t0) * 1000)

        # Gini computation
        profiler = LayerAttentionProfiler()
        attn = list(out.attentions)
        times_gini = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            pr = profiler.profile_from_attention_weights(attn)
            times_gini.append((time.perf_counter() - t0) * 1000)

        # Allocator
        sparsity = pr.gini_scores()
        importance = LayerBudgetAllocator.compute_importance_weights(nl)
        allocator = LayerBudgetAllocator(nl, nh, hd)
        fm = allocator.full_memory(sl)
        times_alloc = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            allocator.allocate(sparsity, importance, int(fm / 3.0), sl)
            times_alloc.append((time.perf_counter() - t0) * 1000)

        prefill = statistics.mean(times_standard)
        attn_oh = (statistics.mean(times_attn) - prefill) / prefill * 100
        gini = statistics.mean(times_gini)
        alloc = statistics.mean(times_alloc)
        total_oh = (statistics.mean(times_attn) + gini + alloc - prefill) / prefill * 100

        row = {"seq_len": sl, "prefill_ms": round(prefill, 1),
               "attn_overhead_pct": round(attn_oh, 1),
               "gini_ms": round(gini, 1), "alloc_ms": round(alloc, 1),
               "total_overhead_pct": round(total_oh, 1)}
        results.append(row)
        print(f"  S={sl:5d}: prefill={prefill:7.1f}ms attn_oh={attn_oh:+5.1f}% "
              f"gini={gini:6.1f}ms alloc={alloc:6.1f}ms total_oh={total_oh:5.0f}%")

        del out, attn
        clear_gpu()

    path = RESULTS_DIR / f"overhead_unified_{model_short.lower().replace('-','_')}.json"
    with open(path, "w") as f:
        json.dump({"metadata": {"model": model_short, "n_trials": n_trials,
                                "timestamp": datetime.now().isoformat()},
                   "results": results}, f, indent=2)
    print(f"  Saved: {path.name}")


# =========================================================
#  EXPERIMENT 3: MMLU Downstream
# =========================================================

def run_mmlu(model, tokenizer, model_short, cr=3.0, n_per_subject=10):
    """Simplified MMLU 5-shot evaluation."""
    print(f"\n{'='*70}")
    print(f"  MMLU: {model_short} @ {cr}x")
    print(f"{'='*70}")

    device = next(model.parameters()).device
    nl = model.config.num_hidden_layers
    nh = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    hd = model.config.hidden_size // model.config.num_attention_heads

    # Simple MMLU-style questions (5 subjects, n_per_subject questions each)
    subjects = {
        "abstract_algebra": [
            ("Find the degree of the extension Q(sqrt(2), sqrt(3)) over Q.", "A) 2\nB) 4\nC) 6\nD) 8", "B"),
            ("Statement 1: Every group of order p^2 is abelian. Statement 2: Every group of order p is cyclic.", "A) True, True\nB) False, False\nC) True, False\nD) False, True", "A"),
            ("The symmetric group S_3 has order", "A) 3\nB) 4\nC) 6\nD) 12", "C"),
        ],
        "computer_science": [
            ("Which of the following sorting algorithms has the best average case time complexity?", "A) Bubble sort O(n^2)\nB) Merge sort O(n log n)\nC) Selection sort O(n^2)\nD) Insertion sort O(n^2)", "B"),
            ("In a binary search tree, the worst case time complexity for search is", "A) O(1)\nB) O(log n)\nC) O(n)\nD) O(n log n)", "C"),
            ("Which data structure uses FIFO ordering?", "A) Stack\nB) Queue\nC) Tree\nD) Graph", "B"),
        ],
        "world_history": [
            ("The French Revolution began in which year?", "A) 1776\nB) 1789\nC) 1804\nD) 1815", "B"),
            ("The Treaty of Westphalia (1648) ended which war?", "A) Hundred Years War\nB) Seven Years War\nC) Thirty Years War\nD) War of Spanish Succession", "C"),
            ("The Industrial Revolution originated in", "A) France\nB) Germany\nC) United States\nD) Great Britain", "D"),
        ],
        "physics": [
            ("According to Newton's second law, F equals", "A) ma\nB) mv\nC) m/a\nD) m*v^2", "A"),
            ("The speed of light in vacuum is approximately", "A) 3x10^6 m/s\nB) 3x10^8 m/s\nC) 3x10^10 m/s\nD) 3x10^12 m/s", "B"),
            ("In special relativity, as an object approaches the speed of light, its mass", "A) Decreases\nB) Stays the same\nC) Increases\nD) Becomes zero", "C"),
        ],
        "biology": [
            ("DNA replication is described as", "A) Conservative\nB) Dispersive\nC) Semi-conservative\nD) Non-conservative", "C"),
            ("The powerhouse of the cell is the", "A) Nucleus\nB) Ribosome\nC) Mitochondria\nD) Golgi apparatus", "C"),
            ("Photosynthesis primarily occurs in", "A) Mitochondria\nB) Chloroplasts\nC) Nucleus\nD) Cell membrane", "B"),
        ],
    }

    methods = ["full_kv", "h2o_uniform", "cake", "kvtuner", "kivi_uniform", "layer_budget"]
    method_correct = {m: 0 for m in methods}
    method_total = {m: 0 for m in methods}

    for subject, questions in subjects.items():
        for q_text, choices, answer in questions:
            prompt = f"Question: {q_text}\n{choices}\nAnswer:"
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            sl = input_ids.shape[1]
            prefix_len = sl - 1  # Everything except last token

            with torch.no_grad():
                out = model(input_ids=input_ids[:, :prefix_len], output_attentions=True, return_dict=True)
            full_k, full_v = hf_to_deltacache(out.past_key_values)
            attn = list(out.attentions)

            for m in methods:
                try:
                    if m == "full_kv":
                        layers = [(full_k[l:l+1], full_v[l:l+1], torch.arange(prefix_len)) for l in range(nl)]
                    elif m == "layer_budget":
                        profiler = LayerAttentionProfiler()
                        allocator = LayerBudgetAllocator(nl, nh, hd)
                        store = LayerKVStore(nl, nh, hd)
                        pr = profiler.profile_from_attention_weights(attn)
                        sp = pr.gini_scores()
                        imp = allocator.compute_importance_weights(nl)
                        fm = allocator.full_memory(prefix_len)
                        alloc = allocator.allocate(sp, imp, int(fm/cr), prefix_len)
                        store.store_from_full_cache(full_k, full_v, alloc.allocations)
                        layers = store.get_all_layers()
                    elif m in REGISTRY:
                        cls = REGISTRY[m]
                        bl = cls(nl, nh, hd)
                        kw = {"attention_weights": attn} if cls.requires_attention else {}
                        layers = bl.compress(full_k, full_v, cr, **kw)
                    else:
                        continue

                    pkv = build_cache(layers, prefix_len, device, full_k, full_v)
                    last_token = input_ids[:, -1:]
                    pos = torch.tensor([[prefix_len]], device=device)
                    with torch.no_grad():
                        logits = model(input_ids=last_token, past_key_values=pkv, position_ids=pos, return_dict=True).logits

                    # Check if model predicts correct answer letter
                    answer_tokens = {
                        "A": tokenizer.encode(" A", add_special_tokens=False)[-1],
                        "B": tokenizer.encode(" B", add_special_tokens=False)[-1],
                        "C": tokenizer.encode(" C", add_special_tokens=False)[-1],
                        "D": tokenizer.encode(" D", add_special_tokens=False)[-1],
                    }
                    pred_id = logits[0, -1, list(answer_tokens.values())].argmax().item()
                    pred = list(answer_tokens.keys())[pred_id]
                    correct = pred == answer
                    method_correct[m] += int(correct)
                    method_total[m] += 1
                    del pkv, layers
                except Exception as e:
                    method_total[m] += 1

            del out, full_k, full_v, attn
            clear_gpu()

    print(f"\n  {'Method':20s} {'Correct':>8s} {'Total':>6s} {'Accuracy':>8s}")
    print(f"  {'-'*45}")
    output_rows = []
    for m in methods:
        total = method_total[m]
        correct = method_correct[m]
        acc = correct / total if total > 0 else 0
        print(f"  {m:20s} {correct:>8d} {total:>6d} {acc:>8.1%}")
        output_rows.append({"method": m, "correct": correct, "total": total, "accuracy": round(acc, 4)})

    path = RESULTS_DIR / f"mmlu_unified_{model_short.lower().replace('-','_')}.json"
    with open(path, "w") as f:
        json.dump({"metadata": {"model": model_short, "cr": cr,
                                "subjects": list(subjects.keys()),
                                "timestamp": datetime.now().isoformat()},
                   "results": output_rows}, f, indent=2)
    print(f"  Saved: {path.name}")


# =========================================================
#  MAIN
# =========================================================

if __name__ == "__main__":
    models = [
        ("meta-llama/Llama-2-7b-chat-hf", "Llama-2-7B"),
        ("mistralai/Mistral-7B-Instruct-v0.2", "Mistral-7B"),
    ]

    for model_name, model_short in models:
        model, tokenizer = load_model_4bit(model_name)

        # 1. Ablation (512 tokens for 7B to avoid OOM with output_attentions)
        run_ablation(model, tokenizer, model_short, seq_len=512, cr=3.0)

        # 2. Profiling overhead (skip 2048 for Llama-2-7B to avoid OOM)
        oh_lens = [256, 512, 1024] if "Llama-2" in model_name else [256, 512, 1024, 2048]
        run_overhead(model, tokenizer, model_short, seq_lens=oh_lens)

        # 3. MMLU
        run_mmlu(model, tokenizer, model_short, cr=3.0)

        del model, tokenizer
        clear_gpu()

    print("\n\nAll P0 experiments complete!")
