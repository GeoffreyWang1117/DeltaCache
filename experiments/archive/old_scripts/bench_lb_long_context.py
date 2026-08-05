#!/usr/bin/env python3
"""Long-context validation for LayerBudget.

Validates LayerBudget at longer sequences (512-4096 tokens) to address
the short-sequence limitation noted in the paper.

Measures:
  1. Perplexity at different context lengths
  2. Gini coefficient stability across context lengths
  3. Compression ratio effect at long context
"""

import gc
import json
import math
import sys
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from deltacache.core.layer_profiler import LayerAttentionProfiler
from deltacache.core.layer_budget_allocator import LayerBudgetAllocator
from deltacache.core.layer_kv_store import LayerKVStore
from deltacache.hf_integration.kv_format import hf_to_deltacache, deltacache_to_hf

from baselines.cake import CAKEBaseline

RESULTS_DIR = Path(__file__).parent / "results" / "paper"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LONG_TEXTS = [
    "The history of artificial intelligence began in antiquity, with myths, stories "
    "and rumors of artificial beings endowed with intelligence or consciousness by "
    "master craftsmen. The seeds of modern AI were planted by philosophers who "
    "attempted to describe the process of human thinking as the mechanical manipulation "
    "of symbols. This work culminated in the invention of the programmable digital "
    "computer in the 1940s, a machine based on the abstract essence of mathematical "
    "reasoning. The field of AI research was born at a workshop at Dartmouth College "
    "in 1956, where the term artificial intelligence was coined. In the following decades, "
    "AI research explored several approaches including symbolic AI, neural networks, and "
    "evolutionary algorithms. The field experienced periods of optimism followed by "
    "disappointment and loss of funding, known as AI winters. In the early 2000s, "
    "machine learning revitalized the field with practical applications in speech "
    "recognition, computer vision, and natural language processing. The deep learning "
    "revolution, powered by large datasets and GPU computing, led to breakthroughs "
    "in image classification, machine translation, and game playing. By the 2020s, "
    "large language models trained on vast corpora of text demonstrated remarkable "
    "capabilities in generating coherent text, answering questions, and even writing "
    "code. These models raised important questions about the nature of intelligence "
    "and the future of human-computer interaction. ",

    "In computer science, algorithms and data structures form the foundation of "
    "efficient computing. A hash table is a data structure that implements an "
    "associative array, mapping keys to values with average O(1) lookup time. "
    "Binary search trees maintain sorted data with O(log n) operations. Graph "
    "algorithms like Dijkstra's shortest path and breadth-first search are fundamental "
    "to network analysis and routing. Dynamic programming breaks complex problems "
    "into simpler subproblems, enabling efficient solutions to optimization challenges. "
    "The study of computational complexity reveals inherent limits on what can be "
    "efficiently computed. Problems in the class P can be solved in polynomial time, "
    "while NP-complete problems appear to require exponential time. The P versus NP "
    "question remains one of the great open problems in mathematics. ",

    "The natural world is governed by physical laws that describe how matter and energy "
    "interact across all scales, from subatomic particles to the cosmic web of galaxies. "
    "Quantum mechanics describes the behavior of matter at the smallest scales, where "
    "particles exhibit wave-particle duality and measurements are inherently probabilistic. "
    "The uncertainty principle establishes fundamental limits on how precisely we can "
    "simultaneously know certain pairs of physical properties. At larger scales, "
    "classical mechanics provides an excellent approximation, with Newton's laws "
    "describing motion and gravity. Einstein's special relativity unified space and "
    "time into spacetime, showing that the speed of light is the same for all observers. "
    "General relativity extended this to include gravity as the curvature of spacetime "
    "caused by mass and energy. These theories have been confirmed by countless "
    "experiments, from the bending of light around massive objects to the detection "
    "of gravitational waves from merging black holes. ",
]


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_long_eval_text(tokenizer, target_len):
    full_text = " ".join(LONG_TEXTS)
    while len(tokenizer.encode(full_text)) < target_len + 100:
        full_text = full_text + " " + full_text
    return tokenizer.encode(full_text)[:target_len]


def _build_hf_past_kv(layers_data, device, full_seq_len=None):
    from transformers.cache_utils import DynamicCache
    token_counts = [k.shape[1] if k.dim() == 4 else k.shape[0] for k, v, _ in layers_data]
    seq_len = full_seq_len if full_seq_len else max(token_counts)
    cache = DynamicCache()
    for layer_idx, (keys, values, indices) in enumerate(layers_data):
        k = keys.to(device)
        v = values.to(device)
        if k.dim() == 3: k = k.unsqueeze(0); v = v.unsqueeze(0)
        n_tokens, num_heads, head_dim = k.shape[1], k.shape[2], k.shape[3]
        if n_tokens == seq_len:
            k_out, v_out = k.transpose(1, 2), v.transpose(1, 2)
        else:
            k_full = torch.zeros(1, seq_len, num_heads, head_dim, dtype=k.dtype, device=device)
            v_full = torch.zeros(1, seq_len, num_heads, head_dim, dtype=v.dtype, device=device)
            idx = indices.long().to(device)
            valid_idx = idx[idx < seq_len]
            if valid_idx.shape[0] > 0:
                k_full[0, valid_idx] = k[0, :valid_idx.shape[0]]
                v_full[0, valid_idx] = v[0, :valid_idx.shape[0]]
            k_out, v_out = k_full.transpose(1, 2), v_full.transpose(1, 2)
        cache.update(k_out, v_out, layer_idx)
    return cache


def compute_ppl(model, input_ids, past_kv, prefix_len):
    suffix_ids = input_ids[:, prefix_len:]
    if suffix_ids.shape[1] <= 1:
        return 1.0
    with torch.no_grad():
        position_ids = torch.arange(prefix_len, prefix_len + suffix_ids.shape[1],
                                    device=input_ids.device).unsqueeze(0)
        outputs = model(input_ids=suffix_ids, past_key_values=past_kv,
                        position_ids=position_ids, return_dict=True)
    logits = outputs.logits
    loss = F.cross_entropy(
        logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
        suffix_ids[:, 1:].contiguous().view(-1), reduction="mean")
    return math.exp(min(loss.item(), 20))


def run_long_context(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device="cuda:0",
    context_lengths=None,
    compression_ratios=None,
    load_in_4bit=False,
):
    if context_lengths is None:
        context_lengths = [256, 512, 1024, 2048]
    if compression_ratios is None:
        compression_ratios = [3.0, 4.0]

    print(f"\n{'='*70}")
    print(f"Long-Context Validation: {model_name}")
    print(f"Lengths: {context_lengths}   CRs: {compression_ratios}")
    print(f"{'='*70}\n")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    load_kwargs = dict(torch_dtype=torch.float16, trust_remote_code=True,
                       attn_implementation="eager")
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = device
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()

    num_layers = model.config.num_hidden_layers
    num_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    all_results = []
    gini_by_length = {}

    for ctx_len in context_lengths:
        print(f"\n--- Context: {ctx_len} tokens ---")
        tokens = get_long_eval_text(tokenizer, ctx_len)
        input_ids = torch.tensor([tokens], device=device)
        prefix_len = int(len(tokens) * 0.7)
        prefix_ids = input_ids[:, :prefix_len]

        with torch.no_grad():
            outputs = model(input_ids=prefix_ids, output_attentions=True, return_dict=True)
        kv_cache = outputs.past_key_values
        attention_weights = list(outputs.attentions)
        full_keys, full_values = hf_to_deltacache(kv_cache)
        del outputs, kv_cache
        clear_gpu()

        profiler = LayerAttentionProfiler()
        profile_result = profiler.profile_from_attention_weights(attention_weights)
        gini = profile_result.gini_scores()
        gini_by_length[ctx_len] = {str(k): round(v, 4) for k, v in gini.items()}

        past_kv_full = deltacache_to_hf(full_keys, full_values, add_batch_dim=True)
        ppl_full = compute_ppl(model, input_ids, past_kv_full, prefix_len)
        del past_kv_full; clear_gpu()

        for cr in compression_ratios:
            # LayerBudget
            allocator = LayerBudgetAllocator(num_layers, num_heads, head_dim)
            store = LayerKVStore(num_layers, num_heads, head_dim)
            importance = allocator.compute_importance_weights(num_layers)
            budget = int(allocator.full_memory(prefix_len) / cr)
            allocation = allocator.allocate(gini, importance, budget, prefix_len)
            store.store_from_full_cache(full_keys, full_values, allocation.allocations)
            past_kv = _build_hf_past_kv(store.get_all_layers(), device, full_seq_len=prefix_len)
            ppl_lb = compute_ppl(model, input_ids, past_kv, prefix_len)
            mem_lb = store.memory_usage()
            del past_kv, store; clear_gpu()

            # H2O uniform
            n_tok = max(1, int(prefix_len / cr))
            store_h2o = LayerKVStore(num_layers, num_heads, head_dim)
            for l in range(num_layers):
                idx = store_h2o._default_token_selection(
                    full_keys[l:l+1], full_values[l:l+1], n_tok, prefix_len)
                store_h2o.store_layer(l, full_keys[l:l+1], full_values[l:l+1], idx, quant_bits=16)
            past_kv_h2o = _build_hf_past_kv(store_h2o.get_all_layers(), device, full_seq_len=prefix_len)
            ppl_h2o = compute_ppl(model, input_ids, past_kv_h2o, prefix_len)
            del past_kv_h2o, store_h2o; clear_gpu()

            # CAKE
            cake = CAKEBaseline(num_layers, num_heads, head_dim)
            cake_layers = cake.compress(full_keys, full_values, attention_weights, cr)
            past_kv_cake = _build_hf_past_kv(cake_layers, device, full_seq_len=prefix_len)
            ppl_cake = compute_ppl(model, input_ids, past_kv_cake, prefix_len)
            del past_kv_cake; clear_gpu()

            entry = {
                "context_length": ctx_len, "prefix_len": prefix_len,
                "compression_ratio": cr,
                "ppl_full": round(ppl_full, 4),
                "ppl_layer_budget": round(ppl_lb, 4),
                "ppl_h2o": round(ppl_h2o, 4),
                "ppl_cake": round(ppl_cake, 4),
                "ratio_lb": round(ppl_lb / ppl_full, 4),
                "ratio_h2o": round(ppl_h2o / ppl_full, 4),
                "ratio_cake": round(ppl_cake / ppl_full, 4),
            }
            all_results.append(entry)
            print(f"  CR={cr}x: Full={ppl_full:.2f} | LB={ppl_lb:.2f} ({ppl_lb/ppl_full:.3f}x) | "
                  f"H2O={ppl_h2o:.2f} ({ppl_h2o/ppl_full:.3f}x) | "
                  f"CAKE={ppl_cake:.2f} ({ppl_cake/ppl_full:.3f}x)")

        del full_keys, full_values, attention_weights; clear_gpu()

    # Print summary
    print(f"\n{'='*70}")
    print(f"{'CtxLen':>8s} {'CR':>4s} {'Full':>8s} {'LB':>8s} {'LB%':>8s} "
          f"{'H2O':>8s} {'H2O%':>8s} {'CAKE':>8s} {'CAKE%':>8s}")
    for r in all_results:
        print(f"{r['context_length']:>8d} {r['compression_ratio']:>3.0f}x "
              f"{r['ppl_full']:>7.2f} {r['ppl_layer_budget']:>7.2f} {r['ratio_lb']:>7.3f}x "
              f"{r['ppl_h2o']:>7.2f} {r['ratio_h2o']:>7.3f}x "
              f"{r['ppl_cake']:>7.2f} {r['ratio_cake']:>7.3f}x")

    # Gini stability
    print(f"\nGini stability (std across lengths):")
    for l in [0, num_layers//4, num_layers//2, 3*num_layers//4, num_layers-1]:
        vals = [gini_by_length[cl].get(str(l), 0) for cl in context_lengths]
        std = statistics.stdev(vals) if len(vals) > 1 else 0
        print(f"  Layer {l}: {' → '.join(f'{v:.3f}' for v in vals)} (std={std:.4f})")

    output = {
        "metadata": {
            "experiment": "long_context_validation", "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        },
        "results": all_results,
        "gini_by_length": gini_by_length,
    }
    safe_model = model_name.split("/")[-1].lower().replace("-", "_")
    out_path = RESULTS_DIR / f"long_context_lb_{safe_model}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()
    run_long_context(model_name=args.model, device=args.device, load_in_4bit=args.load_in_4bit)
