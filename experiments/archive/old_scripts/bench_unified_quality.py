#!/usr/bin/env python3
"""Unified quality benchmark: perplexity comparison across all baselines.

Extends bench_layer_budget_quality.py to support all registered baselines.
Uses the proven 70/30 prefix-suffix split with perplexity evaluation.

Usage:
    python bench_unified_quality.py --model tinyllama --cr 2,3,4,6
    python bench_unified_quality.py --model mistral7b --cr 3 --load-in-4bit
"""

import argparse
import gc
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from baselines import REGISTRY, BaselineMethod
from deltacache.core.layer_profiler import LayerAttentionProfiler
from deltacache.core.layer_budget_allocator import LayerBudgetAllocator
from deltacache.core.layer_kv_store import LayerKVStore
from deltacache.hf_integration.kv_format import hf_to_deltacache, deltacache_to_hf

RESULTS_DIR = Path(__file__).parent / "results" / "paper"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_eval_texts(tokenizer, max_texts=20, max_tokens=256):
    """Diverse evaluation texts, tokenized to 100-256 tokens."""
    prompts = [
        "The history of artificial intelligence began in antiquity, with myths, stories "
        "and rumors of artificial beings endowed with intelligence or consciousness by "
        "master craftsmen. The seeds of modern AI were planted by philosophers who "
        "attempted to describe the process of human thinking as the mechanical manipulation "
        "of symbols. This work culminated in the invention of the programmable digital "
        "computer in the 1940s, a machine based on the abstract essence of mathematical "
        "reasoning. The field of AI research was founded at a workshop held on the campus "
        "of Dartmouth College during the summer of 1956.",

        "In computer science, a hash table is a data structure that implements an "
        "associative array abstract data type, a structure that can map keys to values. "
        "A hash table uses a hash function to compute an index, also called a hash code, "
        "into an array of buckets or slots, from which the desired value can be found. "
        "During lookup, the key is hashed and the resulting hash indicates where the "
        "corresponding value is stored. Ideally, the hash function will assign each key "
        "to a unique bucket, but most hash table designs employ an imperfect hash function.",

        "Photosynthesis is a process used by plants and other organisms to convert light "
        "energy into chemical energy that, through cellular respiration, can later be "
        "released to fuel the organism's activities. Some of this chemical energy is stored "
        "in carbohydrate molecules, such as sugars and starches, which are synthesized from "
        "carbon dioxide and water. In most cases, oxygen is also released as a waste product "
        "that sustains aerobic life. Most plants, algae, and cyanobacteria perform photosynthesis.",

        "The theory of general relativity describes gravity not as a force, as understood "
        "by Newtonian physics, but as a consequence of the curvature of spacetime caused "
        "by the uneven distribution of mass. The theory's predictions have been confirmed "
        "in many experiments since Einstein first published the theory in 1915. Among the "
        "most famous tests are the perihelion precession of Mercury, the deflection of light "
        "by the Sun, and the gravitational redshift of light.",

        "Machine learning algorithms build a model based on sample data, known as training "
        "data, in order to make predictions or decisions without being explicitly programmed "
        "to do so. Machine learning algorithms are used in a wide variety of applications, "
        "such as in medicine, email filtering, speech recognition, agriculture, and "
        "computer vision, where it is difficult or infeasible to develop conventional "
        "algorithms to perform the needed tasks. A subset of machine learning is closely "
        "related to computational statistics.",

        "The operating system serves as an intermediary between the user and the computer "
        "hardware. The purpose of an operating system is to provide an environment in which "
        "a user can execute programs conveniently and efficiently. Memory management is crucial "
        "for operating system performance, particularly the virtual memory system which allows "
        "programs to use more memory than physically available. The operating system must handle "
        "page faults and manage the translation lookaside buffer for efficient address mapping.",

        "Quantum computing is a type of computation whose operations can harness the phenomena "
        "of quantum mechanics, such as superposition, interference, and entanglement. Devices "
        "that perform quantum computations are known as quantum computers. Though current "
        "quantum computers may be too small to outperform usual classical computers for "
        "practical applications, larger realizations are believed to be capable of solving "
        "certain computational problems substantially faster than any classical computer.",

        "The transformer architecture has revolutionized natural language processing since its "
        "introduction in 2017. The key innovation is the self-attention mechanism, which allows "
        "the model to weigh the importance of different parts of the input sequence when producing "
        "each output element. This parallel processing capability makes transformers significantly "
        "more efficient to train than recurrent architectures. Modern large language models like "
        "GPT and BERT are all based on the transformer architecture.",
    ]

    tokenized = []
    for text in prompts[:max_texts]:
        ids = tokenizer.encode(text)
        if len(ids) > max_tokens:
            ids = ids[:max_tokens]
        if len(ids) >= 64:
            tokenized.append(ids)
    return tokenized


def compute_perplexity_with_compressed_kv(model, input_ids, compressed_past_kv, prefix_len):
    """Perplexity on suffix tokens using compressed prefix KV."""
    suffix_ids = input_ids[:, prefix_len:]
    if suffix_ids.shape[1] <= 1:
        return 1.0

    with torch.no_grad():
        position_ids = torch.arange(
            prefix_len, prefix_len + suffix_ids.shape[1],
            device=input_ids.device,
        ).unsqueeze(0)

        outputs = model(
            input_ids=suffix_ids,
            past_key_values=compressed_past_kv,
            position_ids=position_ids,
            return_dict=True,
        )

    logits = outputs.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = suffix_ids[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="mean",
    )
    return math.exp(min(loss.item(), 20))


def build_hf_past_kv(layers_data, device, full_seq_len,
                     full_keys=None, full_values=None):
    """Convert per-layer (K, V, indices) to HF DynamicCache."""
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()

    for layer_idx, (keys, values, indices) in enumerate(layers_data):
        k = keys.to(device)
        v = values.to(device)
        if k.dim() == 3:
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)

        n_tokens = k.shape[1]
        num_heads = k.shape[2]
        head_dim = k.shape[3]

        if n_tokens == full_seq_len:
            k_out = k.transpose(1, 2)
            v_out = v.transpose(1, 2)
        else:
            k_full = torch.zeros(1, full_seq_len, num_heads, head_dim, dtype=k.dtype, device=device)
            v_full = torch.zeros(1, full_seq_len, num_heads, head_dim, dtype=v.dtype, device=device)

            if full_keys is not None and full_values is not None:
                k_full[0] = full_keys[layer_idx].to(device)
                v_full[0] = full_values[layer_idx].to(device)

            idx = indices.long().to(device)
            valid_idx = idx[idx < full_seq_len]
            valid_count = valid_idx.shape[0]
            if valid_count > 0:
                k_full[0, valid_idx] = k[0, :valid_count]
                v_full[0, valid_idx] = v[0, :valid_count]

            k_out = k_full.transpose(1, 2)
            v_out = v_full.transpose(1, 2)

        cache.update(k_out, v_out, layer_idx)
    return cache


def compress_with_baseline(
    method_name, full_keys, full_values, attention_weights,
    num_layers, num_heads, head_dim, seq_len, compression_ratio,
    hidden_states=None,
):
    """Compress using any baseline method. Returns (layers_data, memory_bytes)."""
    if method_name == "full_kv":
        layers = [(full_keys[l:l+1], full_values[l:l+1], torch.arange(seq_len))
                  for l in range(num_layers)]
        mem = full_keys.numel() * full_keys.element_size() * 2
        return layers, mem

    elif method_name == "layer_budget":
        profiler = LayerAttentionProfiler()
        allocator = LayerBudgetAllocator(num_layers, num_heads, head_dim)
        store = LayerKVStore(num_layers, num_heads, head_dim)

        profile_result = profiler.profile_from_attention_weights(attention_weights)
        sparsity = profile_result.gini_scores()
        importance = allocator.compute_importance_weights(num_layers)
        full_mem = allocator.full_memory(seq_len)
        budget = int(full_mem / compression_ratio)
        allocation = allocator.allocate(sparsity, importance, budget, seq_len)
        store.store_from_full_cache(full_keys, full_values, allocation.allocations)

        layers = store.get_all_layers()
        return layers, store.memory_usage()

    elif method_name in REGISTRY:
        cls = REGISTRY[method_name]
        baseline = cls(num_layers, num_heads, head_dim)
        kwargs = {}
        if cls.requires_attention:
            kwargs["attention_weights"] = attention_weights
        if getattr(cls, "requires_hidden_states", False):
            kwargs["hidden_states"] = hidden_states
        layers = baseline.compress(full_keys, full_values, compression_ratio, **kwargs)
        mem = baseline.memory_bytes(layers)
        return layers, mem

    raise ValueError(f"Unknown method: {method_name}")


def run_experiment(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device="cuda:0",
    compression_ratios=None,
    methods=None,
    load_in_4bit=False,
    max_texts=8,
):
    """Run perplexity evaluation across all methods and CRs."""
    if compression_ratios is None:
        compression_ratios = [2.0, 3.0, 4.0, 6.0]

    print(f"\n{'='*80}")
    print(f"Unified Quality Benchmark")
    print(f"Model: {model_name}")
    print(f"CRs: {compression_ratios}")
    print(f"4-bit: {load_in_4bit}")
    print(f"{'='*80}\n")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        )
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = device
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()
    model_device = next(model.parameters()).device

    num_layers = model.config.num_hidden_layers
    num_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    print(f"Config: {num_layers}L, {num_heads}H, {head_dim}D")

    eval_texts = get_eval_texts(tokenizer, max_texts=max_texts)
    print(f"Eval texts: {len(eval_texts)}, avg tokens: {statistics.mean(len(t) for t in eval_texts):.0f}")

    # Resolve methods
    if methods is None or methods == ["all"]:
        methods = ["full_kv"] + sorted(REGISTRY.keys()) + ["layer_budget"]
    elif "full_kv" not in methods:
        methods = ["full_kv"] + methods

    # Remove evolkv by default (too slow for quality eval)
    if "evolkv" in methods:
        methods.remove("evolkv")
        print("Note: Skipping evolkv (too slow for PPL evaluation)")

    print(f"Methods ({len(methods)}): {methods}\n")

    # Results: method -> cr -> list of ppl values
    all_results = {}

    for tidx, tokens in enumerate(tqdm(eval_texts, desc="Texts")):
        input_ids = torch.tensor([tokens], device=model_device)
        seq_len = len(tokens)
        prefix_len = int(seq_len * 0.7)
        if prefix_len < 16 or seq_len - prefix_len < 8:
            continue

        # Get full KV + attention + hidden states
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids[:, :prefix_len],
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )

        full_keys, full_values = hf_to_deltacache(outputs.past_key_values)
        attention_weights = list(outputs.attentions) if outputs.attentions else None
        hidden_states = list(outputs.hidden_states) if outputs.hidden_states else None

        # Reference PPL (full KV)
        ref_kv = deltacache_to_hf(full_keys, full_values, add_batch_dim=True)
        ref_ppl = compute_perplexity_with_compressed_kv(model, input_ids, ref_kv, prefix_len)
        del ref_kv

        for cr in compression_ratios:
            for method in methods:
                key = f"{method}@{cr}"
                if key not in all_results:
                    all_results[key] = {"method": method, "cr": cr, "ppls": [], "ratios": [], "mems": []}

                try:
                    actual_cr = 1.0 if method == "full_kv" else cr
                    layers, mem = compress_with_baseline(
                        method, full_keys, full_values, attention_weights,
                        num_layers, num_heads, head_dim, prefix_len, actual_cr,
                        hidden_states=hidden_states,
                    )

                    past_kv = build_hf_past_kv(
                        layers, model_device, prefix_len,
                        full_keys=full_keys, full_values=full_values,
                    )
                    ppl = compute_perplexity_with_compressed_kv(model, input_ids, past_kv, prefix_len)
                    ratio = ppl / max(1e-6, ref_ppl)
                    del past_kv

                    all_results[key]["ppls"].append(ppl)
                    all_results[key]["ratios"].append(ratio)
                    all_results[key]["mems"].append(mem)

                except Exception as e:
                    print(f"  ERROR {method}@{cr}x text {tidx}: {e}")
                    import traceback
                    traceback.print_exc()

        del outputs, full_keys, full_values, attention_weights, hidden_states
        clear_gpu()

    # Build summary
    summary = []
    for key, data in sorted(all_results.items()):
        if not data["ppls"]:
            continue
        summary.append({
            "method": data["method"],
            "compression_ratio": data["cr"],
            "mean_ppl": round(statistics.mean(data["ppls"]), 4),
            "std_ppl": round(statistics.stdev(data["ppls"]), 4) if len(data["ppls"]) > 1 else 0,
            "mean_ppl_ratio": round(statistics.mean(data["ratios"]), 4),
            "mean_memory_kb": round(statistics.mean(data["mems"]) / 1024, 1),
            "n_texts": len(data["ppls"]),
        })

    # Sort by CR then PPL ratio
    summary.sort(key=lambda x: (x["compression_ratio"], x["mean_ppl_ratio"]))

    # Print table
    print(f"\n{'='*90}")
    print(f"{'Method':20s} {'CR':>4s} {'PPL':>8s} {'Ratio':>8s} {'Memory':>8s} {'N':>3s}")
    print("-" * 90)
    for s in summary:
        print(f"{s['method']:20s} {s['compression_ratio']:>4.0f}x {s['mean_ppl']:>8.2f} "
              f"{s['mean_ppl_ratio']:>8.4f} {s['mean_memory_kb']:>7.1f}KB {s['n_texts']:>3d}")
    print("=" * 90)

    # Save
    output = {
        "metadata": {
            "experiment": "unified_quality",
            "model": model_name,
            "device": str(model_device),
            "compression_ratios": compression_ratios,
            "methods": methods,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "num_eval_texts": len(eval_texts),
            "load_in_4bit": load_in_4bit,
            "timestamp": datetime.now().isoformat(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        },
        "summary": summary,
        "detailed_results": {
            k: {"method": v["method"], "cr": v["cr"],
                "ppls": [round(p, 4) for p in v["ppls"]],
                "ratios": [round(r, 4) for r in v["ratios"]]}
            for k, v in all_results.items()
        },
    }

    safe_model = model_name.split("/")[-1].lower().replace("-", "_")
    out_path = RESULTS_DIR / f"unified_quality_{safe_model}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cr", default="2,3,4,6")
    parser.add_argument("--methods", default="all")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--texts", type=int, default=8)
    args = parser.parse_args()

    crs = [float(x) for x in args.cr.split(",")]
    methods = args.methods.split(",") if args.methods != "all" else None

    run_experiment(
        model_name=args.model,
        device=args.device,
        compression_ratios=crs,
        methods=methods,
        load_in_4bit=args.load_in_4bit,
        max_texts=args.texts,
    )
