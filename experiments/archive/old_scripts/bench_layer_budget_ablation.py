#!/usr/bin/env python3
"""LayerBudget ablation study.

Five ablations:
  1. Component: eviction-only vs quant-only vs joint (LayerBudget)
  2. Signal: sparsity-only vs importance-only vs combined
  3. Precision levels: {4,16} vs {4,8,16}
  4. Token selection: H2O value-norm vs random
  5. Profiling: per-input vs fixed-profile

Each ablation fixes compression ratio at 3x and evaluates perplexity.
"""

import gc
import json
import math
import os
import sys
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from deltacache.core.layer_profiler import LayerAttentionProfiler
from deltacache.core.layer_budget_allocator import (
    LayerBudgetAllocator, sigmoid_importance,
)
from deltacache.core.layer_kv_store import LayerKVStore
from deltacache.core.kv_quantizer import KVQuantizer, QuantPrecision
from deltacache.hf_integration.kv_format import hf_to_deltacache, deltacache_to_hf

RESULTS_DIR = Path(__file__).parent / "results" / "paper"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


EVAL_TEXTS = [
    "The history of artificial intelligence began in antiquity, with myths, stories "
    "and rumors of artificial beings endowed with intelligence or consciousness by "
    "master craftsmen. The seeds of modern AI were planted by philosophers who "
    "attempted to describe the process of human thinking as the mechanical manipulation "
    "of symbols. This work culminated in the invention of the programmable digital "
    "computer in the 1940s, a machine based on the abstract essence of mathematical "
    "reasoning.",

    "In computer science, a hash table is a data structure that implements an "
    "associative array abstract data type, a structure that can map keys to values. "
    "A hash table uses a hash function to compute an index, also called a hash code, "
    "into an array of buckets or slots, from which the desired value can be found. "
    "During lookup, the key is hashed and the resulting hash indicates where the "
    "corresponding value is stored.",

    "Photosynthesis is a process used by plants and other organisms to convert light "
    "energy into chemical energy that, through cellular respiration, can later be "
    "released to fuel the organism's activities. Some of this chemical energy is stored "
    "in carbohydrate molecules, such as sugars and starches, which are synthesized from "
    "carbon dioxide and water.",

    "The theory of general relativity describes gravity not as a force, as understood "
    "by Newtonian physics, but as a consequence of the curvature of spacetime caused "
    "by the uneven distribution of mass. The theory's predictions have been confirmed "
    "in many experiments since Einstein first published the theory in 1915.",

    "Machine learning algorithms build a model based on sample data, known as training "
    "data, in order to make predictions or decisions without being explicitly programmed "
    "to do so. Machine learning algorithms are used in a wide variety of applications, "
    "such as in medicine, email filtering, speech recognition, agriculture, and "
    "computer vision, where it is difficult or infeasible to develop conventional "
    "algorithms to perform the needed tasks.",

    "Quantum computing is a type of computation whose operations can harness the phenomena "
    "of quantum mechanics, such as superposition, interference, and entanglement. Devices "
    "that perform quantum computations are known as quantum computers. Though current "
    "quantum computers may be too small to outperform usual classical computers for "
    "practical applications, larger realizations are believed to be capable of solving "
    "certain computational problems.",

    "DNA is a molecule composed of two polynucleotide chains that coil around each other "
    "to form a double helix. The molecule carries genetic instructions for the development, "
    "functioning, growth and reproduction of all known organisms and many viruses.",

    "The Internet protocol suite, commonly known as TCP/IP, is the set of communication "
    "protocols used in the Internet and similar computer networks. The current foundational "
    "protocols in the suite are the Transmission Control Protocol and the Internet Protocol.",
]


def get_eval_tokens(tokenizer, max_tokens=256):
    tokenized = []
    for text in EVAL_TEXTS:
        tokens = tokenizer.encode(text)[:max_tokens]
        if len(tokens) >= 32:
            tokenized.append(tokens)
    return tokenized


def _build_hf_past_kv(
    layers_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
    full_seq_len: Optional[int] = None,
) -> Any:
    """Convert per-layer (K, V, indices) to HF DynamicCache."""
    from transformers.cache_utils import DynamicCache

    token_counts = [k.shape[1] if k.dim() == 4 else k.shape[0] for k, v, _ in layers_data]
    seq_len = full_seq_len if full_seq_len else max(token_counts)

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

        if n_tokens == seq_len:
            k_out = k.transpose(1, 2)
            v_out = v.transpose(1, 2)
        else:
            k_full = torch.zeros(1, seq_len, num_heads, head_dim, dtype=k.dtype, device=device)
            v_full = torch.zeros(1, seq_len, num_heads, head_dim, dtype=v.dtype, device=device)
            idx = indices.long().to(device)
            valid_idx = idx[idx < seq_len]
            valid_count = valid_idx.shape[0]
            if valid_count > 0:
                k_full[0, valid_idx] = k[0, :valid_count]
                v_full[0, valid_idx] = v[0, :valid_count]
            k_out = k_full.transpose(1, 2)
            v_out = v_full.transpose(1, 2)
        cache.update(k_out, v_out, layer_idx)
    return cache


def compute_ppl(model, input_ids, compressed_past_kv, prefix_len):
    """Compute perplexity on suffix tokens using compressed prefix KV."""
    suffix_ids = input_ids[:, prefix_len:]
    if suffix_ids.shape[1] == 0:
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


def compress_layer_budget(
    full_keys, full_values, attention_weights,
    num_layers, num_heads, head_dim, seq_len,
    compression_ratio, device,
    *,
    available_bits=None,
    sparsity_mode="gini",
    importance_mode="sigmoid",
    token_selection="h2o",
):
    """Flexible LayerBudget compression with ablation controls.

    Args:
        sparsity_mode: "gini" (default), "uniform" (all 0.5), or "inverted" (1 - gini)
        importance_mode: "sigmoid" (default), "uniform" (all 0.5)
        available_bits: List of available precision levels, e.g. [4, 8, 16]
        token_selection: "h2o" (value-norm based) or "random"
    """
    if available_bits is None:
        available_bits = [4, 8, 16]

    profiler = LayerAttentionProfiler()
    allocator = LayerBudgetAllocator(
        num_layers, num_heads, head_dim,
        available_bits=available_bits,
    )
    store = LayerKVStore(num_layers, num_heads, head_dim)

    # Sparsity signal
    profile_result = profiler.profile_from_attention_weights(attention_weights)
    gini = profile_result.gini_scores()

    if sparsity_mode == "gini":
        sparsity = gini
    elif sparsity_mode == "uniform":
        sparsity = {l: 0.5 for l in range(num_layers)}
    elif sparsity_mode == "inverted":
        sparsity = {l: 1.0 - gini.get(l, 0.5) for l in range(num_layers)}
    else:
        sparsity = gini

    # Importance signal
    if importance_mode == "sigmoid":
        importance = allocator.compute_importance_weights(num_layers)
    elif importance_mode == "uniform":
        importance = {l: 0.5 for l in range(num_layers)}
    else:
        importance = allocator.compute_importance_weights(num_layers)

    full_mem = allocator.full_memory(seq_len)
    budget = int(full_mem / compression_ratio)
    allocation = allocator.allocate(sparsity, importance, budget, seq_len)

    # Token selection
    if token_selection == "random":
        def random_selector(layer_idx, keys, values, n_tokens):
            s = keys.shape[1]
            if n_tokens >= s:
                return torch.arange(s)
            perm = torch.randperm(s)[:n_tokens]
            return perm.sort()[0]
        store.store_from_full_cache(
            full_keys, full_values, allocation.allocations,
            token_selector=random_selector,
        )
    else:
        store.store_from_full_cache(full_keys, full_values, allocation.allocations)

    layers_data = store.get_all_layers()
    past_kv = _build_hf_past_kv(layers_data, device, full_seq_len=seq_len)
    return past_kv, store.memory_usage()


def compress_eviction_only(
    full_keys, full_values, attention_weights,
    num_layers, num_heads, head_dim, seq_len,
    compression_ratio, device,
):
    """Eviction-only: per-layer token budgets, all FP16 (no quantization)."""
    return compress_layer_budget(
        full_keys, full_values, attention_weights,
        num_layers, num_heads, head_dim, seq_len,
        compression_ratio, device,
        available_bits=[16],  # Force FP16 only
    )


def compress_quant_only(
    full_keys, full_values, attention_weights,
    num_layers, num_heads, head_dim, seq_len,
    compression_ratio, device,
):
    """Quant-only: keep all tokens, per-layer quantization precision."""
    profiler = LayerAttentionProfiler()
    allocator = LayerBudgetAllocator(
        num_layers, num_heads, head_dim,
        available_bits=[4, 8, 16],
    )
    store = LayerKVStore(num_layers, num_heads, head_dim)

    profile_result = profiler.profile_from_attention_weights(attention_weights)
    sparsity = profile_result.gini_scores()
    importance = allocator.compute_importance_weights(num_layers)
    full_mem = allocator.full_memory(seq_len)
    budget = int(full_mem / compression_ratio)

    # Force all layers to keep all tokens → only quantization can reduce memory
    # Use allocator but set token_step very high so it never adds tokens
    # Instead, manually compute per-layer quantization from sparsity/importance
    # Simple heuristic: rank layers by importance, assign highest bits to most important
    layer_scores = []
    for l in range(num_layers):
        imp = importance.get(l, 0.5)
        gin = sparsity.get(l, 0.5)
        # Higher importance + lower sparsity → needs higher precision
        score = imp * (1.0 - gin)
        layer_scores.append((l, score))
    layer_scores.sort(key=lambda x: x[1], reverse=True)

    # Greedily assign precision: start all at INT4, upgrade to INT8/FP16
    bits_assigned = [4] * num_layers
    total_mem = sum(allocator.memory_cost(seq_len, 4) for _ in range(num_layers))

    for l, score in layer_scores:
        if total_mem >= budget:
            break
        # Try upgrading to INT8
        old = allocator.memory_cost(seq_len, bits_assigned[l])
        new8 = allocator.memory_cost(seq_len, 8)
        if total_mem - old + new8 <= budget:
            total_mem = total_mem - old + new8
            bits_assigned[l] = 8
            # Try upgrading to FP16
            new16 = allocator.memory_cost(seq_len, 16)
            if total_mem - new8 + new16 <= budget:
                total_mem = total_mem - new8 + new16
                bits_assigned[l] = 16

    for l in range(num_layers):
        indices = torch.arange(seq_len)
        store.store_layer(l, full_keys[l:l+1], full_values[l:l+1], indices, quant_bits=bits_assigned[l])

    layers_data = store.get_all_layers()
    past_kv = _build_hf_past_kv(layers_data, device, full_seq_len=seq_len)
    return past_kv, store.memory_usage()


def compress_full_kv(full_keys, full_values, device):
    """Full KV baseline (no compression)."""
    past_kv = deltacache_to_hf(full_keys, full_values, add_batch_dim=True)
    mem = full_keys.numel() * full_keys.element_size() * 2
    return past_kv, mem


# ─── Ablation runners ───────────────────────────────────────────

def run_ablation_1_components(
    model, tokenizer, eval_tokens,
    num_layers, num_heads, head_dim,
    device, compression_ratio,
):
    """Ablation 1: eviction-only vs quant-only vs joint."""
    print("\n--- Ablation 1: Component Ablation ---")
    methods = {
        "full_kv": None,
        "eviction_only": lambda fk, fv, aw, sl: compress_eviction_only(
            fk, fv, aw, num_layers, num_heads, head_dim, sl, compression_ratio, device),
        "quant_only": lambda fk, fv, aw, sl: compress_quant_only(
            fk, fv, aw, num_layers, num_heads, head_dim, sl, compression_ratio, device),
        "joint": lambda fk, fv, aw, sl: compress_layer_budget(
            fk, fv, aw, num_layers, num_heads, head_dim, sl, compression_ratio, device),
    }
    return _run_ablation(model, tokenizer, eval_tokens, methods, device, num_layers, num_heads, head_dim)


def run_ablation_2_signals(
    model, tokenizer, eval_tokens,
    num_layers, num_heads, head_dim,
    device, compression_ratio,
):
    """Ablation 2: sparsity-only vs importance-only vs combined."""
    print("\n--- Ablation 2: Signal Ablation ---")
    methods = {
        "full_kv": None,
        "sparsity_only": lambda fk, fv, aw, sl: compress_layer_budget(
            fk, fv, aw, num_layers, num_heads, head_dim, sl, compression_ratio, device,
            sparsity_mode="gini", importance_mode="uniform"),
        "importance_only": lambda fk, fv, aw, sl: compress_layer_budget(
            fk, fv, aw, num_layers, num_heads, head_dim, sl, compression_ratio, device,
            sparsity_mode="uniform", importance_mode="sigmoid"),
        "combined": lambda fk, fv, aw, sl: compress_layer_budget(
            fk, fv, aw, num_layers, num_heads, head_dim, sl, compression_ratio, device,
            sparsity_mode="gini", importance_mode="sigmoid"),
    }
    return _run_ablation(model, tokenizer, eval_tokens, methods, device, num_layers, num_heads, head_dim)


def run_ablation_3_precision_levels(
    model, tokenizer, eval_tokens,
    num_layers, num_heads, head_dim,
    device, compression_ratio,
):
    """Ablation 3: precision level sets {4,16} vs {4,8,16}."""
    print("\n--- Ablation 3: Precision Levels ---")
    methods = {
        "full_kv": None,
        "bits_4_16": lambda fk, fv, aw, sl: compress_layer_budget(
            fk, fv, aw, num_layers, num_heads, head_dim, sl, compression_ratio, device,
            available_bits=[4, 16]),
        "bits_4_8_16": lambda fk, fv, aw, sl: compress_layer_budget(
            fk, fv, aw, num_layers, num_heads, head_dim, sl, compression_ratio, device,
            available_bits=[4, 8, 16]),
    }
    return _run_ablation(model, tokenizer, eval_tokens, methods, device, num_layers, num_heads, head_dim)


def run_ablation_4_token_selection(
    model, tokenizer, eval_tokens,
    num_layers, num_heads, head_dim,
    device, compression_ratio,
):
    """Ablation 4: H2O token selection vs random."""
    print("\n--- Ablation 4: Token Selection ---")
    methods = {
        "full_kv": None,
        "h2o_selection": lambda fk, fv, aw, sl: compress_layer_budget(
            fk, fv, aw, num_layers, num_heads, head_dim, sl, compression_ratio, device,
            token_selection="h2o"),
        "random_selection": lambda fk, fv, aw, sl: compress_layer_budget(
            fk, fv, aw, num_layers, num_heads, head_dim, sl, compression_ratio, device,
            token_selection="random"),
    }
    return _run_ablation(model, tokenizer, eval_tokens, methods, device, num_layers, num_heads, head_dim)


def run_ablation_5_profile_stability(
    model, tokenizer, eval_tokens,
    num_layers, num_heads, head_dim,
    device, compression_ratio,
):
    """Ablation 5: per-input profiling vs fixed profile from first input."""
    print("\n--- Ablation 5: Profile Stability ---")

    # Get fixed profile from first eval text
    first_ids = torch.tensor([eval_tokens[0]], device=device)
    with torch.no_grad():
        out = model(
            input_ids=first_ids,
            output_attentions=True,
            return_dict=True,
        )
    fixed_attn = list(out.attentions)
    profiler = LayerAttentionProfiler()
    fixed_profile = profiler.profile_from_attention_weights(fixed_attn)
    fixed_gini = fixed_profile.gini_scores()
    del out
    clear_gpu()

    methods = {
        "full_kv": None,
        "per_input_profile": lambda fk, fv, aw, sl: compress_layer_budget(
            fk, fv, aw, num_layers, num_heads, head_dim, sl, compression_ratio, device),
        "fixed_profile": lambda fk, fv, aw, sl: _compress_with_fixed_profile(
            fk, fv, fixed_gini,
            num_layers, num_heads, head_dim, sl, compression_ratio, device),
    }
    return _run_ablation(model, tokenizer, eval_tokens, methods, device, num_layers, num_heads, head_dim)


def _compress_with_fixed_profile(
    full_keys, full_values, fixed_gini,
    num_layers, num_heads, head_dim, seq_len,
    compression_ratio, device,
):
    """Compress using a fixed sparsity profile (not input-specific)."""
    allocator = LayerBudgetAllocator(num_layers, num_heads, head_dim)
    store = LayerKVStore(num_layers, num_heads, head_dim)
    importance = allocator.compute_importance_weights(num_layers)
    full_mem = allocator.full_memory(seq_len)
    budget = int(full_mem / compression_ratio)
    allocation = allocator.allocate(fixed_gini, importance, budget, seq_len)
    store.store_from_full_cache(full_keys, full_values, allocation.allocations)
    layers_data = store.get_all_layers()
    past_kv = _build_hf_past_kv(layers_data, device, full_seq_len=seq_len)
    return past_kv, store.memory_usage()


def _run_ablation(model, tokenizer, eval_tokens, methods, device, num_layers, num_heads, head_dim):
    """Common ablation runner: evaluate all methods on all eval texts."""
    results = {m: [] for m in methods}
    full_ppls = []

    for tidx, tokens in enumerate(tqdm(eval_tokens, desc="  Texts")):
        input_ids = torch.tensor([tokens], device=device)
        seq_len = len(tokens)
        prefix_len = int(seq_len * 0.6)
        prefix_ids = input_ids[:, :prefix_len]

        with torch.no_grad():
            outputs = model(
                input_ids=prefix_ids,
                output_attentions=True,
                return_dict=True,
            )

        kv_cache = outputs.past_key_values
        attention_weights = list(outputs.attentions)
        full_keys, full_values = hf_to_deltacache(kv_cache)
        del outputs, kv_cache
        clear_gpu()

        for method_name, compress_fn in methods.items():
            if method_name == "full_kv":
                past_kv, mem = compress_full_kv(full_keys, full_values, device)
            else:
                past_kv, mem = compress_fn(full_keys, full_values, attention_weights, prefix_len)

            ppl = compute_ppl(model, input_ids, past_kv, prefix_len)
            results[method_name].append({"ppl": ppl, "memory_bytes": mem})

            if method_name == "full_kv":
                full_ppls.append(ppl)

            del past_kv
            clear_gpu()

        del full_keys, full_values, attention_weights
        clear_gpu()

    # Aggregate
    summary = {}
    ref_ppl = statistics.mean(full_ppls) if full_ppls else 1.0
    for method_name, entries in results.items():
        avg_ppl = statistics.mean(e["ppl"] for e in entries)
        avg_mem = statistics.mean(e["memory_bytes"] for e in entries)
        summary[method_name] = {
            "perplexity": round(avg_ppl, 4),
            "ppl_ratio": round(avg_ppl / ref_ppl, 4) if ref_ppl > 0 else 0,
            "memory_kb": round(avg_mem / 1024, 1),
        }

    return summary


def run_all_ablations(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device: str = "cuda:0",
    compression_ratio: float = 3.0,
    load_in_4bit: bool = False,
):
    """Run all ablation experiments."""
    print(f"\n{'='*70}")
    print(f"LayerBudget Ablation Study")
    print(f"Model: {model_name}   CR: {compression_ratio}x   4-bit: {load_in_4bit}")
    print(f"{'='*70}\n")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    load_kwargs = dict(
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = device
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()

    num_layers = model.config.num_hidden_layers
    num_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    print(f"Model: {num_layers}L, {num_heads}H, {head_dim}D")

    eval_tokens = get_eval_tokens(tokenizer)
    print(f"Eval texts: {len(eval_tokens)}")

    common = (model, tokenizer, eval_tokens, num_layers, num_heads, head_dim, device, compression_ratio)

    all_results = {}

    # Ablation 1: Components
    all_results["component"] = run_ablation_1_components(*common)
    _print_results("Component", all_results["component"])

    # Ablation 2: Signals
    all_results["signal"] = run_ablation_2_signals(*common)
    _print_results("Signal", all_results["signal"])

    # Ablation 3: Precision levels
    all_results["precision"] = run_ablation_3_precision_levels(*common)
    _print_results("Precision Levels", all_results["precision"])

    # Ablation 4: Token selection
    all_results["selection"] = run_ablation_4_token_selection(*common)
    _print_results("Token Selection", all_results["selection"])

    # Ablation 5: Profile stability
    all_results["profile"] = run_ablation_5_profile_stability(*common)
    _print_results("Profile Stability", all_results["profile"])

    # Save
    output = {
        "metadata": {
            "experiment": "layer_budget_ablation",
            "model": model_name,
            "compression_ratio": compression_ratio,
            "load_in_4bit": load_in_4bit,
            "num_layers": num_layers,
            "num_texts": len(eval_tokens),
            "timestamp": datetime.now().isoformat(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        },
        "ablations": all_results,
    }

    safe_model = model_name.split("/")[-1].lower().replace("-", "_")
    out_path = RESULTS_DIR / f"ablation_{safe_model}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nAll results saved to {out_path}")
    return output


def _print_results(name, results):
    print(f"\n  {name}:")
    print(f"  {'Method':<25s} {'PPL':>10s} {'Ratio':>10s} {'Memory':>10s}")
    print(f"  {'-'*55}")
    for method, data in results.items():
        print(f"  {method:<25s} {data['perplexity']:10.4f} {data['ppl_ratio']:9.4f}x {data['memory_kb']:8.1f}KB")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--compression", type=float, default=3.0)
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    run_all_ablations(
        model_name=args.model,
        device=args.device,
        compression_ratio=args.compression,
        load_in_4bit=args.load_in_4bit,
    )
