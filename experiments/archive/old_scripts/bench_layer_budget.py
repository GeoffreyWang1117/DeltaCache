#!/usr/bin/env python3
"""LayerBudget main experiment: per-layer (token_budget, quant_bits) allocation.

Compares LayerBudget against uniform baselines and per-layer baselines
(CAKE for eviction, KVTuner for quantization) across compression ratios.

Baselines:
  1. Full KV (no compression)
  2. H2O uniform (uniform token budget, FP16)
  3. KIVI uniform (all tokens, uniform INT4)
  4. H2O + KIVI naive (uniform tokens, uniform INT4)
  5. CAKE (per-layer eviction, FP16)
  6. KVTuner (all tokens, per-layer bits)
  7. LayerBudget (per-layer tokens + per-layer bits) — ours

Metrics: KV reconstruction cosine similarity, memory compression ratio.
"""

import gc
import json
import os
import sys
import time
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from deltacache.core.layer_profiler import LayerAttentionProfiler
from deltacache.core.layer_budget_allocator import LayerBudgetAllocator
from deltacache.core.layer_kv_store import LayerKVStore
from deltacache.core.kv_quantizer import KVQuantizer, QuantPrecision
from deltacache.hf_integration import LlamaStyleAdapter
from deltacache.hf_integration.kv_format import hf_to_deltacache

from baselines.cake import CAKEBaseline
from baselines.kvtuner import KVTunerBaseline

RESULTS_DIR = Path(__file__).parent / "results" / "paper"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


PROMPTS = [
    (
        "You are an expert software engineer. You write clean, well-tested Python code "
        "following PEP 8 guidelines. You include type hints and docstrings. "
        "You prefer composition over inheritance.\n\n"
        "Write a function to parse CSV files."
    ),
    (
        "The Transformer architecture uses self-attention where each position attends to "
        "all other positions. Multi-head attention projects Q, K, V through parallel heads. "
        "Feed-forward networks apply position-wise transformations with ReLU or SwiGLU. "
        "Layer normalization improves training stability.\n\n"
        "Summarize the key components."
    ),
    (
        "import asyncio\nfrom dataclasses import dataclass\n\n"
        "@dataclass\nclass Config:\n    host: str = 'localhost'\n    port: int = 5432\n\n"
        "class Pool:\n    def __init__(self, cfg: Config):\n"
        "        self._cfg = cfg\n        self._pool = None\n\n"
        "    async def init(self):\n        self._pool = await create_pool(self._cfg.host)\n\n"
        "Add error handling to this code."
    ),
    (
        "Classify the sentiment:\n"
        "Text: 'The movie was fantastic!' → Positive\n"
        "Text: 'Terrible service, never again.' → Negative\n"
        "Text: 'It was okay, nothing special.' → Neutral\n"
        "Text: 'Best purchase I ever made!' → Positive\n"
        "Text: 'The food was bland and overpriced.' →"
    ),
    (
        "Paxos consensus protocol: Phase 1 (Prepare) - proposer sends prepare(n) to "
        "acceptors. Acceptors promise not to accept proposals < n. Phase 2 (Accept) - "
        "if majority promises, proposer sends accept(n, v). Multi-Paxos elects a stable "
        "leader to skip Phase 1 for subsequent rounds. Raft simplifies this with leader "
        "election via randomized timeouts and log replication.\n\n"
        "Compare Paxos and Raft."
    ),
    (
        "User: What is machine learning?\n"
        "Assistant: Machine learning is a subset of AI where systems learn patterns from "
        "data rather than being explicitly programmed. It includes supervised learning, "
        "unsupervised learning, and reinforcement learning.\n"
        "User: How is deep learning different?\n"
        "Assistant:"
    ),
]


@dataclass
class CompressionResult:
    """Result for one method at one compression ratio."""

    method: str
    compression_ratio: float
    actual_compression: float
    mean_cosine_sim: float  # Average cosine similarity of compressed vs full KV
    std_cosine_sim: float
    per_layer_cosine: List[float]
    memory_bytes: int
    full_memory_bytes: int
    compress_time_ms: float
    num_prompts: int


def compute_kv_similarity(
    full_keys: torch.Tensor,
    full_values: torch.Tensor,
    compressed_layers: list,
) -> Tuple[float, List[float]]:
    """Compute cosine similarity between full and compressed KV per layer.

    For layers with fewer tokens in the compressed version, we compare
    only the retained positions.

    Returns:
        (mean_cosine_sim, per_layer_cosine_sims)
    """
    per_layer = []
    num_layers = full_keys.shape[0]

    for l in range(num_layers):
        if l < len(compressed_layers):
            comp_k, comp_v, indices = compressed_layers[l]

            # Get corresponding full KV at retained positions
            if isinstance(indices, torch.Tensor):
                idx = indices.long()
            else:
                idx = torch.arange(comp_k.shape[1])

            # Handle shape: comp_k may be (1, n, H, D)
            if comp_k.dim() == 4:
                comp_k_flat = comp_k[0].reshape(-1).float()
                comp_v_flat = comp_v[0].reshape(-1).float()
            else:
                comp_k_flat = comp_k.reshape(-1).float()
                comp_v_flat = comp_v.reshape(-1).float()

            full_k_selected = full_keys[l, idx].reshape(-1).float()
            full_v_selected = full_values[l, idx].reshape(-1).float()

            # Concatenate K and V for overall similarity
            comp_flat = torch.cat([comp_k_flat, comp_v_flat])
            full_flat = torch.cat([full_k_selected, full_v_selected])

            if comp_flat.numel() > 0 and full_flat.numel() > 0:
                cos = F.cosine_similarity(
                    comp_flat.unsqueeze(0),
                    full_flat.unsqueeze(0),
                ).item()
            else:
                cos = 1.0
        else:
            cos = 1.0

        per_layer.append(cos)

    return statistics.mean(per_layer), per_layer


def run_method_full(
    full_keys: torch.Tensor,
    full_values: torch.Tensor,
    attention_weights: list,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    seq_len: int,
) -> Tuple[float, List[float], int, float]:
    """Full KV baseline (no compression)."""
    layers = []
    for l in range(num_layers):
        layers.append((
            full_keys[l:l+1],
            full_values[l:l+1],
            torch.arange(seq_len),
        ))
    mem = full_keys.numel() * full_keys.element_size() * 2
    sim, per_layer = compute_kv_similarity(full_keys, full_values, layers)
    return sim, per_layer, mem, 0.0


def run_method_h2o_uniform(
    full_keys: torch.Tensor,
    full_values: torch.Tensor,
    attention_weights: list,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    seq_len: int,
    compression_ratio: float,
) -> Tuple[float, List[float], int, float]:
    """H2O uniform: same token budget per layer, FP16."""
    token_frac = 1.0 / compression_ratio
    n_tokens = max(1, int(seq_len * token_frac))
    store = LayerKVStore(num_layers, num_heads, head_dim)

    t0 = time.perf_counter()
    for l in range(num_layers):
        indices = store._default_token_selection(
            full_keys[l:l+1], full_values[l:l+1], n_tokens, seq_len,
        )
        store.store_layer(l, full_keys[l:l+1], full_values[l:l+1], indices, quant_bits=16)
    elapsed = (time.perf_counter() - t0) * 1000

    layers = store.get_all_layers()
    sim, per_layer = compute_kv_similarity(full_keys, full_values, layers)
    return sim, per_layer, store.memory_usage(), elapsed


def run_method_kivi_uniform(
    full_keys: torch.Tensor,
    full_values: torch.Tensor,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    seq_len: int,
) -> Tuple[float, List[float], int, float]:
    """KIVI uniform: all tokens, INT4."""
    store = LayerKVStore(num_layers, num_heads, head_dim)

    t0 = time.perf_counter()
    for l in range(num_layers):
        indices = torch.arange(seq_len)
        store.store_layer(l, full_keys[l:l+1], full_values[l:l+1], indices, quant_bits=4)
    elapsed = (time.perf_counter() - t0) * 1000

    layers = store.get_all_layers()
    sim, per_layer = compute_kv_similarity(full_keys, full_values, layers)
    return sim, per_layer, store.memory_usage(), elapsed


def run_method_h2o_kivi_naive(
    full_keys: torch.Tensor,
    full_values: torch.Tensor,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    seq_len: int,
    compression_ratio: float,
) -> Tuple[float, List[float], int, float]:
    """H2O + KIVI naive: uniform token budget + uniform INT4."""
    # Split compression: sqrt for each dimension
    token_ratio = max(1.0, compression_ratio / 2.5)  # ~2.5x from INT4
    token_frac = 1.0 / token_ratio
    n_tokens = max(1, int(seq_len * token_frac))
    store = LayerKVStore(num_layers, num_heads, head_dim)

    t0 = time.perf_counter()
    for l in range(num_layers):
        indices = store._default_token_selection(
            full_keys[l:l+1], full_values[l:l+1], n_tokens, seq_len,
        )
        store.store_layer(l, full_keys[l:l+1], full_values[l:l+1], indices, quant_bits=4)
    elapsed = (time.perf_counter() - t0) * 1000

    layers = store.get_all_layers()
    sim, per_layer = compute_kv_similarity(full_keys, full_values, layers)
    return sim, per_layer, store.memory_usage(), elapsed


def run_method_cake(
    full_keys: torch.Tensor,
    full_values: torch.Tensor,
    attention_weights: list,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    seq_len: int,
    compression_ratio: float,
) -> Tuple[float, List[float], int, float]:
    """CAKE: per-layer eviction, FP16."""
    cake = CAKEBaseline(num_layers, num_heads, head_dim)

    t0 = time.perf_counter()
    layers = cake.compress(full_keys, full_values, attention_weights, compression_ratio)
    elapsed = (time.perf_counter() - t0) * 1000

    mem = sum(k.numel() * k.element_size() + v.numel() * v.element_size()
              for k, v, _ in layers)
    sim, per_layer = compute_kv_similarity(full_keys, full_values, layers)
    return sim, per_layer, mem, elapsed


def run_method_kvtuner(
    full_keys: torch.Tensor,
    full_values: torch.Tensor,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    seq_len: int,
    compression_ratio: float,
) -> Tuple[float, List[float], int, float]:
    """KVTuner: all tokens, per-layer bits."""
    kvtuner = KVTunerBaseline(num_layers, num_heads, head_dim)

    t0 = time.perf_counter()
    result = kvtuner.compress(full_keys, full_values, compression_ratio)
    elapsed = (time.perf_counter() - t0) * 1000

    layers = []
    mem = 0
    for l, (k, v, bits) in enumerate(result):
        indices = torch.arange(seq_len)
        layers.append((k, v, indices))
        mem += k.numel() * k.element_size() + v.numel() * v.element_size()

    sim, per_layer = compute_kv_similarity(full_keys, full_values, layers)
    return sim, per_layer, mem, elapsed


def run_method_layer_budget(
    full_keys: torch.Tensor,
    full_values: torch.Tensor,
    attention_weights: list,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    seq_len: int,
    compression_ratio: float,
) -> Tuple[float, List[float], int, float]:
    """LayerBudget (ours): per-layer tokens + per-layer bits."""
    profiler = LayerAttentionProfiler()
    allocator = LayerBudgetAllocator(num_layers, num_heads, head_dim)
    store = LayerKVStore(num_layers, num_heads, head_dim)

    t0 = time.perf_counter()

    # Profile: extract sparsity from attention weights
    profile_result = profiler.profile_from_attention_weights(attention_weights)
    sparsity = profile_result.gini_scores()

    # Importance: sigmoid weights
    importance = allocator.compute_importance_weights(num_layers)

    # Budget
    full_mem = allocator.full_memory(seq_len)
    budget = int(full_mem / compression_ratio)

    # Allocate
    allocation = allocator.allocate(sparsity, importance, budget, seq_len)

    # Store per-layer
    store.store_from_full_cache(full_keys, full_values, allocation.allocations)

    elapsed = (time.perf_counter() - t0) * 1000

    layers = store.get_all_layers()
    sim, per_layer = compute_kv_similarity(full_keys, full_values, layers)
    return sim, per_layer, store.memory_usage(), elapsed


def run_experiment(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device: str = "cuda:0",
    compression_ratios: Optional[List[float]] = None,
    load_in_4bit: bool = False,
):
    """Run full LayerBudget comparison experiment."""
    if compression_ratios is None:
        compression_ratios = [2.0, 3.0, 4.0, 6.0]

    print(f"\n{'='*70}")
    print(f"LayerBudget Experiment")
    print(f"Model: {model_name}   Device: {device}   4-bit: {load_in_4bit}")
    print(f"Compression ratios: {compression_ratios}")
    print(f"{'='*70}\n")

    # Load model
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
    num_heads = getattr(model.config, "num_key_value_heads",
                        model.config.num_attention_heads)
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    print(f"Model: {num_layers} layers, {num_heads} KV heads, {head_dim} head_dim")

    all_results = []

    for pidx, prompt in enumerate(tqdm(PROMPTS, desc="Processing prompts")):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        seq_len = inputs["input_ids"].shape[1]
        print(f"\n  Prompt {pidx}: {seq_len} tokens")

        # Get full KV and attention weights
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                output_attentions=True,
                return_dict=True,
            )

        # Extract KV cache from past_key_values
        past_kv = outputs.past_key_values
        if past_kv is not None:
            full_keys, full_values = hf_to_deltacache(past_kv)
        else:
            # Fallback: create from hidden states
            print("  Warning: no past_key_values, creating synthetic KV")
            full_keys = torch.randn(num_layers, seq_len, num_heads, head_dim,
                                   dtype=torch.float16, device=device)
            full_values = torch.randn_like(full_keys)

        attention_weights = list(outputs.attentions) if outputs.attentions else []
        full_mem = full_keys.numel() * full_keys.element_size() * 2

        for cr in compression_ratios:
            prompt_results = {}

            # 1. Full KV
            if cr == compression_ratios[0]:  # Only once
                sim, per_layer, mem, t = run_method_full(
                    full_keys, full_values, attention_weights,
                    num_layers, num_heads, head_dim, seq_len,
                )
                all_results.append(CompressionResult(
                    method="full_kv", compression_ratio=1.0,
                    actual_compression=1.0,
                    mean_cosine_sim=sim, std_cosine_sim=0.0,
                    per_layer_cosine=per_layer,
                    memory_bytes=mem, full_memory_bytes=full_mem,
                    compress_time_ms=t, num_prompts=pidx + 1,
                ))

            # 2. H2O uniform
            sim, per_layer, mem, t = run_method_h2o_uniform(
                full_keys, full_values, attention_weights,
                num_layers, num_heads, head_dim, seq_len, cr,
            )
            all_results.append(CompressionResult(
                method="h2o_uniform", compression_ratio=cr,
                actual_compression=full_mem / max(1, mem),
                mean_cosine_sim=sim, std_cosine_sim=0.0,
                per_layer_cosine=per_layer,
                memory_bytes=mem, full_memory_bytes=full_mem,
                compress_time_ms=t, num_prompts=pidx + 1,
            ))

            # 3. KIVI uniform (fixed compression ~2.5x for INT4)
            if cr == compression_ratios[0]:
                sim, per_layer, mem, t = run_method_kivi_uniform(
                    full_keys, full_values,
                    num_layers, num_heads, head_dim, seq_len,
                )
                all_results.append(CompressionResult(
                    method="kivi_uniform", compression_ratio=cr,
                    actual_compression=full_mem / max(1, mem),
                    mean_cosine_sim=sim, std_cosine_sim=0.0,
                    per_layer_cosine=per_layer,
                    memory_bytes=mem, full_memory_bytes=full_mem,
                    compress_time_ms=t, num_prompts=pidx + 1,
                ))

            # 4. H2O + KIVI naive
            sim, per_layer, mem, t = run_method_h2o_kivi_naive(
                full_keys, full_values,
                num_layers, num_heads, head_dim, seq_len, cr,
            )
            all_results.append(CompressionResult(
                method="h2o_kivi_naive", compression_ratio=cr,
                actual_compression=full_mem / max(1, mem),
                mean_cosine_sim=sim, std_cosine_sim=0.0,
                per_layer_cosine=per_layer,
                memory_bytes=mem, full_memory_bytes=full_mem,
                compress_time_ms=t, num_prompts=pidx + 1,
            ))

            # 5. CAKE
            if attention_weights:
                sim, per_layer, mem, t = run_method_cake(
                    full_keys, full_values, attention_weights,
                    num_layers, num_heads, head_dim, seq_len, cr,
                )
                all_results.append(CompressionResult(
                    method="cake", compression_ratio=cr,
                    actual_compression=full_mem / max(1, mem),
                    mean_cosine_sim=sim, std_cosine_sim=0.0,
                    per_layer_cosine=per_layer,
                    memory_bytes=mem, full_memory_bytes=full_mem,
                    compress_time_ms=t, num_prompts=pidx + 1,
                ))

            # 6. KVTuner
            sim, per_layer, mem, t = run_method_kvtuner(
                full_keys, full_values,
                num_layers, num_heads, head_dim, seq_len, cr,
            )
            all_results.append(CompressionResult(
                method="kvtuner", compression_ratio=cr,
                actual_compression=full_mem / max(1, mem),
                mean_cosine_sim=sim, std_cosine_sim=0.0,
                per_layer_cosine=per_layer,
                memory_bytes=mem, full_memory_bytes=full_mem,
                compress_time_ms=t, num_prompts=pidx + 1,
            ))

            # 7. LayerBudget (ours)
            if attention_weights:
                sim, per_layer, mem, t = run_method_layer_budget(
                    full_keys, full_values, attention_weights,
                    num_layers, num_heads, head_dim, seq_len, cr,
                )
                all_results.append(CompressionResult(
                    method="layer_budget", compression_ratio=cr,
                    actual_compression=full_mem / max(1, mem),
                    mean_cosine_sim=sim, std_cosine_sim=0.0,
                    per_layer_cosine=per_layer,
                    memory_bytes=mem, full_memory_bytes=full_mem,
                    compress_time_ms=t, num_prompts=pidx + 1,
                ))

        del outputs, past_kv, attention_weights
        clear_gpu()

    # Aggregate results by method × compression_ratio
    aggregated = {}
    for r in all_results:
        key = (r.method, r.compression_ratio)
        if key not in aggregated:
            aggregated[key] = []
        aggregated[key].append(r)

    print(f"\n\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Method':20s} {'CR':>6s} {'CosSimKV':>10s} {'Memory':>12s} {'Time(ms)':>10s}")
    print("-" * 65)

    summary_results = []
    for (method, cr), results in sorted(aggregated.items()):
        sims = [r.mean_cosine_sim for r in results]
        mems = [r.memory_bytes for r in results]
        times = [r.compress_time_ms for r in results]

        avg_sim = statistics.mean(sims)
        avg_mem = statistics.mean(mems)
        avg_time = statistics.mean(times)

        print(f"{method:20s} {cr:6.1f}x {avg_sim:10.6f} {avg_mem/1024:9.1f}KB {avg_time:10.2f}")

        summary_results.append({
            "method": method,
            "compression_ratio": cr,
            "mean_cosine_sim": round(avg_sim, 6),
            "std_cosine_sim": round(statistics.stdev(sims), 6) if len(sims) > 1 else 0,
            "mean_memory_bytes": int(avg_mem),
            "mean_compress_time_ms": round(avg_time, 2),
            "n_prompts": len(results),
        })

    # Save results
    output = {
        "metadata": {
            "experiment": "layer_budget_comparison",
            "model": model_name,
            "device": device,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "num_prompts": len(PROMPTS),
            "compression_ratios": compression_ratios,
            "timestamp": datetime.now().isoformat(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        },
        "summary": summary_results,
        "detailed_results": [asdict(r) for r in all_results],
    }

    safe_model = model_name.split("/")[-1].lower().replace("-", "_")
    out_path = RESULTS_DIR / f"layer_budget_{safe_model}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--compression", nargs="+", type=float, default=[2.0, 3.0, 4.0, 6.0])
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    run_experiment(
        model_name=args.model,
        device=args.device,
        compression_ratios=args.compression,
        load_in_4bit=args.load_in_4bit,
    )
