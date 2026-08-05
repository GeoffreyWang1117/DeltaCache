#!/usr/bin/env python3
"""Unified benchmark: all baselines under controlled comparison.

Compares all registered KV cache compression baselines using identical
model, prompts, metrics, and hardware. Outputs a single JSON with
per-method, per-CR results.

Usage:
    python bench_unified.py --model tinyllama --cr 2,3,4,6
    python bench_unified.py --model mistral7b --methods h2o_uniform,cake,layer_budget
    python bench_unified.py --model tinyllama --cr 3 --methods all --eval ppl
"""

import argparse
import gc
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import baseline registry (triggers all @register_baseline)
from baselines import REGISTRY, BaselineMethod

from deltacache.core.layer_profiler import LayerAttentionProfiler
from deltacache.core.layer_budget_allocator import LayerBudgetAllocator
from deltacache.core.layer_kv_store import LayerKVStore
from deltacache.hf_integration.kv_format import hf_to_deltacache, deltacache_to_hf

RESULTS_DIR = Path(__file__).parent / "results" / "paper"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CONFIGS = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "mistral7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "llama2_7b": "meta-llama/Llama-2-7b-chat-hf",
}


# =========================================================
#  Evaluation texts
# =========================================================

EVAL_PROMPTS = [
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
    "computer vision.",

    "The Internet protocol suite, commonly known as TCP/IP, is the set of communication "
    "protocols used in the Internet and similar computer networks. The current foundational "
    "protocols in the suite are the Transmission Control Protocol and the Internet Protocol. "
    "TCP provides reliable, ordered, and error-checked delivery of a stream of octets "
    "between applications running on hosts communicating via an IP network.",

    "Quantum computing is a type of computation whose operations can harness the phenomena "
    "of quantum mechanics, such as superposition, interference, and entanglement. Devices "
    "that perform quantum computations are known as quantum computers.",

    "The transformer architecture has revolutionized natural language processing since its "
    "introduction in 2017. The key innovation is the self-attention mechanism, which allows "
    "the model to weigh the importance of different parts of the input sequence when producing "
    "each output element. This parallel processing capability makes transformers significantly "
    "more efficient to train than recurrent architectures.",

    "Deep reinforcement learning combines reinforcement learning and deep learning. The field "
    "of research was established after Google DeepMind used deep RL to achieve superhuman "
    "performance in Atari games. Since then, deep RL has been applied to robotics, natural "
    "language processing, computer vision, and game playing. The key challenge in deep RL "
    "is the sample efficiency problem, where agents require millions of interactions with "
    "the environment to learn effective policies. Various methods have been proposed to "
    "address this, including model-based approaches, meta-learning, and transfer learning. "
    "Recent advances have also explored the use of large language models as components in "
    "reinforcement learning systems, enabling more structured reasoning about actions.",

    "The operating system serves as an intermediary between the user and the computer hardware. "
    "The purpose of an operating system is to provide an environment in which a user can "
    "execute programs conveniently and efficiently. An operating system is software that "
    "manages computer hardware. The hardware must provide appropriate mechanisms to ensure "
    "the correct operation of the computer system and to prevent user programs from interfering "
    "with the proper operation of the system. A more common definition is that the operating "
    "system is the one program running at all times on the computer, usually called the kernel. "
    "Memory management is crucial for operating system performance, particularly the virtual "
    "memory system which allows programs to use more memory than physically available.",

    "Cryptography is the practice and study of techniques for secure communication in the "
    "presence of adversarial behavior. More generally, cryptography is about constructing and "
    "analyzing protocols that prevent third parties from reading private messages. Modern "
    "cryptography exists at the intersection of the disciplines of mathematics, computer "
    "science, information security, electrical engineering, digital signal processing, and "
    "physics. Core concepts include encryption, hash functions, digital signatures, and "
    "key exchange protocols. The RSA algorithm, one of the first public-key cryptosystems, "
    "is widely used for secure data transmission and is based on the practical difficulty "
    "of factoring the product of two large prime numbers.",

    "Natural language processing is a subfield of linguistics, computer science, and "
    "artificial intelligence concerned with the interactions between computers and human "
    "language, in particular how to program computers to process and analyze large amounts "
    "of natural language data. The result is a computer capable of understanding the contents "
    "of documents, including the contextual nuances of the language within them. Challenges "
    "in natural language processing frequently involve speech recognition, natural language "
    "understanding, and natural language generation. Recent transformer-based models such as "
    "BERT, GPT, and T5 have achieved state-of-the-art results on many NLP benchmarks.",
]


# =========================================================
#  Data classes
# =========================================================

@dataclass
class MethodResult:
    method: str
    category: str
    is_per_layer: bool
    compression_ratio: float
    actual_compression: float
    # Cosine similarity
    mean_cosine_sim: float
    std_cosine_sim: float
    per_layer_cosine: List[float]
    # Perplexity (optional, requires --eval ppl)
    mean_ppl: Optional[float] = None
    std_ppl: Optional[float] = None
    mean_ppl_ratio: Optional[float] = None
    # Downstream (optional)
    mean_top1_agreement: Optional[float] = None
    # Resources
    memory_bytes: int = 0
    compress_time_ms: float = 0.0
    n_prompts: int = 0


# =========================================================
#  Core evaluation functions
# =========================================================

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def compute_kv_similarity(
    full_keys: torch.Tensor,
    full_values: torch.Tensor,
    compressed_layers: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Tuple[float, List[float]]:
    """Cosine similarity between full and compressed KV per layer."""
    per_layer = []
    num_layers = full_keys.shape[0]

    for l in range(num_layers):
        if l < len(compressed_layers):
            comp_k, comp_v, indices = compressed_layers[l]
            if isinstance(indices, torch.Tensor):
                idx = indices.long()
            else:
                idx = torch.arange(comp_k.shape[1] if comp_k.dim() == 4 else comp_k.shape[0])

            if comp_k.dim() == 4:
                comp_flat = torch.cat([comp_k[0].reshape(-1), comp_v[0].reshape(-1)]).float()
            else:
                comp_flat = torch.cat([comp_k.reshape(-1), comp_v.reshape(-1)]).float()

            full_flat = torch.cat([
                full_keys[l, idx].reshape(-1),
                full_values[l, idx].reshape(-1),
            ]).float()

            if comp_flat.numel() > 0 and full_flat.numel() > 0:
                cos = F.cosine_similarity(comp_flat.unsqueeze(0), full_flat.unsqueeze(0)).item()
            else:
                cos = 1.0
        else:
            cos = 1.0
        per_layer.append(cos)

    return statistics.mean(per_layer), per_layer


def build_hf_past_kv(
    layers_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
    full_seq_len: int,
    full_keys: Optional[torch.Tensor] = None,
    full_values: Optional[torch.Tensor] = None,
):
    """Convert per-layer compressed data to HF DynamicCache."""
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


def compute_perplexity_with_compressed_kv(
    model, input_ids, compressed_past_kv, prefix_len,
) -> float:
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


# =========================================================
#  LayerBudget (ours) — implements BaselineMethod-compatible compress
# =========================================================

def run_layer_budget(
    full_keys, full_values, attention_weights,
    num_layers, num_heads, head_dim, seq_len, compression_ratio,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Run LayerBudget and return in standard format."""
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

    return store.get_all_layers()


# =========================================================
#  Main benchmark
# =========================================================

def load_model(model_name: str, device: str = "cuda"):
    """Load model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = MODEL_CONFIGS.get(model_name, model_name)
    print(f"Loading {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(
        torch_dtype=torch.float16,
        device_map=device,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    # Use 4-bit for 7B models on 24GB GPU
    if "7b" in model_path.lower() or "7B" in model_path:
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            del load_kwargs["device_map"]
            load_kwargs["device_map"] = "auto"
        except ImportError:
            pass

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model.eval()
    return model, tokenizer


def extract_model_config(model) -> Tuple[int, int, int]:
    """Extract (num_layers, num_kv_heads, head_dim) from model."""
    config = model.config
    num_layers = config.num_hidden_layers
    num_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads
    return num_layers, num_heads, head_dim


def run_benchmark(
    model_name: str,
    methods: List[str],
    compression_ratios: List[float],
    eval_ppl: bool = False,
    max_prompts: int = 8,
    device: str = "cuda",
):
    """Main benchmark loop."""
    model, tokenizer = load_model(model_name, device)
    num_layers, num_heads, head_dim = extract_model_config(model)
    model_device = next(model.parameters()).device

    print(f"Model: {model_name} ({num_layers}L, {num_heads}H, {head_dim}D)")
    print(f"Methods: {methods}")
    print(f"Compression ratios: {compression_ratios}")
    print(f"Eval PPL: {eval_ppl}")
    print(f"Registered baselines: {list(REGISTRY.keys())}")
    print()

    # Resolve methods
    if methods == ["all"]:
        active_methods = list(REGISTRY.keys()) + ["layer_budget"]
    else:
        active_methods = methods

    # Check which need attention / hidden states
    needs_attention = any(
        m in REGISTRY and REGISTRY[m].requires_attention
        for m in active_methods
    ) or "layer_budget" in active_methods
    needs_hidden_states = any(
        m in REGISTRY and getattr(REGISTRY[m], "requires_hidden_states", False)
        for m in active_methods
    )

    # Tokenize prompts — concatenate short prompts to reach min length
    min_seq_len = 128
    texts = EVAL_PROMPTS[:max_prompts]
    all_results: Dict[str, Dict[float, List[dict]]] = {}

    for text_idx, text in enumerate(texts):
        input_ids = tokenizer.encode(text, return_tensors="pt").to(model_device)
        # If too short, repeat text to reach minimum length
        while input_ids.shape[1] < min_seq_len:
            extra = tokenizer.encode(text, return_tensors="pt").to(model_device)
            input_ids = torch.cat([input_ids, extra[:, 1:]], dim=1)
        # Truncate to reasonable length
        if input_ids.shape[1] > 512:
            input_ids = input_ids[:, :512]
        seq_len = input_ids.shape[1]
        prefix_len = int(seq_len * 0.6)

        print(f"[{text_idx+1}/{len(texts)}] seq_len={seq_len}, prefix={prefix_len}")

        # Prefill: get full KV + attention weights
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids[:, :prefix_len],
                output_attentions=needs_attention,
                output_hidden_states=needs_hidden_states,
                return_dict=True,
            )

        full_keys, full_values = hf_to_deltacache(outputs.past_key_values)
        attention_weights = list(outputs.attentions) if needs_attention and outputs.attentions else None
        hidden_states = list(outputs.hidden_states) if needs_hidden_states and hasattr(outputs, 'hidden_states') and outputs.hidden_states else None

        for cr in compression_ratios:
            for method_name in active_methods:
                try:
                    if method_name == "layer_budget":
                        t0 = time.perf_counter()
                        layers = run_layer_budget(
                            full_keys, full_values, attention_weights,
                            num_layers, num_heads, head_dim, prefix_len, cr,
                        )
                        elapsed = (time.perf_counter() - t0) * 1000
                        mem = sum(k.numel() * k.element_size() + v.numel() * v.element_size()
                                  for k, v, _ in layers)
                        category = "joint"
                        is_per_layer = True
                    elif method_name in REGISTRY:
                        cls = REGISTRY[method_name]
                        baseline = cls(num_layers, num_heads, head_dim)
                        result = baseline.compress_timed(
                            full_keys, full_values, cr,
                            attention_weights=attention_weights,
                            hidden_states=hidden_states,
                        )
                        layers = result.layers
                        elapsed = result.compress_time_ms
                        mem = result.memory_bytes
                        category = cls.category
                        is_per_layer = cls.is_per_layer
                    else:
                        print(f"  Skipping unknown method: {method_name}")
                        continue

                    # Cosine similarity
                    sim, per_layer_sim = compute_kv_similarity(full_keys, full_values, layers)
                    full_mem = full_keys.numel() * full_keys.element_size() * 2
                    actual_cr = full_mem / max(1, mem)

                    # Perplexity (optional)
                    ppl = None
                    ppl_ratio = None
                    if eval_ppl:
                        past_kv = build_hf_past_kv(
                            layers, model_device, prefix_len,
                            full_keys=full_keys, full_values=full_values,
                        )
                        ppl = compute_perplexity_with_compressed_kv(
                            model, input_ids, past_kv, prefix_len,
                        )
                        # Get full KV perplexity for ratio
                        full_past_kv = build_hf_past_kv(
                            [(full_keys[l:l+1], full_values[l:l+1], torch.arange(prefix_len))
                             for l in range(num_layers)],
                            model_device, prefix_len,
                        )
                        ppl_full = compute_perplexity_with_compressed_kv(
                            model, input_ids, full_past_kv, prefix_len,
                        )
                        ppl_ratio = ppl / max(ppl_full, 1e-6)
                        del past_kv, full_past_kv

                    entry = {
                        "method": method_name,
                        "category": category,
                        "is_per_layer": is_per_layer,
                        "text_idx": text_idx,
                        "compression_ratio": cr,
                        "actual_compression": round(actual_cr, 2),
                        "cosine_sim": round(sim, 6),
                        "per_layer_cosine": [round(s, 4) for s in per_layer_sim],
                        "memory_bytes": mem,
                        "compress_time_ms": round(elapsed, 2),
                        "perplexity": round(ppl, 4) if ppl else None,
                        "ppl_ratio": round(ppl_ratio, 4) if ppl_ratio else None,
                    }

                    key = f"{method_name}_{cr}"
                    if key not in all_results:
                        all_results[key] = []
                    all_results[key].append(entry)

                    print(f"  {method_name} @ {cr}x: sim={sim:.4f} mem={mem/1024:.0f}KB "
                          f"actual_cr={actual_cr:.1f}x t={elapsed:.1f}ms"
                          + (f" ppl={ppl:.2f}" if ppl else ""))

                except Exception as e:
                    print(f"  ERROR {method_name} @ {cr}x: {e}")
                    import traceback
                    traceback.print_exc()

        clear_gpu()

    # Aggregate results
    summary = []
    for key, entries in all_results.items():
        method = entries[0]["method"]
        cr = entries[0]["compression_ratio"]
        sims = [e["cosine_sim"] for e in entries]
        ppls = [e["perplexity"] for e in entries if e["perplexity"] is not None]
        ppl_ratios = [e["ppl_ratio"] for e in entries if e["ppl_ratio"] is not None]

        summary.append({
            "method": method,
            "category": entries[0]["category"],
            "is_per_layer": entries[0]["is_per_layer"],
            "compression_ratio": cr,
            "mean_cosine_sim": round(statistics.mean(sims), 6),
            "std_cosine_sim": round(statistics.stdev(sims), 6) if len(sims) > 1 else 0.0,
            "mean_actual_cr": round(statistics.mean(e["actual_compression"] for e in entries), 2),
            "mean_memory_bytes": int(statistics.mean(e["memory_bytes"] for e in entries)),
            "mean_compress_time_ms": round(statistics.mean(e["compress_time_ms"] for e in entries), 2),
            "mean_ppl": round(statistics.mean(ppls), 4) if ppls else None,
            "std_ppl": round(statistics.stdev(ppls), 4) if len(ppls) > 1 else None,
            "mean_ppl_ratio": round(statistics.mean(ppl_ratios), 4) if ppl_ratios else None,
            "n_prompts": len(entries),
        })

    # Sort summary: by CR then by cosine sim (descending)
    summary.sort(key=lambda x: (x["compression_ratio"], -x["mean_cosine_sim"]))

    # Save
    output = {
        "metadata": {
            "experiment": "unified_baseline_comparison",
            "model": MODEL_CONFIGS.get(model_name, model_name),
            "model_config": {"num_layers": num_layers, "num_heads": num_heads, "head_dim": head_dim},
            "compression_ratios": compression_ratios,
            "methods": active_methods,
            "eval_ppl": eval_ppl,
            "n_prompts": len(texts),
            "timestamp": datetime.now().isoformat(),
            "device": str(model_device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
        "summary": summary,
        "detailed_results": {k: v for k, v in all_results.items()},
    }

    safe_name = model_name.replace("/", "_").replace("-", "_")
    output_path = RESULTS_DIR / f"unified_{safe_name}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Print summary table
    print("\n" + "=" * 100)
    print(f"{'Method':<20} {'Cat':>5} {'CR':>4} {'CosSim':>8} {'ActCR':>6} {'Mem(KB)':>8} {'Time(ms)':>8}"
          + (" {'PPL':>8} {'PPL_R':>7}" if eval_ppl else ""))
    print("-" * 100)
    for s in summary:
        line = (f"{s['method']:<20} {s['category']:>5} {s['compression_ratio']:>4.0f}x "
                f"{s['mean_cosine_sim']:>8.4f} {s['mean_actual_cr']:>5.1f}x "
                f"{s['mean_memory_bytes']/1024:>7.0f} {s['mean_compress_time_ms']:>8.1f}")
        if eval_ppl and s["mean_ppl"]:
            line += f" {s['mean_ppl']:>8.2f} {s['mean_ppl_ratio']:>7.3f}"
        print(line)
    print("=" * 100)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified KV cache baseline benchmark")
    parser.add_argument("--model", default="tinyllama", help="Model name or path")
    parser.add_argument("--methods", default="all", help="Comma-separated methods or 'all'")
    parser.add_argument("--cr", default="2,3,4,6", help="Comma-separated compression ratios")
    parser.add_argument("--eval", default="cosine", choices=["cosine", "ppl", "both"],
                        help="Evaluation mode")
    parser.add_argument("--prompts", type=int, default=8, help="Number of evaluation prompts")
    parser.add_argument("--device", default="cuda", help="Device")
    args = parser.parse_args()

    methods = args.methods.split(",")
    crs = [float(x) for x in args.cr.split(",")]
    eval_ppl = args.eval in ("ppl", "both")

    run_benchmark(
        model_name=args.model,
        methods=methods,
        compression_ratios=crs,
        eval_ppl=eval_ppl,
        max_prompts=args.prompts,
        device=args.device,
    )
