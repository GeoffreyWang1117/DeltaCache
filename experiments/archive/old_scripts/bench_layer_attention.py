#!/usr/bin/env python3
"""Layer attention distribution analysis on real models.

Empirically validates the layer heterogeneity hypothesis that the paper
relies on: late layers should show higher/more concentrated attention
scores than early layers, making them more valuable for cache retention.

Measures per-layer attention statistics across diverse prompts:
  - Mean attention entropy (uniform → high entropy, concentrated → low)
  - Mean attention to prefix tokens vs suffix tokens
  - Top-k attention mass (fraction of attention captured by top 10% tokens)
  - Layer-by-layer cosine similarity of attention patterns across prompts

Output: JSON with per-layer statistics + summary for paper Table/Figure.
"""

import gc
import json
import math
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

RESULTS_DIR = Path(__file__).parent / "results" / "paper"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Prompts for analysis
# ---------------------------------------------------------------------------

PROMPTS = [
    # Long system prompt + query
    (
        "You are an expert software engineer. You write clean, well-tested Python code "
        "following PEP 8 guidelines. You include type hints and docstrings. "
        "You prefer composition over inheritance.\n\n"
        "Write a function to parse CSV files."
    ),
    # RAG-style
    (
        "The Transformer architecture uses self-attention where each position attends to "
        "all other positions. Multi-head attention projects Q, K, V through parallel heads. "
        "Feed-forward networks apply position-wise transformations with ReLU or SwiGLU. "
        "Layer normalization improves training stability.\n\n"
        "Summarize the key components."
    ),
    # Code context
    (
        "import asyncio\nfrom dataclasses import dataclass\n\n"
        "@dataclass\nclass Config:\n    host: str = 'localhost'\n    port: int = 5432\n\n"
        "class Pool:\n    def __init__(self, cfg: Config):\n"
        "        self._cfg = cfg\n        self._pool = None\n\n"
        "    async def init(self):\n        self._pool = await create_pool(self._cfg.host)\n\n"
        "Add error handling to this code."
    ),
    # Few-shot
    (
        "Classify the sentiment:\n"
        "Text: 'The movie was fantastic!' → Positive\n"
        "Text: 'Terrible service, never again.' → Negative\n"
        "Text: 'It was okay, nothing special.' → Neutral\n"
        "Text: 'Best purchase I ever made!' → Positive\n"
        "Text: 'The food was bland and overpriced.' →"
    ),
    # Conversation
    (
        "User: What is machine learning?\n"
        "Assistant: Machine learning is a subset of AI where systems learn patterns from "
        "data rather than being explicitly programmed. It includes supervised learning "
        "(labeled data), unsupervised learning (finding structure), and reinforcement "
        "learning (reward-based optimization).\n"
        "User: How is deep learning different?\n"
        "Assistant:"
    ),
    # Technical document
    (
        "Paxos consensus protocol: Phase 1 (Prepare) - proposer sends prepare(n) to "
        "acceptors. Acceptors promise not to accept proposals < n. Phase 2 (Accept) - "
        "if majority promises, proposer sends accept(n, v). Multi-Paxos elects a stable "
        "leader to skip Phase 1 for subsequent rounds. Raft simplifies this with leader "
        "election via randomized timeouts and log replication.\n\n"
        "Compare Paxos and Raft."
    ),
]


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@dataclass
class LayerStats:
    """Per-layer attention statistics."""
    layer_idx: int
    mean_entropy: float            # avg Shannon entropy of attn distribution
    std_entropy: float
    mean_top10_mass: float         # fraction of attn in top 10% positions
    std_top10_mass: float
    mean_prefix_attention: float   # avg attn mass on prefix (first 80%) tokens
    std_prefix_attention: float
    mean_max_attention: float      # avg of max attention weight per head
    std_max_attention: float


def compute_attention_stats(
    model,
    tokenizer,
    device: str,
    prompts: List[str],
) -> Tuple[List[LayerStats], Dict]:
    """Run prompts through model and collect per-layer attention statistics."""

    all_layer_entropies = {}     # layer_idx -> list of entropy values
    all_layer_top10 = {}
    all_layer_prefix_attn = {}
    all_layer_max_attn = {}

    num_layers = model.config.num_hidden_layers

    for layer_idx in range(num_layers):
        all_layer_entropies[layer_idx] = []
        all_layer_top10[layer_idx] = []
        all_layer_prefix_attn[layer_idx] = []
        all_layer_max_attn[layer_idx] = []

    prompt_details = []

    for pidx, prompt in enumerate(tqdm(prompts, desc="Analyzing prompts")):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        seq_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                output_attentions=True,
                return_dict=True,
            )

        attentions = outputs.attentions  # tuple of (batch, heads, seq, seq)
        if attentions is None:
            raise RuntimeError(f"Model did not return attentions. Check model config.")

        prefix_len = int(seq_len * 0.8)  # first 80% = "prefix"

        prompt_info = {
            "prompt_idx": pidx,
            "seq_len": seq_len,
            "prefix_len": prefix_len,
            "per_layer": {},
        }

        for layer_idx, attn in enumerate(attentions):
            # attn shape: (1, num_heads, seq_len, seq_len)
            attn_weights = attn[0]  # (num_heads, seq_len, seq_len)

            # Use the attention from the last token (most relevant for generation)
            last_tok_attn = attn_weights[:, -1, :]  # (num_heads, seq_len)

            # Average across heads
            avg_attn = last_tok_attn.mean(dim=0)  # (seq_len,)

            # 1. Shannon entropy
            # Clamp to avoid log(0)
            attn_clamped = avg_attn.clamp(min=1e-10)
            entropy = -(attn_clamped * attn_clamped.log()).sum().item()
            # Normalize by max entropy (log(seq_len))
            max_entropy = math.log(seq_len) if seq_len > 1 else 1.0
            norm_entropy = entropy / max_entropy

            # 2. Top-10% mass
            k = max(1, seq_len // 10)
            top_k_vals, _ = avg_attn.topk(k)
            top10_mass = top_k_vals.sum().item()

            # 3. Prefix attention mass
            prefix_mass = avg_attn[:prefix_len].sum().item()

            # 4. Max attention per head (averaged)
            max_per_head = last_tok_attn.max(dim=-1).values.mean().item()

            all_layer_entropies[layer_idx].append(norm_entropy)
            all_layer_top10[layer_idx].append(top10_mass)
            all_layer_prefix_attn[layer_idx].append(prefix_mass)
            all_layer_max_attn[layer_idx].append(max_per_head)

            prompt_info["per_layer"][layer_idx] = {
                "entropy": round(norm_entropy, 4),
                "top10_mass": round(top10_mass, 4),
                "prefix_attn": round(prefix_mass, 4),
                "max_attn": round(max_per_head, 4),
            }

        prompt_details.append(prompt_info)

        # Free attention tensors
        del outputs, attentions
        clear_gpu()

    # Aggregate per-layer stats
    layer_stats = []
    for l in range(num_layers):
        layer_stats.append(LayerStats(
            layer_idx=l,
            mean_entropy=statistics.mean(all_layer_entropies[l]),
            std_entropy=statistics.stdev(all_layer_entropies[l]) if len(all_layer_entropies[l]) > 1 else 0,
            mean_top10_mass=statistics.mean(all_layer_top10[l]),
            std_top10_mass=statistics.stdev(all_layer_top10[l]) if len(all_layer_top10[l]) > 1 else 0,
            mean_prefix_attention=statistics.mean(all_layer_prefix_attn[l]),
            std_prefix_attention=statistics.stdev(all_layer_prefix_attn[l]) if len(all_layer_prefix_attn[l]) > 1 else 0,
            mean_max_attention=statistics.mean(all_layer_max_attn[l]),
            std_max_attention=statistics.stdev(all_layer_max_attn[l]) if len(all_layer_max_attn[l]) > 1 else 0,
        ))

    return layer_stats, prompt_details


def compute_cross_prompt_similarity(
    model,
    tokenizer,
    device: str,
    prompts: List[str],
) -> Dict[int, float]:
    """Compute per-layer cosine similarity of attention across prompts.

    Higher similarity = more stable pattern = more cacheable.
    """
    num_layers = model.config.num_hidden_layers
    # Collect attention vectors per layer per prompt
    layer_vectors = {l: [] for l in range(num_layers)}

    for prompt in tqdm(prompts[:4], desc="Cross-prompt similarity"):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                output_attentions=True,
                return_dict=True,
            )

        for l, attn in enumerate(outputs.attentions):
            # Last token, average over heads
            vec = attn[0, :, -1, :].mean(dim=0)  # (seq_len,)
            layer_vectors[l].append(vec.cpu())

        del outputs
        clear_gpu()

    # Compute pairwise cosine similarity per layer
    layer_sim = {}
    for l in range(num_layers):
        vecs = layer_vectors[l]
        if len(vecs) < 2:
            layer_sim[l] = 0.0
            continue

        # Pad to same length
        max_len = max(v.shape[0] for v in vecs)
        padded = [F.pad(v, (0, max_len - v.shape[0])) for v in vecs]

        sims = []
        for i in range(len(padded)):
            for j in range(i + 1, len(padded)):
                cos = F.cosine_similarity(padded[i].unsqueeze(0), padded[j].unsqueeze(0)).item()
                sims.append(cos)
        layer_sim[l] = statistics.mean(sims) if sims else 0.0

    return layer_sim


def run_experiment(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device: str = "cuda:0",
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*70}")
    print(f"Layer Attention Distribution Analysis")
    print(f"Model: {model_name}   Device: {device}")
    print(f"{'='*70}\n")

    print("Loading model with output_attentions support...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager",  # Required to get attention weights
    )
    model.eval()
    num_layers = model.config.num_hidden_layers
    print(f"Loaded: {num_layers} layers")

    # 1. Per-layer attention stats
    print("\n--- Per-layer attention statistics ---")
    layer_stats, prompt_details = compute_attention_stats(
        model, tokenizer, device, PROMPTS,
    )

    print(f"\n{'Layer':>6s}  {'Entropy':>8s}  {'Top10%':>8s}  {'PrefixAttn':>10s}  {'MaxAttn':>8s}")
    print("-" * 50)
    for ls in layer_stats:
        print(f"{ls.layer_idx:6d}  {ls.mean_entropy:8.4f}  {ls.mean_top10_mass:8.4f}  "
              f"{ls.mean_prefix_attention:10.4f}  {ls.mean_max_attention:8.4f}")

    # Summarize early vs late
    early_end = int(num_layers * 0.3)
    late_start = int(num_layers * 0.7)

    early_entropy = statistics.mean([ls.mean_entropy for ls in layer_stats[:early_end]])
    late_entropy = statistics.mean([ls.mean_entropy for ls in layer_stats[late_start:]])
    early_top10 = statistics.mean([ls.mean_top10_mass for ls in layer_stats[:early_end]])
    late_top10 = statistics.mean([ls.mean_top10_mass for ls in layer_stats[late_start:]])
    early_prefix = statistics.mean([ls.mean_prefix_attention for ls in layer_stats[:early_end]])
    late_prefix = statistics.mean([ls.mean_prefix_attention for ls in layer_stats[late_start:]])

    print(f"\n--- Early (layers 0-{early_end-1}) vs Late (layers {late_start}-{num_layers-1}) ---")
    print(f"  Entropy:      early={early_entropy:.4f}  late={late_entropy:.4f}")
    print(f"  Top-10% mass: early={early_top10:.4f}  late={late_top10:.4f}")
    print(f"  Prefix attn:  early={early_prefix:.4f}  late={late_prefix:.4f}")

    # 2. Cross-prompt similarity
    print("\n--- Cross-prompt attention similarity ---")
    layer_sim = compute_cross_prompt_similarity(model, tokenizer, device, PROMPTS)
    early_sim = statistics.mean([layer_sim[l] for l in range(early_end)])
    late_sim = statistics.mean([layer_sim[l] for l in range(late_start, num_layers)])
    print(f"  Avg cosine sim: early={early_sim:.4f}  late={late_sim:.4f}")

    for l in range(num_layers):
        marker = ""
        if l < early_end:
            marker = " (early)"
        elif l >= late_start:
            marker = " (late)"
        print(f"  Layer {l:2d}: sim={layer_sim[l]:.4f}{marker}")

    # Build hypothesis validation
    # Paper claims: late layers are more important → they should have
    # lower entropy (more concentrated), higher top-k mass, and
    # possibly higher cross-prompt consistency.
    hypothesis = {
        "entropy_decreases_with_depth": late_entropy < early_entropy,
        "top10_increases_with_depth": late_top10 > early_top10,
        "late_layers_more_consistent": late_sim > early_sim,
        "early_entropy": round(early_entropy, 4),
        "late_entropy": round(late_entropy, 4),
        "early_top10_mass": round(early_top10, 4),
        "late_top10_mass": round(late_top10, 4),
        "early_cross_sim": round(early_sim, 4),
        "late_cross_sim": round(late_sim, 4),
        "early_prefix_attn": round(early_prefix, 4),
        "late_prefix_attn": round(late_prefix, 4),
    }

    print(f"\n--- Hypothesis validation ---")
    for k, v in hypothesis.items():
        print(f"  {k}: {v}")

    # Save results
    output = {
        "metadata": {
            "experiment": "layer_attention_distribution",
            "model": model_name,
            "device": device,
            "num_layers": num_layers,
            "num_prompts": len(PROMPTS),
            "early_layer_end": early_end,
            "late_layer_start": late_start,
            "timestamp": datetime.now().isoformat(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        },
        "per_layer_stats": [asdict(ls) for ls in layer_stats],
        "cross_prompt_similarity": {str(k): round(v, 6) for k, v in layer_sim.items()},
        "hypothesis_validation": hypothesis,
        "prompt_details": prompt_details,
    }

    safe_model = model_name.split("/")[-1].lower().replace("-", "_")
    out_path = RESULTS_DIR / f"layer_attention_{safe_model}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    run_experiment(model_name=args.model, device=args.device)
