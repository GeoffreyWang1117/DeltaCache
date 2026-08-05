"""Exp B — Quantize-First Optimality Verification (NeurIPS Theory Section).

Goal
────
Validate that the "quantize-first, evict-second" policy is optimal or
near-optimal under fixed memory budgets, across layers and models.

We compare four policies (all baselines in this suite) at each layer × budget:

  Policy            | Token budget n  | Quantization bits b
  ────────────────  | ─────────────── | ───────────────────
  token_only        | varies          | 16
  quant_only        | seq_len (full)  | varies (4, 8, 16)
  token_first       | evict to target | then quantize to 4
  quant_first (ours)| quantize to b*  | then evict

We measure KV reconstruction error: ‖K_compressed − K_full‖² / ‖K_full‖²
for each (policy, memory_budget, layer), then check:

  1. What fraction of (layer, budget) grid does quant-first Pareto-dominate?
  2. On Qwen3-0.6B: brute-force exhaustive search vs greedy gap.

Usage
─────
    python experiments/scripts/exp_b_quantize_first.py \\
        --model qwen3-0.6b --seq-len 1024 --device cuda

Output
──────
    experiments/results/exp_b/<short_name>.json
    experiments/results/exp_b/<short_name>_summary.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))          # experiments/
sys.path.insert(0, str(_HERE.parent.parent))   # project root


# ── Quantization simulation ─────────────────────────────────────────

def quantize_dequant(tensor: Tensor, bits: int) -> Tensor:
    """Simulate quantization → dequant: asymmetric per-channel for K, per-token for V."""
    if bits >= 16:
        return tensor.clone()
    t = tensor.float()
    # Per-channel asymmetric: quantize along the last dim
    vmin = t.amin(dim=-1, keepdim=True)
    vmax = t.amax(dim=-1, keepdim=True)
    scale = (vmax - vmin) / (2 ** bits - 1)
    scale = scale.clamp(min=1e-8)
    q = ((t - vmin) / scale).round().clamp(0, 2 ** bits - 1)
    return (q * scale + vmin).to(tensor.dtype)


def reconstruction_error(original: Tensor, compressed: Tensor) -> float:
    """Normalized reconstruction error: ‖Δ‖² / ‖orig‖²."""
    diff = (original.float() - compressed.float())
    return (diff.norm() ** 2 / original.float().norm() ** 2).item()


# ── Token selection ──────────────────────────────────────────────────

def select_uniform(seq_len: int, n_keep: int, sink: int = 4) -> Tensor:
    """Sink + uniformly spaced + recent tail."""
    if n_keep >= seq_len:
        return torch.arange(seq_len, dtype=torch.long)
    s = min(sink, n_keep)
    recent = min(max(1, n_keep // 4), n_keep - s)
    middle_budget = n_keep - s - recent
    idx = list(range(s))
    if middle_budget > 0:
        middle_range = list(range(s, seq_len - recent))
        step = max(1, len(middle_range) // middle_budget)
        idx.extend(middle_range[::step][:middle_budget])
    idx.extend(range(max(s, seq_len - recent), seq_len))
    return torch.tensor(sorted(set(idx))[:n_keep], dtype=torch.long)


# ── Four policies ────────────────────────────────────────────────────

def memory_for(n_tokens: int, bits: int, num_heads: int, head_dim: int) -> int:
    """Bytes for K+V at one layer."""
    return 2 * n_tokens * num_heads * head_dim * bits // 8


def _reconstruct_full(original: Tensor, compressed: Tensor,
                       indices: Tensor) -> Tensor:
    """Place compressed tokens back at their positions in a zero-filled
    full-size tensor (same shape as original)."""
    out = torch.zeros_like(original)
    out[:, indices] = compressed
    return out


def apply_policy_token_only(k: Tensor, v: Tensor, budget_bytes: int,
                             num_heads: int, head_dim: int, seq_len: int):
    """Evict tokens only (FP16). Return full-size reconstruction."""
    bits = 16
    max_tok = budget_bytes // max(1, 2 * num_heads * head_dim * bits // 8)
    n_keep = max(1, min(max_tok, seq_len))
    idx = select_uniform(seq_len, n_keep)
    kr = _reconstruct_full(k, k[:, idx], idx)
    vr = _reconstruct_full(v, v[:, idx], idx)
    return kr, vr, n_keep, bits


def apply_policy_quant_only(k: Tensor, v: Tensor, budget_bytes: int,
                             num_heads: int, head_dim: int, seq_len: int):
    """Quantize only (keep all tokens)."""
    for bits in [4, 8, 16]:
        mem = memory_for(seq_len, bits, num_heads, head_dim)
        if mem <= budget_bytes:
            return quantize_dequant(k, bits), quantize_dequant(v, bits), seq_len, bits
    return quantize_dequant(k, 4), quantize_dequant(v, 4), seq_len, 4


def apply_policy_token_first(k: Tensor, v: Tensor, budget_bytes: int,
                              num_heads: int, head_dim: int, seq_len: int):
    """Evict tokens first, then quantize to 4-bit."""
    final_bits = 4
    max_tok_4b = budget_bytes // max(1, 2 * num_heads * head_dim * final_bits // 8)
    n_keep = max(1, min(max_tok_4b, seq_len))
    idx = select_uniform(seq_len, n_keep)
    kq = quantize_dequant(k[:, idx], final_bits)
    vq = quantize_dequant(v[:, idx], final_bits)
    kr = _reconstruct_full(k, kq, idx)
    vr = _reconstruct_full(v, vq, idx)
    return kr, vr, n_keep, final_bits


def apply_policy_quant_first(k: Tensor, v: Tensor, budget_bytes: int,
                              num_heads: int, head_dim: int, seq_len: int):
    """Quantize first (find best bits), then evict. LayerBudget's policy."""
    best = (None, None, 0, 16, float("inf"))
    for bits in [4, 8, 16]:
        max_tok = budget_bytes // max(1, 2 * num_heads * head_dim * bits // 8)
        n_keep = max(1, min(max_tok, seq_len))
        idx = select_uniform(seq_len, n_keep)
        kq = quantize_dequant(k[:, idx], bits)
        vq = quantize_dequant(v[:, idx], bits)
        kr = _reconstruct_full(k, kq, idx)
        vr = _reconstruct_full(v, vq, idx)
        err = reconstruction_error(k, kr) + reconstruction_error(v, vr)
        if err < best[4]:
            best = (kr, vr, n_keep, bits, err)
    return best[0], best[1], best[2], best[3]


# ── Brute-force optimal (small models only) ─────────────────────────

def brute_force_optimal(k: Tensor, v: Tensor, budget_bytes: int,
                         num_heads: int, head_dim: int, seq_len: int,
                         token_step: int = 8):
    """Exhaustive search over all (n_tokens, bits) combos."""
    best_err = float("inf")
    best_config = (seq_len, 16)
    for bits in [4, 8, 16]:
        max_tok = min(budget_bytes // max(1, 2 * num_heads * head_dim * bits // 8), seq_len)
        # Always include max_tok so we don't miss the boundary
        candidates = list(range(1, max_tok + 1, token_step))
        if max_tok > 0 and max_tok not in candidates:
            candidates.append(max_tok)
        for n_tok in candidates:
            idx = select_uniform(seq_len, n_tok)
            kq = quantize_dequant(k[:, idx], bits)
            vq = quantize_dequant(v[:, idx], bits)
            kr = _reconstruct_full(k, kq, idx)
            vr = _reconstruct_full(v, vq, idx)
            err = reconstruction_error(k, kr) + reconstruction_error(v, vr)
            if err < best_err:
                best_err = err
                best_config = (n_tok, bits)
    return best_config, best_err


# ── Main driver ──────────────────────────────────────────────────────

def run(model_key: str, seq_len: int, device: str, out_dir: Path,
        budget_fractions: List[float] | None = None,
        do_brute_force: bool = True, token_step: int = 8):
    from suite.config import MODEL_ZOO
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = MODEL_ZOO[model_key]
    print(f"\n{'='*60}\n  Exp B — Quantize-First Optimality: {spec.short_name}\n{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(spec.hf_name)
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_name, dtype=torch.float16, device_map=device,
    )
    model.eval()

    if budget_fractions is None:
        budget_fractions = [0.10, 0.15, 0.20, 0.25, 0.33, 0.50, 0.75]

    # Get a calibration sequence
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = " ".join(x["text"] for x in ds if x["text"].strip())
    input_ids = tokenizer.encode(text, add_special_tokens=True,
                                  max_length=seq_len, truncation=True)
    input_ids = torch.tensor([input_ids], device=device)
    actual_len = input_ids.shape[1]
    print(f"  seq_len={actual_len}  layers={spec.num_layers}")

    # Extract full KV cache
    with torch.no_grad():
        out = model(input_ids, use_cache=True, output_attentions=False)
    dyn_cache = out.past_key_values
    # Convert DynamicCache → list of (key, value) tuples.
    # transformers 4.57+ uses DynamicCache.layers (list of CacheLayer).
    # transformers 4.57+: DynamicCache.layers → list of DynamicLayer
    # with .keys/.values attrs, each of shape (B, heads, seq, head_dim).
    if hasattr(dyn_cache, 'layers') and len(dyn_cache.layers) > 0:
        kv = [(lay.keys, lay.values) for lay in dyn_cache.layers]
    elif hasattr(dyn_cache, 'key_cache'):
        kv = [(dyn_cache.key_cache[l], dyn_cache.value_cache[l])
              for l in range(len(dyn_cache.key_cache))]
    else:
        kv = [(dyn_cache[l][0], dyn_cache[l][1]) for l in range(len(dyn_cache))]
    del out, dyn_cache

    full_mem_per_layer = memory_for(
        actual_len, 16, spec.num_kv_heads, spec.head_dim)

    policies = {
        "token_only": apply_policy_token_only,
        "quant_only": apply_policy_quant_only,
        "token_first": apply_policy_token_first,
        "quant_first": apply_policy_quant_first,
    }

    results = []
    pareto_wins = {"quant_first": 0, "total": 0}

    for layer_idx in range(spec.num_layers):
        # kv layout: (1, heads, seq_len, head_dim) from DynamicLayer.
        # Rearrange to (1, seq_len, heads, head_dim) for consistency
        # with our compression functions.
        k_full = kv[layer_idx][0].permute(0, 2, 1, 3)  # (1, S, H, D)
        v_full = kv[layer_idx][1].permute(0, 2, 1, 3)

        for frac in budget_fractions:
            budget = int(full_mem_per_layer * frac)
            row = {
                "layer": layer_idx,
                "budget_fraction": frac,
                "budget_bytes": budget,
            }

            best_err = float("inf")
            best_policy = None

            for pname, pfn in policies.items():
                kc, vc, n_tok, bits = pfn(
                    k_full, v_full, budget,
                    spec.num_kv_heads, spec.head_dim, actual_len,
                )
                err = reconstruction_error(k_full, kc) + reconstruction_error(v_full, vc)
                row[f"{pname}_err"] = round(err, 6)
                row[f"{pname}_n_tok"] = n_tok
                row[f"{pname}_bits"] = bits
                if err < best_err:
                    best_err = err
                    best_policy = pname

            row["best_policy"] = best_policy
            pareto_wins["total"] += 1
            if best_policy == "quant_first":
                pareto_wins["quant_first"] += 1

            # Brute-force optimal (expensive, only for small models)
            if do_brute_force:
                bf_config, bf_err = brute_force_optimal(
                    k_full, v_full, budget,
                    spec.num_kv_heads, spec.head_dim, actual_len,
                    token_step=token_step,
                )
                row["bf_optimal_n_tok"] = bf_config[0]
                row["bf_optimal_bits"] = bf_config[1]
                row["bf_optimal_err"] = round(bf_err, 6)
                greedy_gap = (row["quant_first_err"] - bf_err) / max(bf_err, 1e-12)
                row["greedy_gap_pct"] = round(greedy_gap * 100, 2)

            results.append(row)

        if (layer_idx + 1) % 4 == 0:
            pct = pareto_wins["quant_first"] / max(1, pareto_wins["total"]) * 100
            print(f"  layer {layer_idx+1}/{spec.num_layers}  "
                  f"quant-first wins {pct:.0f}%")

    # Aggregate
    pareto_pct = pareto_wins["quant_first"] / max(1, pareto_wins["total"]) * 100
    if do_brute_force:
        gaps = [r["greedy_gap_pct"] for r in results if "greedy_gap_pct" in r]
        mean_gap = sum(gaps) / len(gaps) if gaps else float("nan")
        max_gap = max(gaps) if gaps else float("nan")
    else:
        mean_gap = max_gap = float("nan")

    summary = {
        "model": spec.short_name,
        "seq_len": actual_len,
        "num_layers": spec.num_layers,
        "budget_fractions": budget_fractions,
        "quant_first_pareto_dominance_pct": round(pareto_pct, 1),
        "greedy_vs_optimal_mean_gap_pct": round(mean_gap, 2),
        "greedy_vs_optimal_max_gap_pct": round(max_gap, 2),
        "total_grid_points": pareto_wins["total"],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{spec.short_name}.json"
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "grid": results}, f, indent=2)
    print(f"\n  saved → {json_path}")

    txt_path = out_dir / f"{spec.short_name}_summary.txt"
    with open(txt_path, "w") as f:
        f.write(f"Exp B — Quantize-First Optimality: {spec.short_name}\n")
        f.write(f"seq_len={actual_len}  layers={spec.num_layers}\n\n")
        f.write(f"Quant-first Pareto dominance: {pareto_pct:.1f}%\n")
        f.write(f"Greedy vs optimal mean gap:   {mean_gap:.2f}%\n")
        f.write(f"Greedy vs optimal max gap:    {max_gap:.2f}%\n\n")
        # Per-budget breakdown
        f.write(f"{'budget':>8s}  {'token':>8s}  {'quant':>8s}  {'tok1st':>8s}  {'qnt1st':>8s}  {'opt':>8s}  {'gap%':>6s}\n")
        for frac in budget_fractions:
            subset = [r for r in results if r["budget_fraction"] == frac]
            te = sum(r["token_only_err"] for r in subset) / len(subset)
            qe = sum(r["quant_only_err"] for r in subset) / len(subset)
            tfe = sum(r["token_first_err"] for r in subset) / len(subset)
            qfe = sum(r["quant_first_err"] for r in subset) / len(subset)
            if do_brute_force:
                oe = sum(r["bf_optimal_err"] for r in subset) / len(subset)
                gg = sum(r["greedy_gap_pct"] for r in subset) / len(subset)
            else:
                oe = gg = float("nan")
            f.write(f"{frac:8.2f}  {te:8.4f}  {qe:8.4f}  {tfe:8.4f}  {qfe:8.4f}  {oe:8.4f}  {gg:6.2f}\n")
    print(f"  saved → {txt_path}")
    print()
    with open(txt_path) as f:
        print(f.read())


def main():
    parser = argparse.ArgumentParser(description="Exp B: quantize-first optimality")
    parser.add_argument("--model", required=True)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default="experiments/results/exp_b")
    parser.add_argument("--no-brute-force", action="store_true")
    parser.add_argument("--token-step", type=int, default=8,
                        help="Step for brute-force n_tokens sweep")
    args = parser.parse_args()

    run(args.model, args.seq_len, args.device, Path(args.out_dir),
        do_brute_force=not args.no_brute_force,
        token_step=args.token_step)


if __name__ == "__main__":
    main()
