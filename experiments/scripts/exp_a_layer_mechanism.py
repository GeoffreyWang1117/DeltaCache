"""Exp A — Layer Mechanism Analysis (NeurIPS Theory Section).

Goal
────
Upgrade LayerBudget's "early layers are more important" observation from
a single ablation row to a Transformer information-flow principle, by
measuring six per-layer quantities on real models and showing they all
agree with each other AND with LayerBudget's heuristic allocations:

  1. Activation entropy        (uni_layer.ActivationEntropy)
  2. Effective rank            (uni_layer.EffectiveRank)
  3. Block influence (LOO)     (uni_layer.BlockInfluence)
  4. Fisher information trace  (uni_layer.FisherInformation)
  5. Gradient norm             (uni_layer.GradientNorm)
  6. Quantization sensitivity  (uni_layer.QuantizationSensitivity)

For each model we then compute the Spearman rank correlation between each
of the six metrics and:

  • LayerBudget gini score      (sparsity-based heuristic)
  • LayerBudget importance      (sigmoid-based inverse-depth heuristic)

A high correlation (≥ 0.6) is the empirical-theory plank: it lets us
write "the heuristic Q(l, n, b) is a first-order approximation of
mutual information / Fisher information / effective rank" instead of
calling it a fitted constant.

Usage
─────
    python experiments/scripts/exp_a_layer_mechanism.py \\
        --model qwen3-0.6b --n-samples 5 --seq-len 1024
    python experiments/scripts/exp_a_layer_mechanism.py \\
        --model llama3.1-8b --n-samples 5 --seq-len 2048 --device cuda

Output
──────
    experiments/results/exp_a/<short_name>.json
    experiments/results/exp_a/<short_name>_summary.txt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))         # experiments/
sys.path.insert(0, str(_HERE.parent.parent))  # project root


# ── Calibration dataset ──────────────────────────────────────────────

class TextWindowDataset(Dataset):
    """Fixed-length token windows over WikiText for layer profiling."""

    def __init__(self, tokenizer, seq_len: int, n_samples: int):
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = " ".join(x["text"] for x in ds if x["text"].strip())
        ids = tokenizer.encode(text, add_special_tokens=False)
        self.windows = []
        for i in range(n_samples):
            chunk = ids[i * seq_len:(i + 1) * seq_len]
            if len(chunk) < seq_len:
                break
            self.windows.append(torch.tensor(chunk, dtype=torch.long))
        if len(self.windows) == 0:
            raise RuntimeError("Not enough tokens in WikiText to build samples")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        # uni_layer expects (input, target) tuples for generation tasks;
        # for next-token-prediction we use the same window for both.
        x = self.windows[i]
        return x, x


def collate(batch):
    xs = torch.stack([b[0] for b in batch])
    ys = torch.stack([b[1] for b in batch])
    return xs, ys


# ── Spearman correlation (no scipy dependency) ───────────────────────

def spearman(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n < 3:
        return float("nan")

    def ranks(vals):
        order = sorted(range(n), key=lambda i: vals[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx * dy > 0 else float("nan")


# ── LayerBudget reference scores ─────────────────────────────────────

def compute_layerbudget_scores(model, tokenizer, spec, device: str,
                                n_samples: int = 2, seq_len: int = 512):
    """Reproduce the gini + importance scores LayerBudget uses at runtime."""
    from deltacache.core.layer_budget_allocator import LayerBudgetAllocator
    from deltacache.core.layer_profiler import LayerAttentionProfiler
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = " ".join(x["text"] for x in ds if x["text"].strip())
    tokens = tokenizer.encode(text, add_special_tokens=False)

    gini_accum: Dict[int, List[float]] = {}
    profiler = LayerAttentionProfiler()
    profile_len = min(seq_len, len(tokens) // n_samples)
    for i in range(n_samples):
        chunk = tokens[i * profile_len:(i + 1) * profile_len]
        input_ids = torch.tensor([chunk], device=device)
        result = profiler.profile(input_ids, model, device=device)
        for l, g in result.gini_scores().items():
            gini_accum.setdefault(l, []).append(g)

    gini = {l: sum(v) / len(v) for l, v in gini_accum.items()}

    nl = spec.num_layers
    sigmoid = LayerBudgetAllocator.compute_importance_weights(nl)
    importance = {l: sigmoid[nl - 1 - l] for l in range(nl)}

    return gini, importance


# ── LM criterion for gradient-based metrics ──────────────────────────

class LMCriterion(torch.nn.Module):
    """Cross-entropy for next-token prediction (shifted labels)."""

    def __init__(self):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, output, target):
        logits = getattr(output, "logits", output)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = target[..., 1:].contiguous()
        return self.ce(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )


# ── uni_layer wrapper ────────────────────────────────────────────────

def run_uni_layer(model, dataloader, device: str) -> Dict[str, Dict[str, float]]:
    """Run BlockInfluence + EffectiveRank + ActivationEntropy + GradientNorm
    + FisherInformation + QuantizationSensitivity on the model.

    Returns: {metric_name: {layer_name: value}}
    """
    from uni_layer import LayerAnalyzer
    from uni_layer import (
        BlockInfluence, EffectiveRank, ActivationEntropy,
        GradientNorm, FisherInformation, QuantizationSensitivity,
    )

    # task_type='language_modeling' uses cross-entropy with shifted-token
    # targets and avoids the bogus MSE warning that 'generation' produces.
    analyzer = LayerAnalyzer(
        model, task_type="language_modeling", device=device,
        criterion=LMCriterion(),
    )
    # Run gradient-based and activation-based metrics in separate calls so
    # the activation cache doesn't short-circuit gradient-needing metrics.
    activation_metrics = [
        BlockInfluence(), EffectiveRank(), ActivationEntropy(),
        QuantizationSensitivity(),
    ]
    gradient_metrics = [GradientNorm(), FisherInformation()]

    out = analyzer.compute_metrics(
        metrics=activation_metrics,
        data_loader=dataloader,
        num_batches=len(dataloader),
        verbose=False,
        use_cache=False,
    )
    out2 = analyzer.compute_metrics(
        metrics=gradient_metrics,
        data_loader=dataloader,
        num_batches=len(dataloader),
        verbose=False,
        use_cache=False,
    )
    # Merge: out and out2 are both {layer_name: {field: value}}
    for layer_name, fields in out2.items():
        if layer_name in out:
            out[layer_name].update(fields)
        else:
            out[layer_name] = fields
    return out


# ── Per-layer reduction ──────────────────────────────────────────────

def invert_uni_layer_output(
    raw: Dict[str, Dict[str, float]], num_layers: int,
) -> Dict[str, List[float]]:
    """uni_layer returns OrderedDict[layer_name, OrderedDict[metric_field, value]].
    Invert and reduce to {metric_field: [value_for_layer_0, ..., value_for_layer_{N-1}]}.
    """
    import re
    by_metric: Dict[str, List[float]] = {}
    for layer_name, fields in raw.items():
        m = re.search(r"layers\.(\d+)(?:\.|$)", layer_name)
        if not m:
            continue
        l = int(m.group(1))
        if not (0 <= l < num_layers):
            continue
        for fname, v in fields.items():
            if not isinstance(v, (int, float)):
                continue
            if isinstance(v, float) and math.isnan(v):
                continue
            if fname in ("layer_idx",):  # skip metadata
                continue
            arr = by_metric.setdefault(fname, [float("nan")] * num_layers)
            arr[l] = float(v)
    return by_metric


# ── Main driver ──────────────────────────────────────────────────────

def run(model_key: str, n_samples: int, seq_len: int, device: str,
        out_dir: Path):
    from suite.config import MODEL_ZOO
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = MODEL_ZOO[model_key]
    print(f"\n{'=' * 60}\n  Exp A — {spec.short_name}\n{'=' * 60}")
    print(f"  layers={spec.num_layers}  kv_heads={spec.num_kv_heads}")
    print(f"  loading in fp16 on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(spec.hf_name)
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_name, dtype=torch.float16, device_map=device,
        attn_implementation="eager",
    )
    model.eval()

    print(f"  building calibration set ({n_samples} × {seq_len} tokens)...")
    ds = TextWindowDataset(tokenizer, seq_len=seq_len, n_samples=n_samples)
    dl = DataLoader(ds, batch_size=1, collate_fn=collate)

    print(f"  running uni_layer metrics ({len(ds)} batches)...")
    raw = run_uni_layer(model, dl, device)
    nl = spec.num_layers
    per_layer = invert_uni_layer_output(raw, nl)
    print(f"  got metrics ({len(per_layer)}): {sorted(per_layer.keys())}")

    # LayerBudget reference scores
    print("  computing LayerBudget reference scores...")
    gini_dict, imp_dict = compute_layerbudget_scores(
        model, tokenizer, spec, device,
        n_samples=2, seq_len=min(512, seq_len),
    )
    gini = [gini_dict.get(l, float("nan")) for l in range(nl)]
    importance = [imp_dict.get(l, float("nan")) for l in range(nl)]
    per_layer["LayerBudget_gini"] = gini
    per_layer["LayerBudget_importance"] = importance

    # Spearman correlations between every pair (focus on metric vs LayerBudget)
    correlations = {}
    for metric_name, vals in per_layer.items():
        if metric_name.startswith("LayerBudget"):
            continue
        clean_v = [v for v in vals if not math.isnan(v)]
        if len(clean_v) < 3:
            continue
        valid_idx = [i for i, v in enumerate(vals) if not math.isnan(v)]
        v_clean = [vals[i] for i in valid_idx]
        g_clean = [gini[i] for i in valid_idx]
        i_clean = [importance[i] for i in valid_idx]
        correlations[metric_name] = {
            "spearman_vs_gini": round(spearman(v_clean, g_clean), 3),
            "spearman_vs_importance": round(spearman(v_clean, i_clean), 3),
        }

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{spec.short_name}.json"
    with open(json_path, "w") as f:
        json.dump({
            "model": spec.short_name,
            "num_layers": nl,
            "n_calib_samples": n_samples,
            "seq_len": seq_len,
            "per_layer": per_layer,
            "correlations": correlations,
        }, f, indent=2)
    print(f"\n  saved → {json_path}")

    # Summary text
    summary_path = out_dir / f"{spec.short_name}_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Exp A — Layer Mechanism Analysis: {spec.short_name}\n")
        f.write(f"Layers: {nl}, calibration: {n_samples} × {seq_len} tok\n\n")
        f.write(f"{'metric':<25s} {'Spearman vs gini':<18s} "
                f"{'Spearman vs importance':<22s}\n")
        f.write("─" * 65 + "\n")
        for m, c in correlations.items():
            f.write(f"{m:<25s} {c['spearman_vs_gini']:<+18.3f} "
                    f"{c['spearman_vs_importance']:<+22.3f}\n")
    print(f"  saved → {summary_path}")
    print()
    with open(summary_path) as f:
        print(f.read())


def main():
    parser = argparse.ArgumentParser(description="Exp A: layer mechanism")
    parser.add_argument("--model", required=True,
                        help="Model key from MODEL_ZOO")
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default="experiments/results/exp_a")
    args = parser.parse_args()

    run(args.model, args.n_samples, args.seq_len, args.device,
        Path(args.out_dir))


if __name__ == "__main__":
    main()
