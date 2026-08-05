#!/usr/bin/env python3
"""Generate paper figures from all unified quality results."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results" / "paper"
PAPER_FIG = Path(__file__).parent.parent / "paper" / "figures"
PAPER_FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.labelsize": 11,
    "legend.fontsize": 7, "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

EVICTION = ["h2o_uniform", "snapkv", "pyramidkv", "d2o", "squeeze_attention",
            "adakv", "dynamickv", "cake", "lava"]
QUANT = ["kivi_uniform", "kvtuner"]
OURS = ["layer_budget"]

COLORS = {"kivi_uniform": "#fdae6b", "kvtuner": "#e6550d", "layer_budget": "#e41a1c"}
LABELS = {
    "kivi_uniform": "KIVI", "kvtuner": "KVTuner",
    "layer_budget": "LayerBudget (Ours)", "full_kv": "Full KV",
    "h2o_uniform": "H2O", "cake": "CAKE",
}


def load(fname):
    p = RESULTS_DIR / fname
    return json.load(open(p)) if p.exists() else None


def save_fig(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(PAPER_FIG / f"{name}.{ext}")
    plt.close()
    print(f"  {name}.pdf")


def fig1_cross_model_ppl():
    """Main figure: PPL ratio across models and seq lengths."""
    configs = [
        ("Llama-2-7B\n512 tok", "unified_quality_512tok_llama2_7b.json"),
        ("Llama-2-7B\n1024 tok", "unified_quality_1024tok_llama_2_7b_chat_hf.json"),
        ("Llama-2-7B\n2048 tok", "unified_quality_2048tok_llama2_7b.json"),
        ("Mistral-7B\n512 tok", "unified_quality_512tok_mistral_7b_instruct_v0.2.json"),
        ("Mistral-7B\n1024 tok", "unified_quality_1024tok_mistral_7b_instruct_v0.2.json"),
        ("Mistral-7B\n2048 tok", "unified_quality_2048tok_mistral_7b_instruct_v0.2.json"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey=True)
    axes = axes.flatten()

    for ax, (label, fname) in zip(axes, configs):
        data = load(fname)
        if data is None:
            ax.set_title(label + "\n(pending)", fontsize=9)
            continue

        # Collect per-method data
        method_crs = {}
        for s in data["summary"]:
            m, cr, r = s["method"], s["cr"], s["mean_ratio"]
            if m not in method_crs:
                method_crs[m] = ([], [])
            method_crs[m][0].append(cr)
            method_crs[m][1].append(r)

        # Plot eviction band (gray)
        if "h2o_uniform" in method_crs:
            crs = method_crs["h2o_uniform"][0]
            ax.fill_between(crs, [0.999]*len(crs), [1.001]*len(crs),
                           alpha=0.3, color="#cccccc", label="Eviction (9 methods)")

        # Plot quant methods
        for m in QUANT:
            if m in method_crs:
                ax.plot(method_crs[m][0], method_crs[m][1],
                       marker="s", color=COLORS[m], linewidth=1.5,
                       markersize=5, label=LABELS.get(m, m))

        # Plot ours
        if "layer_budget" in method_crs:
            ax.plot(method_crs["layer_budget"][0], method_crs["layer_budget"][1],
                   marker="*", color="#e41a1c", linewidth=2.5,
                   markersize=12, label="LayerBudget (Ours)", zorder=10)

        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.set_xticks([2, 3, 4, 6])
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Compression Ratio")

    axes[0].set_ylabel("PPL Ratio")
    axes[3].set_ylabel("PPL Ratio")

    # Legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
              bbox_to_anchor=(0.5, 1.02), fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "ppl_vs_compression_all")


def fig2_heatmap():
    """Heatmap: LayerBudget ratio across all configs."""
    models = ["Llama-2-7B", "Mistral-7B"]
    seqlens = [512, 1024, 2048]
    crs = [2.0, 3.0, 4.0, 6.0]

    files = {
        ("Llama-2-7B", 512): "unified_quality_512tok_llama2_7b.json",
        ("Llama-2-7B", 1024): "unified_quality_1024tok_llama_2_7b_chat_hf.json",
        ("Llama-2-7B", 2048): "unified_quality_2048tok_llama2_7b.json",
        ("Mistral-7B", 512): "unified_quality_512tok_mistral_7b_instruct_v0.2.json",
        ("Mistral-7B", 1024): "unified_quality_1024tok_mistral_7b_instruct_v0.2.json",
        ("Mistral-7B", 2048): "unified_quality_2048tok_mistral_7b_instruct_v0.2.json",
    }

    # Build matrices for LB, KIVI, KVTuner
    methods_to_show = {"layer_budget": "LayerBudget", "kivi_uniform": "KIVI", "kvtuner": "KVTuner"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3), sharey=True)

    for ax, (mkey, mlabel) in zip(axes, methods_to_show.items()):
        matrix = np.full((len(models) * len(seqlens), len(crs)), np.nan)
        ylabels = []
        for i, model in enumerate(models):
            for j, sl in enumerate(seqlens):
                row = i * len(seqlens) + j
                ylabels.append(f"{model}\n{sl}tok")
                data = load(files.get((model, sl), ""))
                if data is None:
                    continue
                for s in data["summary"]:
                    if s["method"] == mkey:
                        ci = crs.index(s["cr"])
                        matrix[row, ci] = s["mean_ratio"]

        im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto",
                      vmin=0.995, vmax=1.02)
        ax.set_xticks(range(len(crs)))
        ax.set_xticklabels([f"{int(c)}x" for c in crs])
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels, fontsize=8)
        ax.set_xlabel("Compression Ratio")
        ax.set_title(mlabel, fontsize=11)

        for r in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                if not np.isnan(matrix[r, c]):
                    val = matrix[r, c]
                    color = "white" if abs(val - 1.0) > 0.005 else "black"
                    weight = "bold" if val < 1.0 else "normal"
                    ax.text(c, r, f"{val:.4f}", ha="center", va="center",
                           fontsize=7, color=color, fontweight=weight)

    plt.colorbar(im, ax=axes, shrink=0.8, label="PPL Ratio", pad=0.02)
    plt.tight_layout()
    save_fig(fig, "method_heatmap_comparison")


def fig3_win_rate():
    """Bar chart: LayerBudget win rate vs each baseline."""
    files = [
        "unified_quality_512tok_llama2_7b.json",
        "unified_quality_1024tok_llama_2_7b_chat_hf.json",
        "unified_quality_2048tok_llama2_7b.json",
        "unified_quality_512tok_mistral_7b_instruct_v0.2.json",
        "unified_quality_1024tok_mistral_7b_instruct_v0.2.json",
        "unified_quality_2048tok_mistral_7b_instruct_v0.2.json",
    ]
    crs = [2.0, 3.0, 4.0, 6.0]
    opponents = ["kivi_uniform", "kvtuner"]
    wins = {o: 0 for o in opponents}
    ties = {o: 0 for o in opponents}
    losses = {o: 0 for o in opponents}
    total = 0

    for fname in files:
        data = load(fname)
        if data is None:
            continue
        for cr in crs:
            lb = [s for s in data["summary"] if s["method"] == "layer_budget" and s["cr"] == cr]
            if not lb:
                continue
            total += 1
            for o in opponents:
                opp = [s for s in data["summary"] if s["method"] == o and s["cr"] == cr]
                if not opp:
                    continue
                if lb[0]["mean_ratio"] < opp[0]["mean_ratio"] - 0.0001:
                    wins[o] += 1
                elif lb[0]["mean_ratio"] > opp[0]["mean_ratio"] + 0.0001:
                    losses[o] += 1
                else:
                    ties[o] += 1

    fig, ax = plt.subplots(figsize=(5, 3))
    x = np.arange(len(opponents))
    w = 0.25
    bars_w = ax.bar(x - w, [wins[o] for o in opponents], w, label="LB wins", color="#2ca02c")
    bars_t = ax.bar(x, [ties[o] for o in opponents], w, label="Tie", color="#cccccc")
    bars_l = ax.bar(x + w, [losses[o] for o in opponents], w, label="LB loses", color="#d62728")

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(o, o) for o in opponents])
    ax.set_ylabel("# Configurations (out of 24)")
    ax.set_title("LayerBudget vs Quantization Baselines")
    ax.legend()

    for bars in [bars_w, bars_t, bars_l]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.3, str(int(h)),
                       ha="center", fontsize=9)

    plt.tight_layout()
    save_fig(fig, "win_rate_chart")


if __name__ == "__main__":
    print("Generating paper figures...")
    fig1_cross_model_ppl()
    fig2_heatmap()
    fig3_win_rate()
    print("Done!")
