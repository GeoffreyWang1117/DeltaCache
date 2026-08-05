#!/usr/bin/env python3
"""Generate paper figures from CORRECTED experiment results."""

import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent / "results" / "paper"
PAPER_FIG = Path(__file__).parent.parent / "paper" / "figures"
PAPER_FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.labelsize": 11,
    "legend.fontsize": 7.5, "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

LABELS = {
    "full_kv": "Full KV", "h2o_uniform": "H2O", "snapkv": "SnapKV",
    "pyramidkv": "PyramidKV", "d2o": "D2O", "squeeze_attention": "SqzAttn",
    "adakv": "Ada-KV", "dynamickv": "DynamicKV", "cake": "CAKE",
    "lava": "LAVa", "kivi_uniform": "KIVI", "kvtuner": "KVTuner",
    "layer_budget": "LayerBudget",
}


def load(fname):
    p = RESULTS_DIR / fname
    return json.load(open(p)) if p.exists() else None


def save_fig(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(PAPER_FIG / f"{name}.{ext}")
    plt.close()
    print(f"  {name}.pdf")


def fig1_ppl_vs_cr():
    """Main figure: PPL ratio vs CR for key methods across 4 configs."""
    configs = [
        ("Llama-2-7B\n512 tok", "corrected_512tok_llama_2_7b.json"),
        ("Llama-2-7B\n1024 tok", "corrected_1024tok_llama_2_7b.json"),
        ("Mistral-7B\n512 tok", "corrected_512tok_mistral_7b.json"),
        ("Mistral-7B\n1024 tok", "corrected_1024tok_mistral_7b.json"),
    ]

    # Show key methods only (not all 12)
    show_methods = ["cake", "h2o_uniform", "kivi_uniform", "kvtuner", "layer_budget"]
    colors = {"cake": "#31a354", "h2o_uniform": "#6baed6", "kivi_uniform": "#fdae6b",
              "kvtuner": "#e6550d", "layer_budget": "#e41a1c"}
    markers = {"cake": "^", "h2o_uniform": "o", "kivi_uniform": "P",
               "kvtuner": "X", "layer_budget": "*"}
    sizes = {"cake": 5, "h2o_uniform": 5, "kivi_uniform": 6,
             "kvtuner": 6, "layer_budget": 12}

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharey=False)

    for ax, (label, fname) in zip(axes, configs):
        data = load(fname)
        if data is None:
            ax.set_title(label + "\n(pending)")
            continue

        method_data = {}
        for s in data["summary"]:
            m = s["method"]
            if m not in method_data:
                method_data[m] = {"crs": [], "ratios": []}
            method_data[m]["crs"].append(s["cr"])
            method_data[m]["ratios"].append(s["mean_ratio"])

        for m in show_methods:
            if m in method_data:
                d = method_data[m]
                lw = 2.5 if m == "layer_budget" else 1.5
                zorder = 10 if m == "layer_budget" else 5
                ax.plot(d["crs"], d["ratios"], marker=markers.get(m, "o"),
                        color=colors.get(m, "#999"), linewidth=lw,
                        markersize=sizes.get(m, 5), label=LABELS.get(m, m),
                        zorder=zorder)

        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.set_xlabel("Compression Ratio")
        ax.set_title(label, fontsize=9)
        ax.set_xticks([2, 3, 4, 6])
        ax.set_yscale("log")
        ax.set_ylim(0.9, 500)

    axes[0].set_ylabel("PPL Ratio (log scale)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.08), fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    save_fig(fig, "ppl_vs_compression_corrected")


def fig2_method_comparison_heatmap():
    """Heatmap: method × config showing PPL ratio."""
    configs = [
        ("L2-512", "corrected_512tok_llama_2_7b.json"),
        ("L2-1024", "corrected_1024tok_llama_2_7b.json"),
        ("M7-512", "corrected_512tok_mistral_7b.json"),
        ("M7-1024", "corrected_1024tok_mistral_7b.json"),
    ]

    show_methods = ["h2o_uniform", "snapkv", "cake", "d2o", "adakv",
                    "kivi_uniform", "kvtuner", "layer_budget"]

    # Build matrix: methods × configs (at CR=3x)
    cr = 3.0
    matrix = np.full((len(show_methods), len(configs)), np.nan)
    for j, (clabel, fname) in enumerate(configs):
        data = load(fname)
        if data is None:
            continue
        for i, m in enumerate(show_methods):
            match = [s for s in data["summary"] if s["method"] == m and s["cr"] == cr]
            if match:
                matrix[i, j] = match[0]["mean_ratio"]

    fig, ax = plt.subplots(figsize=(6, 4))
    # Use log scale for color since values range from 1.0 to 300+
    log_matrix = np.log10(np.clip(matrix, 0.1, 1000))
    im = ax.imshow(log_matrix, cmap="RdYlGn_r", aspect="auto",
                   vmin=0, vmax=2.5)

    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([c[0] for c in configs])
    ax.set_yticks(range(len(show_methods)))
    ax.set_yticklabels([LABELS.get(m, m) for m in show_methods])

    for i in range(len(show_methods)):
        for j in range(len(configs)):
            if not np.isnan(matrix[i, j]):
                val = matrix[i, j]
                text = f"{val:.1f}" if val < 10 else f"{val:.0f}"
                color = "white" if val > 5 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)

    ax.set_xlabel("Configuration")
    ax.set_title("PPL Ratio at 3× Compression (lower = better)")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_ticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])
    cbar.set_ticklabels(["1", "3", "10", "30", "100", "300"])
    cbar.set_label("PPL Ratio")

    plt.tight_layout()
    save_fig(fig, "method_heatmap_corrected")


def fig3_eviction_vs_quant_vs_joint():
    """Bar chart showing the three families at different CRs."""
    data = load("corrected_512tok_mistral_7b.json")
    if data is None:
        print("  Skipping fig3: no data")
        return

    crs = [2.0, 3.0, 4.0, 6.0]
    families = {
        "H2O (uniform evict.)": "h2o_uniform",
        "CAKE (per-layer evict.)": "cake",
        "KVTuner (per-layer quant.)": "kvtuner",
        "LayerBudget (joint)": "layer_budget",
    }
    colors = ["#6baed6", "#31a354", "#e6550d", "#e41a1c"]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(crs))
    width = 0.2

    for i, (flabel, fmethod) in enumerate(families.items()):
        vals = []
        for cr in crs:
            match = [s for s in data["summary"] if s["method"] == fmethod and s["cr"] == cr]
            vals.append(match[0]["mean_ratio"] if match else 0)
        bars = ax.bar(x + i * width, vals, width, label=flabel, color=colors[i])
        for bar, val in zip(bars, vals):
            if val > 0:
                y = bar.get_height()
                text = f"{val:.1f}" if val < 10 else f"{val:.0f}"
                ax.text(bar.get_x() + bar.get_width()/2, y + 0.3, text,
                        ha="center", fontsize=7, rotation=0)

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([f"{int(cr)}×" for cr in crs])
    ax.set_xlabel("Compression Ratio")
    ax.set_ylabel("PPL Ratio")
    ax.set_title("Mistral-7B (512 tokens): Eviction vs Quantization vs Joint")
    ax.legend(fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(0.8, 20)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_fig(fig, "eviction_vs_quant_vs_joint")


if __name__ == "__main__":
    print("Generating CORRECTED figures...")
    fig1_ppl_vs_cr()
    fig2_method_comparison_heatmap()
    fig3_eviction_vs_quant_vs_joint()
    print("Done!")
