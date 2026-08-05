#!/usr/bin/env python3
"""Generate figures for NeurIPS 2026 submission.

Reads from results/main_v2/ (inverted importance, mean-fill) and
results/paper/ (zero-fill corrected) to produce three figures:

1. ppl_vs_compression_corrected.pdf — PPL ratio vs CR for key methods
2. allocation_heatmap.pdf — Per-layer token allocation on TinyLlama
3. signal_curves.pdf — Gini sparsity + importance across layers

Output: paper/neurips2026/figures/
"""

import json
import sys
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path(__file__).parent / "results"
FIG_DIR = Path(__file__).parent.parent / "paper" / "neurips2026" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Also save to ICML figures dir
FIG_DIR_ICML = Path(__file__).parent.parent / "paper" / "figures"

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.labelsize": 11,
    "legend.fontsize": 8, "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

LABELS = {
    "full_kv": "Full KV", "h2o_uniform": "H2O", "snapkv": "SnapKV",
    "cake": "CAKE", "kivi_uniform": "KIVI", "kvtuner": "KVTuner",
    "layer_budget": "LayerBudget (Ours)",
}


def save_fig(fig, name):
    for d in [FIG_DIR, FIG_DIR_ICML]:
        for ext in ["pdf", "png"]:
            fig.savefig(d / f"{name}.{ext}")
    plt.close()
    print(f"  Saved: {name}.pdf")


def load_json(path):
    if path.exists():
        return json.load(open(path))
    return None


# ═══════════════════════════════════════════════════════════════════
#  Figure 1: PPL ratio vs compression ratio
# ═══════════════════════════════════════════════════════════════════

def fig1_ppl_vs_cr():
    """PPL ratio vs CR curves. Uses main_v2 (mean-fill, inverted) data."""
    print("\n[Fig 1] PPL vs Compression Ratio")

    # Use main_v2 data (inverted importance, mean-fill) where available
    # Fall back to paper/ corrected (zero-fill) data
    configs = [
        ("Llama-2-7B\n512 tok", [
            RESULTS_DIR / "main_v2" / "main_v2_512tok_llama_2_7b.json",
            RESULTS_DIR / "paper" / "corrected_512tok_llama_2_7b.json",
        ]),
        ("Llama-2-7B\n1024 tok", [
            RESULTS_DIR / "main_v2" / "main_v2_1024tok_llama_2_7b.json",
            RESULTS_DIR / "paper" / "corrected_1024tok_llama_2_7b.json",
        ]),
        ("Mistral-7B\n512 tok", [
            RESULTS_DIR / "main_v2" / "main_v2_512tok_mistral_7b.json",
            RESULTS_DIR / "paper" / "corrected_512tok_mistral_7b.json",
        ]),
        ("Mistral-7B\n1024 tok", [
            RESULTS_DIR / "main_v2" / "main_v2_1024tok_mistral_7b.json",
            RESULTS_DIR / "paper" / "corrected_1024tok_mistral_7b.json",
        ]),
    ]

    show_methods = ["cake", "h2o_uniform", "kivi_uniform", "kvtuner", "layer_budget"]
    colors = {
        "cake": "#31a354", "h2o_uniform": "#6baed6", "kivi_uniform": "#fdae6b",
        "kvtuner": "#e6550d", "layer_budget": "#e41a1c",
    }
    markers = {"cake": "^", "h2o_uniform": "o", "kivi_uniform": "P",
               "kvtuner": "X", "layer_budget": "*"}
    sizes = {"cake": 5, "h2o_uniform": 5, "kivi_uniform": 6,
             "kvtuner": 6, "layer_budget": 12}

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharey=False)

    for ax, (label, paths) in zip(axes, configs):
        data = None
        for p in paths:
            data = load_json(p)
            if data is not None:
                break

        if data is None:
            ax.set_title(label + "\n(no data)")
            continue

        # Detect key name
        summary_key = "summary"
        if "summary_mean_fill" in data:
            summary_key = "summary_mean_fill"
        elif "summary_zero_fill" in data:
            summary_key = "summary_zero_fill"

        method_data = {}
        for s in data[summary_key]:
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

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1.08), fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    save_fig(fig, "ppl_vs_compression_corrected")


# ═══════════════════════════════════════════════════════════════════
#  Figure 2: Per-layer allocation heatmap
# ═══════════════════════════════════════════════════════════════════

def fig2_allocation_heatmap():
    """Per-layer allocation at different CRs on TinyLlama-1.1B."""
    print("\n[Fig 2] Allocation Heatmap")

    from deltacache.core.layer_budget_allocator import LayerBudgetAllocator

    # TinyLlama-1.1B parameters
    nl, nh, hd, seq_len = 22, 4, 64, 128

    # Gini scores from profiling TinyLlama (stable across contexts)
    gini_scores = {
        0: 0.431, 1: 0.652, 2: 0.789, 3: 0.856, 4: 0.925,
        5: 0.912, 6: 0.878, 7: 0.845, 8: 0.887, 9: 0.821,
        10: 0.756, 11: 0.723, 12: 0.705, 13: 0.689, 14: 0.712,
        15: 0.738, 16: 0.738, 17: 0.751, 18: 0.769, 19: 0.782,
        20: 0.794, 21: 0.810,
    }

    crs = [2.0, 3.0, 4.0, 6.0]
    bit_colors = {16: "#2ca02c", 8: "#ff7f0e", 4: "#d62728"}
    bit_labels = {16: "FP16", 8: "INT8", 4: "INT4"}

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)

    allocator = LayerBudgetAllocator(nl, nh, hd)
    importance = allocator.compute_importance_weights(nl)
    full_mem = allocator.full_memory(seq_len)

    for idx, (cr, ax) in enumerate(zip(crs, axes.flat)):
        budget = int(full_mem / cr)
        alloc = allocator.allocate(gini_scores, importance, budget, seq_len)

        layers = list(range(nl))
        fracs = [alloc.allocations[l].token_budget / seq_len for l in layers]
        bits = [alloc.allocations[l].quant_bits for l in layers]
        bar_colors = [bit_colors[b] for b in bits]

        ax.bar(layers, fracs, color=bar_colors, width=0.8, edgecolor="white", linewidth=0.3)

        # Overlay Gini
        ax2 = ax.twinx()
        gini_vals = [gini_scores[l] for l in layers]
        ax2.plot(layers, gini_vals, "k--", alpha=0.5, linewidth=1, label="Gini")
        ax2.set_ylim(0, 1.1)
        if idx % 2 == 1:
            ax2.set_ylabel("Gini", fontsize=9)
        else:
            ax2.set_yticklabels([])

        ax.set_ylim(0, 1.15)
        ax.set_title(f"{cr:.0f}× compression", fontsize=10)
        if idx >= 2:
            ax.set_xlabel("Layer")
        if idx % 2 == 0:
            ax.set_ylabel("Token Fraction")

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [Patch(color=bit_colors[b], label=bit_labels[b]) for b in [16, 8, 4]]
    handles.append(Line2D([0], [0], color="black", linestyle="--", alpha=0.5, label="Gini"))
    fig.legend(handles=handles, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.02), fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "allocation_heatmap")


# ═══════════════════════════════════════════════════════════════════
#  Figure 3: Signal curves (Gini + importance)
# ═══════════════════════════════════════════════════════════════════

def fig3_signal_curves():
    """Gini sparsity and sigmoid importance across layers."""
    print("\n[Fig 3] Signal Curves")

    from deltacache.core.layer_budget_allocator import LayerBudgetAllocator

    # Load real Mistral-7B Gini data
    signal_path = RESULTS_DIR / "signal_analysis" / "signal_analysis_mistral_7b_instruct_v0.2_20260330_2328.json"
    signal_data = load_json(signal_path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    models = []

    # TinyLlama (22 layers) — hardcoded Gini from profiling
    tl_gini = {
        0: 0.431, 1: 0.652, 2: 0.789, 3: 0.856, 4: 0.925,
        5: 0.912, 6: 0.878, 7: 0.845, 8: 0.887, 9: 0.821,
        10: 0.756, 11: 0.723, 12: 0.705, 13: 0.689, 14: 0.712,
        15: 0.738, 16: 0.738, 17: 0.751, 18: 0.769, 19: 0.782,
        20: 0.794, 21: 0.810,
    }
    models.append(("TinyLlama-1.1B (22L)", 22, tl_gini))

    # Mistral-7B (32 layers) — from signal analysis JSON
    if signal_data:
        m7_gini = {s["layer"]: s["gini_512"] for s in signal_data["gini_stability"]}
        models.append(("Mistral-7B (32L)", 32, m7_gini))
    else:
        print("  WARNING: No Mistral signal data, using TinyLlama only")

    colors_gini = ["#1f77b4", "#ff7f0e"]
    colors_imp = ["#2ca02c", "#d62728"]

    for idx, (ax, (mname, nl, gini)) in enumerate(zip(axes, models)):
        layers = list(range(nl))
        gini_vals = [gini.get(l, 0.5) for l in layers]

        # Compute inverted importance (default)
        allocator = LayerBudgetAllocator(nl, 8, 128)
        imp = allocator.compute_importance_weights(nl)
        imp_vals = [imp.get(l, 0.5) for l in layers]

        ax.plot(layers, gini_vals, "o-", color=colors_gini[idx], markersize=4,
                linewidth=1.5, label="Gini (sparsity)")
        ax.plot(layers, imp_vals, "s--", color=colors_imp[idx], markersize=3,
                linewidth=1.5, label="Inverted importance")

        # Annotate regions
        ax.axvspan(0, nl * 0.3, alpha=0.08, color="red", label="Protected (early)")
        ax.axvspan(nl * 0.7, nl - 1, alpha=0.08, color="blue", label="Compressible (late)")

        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Score")
        ax.set_title(mname, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, loc="lower right")

        # Correlation
        rho = np.corrcoef(gini_vals, imp_vals)[0, 1]
        ax.text(0.02, 0.95, f"$\\rho$ = {rho:.3f}", transform=ax.transAxes,
                fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    save_fig(fig, "signal_curves")


if __name__ == "__main__":
    print("=" * 50)
    print("  NeurIPS 2026 Figure Generation")
    print("=" * 50)
    fig1_ppl_vs_cr()
    fig2_allocation_heatmap()
    fig3_signal_curves()
    print("\nAll figures saved to:")
    print(f"  {FIG_DIR}")
    print(f"  {FIG_DIR_ICML}")
