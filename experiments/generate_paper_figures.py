#!/usr/bin/env python3
"""
Generate publication-quality figures for ICML 2026 paper.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.labelsize'] = 11
matplotlib.rcParams['figure.dpi'] = 150

RESULTS_DIR = Path(__file__).parent / "results" / "paper"
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def load_json(filename):
    path = RESULTS_DIR / filename
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

def fig_memory_comparison():
    models = ['TinyLlama-1.1B', 'Mistral-7B']
    baseline = [2303, 14491]
    deltacache = [2808, 16893]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = np.arange(len(models))
    width = 0.35

    ax.bar(x - width/2, baseline, width, label='Baseline', color='#2196F3', edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, deltacache, width, label='DeltaCache', color='#4CAF50', edgecolor='black', linewidth=0.5)

    for i, (b, d) in enumerate(zip(baseline, deltacache)):
        overhead = (d - b) / b * 100
        ax.annotate(f'+{overhead:.1f}%', (x[i] + width/2, d + 200), ha='center', fontsize=8, color='#2E7D32')

    ax.set_ylabel('Peak GPU Memory (MB)')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(deltacache) * 1.15)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "memory_comparison.pdf", bbox_inches='tight')
    plt.close()
    print("Generated: memory_comparison.pdf")

def fig_long_context_scaling():
    contexts = [501, 1001, 1501, 1801]
    speedups = [2.00, 3.32, 4.18, 5.17]
    baseline_ms = [35.2, 54.3, 69.8, 90.0]
    cached_ms = [17.6, 16.3, 16.7, 17.4]

    fig, ax1 = plt.subplots(figsize=(5, 3.5))

    color1 = '#4CAF50'
    ax1.set_xlabel('Context Length (tokens)')
    ax1.set_ylabel('Cached Speedup (x)', color=color1)
    line1 = ax1.plot(contexts, speedups, 'o-', color=color1, linewidth=2, markersize=8, label='Speedup')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, max(speedups) * 1.2)

    ax2 = ax1.twinx()
    color2 = '#FF5722'
    ax2.set_ylabel('Latency (ms)', color=color2)
    line2 = ax2.plot(contexts, baseline_ms, 's--', color='#2196F3', linewidth=1.5, markersize=6, label='Baseline')
    line3 = ax2.plot(contexts, cached_ms, '^--', color=color2, linewidth=1.5, markersize=6, label='DeltaCache')
    ax2.tick_params(axis='y', labelcolor=color2)

    lines = line1 + line2 + line3
    labels = ['Speedup', 'Baseline', 'DeltaCache']
    ax1.legend(lines, labels, loc='center left', fontsize=8)
    ax1.grid(axis='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "long_context_scaling.pdf", bbox_inches='tight')
    plt.close()
    print("Generated: long_context_scaling.pdf")

def fig_baseline_comparison():
    prefix_tokens = [298, 360, 581, 726, 1169, 1438, 1746, 2171]
    speedups = [2.89, 5.55, 4.55, 8.54, 6.52, 12.33, 8.27, 13.43]
    scenarios = ['RAG', 'Code', 'RAG', 'Code', 'RAG', 'Code', 'RAG', 'Code']

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['#2196F3' if s == 'RAG' else '#4CAF50' for s in scenarios]
    bars = ax.bar(range(len(prefix_tokens)), speedups, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Speedup (x)')
    ax.set_xlabel('Prefix Tokens')
    ax.set_xticks(range(len(prefix_tokens)))
    ax.set_xticklabels([str(t) for t in prefix_tokens], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    for bar, v in zip(bars, speedups):
        ax.annotate(f'{v:.1f}x', (bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3), ha='center', fontsize=7)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2196F3', edgecolor='black', label='RAG'),
                      Patch(facecolor='#4CAF50', edgecolor='black', label='Code')]
    ax.legend(handles=legend_elements, loc='upper left')

    mean_speedup = np.mean(speedups)
    ax.axhline(y=mean_speedup, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.annotate(f'Mean: {mean_speedup:.2f}x', (len(prefix_tokens)-2, mean_speedup + 0.5), color='red', fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "baseline_comparison.pdf", bbox_inches='tight')
    plt.close()
    print("Generated: baseline_comparison.pdf")

def fig_prefix_ablation():
    prefix_lengths = [100, 250, 500, 750, 1000]
    speedups = [0.94, 1.98, 2.77, 3.12, 3.96]
    cached_speedups = [1.14, 2.14, 3.21, 3.76, 5.05]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(prefix_lengths, speedups, 'o-', color='#2196F3', linewidth=2, markersize=8, label='Overall')
    ax.plot(prefix_lengths, cached_speedups, 's-', color='#4CAF50', linewidth=2, markersize=8, label='Cached')
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.fill_between(prefix_lengths, 1, cached_speedups, alpha=0.1, color='#4CAF50')

    ax.set_xlabel('Prefix Length (tokens)')
    ax.set_ylabel('Speedup (x)')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_ylim(0, max(cached_speedups) * 1.15)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "prefix_length_scaling.pdf", bbox_inches='tight')
    plt.close()
    print("Generated: prefix_length_scaling.pdf")

def fig_scenario_comparison():
    models = ['TinyLlama-1.1B', 'Mistral-7B']
    avg_speedup = [2.60, 5.13]
    max_speedup = [4.22, 10.90]

    fig, ax = plt.subplots(figsize=(4, 3.5))
    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, avg_speedup, width, label='Average', color='#2196F3', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, max_speedup, width, label='Maximum', color='#4CAF50', edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Speedup (x)')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}x', (bar.get_x() + bar.get_width()/2, height + 0.2), ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "scenario_comparison.pdf", bbox_inches='tight')
    plt.close()
    print("Generated: scenario_comparison.pdf")

if __name__ == "__main__":
    print("Generating ICML 2026 paper figures...")
    fig_memory_comparison()
    fig_long_context_scaling()
    fig_baseline_comparison()
    fig_prefix_ablation()
    fig_scenario_comparison()
    print(f"\nAll figures saved to: {FIGURES_DIR}")
