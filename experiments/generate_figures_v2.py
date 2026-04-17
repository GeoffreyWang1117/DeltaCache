"""Generate ICML 2026 figures from experimental results."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results" / "paper"
FIGURES_DIR = Path(__file__).parent / "figures"


def load_results():
    """Load all experimental results."""
    results = {}

    # Targeted benchmark results
    for name in ["targeted_benchmark_tinyllama_1.1b_chat_v1.0", "targeted_benchmark_mistral7b"]:
        path = RESULTS_DIR / f"{name}.json"
        if path.exists():
            with open(path) as f:
                results[name] = json.load(f)

    # Ablation results
    path = RESULTS_DIR / "ablation_sensitivity_tinyllama_1.1b_chat_v1.0.json"
    if path.exists():
        with open(path) as f:
            results["ablation"] = json.load(f)

    return results


def plot_scenario_comparison(results):
    """Plot speedup comparison across scenarios for both models."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Get scenarios from TinyLlama results
    tinyllama = results.get("targeted_benchmark_tinyllama_1.1b_chat_v1.0", {}).get("scenarios", {})
    mistral = results.get("targeted_benchmark_mistral7b", {}).get("scenarios", {})

    if not tinyllama or not mistral:
        print("Missing targeted benchmark results")
        return

    # Common scenarios
    scenarios = list(tinyllama.keys())
    x = np.arange(len(scenarios))
    width = 0.35

    tinyllama_speedups = [tinyllama[s]["speedup"]["overall"] for s in scenarios]
    mistral_speedups = [mistral[s]["speedup"]["overall"] for s in scenarios]

    bars1 = ax.bar(x - width/2, tinyllama_speedups, width, label='TinyLlama-1.1B', color='#2196F3')
    bars2 = ax.bar(x + width/2, mistral_speedups, width, label='Mistral-7B', color='#FF5722')

    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_ylabel('Speedup (×)', fontsize=12)
    ax.set_title('DeltaCache Speedup Across Scenarios', fontsize=14, fontweight='bold')
    ax.set_xticks(x)

    # Clean up scenario names for display
    display_names = []
    for s in scenarios:
        name = s.replace('rag_', 'RAG: ').replace('_', ' ').title()
        if 'Fewshot' in name:
            name = 'Few-shot'
        elif 'System Prompt' in name:
            name = 'System Prompt'
        elif 'Code' in name:
            name = 'Code Completion'
        display_names.append(name)

    ax.set_xticklabels(display_names, rotation=30, ha='right', fontsize=10)
    ax.legend(loc='upper right')
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}×',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}×',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'scenario_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'scenario_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: scenario_comparison.pdf/png")


def plot_prefix_length_scaling(results):
    """Plot speedup vs prefix length from ablation."""
    ablation = results.get("ablation", {}).get("ablations", {}).get("prefix_length", {})

    if not ablation:
        print("Missing ablation results")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    prefix_data = ablation.get("prefix_lengths", {})
    lengths = sorted([int(k) for k in prefix_data.keys()])
    speedups = [prefix_data[str(l)]["speedup"] for l in lengths]
    cached_speedups = [prefix_data[str(l)]["cached_speedup"] for l in lengths]

    ax.plot(lengths, speedups, 'o-', linewidth=2, markersize=8, label='Overall Speedup', color='#2196F3')
    ax.plot(lengths, cached_speedups, 's--', linewidth=2, markersize=8, label='Cached Speedup', color='#4CAF50')

    ax.set_xlabel('Prefix Length (tokens)', fontsize=12)
    ax.set_ylabel('Speedup (×)', fontsize=12)
    ax.set_title('Speedup Scaling with Prefix Length', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)

    # Add annotations
    for i, (l, s, c) in enumerate(zip(lengths, speedups, cached_speedups)):
        ax.annotate(f'{s:.1f}×', (l, s), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'prefix_length_scaling.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'prefix_length_scaling.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: prefix_length_scaling.pdf/png")


def plot_eviction_policy_comparison(results):
    """Plot eviction policy comparison."""
    ablation = results.get("ablation", {}).get("ablations", {}).get("eviction_policies", {})

    if not ablation:
        print("Missing eviction policy results")
        return

    fig, ax = plt.subplots(figsize=(8, 4))

    policies = ablation.get("policies", {})
    names = list(policies.keys())
    speedups = [policies[n]["speedup"] for n in names]

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
    bars = ax.bar(names, speedups, color=colors)

    ax.set_xlabel('Eviction Policy', fontsize=12)
    ax.set_ylabel('Speedup (×)', fontsize=12)
    ax.set_title('Eviction Policy Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(speedups) * 1.2)

    # Add value labels
    for bar, speedup in zip(bars, speedups):
        ax.annotate(f'{speedup:.2f}×',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'eviction_policy_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'eviction_policy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: eviction_policy_comparison.pdf/png")


def plot_query_count_scaling(results):
    """Plot speedup vs number of queries."""
    ablation = results.get("ablation", {}).get("ablations", {}).get("query_count", {})

    if not ablation:
        print("Missing query count results")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    query_data = ablation.get("query_counts", {})
    counts = sorted([int(k) for k in query_data.keys()])
    overall = [query_data[str(c)]["overall_speedup"] for c in counts]
    cached = [query_data[str(c)]["cached_speedup"] for c in counts]

    ax.plot(counts, overall, 'o-', linewidth=2, markersize=8, label='Overall Speedup', color='#2196F3')
    ax.plot(counts, cached, 's--', linewidth=2, markersize=8, label='Cached Speedup', color='#4CAF50')

    ax.set_xlabel('Number of Queries', fontsize=12)
    ax.set_ylabel('Speedup (×)', fontsize=12)
    ax.set_title('Speedup vs Query Count (Cache Amortization)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'query_count_scaling.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'query_count_scaling.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: query_count_scaling.pdf/png")


def plot_model_comparison(results):
    """Plot speedup comparison between TinyLlama and Mistral-7B."""
    tinyllama = results.get("targeted_benchmark_tinyllama_1.1b_chat_v1.0", {})
    mistral = results.get("targeted_benchmark_mistral7b", {})

    if not tinyllama or not mistral:
        print("Missing model comparison data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Overall speedup comparison
    ax1 = axes[0]
    models = ['TinyLlama-1.1B', 'Mistral-7B']
    mean_speedups = [
        tinyllama.get("summary", {}).get("mean_speedup", 0),
        mistral.get("summary", {}).get("mean_speedup", 0)
    ]
    max_speedups = [
        tinyllama.get("summary", {}).get("max_speedup", 0),
        mistral.get("summary", {}).get("max_speedup", 0)
    ]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax1.bar(x - width/2, mean_speedups, width, label='Mean', color='#2196F3')
    bars2 = ax1.bar(x + width/2, max_speedups, width, label='Max', color='#FF5722')

    ax1.set_ylabel('Speedup (×)', fontsize=12)
    ax1.set_title('Model Comparison: Speedup', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=11)
    ax1.legend()

    for bar in bars1 + bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}×',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Plot 2: Best scenario comparison (code completion)
    ax2 = axes[1]

    code_tinyllama = tinyllama.get("scenarios", {}).get("code_completion", {})
    code_mistral = mistral.get("scenarios", {}).get("code_completion", {})

    metrics = ['Overall', 'Cached']
    tinyllama_values = [
        code_tinyllama.get("speedup", {}).get("overall", 0),
        code_tinyllama.get("speedup", {}).get("cached_queries", 0)
    ]
    mistral_values = [
        code_mistral.get("speedup", {}).get("overall", 0),
        code_mistral.get("speedup", {}).get("cached_queries", 0)
    ]

    x = np.arange(len(metrics))
    bars1 = ax2.bar(x - width/2, tinyllama_values, width, label='TinyLlama-1.1B', color='#2196F3')
    bars2 = ax2.bar(x + width/2, mistral_values, width, label='Mistral-7B', color='#FF5722')

    ax2.set_ylabel('Speedup (×)', fontsize=12)
    ax2.set_title('Code Completion Scenario', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, fontsize=11)
    ax2.legend()

    for bar in bars1 + bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}×',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'model_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: model_comparison.pdf/png")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    results = load_results()

    print(f"\nFound results: {list(results.keys())}")

    print("\nGenerating figures...")
    plot_scenario_comparison(results)
    plot_prefix_length_scaling(results)
    plot_eviction_policy_comparison(results)
    plot_query_count_scaling(results)
    plot_model_comparison(results)

    print(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
