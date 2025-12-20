"""Generate visualizations for DeltaCache experiment results."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"


def load_results():
    """Load experiment results from JSON."""
    with open(RESULTS_DIR / "cache_experiments.json") as f:
        return json.load(f)


def plot_system_prompt_impact(results, save_path):
    """Plot system prompt length vs performance metrics."""
    data = results["system_prompt"]

    prompt_lengths = [int(r["name"].split("_")[-1]) for r in data]
    reuse_rates = [r["token_reuse_rate"] * 100 for r in data]
    throughputs = [r["throughput_tokens_per_sec"] for r in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Token reuse rate
    bars1 = ax1.bar(range(len(prompt_lengths)), reuse_rates, color='#2ecc71', edgecolor='#27ae60', linewidth=1.5)
    ax1.set_xticks(range(len(prompt_lengths)))
    ax1.set_xticklabels([str(p) for p in prompt_lengths])
    ax1.set_xlabel('System Prompt Length (tokens)', fontsize=11)
    ax1.set_ylabel('Token Reuse Rate (%)', fontsize=11)
    ax1.set_title('Token Reuse vs System Prompt Length', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars1, reuse_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

    # Throughput
    bars2 = ax2.bar(range(len(prompt_lengths)), throughputs, color='#3498db', edgecolor='#2980b9', linewidth=1.5)
    ax2.set_xticks(range(len(prompt_lengths)))
    ax2.set_xticklabels([str(p) for p in prompt_lengths])
    ax2.set_xlabel('System Prompt Length (tokens)', fontsize=11)
    ax2.set_ylabel('Throughput (tokens/sec)', fontsize=11)
    ax2.set_title('Throughput vs System Prompt Length', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars2, throughputs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{val:.0f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_rag_scenario(results, save_path):
    """Plot RAG document caching results."""
    data = results["rag"]

    doc_lengths = [int(r["name"].split("_")[-1]) for r in data]
    reuse_rates = [r["token_reuse_rate"] * 100 for r in data]
    hit_rates = [r["cache_hit_rate"] * 100 for r in data]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(doc_lengths))
    width = 0.35

    bars1 = ax.bar(x - width/2, hit_rates, width, label='Cache Hit Rate',
                   color='#9b59b6', edgecolor='#8e44ad', linewidth=1.5)
    bars2 = ax.bar(x + width/2, reuse_rates, width, label='Token Reuse Rate',
                   color='#e74c3c', edgecolor='#c0392b', linewidth=1.5)

    ax.set_xlabel('Document Length (tokens)', fontsize=11)
    ax.set_ylabel('Rate (%)', fontsize=11)
    ax.set_title('RAG Document Caching Performance', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in doc_lengths])
    ax.legend(loc='lower right')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars1, hit_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, reuse_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_few_shot_learning(results, save_path):
    """Plot few-shot learning results."""
    data = results["few_shot"]

    n_shots = [int(r["name"].split("_")[-1]) for r in data]
    reuse_rates = [r["token_reuse_rate"] * 100 for r in data]
    throughputs = [r["throughput_tokens_per_sec"] for r in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Token reuse rate
    ax1.plot(n_shots, reuse_rates, 'o-', color='#f39c12', linewidth=2, markersize=10)
    ax1.fill_between(n_shots, reuse_rates, alpha=0.2, color='#f39c12')
    ax1.set_xlabel('Number of Few-Shot Examples', fontsize=11)
    ax1.set_ylabel('Token Reuse Rate (%)', fontsize=11)
    ax1.set_title('Token Reuse vs Few-Shot Examples', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(alpha=0.3)
    ax1.set_xticks(n_shots)

    for x, y in zip(n_shots, reuse_rates):
        ax1.annotate(f'{y:.1f}%', (x, y), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=10)

    # Throughput scaling
    ax2.plot(n_shots, throughputs, 's-', color='#1abc9c', linewidth=2, markersize=10)
    ax2.fill_between(n_shots, throughputs, alpha=0.2, color='#1abc9c')
    ax2.set_xlabel('Number of Few-Shot Examples', fontsize=11)
    ax2.set_ylabel('Throughput (tokens/sec)', fontsize=11)
    ax2.set_title('Throughput Scaling with Few-Shot Examples', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.set_xticks(n_shots)

    for x, y in zip(n_shots, throughputs):
        ax2.annotate(f'{y:.0f}', (x, y), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_baseline_comparison(results, save_path):
    """Plot baseline vs DeltaCache comparison."""
    data = results["baseline_comparison"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Time comparison
    times = [data["baseline_time_ms"] / 1000, data["deltacache_time_ms"] / 1000]
    labels = ['Baseline\n(No Cache)', 'DeltaCache']
    colors = ['#e74c3c', '#2ecc71']

    bars1 = ax1.bar(labels, times, color=colors, edgecolor=['#c0392b', '#27ae60'], linewidth=2)
    ax1.set_ylabel('Total Time (seconds)', fontsize=11)
    ax1.set_title('Processing Time Comparison', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars1, times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add speedup annotation
    speedup = data["speedup"]
    ax1.annotate(f'{speedup:.2f}x faster',
                xy=(1, times[1]), xytext=(1.3, times[0] * 0.7),
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2),
                fontsize=12, fontweight='bold', color='#27ae60')

    # Token computation comparison
    tokens = [data["baseline_tokens"], data["deltacache_tokens_computed"]]

    bars2 = ax2.bar(labels, tokens, color=colors, edgecolor=['#c0392b', '#27ae60'], linewidth=2)
    ax2.set_ylabel('Tokens Computed', fontsize=11)
    ax2.set_title('Computation Savings', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars2, tokens):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f'{val:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add savings annotation
    savings = data["token_savings"] * 100
    ax2.annotate(f'{savings:.1f}% saved',
                xy=(1, tokens[1]), xytext=(1.3, tokens[0] * 0.6),
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2),
                fontsize=12, fontweight='bold', color='#27ae60')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_summary(results, save_path):
    """Create a summary visualization of all experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. System Prompt (top-left)
    ax = axes[0, 0]
    data = results["system_prompt"]
    prompt_lengths = [int(r["name"].split("_")[-1]) for r in data]
    reuse_rates = [r["token_reuse_rate"] * 100 for r in data]

    bars = ax.bar(range(len(prompt_lengths)), reuse_rates, color='#2ecc71', edgecolor='#27ae60', linewidth=1.5)
    ax.set_xticks(range(len(prompt_lengths)))
    ax.set_xticklabels([str(p) for p in prompt_lengths])
    ax.set_xlabel('System Prompt Length (tokens)')
    ax.set_ylabel('Token Reuse Rate (%)')
    ax.set_title('1. System Prompt Sharing', fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, reuse_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    # 2. RAG (top-right)
    ax = axes[0, 1]
    data = results["rag"]
    doc_lengths = [int(r["name"].split("_")[-1]) for r in data]
    reuse_rates = [r["token_reuse_rate"] * 100 for r in data]

    bars = ax.bar(range(len(doc_lengths)), reuse_rates, color='#9b59b6', edgecolor='#8e44ad', linewidth=1.5)
    ax.set_xticks(range(len(doc_lengths)))
    ax.set_xticklabels([str(d) for d in doc_lengths])
    ax.set_xlabel('Document Length (tokens)')
    ax.set_ylabel('Token Reuse Rate (%)')
    ax.set_title('2. RAG Document Caching', fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, reuse_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    # 3. Few-Shot (bottom-left)
    ax = axes[1, 0]
    data = results["few_shot"]
    n_shots = [int(r["name"].split("_")[-1]) for r in data]
    reuse_rates = [r["token_reuse_rate"] * 100 for r in data]

    ax.plot(n_shots, reuse_rates, 'o-', color='#f39c12', linewidth=2, markersize=10)
    ax.fill_between(n_shots, reuse_rates, alpha=0.2, color='#f39c12')
    ax.set_xlabel('Number of Few-Shot Examples')
    ax.set_ylabel('Token Reuse Rate (%)')
    ax.set_title('3. Few-Shot Learning', fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.set_xticks(n_shots)
    for x, y in zip(n_shots, reuse_rates):
        ax.annotate(f'{y:.1f}%', (x, y), textcoords="offset points",
                   xytext=(0, 10), ha='center', fontsize=9)

    # 4. Baseline Comparison (bottom-right)
    ax = axes[1, 1]
    data = results["baseline_comparison"]

    metrics = ['Speedup', 'Token\nSavings']
    values = [data["speedup"], data["token_savings"] * 100]
    colors = ['#3498db', '#e74c3c']

    bars = ax.bar(metrics, values, color=colors, edgecolor=['#2980b9', '#c0392b'], linewidth=2)
    ax.set_ylabel('Value')
    ax.set_title('4. Baseline Comparison', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    ax.text(0, values[0] + 0.1, f'{values[0]:.2f}x', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.text(1, values[1] + 2, f'{values[1]:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add secondary y-axis for percentage
    ax.set_ylim(0, max(values) * 1.2)

    fig.suptitle('DeltaCache Experiment Results Summary', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    """Generate all visualizations."""
    print("="*60)
    print("DeltaCache Results Visualization")
    print("="*60)

    # Load results
    results = load_results()

    # Create output directory
    figures_dir = RESULTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Generate individual plots
    print("\nGenerating visualizations...")

    plot_system_prompt_impact(results, figures_dir / "system_prompt_impact.png")
    plot_rag_scenario(results, figures_dir / "rag_scenario.png")
    plot_few_shot_learning(results, figures_dir / "few_shot_learning.png")
    plot_baseline_comparison(results, figures_dir / "baseline_comparison.png")
    plot_summary(results, figures_dir / "summary.png")

    print(f"\nAll figures saved to {figures_dir}")
    print("\nGenerated files:")
    for f in sorted(figures_dir.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
