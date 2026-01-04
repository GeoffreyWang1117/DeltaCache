"""Generate professional figures for ICML 2026 paper.

This script creates publication-quality figures:
1. Speedup vs Prefix Length curves
2. TTFT comparison bar chart
3. Token reuse rate visualization
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# Set up publication-quality style
plt.style.use('seaborn-v0_8-paper')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.pad_inches'] = 0.1

RESULTS_DIR = Path(__file__).parent / "results" / "paper"
FIGURES_DIR = Path(__file__).parent / "figures"


def load_results():
    """Load all experiment results."""
    results = {}

    # TinyLlama results
    tinyllama_path = RESULTS_DIR / "icml_tinyllama_1.1b_chat_v1.0.json"
    if tinyllama_path.exists():
        with open(tinyllama_path) as f:
            results['tinyllama'] = json.load(f)

    # Mistral 8-bit results
    mistral_8bit_path = RESULTS_DIR / "icml_mistral_7b_v0.1_8bit.json"
    if mistral_8bit_path.exists():
        with open(mistral_8bit_path) as f:
            results['mistral_8bit'] = json.load(f)

    return results


def plot_speedup_vs_prefix_length(results, save_path=None):
    """Plot speedup vs prefix length for different models."""
    fig, ax = plt.subplots(figsize=(6, 4))

    colors = {
        'tinyllama': '#2E86AB',
        'mistral_8bit': '#E94F37',
    }
    markers = {
        'tinyllama': 'o',
        'mistral_8bit': 's',
    }
    labels = {
        'tinyllama': 'TinyLlama-1.1B',
        'mistral_8bit': 'Mistral-7B (8-bit)',
    }

    for model_key, data in results.items():
        if 'experiments' not in data or 'prefix_length_scaling' not in data['experiments']:
            continue

        prefix_data = data['experiments']['prefix_length_scaling']['prefix_lengths']

        # Sort by prefix length
        sorted_items = sorted(prefix_data.items(), key=lambda x: int(x[0]))
        prefix_lengths = [int(k) for k, v in sorted_items]
        speedups = [v['speedup'] for k, v in sorted_items]

        ax.plot(prefix_lengths, speedups,
                color=colors.get(model_key, '#666666'),
                marker=markers.get(model_key, 'o'),
                markersize=7,
                linewidth=2,
                label=labels.get(model_key, model_key))

    # Reference line at y=1
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel('Prefix Length (tokens)')
    ax.set_ylabel('Speedup (x)')
    ax.set_title('DeltaCache Speedup vs Prefix Length')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Set axis limits
    ax.set_xlim(0, 1600)
    ax.set_ylim(0, 12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    return fig


def plot_ttft_comparison(results, save_path=None):
    """Plot Time-to-First-Token comparison."""
    fig, ax = plt.subplots(figsize=(6, 4))

    # Collect TTFT data
    models = []
    baseline_ttft = []
    deltacache_ttft = []

    for model_key, data in results.items():
        if 'experiments' not in data or 'ttft' not in data['experiments']:
            continue

        ttft_data = data['experiments']['ttft']['prefix_lengths']

        # Use 500 token prefix as representative
        if '500' in ttft_data:
            point = ttft_data['500']
            if model_key == 'tinyllama':
                models.append('TinyLlama-1.1B')
            elif model_key == 'mistral_8bit':
                models.append('Mistral-7B (8-bit)')
            else:
                models.append(model_key)

            baseline_ttft.append(point['baseline_ttft_ms'])
            deltacache_ttft.append(point['deltacache_ttft_ms'])

    if not models:
        print("No TTFT data available")
        return None

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, baseline_ttft, width, label='Baseline', color='#E94F37', alpha=0.8)
    bars2 = ax.bar(x + width/2, deltacache_ttft, width, label='DeltaCache', color='#2E86AB', alpha=0.8)

    # Add speedup labels
    for i, (bl, dc) in enumerate(zip(baseline_ttft, deltacache_ttft)):
        speedup = bl / dc
        ax.annotate(f'{speedup:.1f}x',
                   xy=(x[i] + width/2, dc),
                   xytext=(0, 5),
                   textcoords='offset points',
                   ha='center', va='bottom',
                   fontsize=9, fontweight='bold')

    ax.set_xlabel('Model')
    ax.set_ylabel('TTFT (ms)')
    ax.set_title('Time-to-First-Token Comparison (500 token prefix)')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    return fig


def plot_token_reuse_rate(results, save_path=None):
    """Plot token reuse rate across experiments."""
    fig, ax = plt.subplots(figsize=(6, 4))

    experiment_types = ['Prefix\nScaling', 'Multi-turn', 'RAG']
    reuse_rates = {}

    for model_key, data in results.items():
        if 'experiments' not in data:
            continue

        rates = []

        # Prefix scaling (average)
        if 'prefix_length_scaling' in data['experiments']:
            prefix_data = data['experiments']['prefix_length_scaling']['prefix_lengths']
            avg_reuse = np.mean([v['token_reuse_rate'] for v in prefix_data.values()])
            rates.append(avg_reuse * 100)
        else:
            rates.append(0)

        # Multi-turn
        if 'multiturn' in data['experiments']:
            rates.append(data['experiments']['multiturn']['summary']['mean_token_reuse'] * 100)
        else:
            rates.append(0)

        # RAG
        if 'rag' in data['experiments']:
            rates.append(data['experiments']['rag']['summary']['mean_token_reuse'] * 100)
        else:
            rates.append(0)

        reuse_rates[model_key] = rates

    x = np.arange(len(experiment_types))
    width = 0.35

    colors = {
        'tinyllama': '#2E86AB',
        'mistral_8bit': '#E94F37',
    }
    labels = {
        'tinyllama': 'TinyLlama-1.1B',
        'mistral_8bit': 'Mistral-7B (8-bit)',
    }

    for i, (model_key, rates) in enumerate(reuse_rates.items()):
        offset = (i - len(reuse_rates)/2 + 0.5) * width
        ax.bar(x + offset, rates, width,
               label=labels.get(model_key, model_key),
               color=colors.get(model_key, '#666666'),
               alpha=0.8)

    ax.set_xlabel('Experiment Type')
    ax.set_ylabel('Token Reuse Rate (%)')
    ax.set_title('Token Reuse Rate Across Experiments')
    ax.set_xticks(x)
    ax.set_xticklabels(experiment_types)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    return fig


def plot_speedup_breakdown(results, save_path=None):
    """Plot speedup breakdown by prefix length with error bars."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    colors = ['#2E86AB', '#E94F37', '#52B788', '#9B5DE5']

    # Combine data from all models
    all_data = {}

    for model_key, data in results.items():
        if 'experiments' not in data or 'prefix_length_scaling' not in data['experiments']:
            continue

        prefix_data = data['experiments']['prefix_length_scaling']['prefix_lengths']

        if model_key == 'tinyllama':
            label = 'TinyLlama-1.1B'
        elif model_key == 'mistral_8bit':
            label = 'Mistral-7B (8-bit)'
        else:
            label = model_key

        all_data[label] = prefix_data

    if not all_data:
        print("No prefix scaling data available")
        return None

    # Get all prefix lengths
    all_prefixes = set()
    for data in all_data.values():
        all_prefixes.update(int(k) for k in data.keys())
    prefix_lengths = sorted(all_prefixes)

    x = np.arange(len(prefix_lengths))
    width = 0.35
    n_models = len(all_data)

    for i, (label, prefix_data) in enumerate(all_data.items()):
        speedups = []
        for pl in prefix_lengths:
            if str(pl) in prefix_data:
                speedups.append(prefix_data[str(pl)]['speedup'])
            else:
                speedups.append(0)

        offset = (i - n_models/2 + 0.5) * width
        ax.bar(x + offset, speedups, width,
               label=label, color=colors[i % len(colors)], alpha=0.85)

    # Add reference line
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='No speedup')

    ax.set_xlabel('Prefix Length (tokens)')
    ax.set_ylabel('Speedup (x)')
    ax.set_title('DeltaCache Speedup by Prefix Length')
    ax.set_xticks(x)
    ax.set_xticklabels([str(pl) for pl in prefix_lengths])
    ax.legend(loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, 12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    return fig


def main():
    """Generate all figures."""
    print("=" * 60)
    print("Generating ICML 2026 Figures")
    print("=" * 60)

    # Create figures directory
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load results
    results = load_results()
    print(f"Loaded results for: {list(results.keys())}")

    if not results:
        print("No results found! Run experiments first.")
        return

    # Generate figures
    print("\n1. Speedup vs Prefix Length (line plot)...")
    plot_speedup_vs_prefix_length(
        results,
        save_path=FIGURES_DIR / "speedup_vs_prefix_length.pdf"
    )
    plot_speedup_vs_prefix_length(
        results,
        save_path=FIGURES_DIR / "speedup_vs_prefix_length.png"
    )

    print("\n2. Speedup breakdown (bar chart)...")
    plot_speedup_breakdown(
        results,
        save_path=FIGURES_DIR / "speedup_breakdown.pdf"
    )
    plot_speedup_breakdown(
        results,
        save_path=FIGURES_DIR / "speedup_breakdown.png"
    )

    print("\n3. TTFT comparison...")
    plot_ttft_comparison(
        results,
        save_path=FIGURES_DIR / "ttft_comparison.pdf"
    )
    plot_ttft_comparison(
        results,
        save_path=FIGURES_DIR / "ttft_comparison.png"
    )

    print("\n4. Token reuse rate...")
    plot_token_reuse_rate(
        results,
        save_path=FIGURES_DIR / "token_reuse_rate.pdf"
    )
    plot_token_reuse_rate(
        results,
        save_path=FIGURES_DIR / "token_reuse_rate.png"
    )

    print("\n" + "=" * 60)
    print("All figures saved to:", FIGURES_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
