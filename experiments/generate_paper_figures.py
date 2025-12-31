"""Generate paper-ready figures and tables from DeltaCache experiment results."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Use a clean, publication-ready style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

RESULTS_DIR = Path(__file__).parent / "results" / "paper"
FIGURES_DIR = RESULTS_DIR / "figures"

# Color palette for consistency
COLORS = {
    'primary': '#2563eb',      # Blue
    'secondary': '#16a34a',    # Green
    'tertiary': '#dc2626',     # Red
    'quaternary': '#9333ea',   # Purple
    'accent': '#f59e0b',       # Amber
    'gray': '#6b7280',
    'light_blue': '#93c5fd',
    'light_green': '#86efac',
}


def load_results():
    """Load all experiment results."""
    results = {}

    for name in ['tinyllama', 'qwen2', 'ablation']:
        path = RESULTS_DIR / f"{name}_results.json"
        if path.exists():
            with open(path) as f:
                results[name] = json.load(f)

    return results


def generate_correctness_table(results):
    """Generate LaTeX table for correctness validation."""
    print("\n" + "="*60)
    print("TABLE 1: Correctness Validation Results")
    print("="*60)

    print("""
\\begin{table}[h]
\\centering
\\caption{Correctness Validation: DeltaCache vs Full Computation}
\\label{tab:correctness}
\\begin{tabular}{lcccc}
\\toprule
Model & Tests & KV Max Diff & KV Mean Diff & Tokens Match \\\\
\\midrule""")

    for model_name, key in [("TinyLlama-1.1B", "tinyllama"), ("Qwen2-1.5B", "qwen2")]:
        if key in results:
            c = results[key]["correctness"]
            print(f"{model_name} & {c['num_tests']} & {c['avg_kv_max_diff']:.4f} & "
                  f"{c['avg_kv_mean_diff']:.4f} & {c['tokens_match_rate']*100:.0f}\\% \\\\")

    print("""\\bottomrule
\\end{tabular}
\\end{table}
""")


def generate_performance_table(results):
    """Generate LaTeX table for performance comparison."""
    print("\n" + "="*60)
    print("TABLE 2: Performance Comparison")
    print("="*60)

    print("""
\\begin{table}[h]
\\centering
\\caption{Performance: DeltaCache vs Baseline (No Cache)}
\\label{tab:performance}
\\begin{tabular}{lccccc}
\\toprule
Model & \\multicolumn{2}{c}{Latency (ms)} & \\multicolumn{2}{c}{Throughput (tok/s)} & Speedup \\\\
\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}
 & Baseline & DeltaCache & Baseline & DeltaCache & \\\\
\\midrule""")

    for model_name, key in [("TinyLlama-1.1B", "tinyllama"), ("Qwen2-1.5B", "qwen2")]:
        if key in results and "baseline_comparison" in results[key]:
            bc = results[key]["baseline_comparison"]
            wc = bc["with_cache"]
            wo = bc["without_cache"]
            speedup = wo["avg_latency_ms"] / wc["avg_latency_ms"]
            print(f"{model_name} & {wo['avg_latency_ms']:.1f} & {wc['avg_latency_ms']:.1f} & "
                  f"{wo['throughput_tok_s']:.0f} & {wc['throughput_tok_s']:.0f} & "
                  f"{speedup:.2f}$\\times$ \\\\")

    print("""\\bottomrule
\\end{tabular}
\\end{table}
""")


def plot_eviction_policies(results, save_path):
    """Plot eviction policy comparison."""
    if 'ablation' not in results:
        print("No ablation results found")
        return

    data = results['ablation']['eviction_policies']

    policies = [d['policy'].upper() for d in data]
    latencies = [d['avg_latency_ms'] for d in data]
    throughputs = [d['throughput_tok_s'] for d in data]
    hit_rates = [d['cache_hit_rate'] * 100 for d in data]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Latency
    ax = axes[0]
    bars = ax.bar(policies, latencies, color=COLORS['primary'], edgecolor='white', linewidth=0.5)
    ax.set_ylabel('Average Latency (ms)')
    ax.set_title('(a) Latency by Policy')
    ax.set_ylim(0, max(latencies) * 1.2)
    for bar, val in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    # Throughput
    ax = axes[1]
    bars = ax.bar(policies, throughputs, color=COLORS['secondary'], edgecolor='white', linewidth=0.5)
    ax.set_ylabel('Throughput (tokens/s)')
    ax.set_title('(b) Throughput by Policy')
    ax.set_ylim(0, max(throughputs) * 1.2)
    for bar, val in zip(bars, throughputs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                f'{val:.0f}', ha='center', va='bottom', fontsize=9)

    # Hit Rate
    ax = axes[2]
    bars = ax.bar(policies, hit_rates, color=COLORS['quaternary'], edgecolor='white', linewidth=0.5)
    ax.set_ylabel('Cache Hit Rate (%)')
    ax.set_title('(c) Hit Rate by Policy')
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, hit_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_prompt_length_scaling(results, save_path):
    """Plot token reuse vs prompt length."""
    if 'ablation' not in results:
        return

    data = results['ablation']['prompt_lengths']

    lengths = [d['prompt_length'] for d in data]
    reuse_rates = [d['token_reuse_rate'] * 100 for d in data]
    latencies = [d['avg_latency_ms'] for d in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Token Reuse Rate
    ax1.plot(lengths, reuse_rates, 'o-', color=COLORS['primary'],
             linewidth=2, markersize=8, markerfacecolor='white', markeredgewidth=2)
    ax1.fill_between(lengths, reuse_rates, alpha=0.15, color=COLORS['primary'])
    ax1.set_xlabel('Prompt Length (tokens)')
    ax1.set_ylabel('Token Reuse Rate (%)')
    ax1.set_title('(a) Token Reuse vs Prompt Length')
    ax1.set_ylim(70, 85)
    ax1.set_xticks(lengths)
    for x, y in zip(lengths, reuse_rates):
        ax1.annotate(f'{y:.1f}%', (x, y), textcoords="offset points",
                    xytext=(0, 8), ha='center', fontsize=9)

    # Latency
    ax2.plot(lengths, latencies, 's-', color=COLORS['tertiary'],
             linewidth=2, markersize=8, markerfacecolor='white', markeredgewidth=2)
    ax2.fill_between(lengths, latencies, alpha=0.15, color=COLORS['tertiary'])
    ax2.set_xlabel('Prompt Length (tokens)')
    ax2.set_ylabel('Average Latency (ms)')
    ax2.set_title('(b) Latency vs Prompt Length')
    ax2.set_xticks(lengths)
    for x, y in zip(lengths, latencies):
        ax2.annotate(f'{y:.1f}', (x, y), textcoords="offset points",
                    xytext=(0, 8), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_query_count_scaling(results, save_path):
    """Plot throughput vs number of queries."""
    if 'ablation' not in results:
        return

    data = results['ablation']['query_counts']

    queries = [d['num_queries'] for d in data]
    hit_rates = [d['cache_hit_rate'] * 100 for d in data]
    throughputs = [d['throughput_tok_s'] for d in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Cache Hit Rate scaling
    ax1.plot(queries, hit_rates, 'o-', color=COLORS['secondary'],
             linewidth=2, markersize=8, markerfacecolor='white', markeredgewidth=2)
    ax1.fill_between(queries, hit_rates, alpha=0.15, color=COLORS['secondary'])
    ax1.set_xlabel('Number of Queries')
    ax1.set_ylabel('Cache Hit Rate (%)')
    ax1.set_title('(a) Cache Efficiency vs Query Volume')
    ax1.set_ylim(50, 80)
    for x, y in zip(queries, hit_rates):
        ax1.annotate(f'{y:.1f}%', (x, y), textcoords="offset points",
                    xytext=(0, 8), ha='center', fontsize=9)

    # Throughput scaling
    ax2.plot(queries, throughputs, 's-', color=COLORS['accent'],
             linewidth=2, markersize=8, markerfacecolor='white', markeredgewidth=2)
    ax2.fill_between(queries, throughputs, alpha=0.15, color=COLORS['accent'])
    ax2.set_xlabel('Number of Queries')
    ax2.set_ylabel('Throughput (tokens/s)')
    ax2.set_title('(b) Throughput Scaling')
    for x, y in zip(queries, throughputs):
        ax2.annotate(f'{y:.0f}', (x, y), textcoords="offset points",
                    xytext=(0, 8), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_baseline_comparison(results, save_path):
    """Plot baseline vs DeltaCache comparison for all models."""
    models = []
    baseline_latencies = []
    cache_latencies = []
    speedups = []

    for model_name, key in [("TinyLlama", "tinyllama"), ("Qwen2", "qwen2")]:
        if key in results and "baseline_comparison" in results[key]:
            bc = results[key]["baseline_comparison"]
            models.append(model_name)
            baseline_latencies.append(bc["without_cache"]["avg_latency_ms"])
            cache_latencies.append(bc["with_cache"]["avg_latency_ms"])
            speedups.append(bc["without_cache"]["avg_latency_ms"] / bc["with_cache"]["avg_latency_ms"])

    if not models:
        print("No baseline comparison data found")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    x = np.arange(len(models))
    width = 0.35

    # Latency comparison
    bars1 = ax1.bar(x - width/2, baseline_latencies, width, label='Baseline (No Cache)',
                    color=COLORS['gray'], edgecolor='white', linewidth=0.5)
    bars2 = ax1.bar(x + width/2, cache_latencies, width, label='DeltaCache',
                    color=COLORS['primary'], edgecolor='white', linewidth=0.5)

    ax1.set_ylabel('Average Latency (ms)')
    ax1.set_title('(a) Latency Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend(loc='upper right')

    for bar, val in zip(bars1, baseline_latencies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, cache_latencies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    # Speedup
    bars = ax2.bar(models, speedups, color=COLORS['secondary'], edgecolor='white', linewidth=0.5)
    ax2.axhline(y=1.0, color=COLORS['gray'], linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_ylabel('Speedup (x)')
    ax2.set_title('(b) DeltaCache Speedup')
    ax2.set_ylim(0, max(speedups) * 1.3)

    for bar, val in zip(bars, speedups):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}x', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_summary_figure(results, save_path):
    """Create a comprehensive summary figure for the paper."""
    fig = plt.figure(figsize=(14, 10))

    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # 1. Correctness (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    models = []
    match_rates = []
    for model_name, key in [("TinyLlama", "tinyllama"), ("Qwen2", "qwen2")]:
        if key in results and "correctness" in results[key]:
            models.append(model_name)
            match_rates.append(results[key]["correctness"]["tokens_match_rate"] * 100)

    bars = ax1.bar(models, match_rates, color=COLORS['secondary'], edgecolor='white')
    ax1.set_ylabel('Token Match Rate (%)')
    ax1.set_title('(a) Correctness Validation')
    ax1.set_ylim(0, 110)
    for bar, val in zip(bars, match_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 2. Speedup (top-middle)
    ax2 = fig.add_subplot(gs[0, 1])
    models = []
    speedups = []
    for model_name, key in [("TinyLlama", "tinyllama"), ("Qwen2", "qwen2")]:
        if key in results and "baseline_comparison" in results[key]:
            bc = results[key]["baseline_comparison"]
            models.append(model_name)
            speedups.append(bc["without_cache"]["avg_latency_ms"] / bc["with_cache"]["avg_latency_ms"])

    bars = ax2.bar(models, speedups, color=COLORS['primary'], edgecolor='white')
    ax2.axhline(y=1.0, color=COLORS['gray'], linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_ylabel('Speedup (x)')
    ax2.set_title('(b) Performance Improvement')
    ax2.set_ylim(0, max(speedups) * 1.3 if speedups else 2)
    for bar, val in zip(bars, speedups):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 3. Token Reuse (top-right)
    ax3 = fig.add_subplot(gs[0, 2])
    if 'ablation' in results:
        data = results['ablation']['prompt_lengths']
        lengths = [d['prompt_length'] for d in data]
        reuse_rates = [d['token_reuse_rate'] * 100 for d in data]
        ax3.plot(lengths, reuse_rates, 'o-', color=COLORS['quaternary'],
                 linewidth=2, markersize=8, markerfacecolor='white', markeredgewidth=2)
        ax3.fill_between(lengths, reuse_rates, alpha=0.15, color=COLORS['quaternary'])
        ax3.set_xlabel('Prompt Length')
        ax3.set_ylabel('Token Reuse Rate (%)')
        ax3.set_title('(c) Token Reuse Scaling')
        ax3.set_ylim(70, 85)

    # 4. Eviction Policies (bottom-left, spanning 2 columns)
    ax4 = fig.add_subplot(gs[1, 0:2])
    if 'ablation' in results:
        data = results['ablation']['eviction_policies']
        policies = [d['policy'].upper() for d in data]
        throughputs = [d['throughput_tok_s'] for d in data]
        latencies = [d['avg_latency_ms'] for d in data]

        x = np.arange(len(policies))
        width = 0.35

        bars1 = ax4.bar(x - width/2, throughputs, width, label='Throughput (tok/s)',
                        color=COLORS['primary'], edgecolor='white')

        ax4_twin = ax4.twinx()
        bars2 = ax4_twin.bar(x + width/2, latencies, width, label='Latency (ms)',
                             color=COLORS['tertiary'], edgecolor='white')

        ax4.set_xticks(x)
        ax4.set_xticklabels(policies)
        ax4.set_ylabel('Throughput (tokens/s)', color=COLORS['primary'])
        ax4_twin.set_ylabel('Latency (ms)', color=COLORS['tertiary'])
        ax4.set_title('(d) Eviction Policy Comparison')

        # Combined legend
        ax4.legend([bars1, bars2], ['Throughput', 'Latency'], loc='upper right')

    # 5. Query Count Scaling (bottom-right)
    ax5 = fig.add_subplot(gs[1, 2])
    if 'ablation' in results:
        data = results['ablation']['query_counts']
        queries = [d['num_queries'] for d in data]
        hit_rates = [d['cache_hit_rate'] * 100 for d in data]

        ax5.plot(queries, hit_rates, 's-', color=COLORS['accent'],
                 linewidth=2, markersize=8, markerfacecolor='white', markeredgewidth=2)
        ax5.fill_between(queries, hit_rates, alpha=0.15, color=COLORS['accent'])
        ax5.set_xlabel('Number of Queries')
        ax5.set_ylabel('Cache Hit Rate (%)')
        ax5.set_title('(e) Cache Efficiency Scaling')
        ax5.set_ylim(50, 80)

    fig.suptitle('DeltaCache: Incremental KV Cache Computation for LLMs',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    """Generate all paper figures and tables."""
    print("="*60)
    print("DeltaCache Paper Figure Generation")
    print("="*60)

    # Load results
    results = load_results()
    print(f"\nLoaded results for: {list(results.keys())}")

    # Create output directory
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Generate LaTeX tables
    generate_correctness_table(results)
    generate_performance_table(results)

    # Generate figures
    print("\nGenerating figures...")

    plot_eviction_policies(results, FIGURES_DIR / "fig_eviction_policies.pdf")
    plot_eviction_policies(results, FIGURES_DIR / "fig_eviction_policies.png")

    plot_prompt_length_scaling(results, FIGURES_DIR / "fig_prompt_scaling.pdf")
    plot_prompt_length_scaling(results, FIGURES_DIR / "fig_prompt_scaling.png")

    plot_query_count_scaling(results, FIGURES_DIR / "fig_query_scaling.pdf")
    plot_query_count_scaling(results, FIGURES_DIR / "fig_query_scaling.png")

    plot_baseline_comparison(results, FIGURES_DIR / "fig_baseline_comparison.pdf")
    plot_baseline_comparison(results, FIGURES_DIR / "fig_baseline_comparison.png")

    plot_summary_figure(results, FIGURES_DIR / "fig_summary.pdf")
    plot_summary_figure(results, FIGURES_DIR / "fig_summary.png")

    print(f"\nAll figures saved to {FIGURES_DIR}")
    print("\nGenerated files:")
    for f in sorted(FIGURES_DIR.glob("*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
