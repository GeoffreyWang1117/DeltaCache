"""Stats command for DeltaCache CLI."""

import json
import click
from pathlib import Path
from typing import Dict, List, Any


@click.command()
@click.option(
    "--results-dir", "-r",
    type=click.Path(exists=True),
    help="Directory containing experiment results"
)
@click.option(
    "--results-file", "-f",
    type=click.Path(exists=True),
    help="Specific results JSON file"
)
@click.option(
    "--format", "fmt",
    type=click.Choice(["table", "json", "csv"]),
    default="table",
    help="Output format"
)
@click.pass_context
def stats(ctx, results_dir, results_file, fmt):
    """Display cache statistics and experiment results.

    \b
    Shows:
    - Experiment results summary
    - Cache hit rates
    - Token reuse rates
    - Performance comparisons

    \b
    Examples:
      deltacache stats --results-dir ./experiments/results
      deltacache stats --results-file results.json --format json
      deltacache stats -r ./results -f table
    """
    verbose = ctx.obj.get("verbose", True)

    if not results_dir and not results_file:
        raise click.ClickException("Please specify --results-dir or --results-file")

    results = {}

    # Load results from file or directory
    if results_file:
        results_path = Path(results_file)
        with open(results_path) as f:
            results = json.load(f)
        click.echo(f"Loaded results from: {results_file}")

    elif results_dir:
        results_path = Path(results_dir)
        # Look for JSON files
        json_files = list(results_path.glob("*.json"))
        if not json_files:
            raise click.ClickException(f"No JSON files found in {results_dir}")

        for jf in json_files:
            with open(jf) as f:
                data = json.load(f)
            results[jf.stem] = data
            click.echo(f"Loaded: {jf.name}")

    # Display based on format
    if fmt == "json":
        click.echo(json.dumps(results, indent=2))

    elif fmt == "csv":
        print_csv(results)

    else:  # table
        print_table(results)


def print_table(results: Dict) -> None:
    """Print results as formatted table."""
    click.echo("\n" + "="*80)
    click.echo("DELTACACHE STATISTICS")
    click.echo("="*80)

    for name, data in results.items():
        click.echo(f"\n{name.upper()}")
        click.echo("-"*60)

        if isinstance(data, dict):
            # Check if it's experiment results format
            if "system_prompt" in data or "rag" in data or "few_shot" in data:
                print_experiment_results(data)
            elif "baseline" in data and "deltacache" in data:
                print_benchmark_result(name, data)
            else:
                # Generic dict
                for key, value in data.items():
                    if isinstance(value, float):
                        click.echo(f"  {key}: {value:.4f}")
                    else:
                        click.echo(f"  {key}: {value}")

        elif isinstance(data, list):
            for i, item in enumerate(data[:10]):  # Limit output
                click.echo(f"  [{i}] {item}")


def print_experiment_results(data: Dict) -> None:
    """Print cache experiment results."""
    # System prompt results
    if "system_prompt" in data:
        click.echo("\nSystem Prompt Experiments:")
        click.echo(f"  {'Config':<20} {'Hit Rate':<12} {'Token Reuse':<12} {'Throughput':<15}")
        click.echo("  " + "-"*55)
        for exp in data["system_prompt"]:
            name = exp.get("name", "unknown")
            hit_rate = exp.get("cache_hit_rate", 0) * 100
            reuse = exp.get("token_reuse_rate", 0) * 100
            throughput = exp.get("throughput_tokens_per_sec", 0)
            click.echo(f"  {name:<20} {hit_rate:>8.1f}%   {reuse:>8.1f}%   {throughput:>10.0f} tok/s")

    # RAG results
    if "rag" in data:
        click.echo("\nRAG Experiments:")
        click.echo(f"  {'Config':<20} {'Hit Rate':<12} {'Token Reuse':<12} {'Throughput':<15}")
        click.echo("  " + "-"*55)
        for exp in data["rag"]:
            name = exp.get("name", "unknown")
            hit_rate = exp.get("cache_hit_rate", 0) * 100
            reuse = exp.get("token_reuse_rate", 0) * 100
            throughput = exp.get("throughput_tokens_per_sec", 0)
            click.echo(f"  {name:<20} {hit_rate:>8.1f}%   {reuse:>8.1f}%   {throughput:>10.0f} tok/s")

    # Few-shot results
    if "few_shot" in data:
        click.echo("\nFew-Shot Experiments:")
        click.echo(f"  {'Config':<20} {'Hit Rate':<12} {'Token Reuse':<12} {'Throughput':<15}")
        click.echo("  " + "-"*55)
        for exp in data["few_shot"]:
            name = exp.get("name", "unknown")
            hit_rate = exp.get("cache_hit_rate", 0) * 100
            reuse = exp.get("token_reuse_rate", 0) * 100
            throughput = exp.get("throughput_tokens_per_sec", 0)
            click.echo(f"  {name:<20} {hit_rate:>8.1f}%   {reuse:>8.1f}%   {throughput:>10.0f} tok/s")

    # Baseline comparison
    if "baseline_comparison" in data:
        bc = data["baseline_comparison"]
        click.echo("\nBaseline Comparison:")
        click.echo(f"  Baseline time: {bc.get('baseline_time_ms', 0):.0f}ms")
        click.echo(f"  DeltaCache time: {bc.get('deltacache_time_ms', 0):.0f}ms")
        click.echo(f"  Speedup: {bc.get('speedup', 0):.2f}x")
        click.echo(f"  Token savings: {bc.get('token_savings', 0)*100:.1f}%")


def print_benchmark_result(name: str, data: Dict) -> None:
    """Print benchmark comparison result."""
    baseline = data.get("baseline", {})
    deltacache = data.get("deltacache", {})

    click.echo(f"\n{name}:")
    click.echo(f"  {'Method':<15} {'Time':<12} {'Tokens':<12} {'Throughput':<15}")
    click.echo("  " + "-"*50)

    b_time = baseline.get("total_time_ms", 0)
    b_tokens = baseline.get("total_tokens", 0)
    b_throughput = baseline.get("throughput_tokens_per_sec", 0)
    click.echo(f"  {'Baseline':<15} {b_time:>8.0f}ms  {b_tokens:>8}     {b_throughput:>10.0f} tok/s")

    d_time = deltacache.get("total_time_ms", 0)
    d_tokens = deltacache.get("total_tokens", 0)
    d_throughput = deltacache.get("throughput_tokens_per_sec", 0)
    d_reuse = deltacache.get("token_reuse_rate", 0) * 100
    click.echo(f"  {'DeltaCache':<15} {d_time:>8.0f}ms  {d_tokens:>8}     {d_throughput:>10.0f} tok/s")

    click.echo(f"\n  Speedup: {data.get('speedup_vs_baseline', 0):.2f}x")
    click.echo(f"  Token reuse: {d_reuse:.1f}%")


def print_csv(results: Dict) -> None:
    """Print results as CSV."""
    # Header
    click.echo("name,method,total_time_ms,total_tokens,throughput,hit_rate,reuse_rate")

    for exp_name, data in results.items():
        if isinstance(data, dict):
            for key in ["system_prompt", "rag", "few_shot"]:
                if key in data:
                    for exp in data[key]:
                        click.echo(
                            f"{exp.get('name', 'unknown')},"
                            f"deltacache,"
                            f"{exp.get('total_time_ms', 0):.2f},"
                            f"{exp.get('total_tokens', 0)},"
                            f"{exp.get('throughput_tokens_per_sec', 0):.2f},"
                            f"{exp.get('cache_hit_rate', 0):.4f},"
                            f"{exp.get('token_reuse_rate', 0):.4f}"
                        )
