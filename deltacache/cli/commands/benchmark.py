"""Benchmark command for DeltaCache CLI."""

from pathlib import Path

import click


@click.command()
@click.option(
    "--model", "-m", default="gpt2", help="Model name (gpt2, gpt2-medium, gpt2-large, gpt2-xl)"
)
@click.option(
    "--device", "-d", default="cpu", type=click.Choice(["cpu", "cuda"]), help="Device to run on"
)
@click.option(
    "--scenario", "-s", default="all", help="Scenario to run (or 'all' for all scenarios)"
)
@click.option("--output-json", "-o", type=click.Path(), help="Save results to JSON file")
@click.option("--list-scenarios", is_flag=True, help="List available scenarios and exit")
@click.pass_context
def benchmark(ctx, model, device, scenario, output_json, list_scenarios):
    """Run performance benchmarks comparing baseline vs DeltaCache.

    \b
    Compares three approaches:
    1. Baseline: Full computation for each request (no caching)
    2. HF Cache: HuggingFace's native past_key_values
    3. DeltaCache: Prefix-aware incremental computation

    \b
    Examples:
      deltacache benchmark --model gpt2 --scenario system_prompt_short
      deltacache benchmark --scenario all --output-json results.json
      deltacache benchmark --list-scenarios
    """
    try:
        from benchmarks.hf_benchmark import HFBenchmark
        from benchmarks.scenarios import SCENARIOS
    except ImportError:
        # Try relative import
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
        from benchmarks.hf_benchmark import HFBenchmark
        from benchmarks.scenarios import SCENARIOS

    verbose = ctx.obj.get("verbose", True)

    if list_scenarios:
        click.echo("Available scenarios:")
        for name, scen in SCENARIOS.items():
            click.echo(f"  {name}: {scen.description}")
        return

    click.echo("DeltaCache Benchmark")
    click.echo(f"Model: {model}")
    click.echo(f"Device: {device}")
    click.echo(f"Scenario: {scenario}")
    click.echo()

    bench = HFBenchmark(
        model_name=model,
        device=device,
        verbose=verbose,
    )

    if scenario == "all":
        results = bench.run_all_scenarios()
    else:
        if scenario not in SCENARIOS:
            available = ", ".join(SCENARIOS.keys())
            raise click.ClickException(f"Unknown scenario: {scenario}. Available: {available}")
        results = {scenario: bench.run_scenario(scenario)}

    if output_json:
        output_path = Path(output_json)
        bench.save_results(results, output_path)
        click.echo(f"\nResults saved to {output_path}")

    # Print summary
    click.echo("\n" + "=" * 80)
    click.echo("BENCHMARK SUMMARY")
    click.echo("=" * 80)
    click.echo(
        f"{'Scenario':<30} {'Baseline':<12} {'DeltaCache':<12} {'Speedup':<10} {'Reuse':<10}"
    )
    click.echo("-" * 80)
    for name, result in results.items():
        click.echo(
            f"{name:<30} "
            f"{result.baseline.total_time_ms:>8.0f}ms  "
            f"{result.deltacache.total_time_ms:>8.0f}ms  "
            f"{result.speedup_vs_baseline:>7.2f}x  "
            f"{result.token_savings:>7.1%}"
        )
