"""Main CLI entry point for DeltaCache."""

import click

from deltacache import __version__


@click.group()
@click.version_option(version=__version__, prog_name="deltacache")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx, verbose):
    """DeltaCache - Incremental computation-aware KV cache management for LLMs.

    DeltaCache reduces redundant computation in LLM inference by intelligently
    reusing cached key-value states across requests with shared prefixes.

    \b
    Examples:
      deltacache benchmark --model gpt2 --scenario system_prompt_short
      deltacache analyze --dataset data.json --tokenizer gpt2
      deltacache stats --results-dir ./results
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


# Import and register commands
from deltacache.cli.commands.benchmark import benchmark
from deltacache.cli.commands.analyze import analyze
from deltacache.cli.commands.stats import stats

cli.add_command(benchmark)
cli.add_command(analyze)
cli.add_command(stats)


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
