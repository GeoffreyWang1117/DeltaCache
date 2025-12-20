"""CLI commands for DeltaCache."""

from deltacache.cli.commands.benchmark import benchmark
from deltacache.cli.commands.analyze import analyze
from deltacache.cli.commands.stats import stats

__all__ = ["benchmark", "analyze", "stats"]
