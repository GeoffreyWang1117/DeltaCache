"""Analyze command for DeltaCache CLI."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import click
import numpy as np


@click.command()
@click.option(
    "--dataset",
    "-d",
    type=click.Path(exists=True),
    required=True,
    help="Dataset file (JSON with conversations/prompts)",
)
@click.option("--tokenizer", "-t", default="gpt2", help="Tokenizer name or path")
@click.option("--max-samples", type=int, default=1000, help="Maximum samples to analyze")
@click.option("--output", "-o", type=click.Path(), help="Output JSON file for results")
@click.option(
    "--system-prompt", type=str, help="Simulate adding this system prompt to all sequences"
)
@click.pass_context
def analyze(ctx, dataset, tokenizer, max_samples, output, system_prompt):
    """Analyze prefix sharing patterns in a dataset.

    \b
    Analyzes:
    - Sequence length distribution
    - Natural prefix sharing between sequences
    - Potential cache reuse with system prompts
    - Common prefix patterns

    \b
    Examples:
      deltacache analyze --dataset conversations.json --tokenizer gpt2
      deltacache analyze -d data.json -t gpt2 --system-prompt "You are helpful."
      deltacache analyze -d data.json -o analysis.json
    """
    verbose = ctx.obj.get("verbose", True)

    click.echo("DeltaCache Prefix Analysis")
    click.echo(f"Dataset: {dataset}")
    click.echo(f"Tokenizer: {tokenizer}")
    click.echo()

    # Load tokenizer
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise click.ClickException(
            "transformers library required. Install with: pip install transformers"
        ) from exc

    click.echo("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(tokenizer)

    # Load dataset
    click.echo("Loading dataset...")
    dataset_path = Path(dataset)
    with open(dataset_path) as f:
        data = json.load(f)

    # Extract text sequences
    sequences = extract_sequences(data, max_samples)
    click.echo(f"Extracted {len(sequences)} sequences")

    # Tokenize
    click.echo("Tokenizing...")
    tokenized = []
    for seq in sequences[:max_samples]:
        tokens = tok.encode(seq, add_special_tokens=True)
        tokenized.append(tokens)

    # Analyze
    results = analyze_prefix_patterns(tokenized, verbose)

    # Add system prompt simulation if requested
    if system_prompt:
        click.echo(f"\nSimulating system prompt: {system_prompt[:50]}...")
        sys_tokens = tok.encode(system_prompt, add_special_tokens=False)
        sys_results = simulate_system_prompt(tokenized, sys_tokens)
        results["system_prompt_simulation"] = sys_results

    # Print results
    print_results(results)

    # Save if requested
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        click.echo(f"\nResults saved to {output_path}")


def extract_sequences(data: any, max_samples: int) -> List[str]:
    """Extract text sequences from various dataset formats."""
    sequences = []

    if isinstance(data, list):
        for item in data[:max_samples]:
            if isinstance(item, str):
                sequences.append(item)
            elif isinstance(item, dict):
                # Try common keys
                if "messages" in item:
                    text = ""
                    for msg in item["messages"]:
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                        text += f"<|{role}|>\n{content}\n"
                    sequences.append(text)
                elif "text" in item:
                    sequences.append(item["text"])
                elif "prompt" in item:
                    sequences.append(item["prompt"])
                elif "content" in item:
                    sequences.append(item["content"])

    return sequences


def analyze_prefix_patterns(tokenized: List[List[int]], verbose: bool) -> Dict:
    """Analyze prefix sharing patterns."""
    results = {
        "dataset_stats": {},
        "prefix_sharing": {},
        "common_prefixes": [],
    }

    if not tokenized:
        return results

    # Basic stats
    lengths = [len(seq) for seq in tokenized]
    results["dataset_stats"] = {
        "total_sequences": len(tokenized),
        "avg_length": float(np.mean(lengths)),
        "min_length": int(min(lengths)),
        "max_length": int(max(lengths)),
        "median_length": float(np.median(lengths)),
    }

    if verbose:
        click.echo("\nDataset Statistics:")
        click.echo(f"  Sequences: {len(tokenized)}")
        click.echo(f"  Avg length: {np.mean(lengths):.0f} tokens")
        click.echo(f"  Min/Max: {min(lengths)}/{max(lengths)} tokens")

    # Prefix sharing analysis (sample pairs)
    if verbose:
        click.echo("\nAnalyzing prefix sharing...")

    n_samples = min(len(tokenized), 200)
    sampled = tokenized[:n_samples]
    shared_lengths = []

    for i in range(len(sampled)):
        for j in range(i + 1, len(sampled)):
            shared = compute_shared_prefix(sampled[i], sampled[j])
            if shared > 0:
                shared_lengths.append(shared)

    if shared_lengths:
        results["prefix_sharing"] = {
            "pairs_analyzed": n_samples * (n_samples - 1) // 2,
            "pairs_with_sharing": len(shared_lengths),
            "sharing_rate": len(shared_lengths) / (n_samples * (n_samples - 1) // 2),
            "avg_shared_length": float(np.mean(shared_lengths)),
            "max_shared_length": int(max(shared_lengths)),
            "median_shared_length": float(np.median(shared_lengths)),
        }

        if verbose:
            click.echo(f"  Pairs with shared prefix: {len(shared_lengths)}")
            click.echo(f"  Avg shared length: {np.mean(shared_lengths):.1f} tokens")

    # Common prefix analysis
    prefix_counts = defaultdict(int)
    for seq in tokenized:
        for prefix_len in [5, 10, 20, 50]:
            if len(seq) >= prefix_len:
                prefix = tuple(seq[:prefix_len])
                prefix_counts[(prefix_len, prefix)] += 1

    # Get most common prefixes
    sorted_prefixes = sorted(prefix_counts.items(), key=lambda x: x[1], reverse=True)
    common = []
    for (length, _prefix), count in sorted_prefixes[:10]:
        if count > 1:
            common.append(
                {
                    "length": length,
                    "count": count,
                    "share": count / len(tokenized),
                }
            )

    results["common_prefixes"] = common

    return results


def compute_shared_prefix(seq1: List[int], seq2: List[int]) -> int:
    """Compute length of shared prefix."""
    shared = 0
    for t1, t2 in zip(seq1, seq2):
        if t1 == t2:
            shared += 1
        else:
            break
    return shared


def simulate_system_prompt(tokenized: List[List[int]], sys_tokens: List[int]) -> Dict:
    """Simulate prefix sharing with a common system prompt."""
    total_tokens = 0
    reusable_tokens = 0

    for i, seq in enumerate(tokenized):
        full_len = len(sys_tokens) + len(seq)
        total_tokens += full_len

        # First request computes everything
        # Subsequent requests reuse system prompt
        if i > 0:
            reusable_tokens += len(sys_tokens)

    return {
        "system_prompt_length": len(sys_tokens),
        "sequences": len(tokenized),
        "total_tokens": total_tokens,
        "reusable_tokens": reusable_tokens,
        "reuse_rate": reusable_tokens / total_tokens if total_tokens > 0 else 0,
    }


def print_results(results: Dict) -> None:
    """Print formatted results."""
    click.echo("\n" + "=" * 60)
    click.echo("ANALYSIS RESULTS")
    click.echo("=" * 60)

    if "dataset_stats" in results:
        stats = results["dataset_stats"]
        click.echo("\nDataset Statistics:")
        click.echo(f"  Total sequences: {stats.get('total_sequences', 0)}")
        click.echo(f"  Average length: {stats.get('avg_length', 0):.0f} tokens")
        click.echo(f"  Range: {stats.get('min_length', 0)} - {stats.get('max_length', 0)} tokens")

    if results.get("prefix_sharing"):
        ps = results["prefix_sharing"]
        click.echo("\nNatural Prefix Sharing:")
        click.echo(f"  Pairs analyzed: {ps.get('pairs_analyzed', 0)}")
        click.echo(f"  Sharing rate: {ps.get('sharing_rate', 0) * 100:.1f}%")
        click.echo(f"  Avg shared length: {ps.get('avg_shared_length', 0):.1f} tokens")

    if results.get("common_prefixes"):
        click.echo("\nCommon Prefixes:")
        for i, p in enumerate(results["common_prefixes"][:5]):
            click.echo(
                f"  {i + 1}. Length {p['length']}: {p['count']} occurrences ({p['share'] * 100:.1f}%)"
            )

    if "system_prompt_simulation" in results:
        sim = results["system_prompt_simulation"]
        click.echo("\nSystem Prompt Simulation:")
        click.echo(f"  Prompt length: {sim.get('system_prompt_length', 0)} tokens")
        click.echo(f"  Reusable tokens: {sim.get('reusable_tokens', 0):,}")
        click.echo(f"  Potential reuse rate: {sim.get('reuse_rate', 0) * 100:.1f}%")
