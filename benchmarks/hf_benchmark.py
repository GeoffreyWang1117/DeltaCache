"""HuggingFace model benchmarks comparing baseline vs DeltaCache."""

import time
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
from pathlib import Path

import torch
from torch import Tensor
from tqdm import tqdm

from deltacache import DeltaCacheManager, DeltaCacheConfig
from deltacache.hf_integration import GPT2Adapter, hf_to_deltacache, deltacache_to_hf
from benchmarks.scenarios import BenchmarkScenario, SCENARIOS, get_scenario


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    scenario_name: str
    method: str  # "baseline", "hf_cache", "deltacache"
    num_requests: int
    total_tokens: int
    total_time_ms: float
    avg_latency_ms: float
    throughput_tokens_per_sec: float
    cached_tokens: int = 0
    computed_tokens: int = 0
    cache_hit_rate: float = 0.0
    token_reuse_rate: float = 0.0


@dataclass
class ComparisonResult:
    """Comparison results across all methods."""

    scenario_name: str
    baseline: BenchmarkResult
    hf_cache: BenchmarkResult
    deltacache: BenchmarkResult
    speedup_vs_baseline: float
    speedup_vs_hf_cache: float
    token_savings: float


class HFBenchmark:
    """Benchmark runner for HuggingFace models with DeltaCache.

    Compares three approaches:
    1. Baseline: Full computation for each request (no caching)
    2. HF Cache: HuggingFace's native past_key_values (single request)
    3. DeltaCache: Prefix-aware incremental computation

    Example:
        >>> benchmark = HFBenchmark(model_name="gpt2", device="cpu")
        >>> result = benchmark.run_scenario("system_prompt_short")
        >>> print(f"DeltaCache speedup: {result.speedup_vs_baseline:.2f}x")
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "cpu",
        dtype: Optional[torch.dtype] = None,
        verbose: bool = True,
    ):
        """Initialize benchmark.

        Args:
            model_name: HuggingFace model name
            device: Device to run on ("cpu" or "cuda")
            dtype: Model dtype (None for default)
            verbose: Whether to print progress
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype or torch.float32
        self.verbose = verbose

        self._adapter: Optional[GPT2Adapter] = None
        self._delta_manager: Optional[DeltaCacheManager] = None

    def _log(self, msg: str) -> None:
        """Print message if verbose."""
        if self.verbose:
            print(msg)

    def _get_adapter(self) -> GPT2Adapter:
        """Get or create model adapter."""
        if self._adapter is None:
            self._log(f"Loading model: {self.model_name}")
            self._adapter = GPT2Adapter.from_pretrained(
                self.model_name,
                device=self.device,
                dtype=self.dtype,
            )
        return self._adapter

    def _get_delta_manager(self) -> DeltaCacheManager:
        """Get or create DeltaCache manager."""
        if self._delta_manager is None:
            adapter = self._get_adapter()
            self._delta_manager = DeltaCacheManager(adapter.config)
        return self._delta_manager

    def _reset_delta_manager(self) -> None:
        """Reset DeltaCache manager for new benchmark."""
        self._delta_manager = None

    def _tokenize_prompts(
        self,
        scenario: BenchmarkScenario,
    ) -> List[List[int]]:
        """Tokenize all prompts in a scenario.

        Args:
            scenario: Benchmark scenario

        Returns:
            List of token ID lists
        """
        adapter = self._get_adapter()
        tokenized = []

        for prompt in scenario.prompts:
            # Add system prompt if present
            if scenario.system_prompt:
                full_prompt = f"{scenario.system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt

            tokens = adapter.tokenize(full_prompt, add_special_tokens=True)
            tokenized.append(tokens[0].tolist())

        return tokenized

    def run_baseline(
        self,
        token_sequences: List[List[int]],
        scenario_name: str,
    ) -> BenchmarkResult:
        """Run baseline benchmark (no caching, full computation each time).

        Args:
            token_sequences: List of token sequences to process
            scenario_name: Name for result labeling

        Returns:
            Benchmark result
        """
        adapter = self._get_adapter()
        total_time = 0.0
        total_tokens = 0
        latencies = []

        desc = "Baseline (no cache)"
        iterator = tqdm(token_sequences, desc=desc, disable=not self.verbose)

        for tokens in iterator:
            input_ids = torch.tensor([tokens], device=self.device)
            position_ids = adapter.get_position_ids(len(tokens))

            start = time.perf_counter()
            # Full computation - no past_key_values
            with torch.no_grad():
                outputs = adapter.model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    use_cache=True,
                    return_dict=True,
                )
            elapsed = (time.perf_counter() - start) * 1000

            total_time += elapsed
            total_tokens += len(tokens)
            latencies.append(elapsed)

        return BenchmarkResult(
            scenario_name=scenario_name,
            method="baseline",
            num_requests=len(token_sequences),
            total_tokens=total_tokens,
            total_time_ms=total_time,
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
            throughput_tokens_per_sec=total_tokens / (total_time / 1000) if total_time > 0 else 0,
            cached_tokens=0,
            computed_tokens=total_tokens,
            cache_hit_rate=0.0,
            token_reuse_rate=0.0,
        )

    def run_hf_cache(
        self,
        token_sequences: List[List[int]],
        scenario_name: str,
    ) -> BenchmarkResult:
        """Run HuggingFace native cache benchmark.

        This uses past_key_values but doesn't share across requests.
        Each request still needs full initial computation.

        Args:
            token_sequences: List of token sequences to process
            scenario_name: Name for result labeling

        Returns:
            Benchmark result
        """
        adapter = self._get_adapter()
        total_time = 0.0
        total_tokens = 0
        latencies = []

        desc = "HF Cache (per-request)"
        iterator = tqdm(token_sequences, desc=desc, disable=not self.verbose)

        for tokens in iterator:
            input_ids = torch.tensor([tokens], device=self.device)
            position_ids = adapter.get_position_ids(len(tokens))

            start = time.perf_counter()
            with torch.no_grad():
                # First token with no cache
                outputs = adapter.model(
                    input_ids=input_ids[:, :1],
                    position_ids=position_ids[:, :1],
                    use_cache=True,
                    return_dict=True,
                )
                past_kv = outputs.past_key_values

                # Remaining tokens with cache
                if len(tokens) > 1:
                    outputs = adapter.model(
                        input_ids=input_ids[:, 1:],
                        position_ids=position_ids[:, 1:],
                        past_key_values=past_kv,
                        use_cache=True,
                        return_dict=True,
                    )

            elapsed = (time.perf_counter() - start) * 1000

            total_time += elapsed
            total_tokens += len(tokens)
            latencies.append(elapsed)

        return BenchmarkResult(
            scenario_name=scenario_name,
            method="hf_cache",
            num_requests=len(token_sequences),
            total_tokens=total_tokens,
            total_time_ms=total_time,
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
            throughput_tokens_per_sec=total_tokens / (total_time / 1000) if total_time > 0 else 0,
            cached_tokens=0,
            computed_tokens=total_tokens,
            cache_hit_rate=0.0,
            token_reuse_rate=0.0,
        )

    def run_deltacache(
        self,
        token_sequences: List[List[int]],
        scenario_name: str,
    ) -> BenchmarkResult:
        """Run DeltaCache benchmark with prefix sharing.

        Args:
            token_sequences: List of token sequences to process
            scenario_name: Name for result labeling

        Returns:
            Benchmark result
        """
        self._reset_delta_manager()
        adapter = self._get_adapter()
        manager = self._get_delta_manager()

        total_time = 0.0
        latencies = []

        desc = "DeltaCache (prefix sharing)"
        iterator = tqdm(token_sequences, desc=desc, disable=not self.verbose)

        for tokens in iterator:
            start = time.perf_counter()
            result = manager.compute_incremental(tokens, adapter)
            elapsed = (time.perf_counter() - start) * 1000

            total_time += elapsed
            latencies.append(elapsed)

        stats = manager.get_stats()
        total_tokens = stats["total_tokens"]

        return BenchmarkResult(
            scenario_name=scenario_name,
            method="deltacache",
            num_requests=len(token_sequences),
            total_tokens=total_tokens,
            total_time_ms=total_time,
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
            throughput_tokens_per_sec=total_tokens / (total_time / 1000) if total_time > 0 else 0,
            cached_tokens=stats["cached_tokens"],
            computed_tokens=stats["computed_tokens"],
            cache_hit_rate=stats["hit_rate"],
            token_reuse_rate=stats["token_reuse_rate"],
        )

    def run_scenario(
        self,
        scenario_name: str,
    ) -> ComparisonResult:
        """Run full comparison for a scenario.

        Args:
            scenario_name: Name of scenario to run

        Returns:
            Comparison result with all methods
        """
        scenario = get_scenario(scenario_name)
        self._log(f"\n{'='*60}")
        self._log(f"Scenario: {scenario.name}")
        self._log(f"Description: {scenario.description}")
        self._log(f"{'='*60}")

        # Tokenize prompts
        token_sequences = self._tokenize_prompts(scenario)
        self._log(f"Prompts: {len(token_sequences)}")
        self._log(f"Total tokens: {sum(len(t) for t in token_sequences)}")

        # Run all methods
        baseline_result = self.run_baseline(token_sequences, scenario_name)
        hf_cache_result = self.run_hf_cache(token_sequences, scenario_name)
        deltacache_result = self.run_deltacache(token_sequences, scenario_name)

        # Calculate comparisons
        speedup_vs_baseline = baseline_result.total_time_ms / deltacache_result.total_time_ms if deltacache_result.total_time_ms > 0 else 0
        speedup_vs_hf = hf_cache_result.total_time_ms / deltacache_result.total_time_ms if deltacache_result.total_time_ms > 0 else 0
        token_savings = deltacache_result.token_reuse_rate

        result = ComparisonResult(
            scenario_name=scenario_name,
            baseline=baseline_result,
            hf_cache=hf_cache_result,
            deltacache=deltacache_result,
            speedup_vs_baseline=speedup_vs_baseline,
            speedup_vs_hf_cache=speedup_vs_hf,
            token_savings=token_savings,
        )

        # Print summary
        self._log(f"\nResults:")
        self._log(f"  Baseline:    {baseline_result.total_time_ms:.1f}ms ({baseline_result.throughput_tokens_per_sec:.0f} tok/s)")
        self._log(f"  HF Cache:    {hf_cache_result.total_time_ms:.1f}ms ({hf_cache_result.throughput_tokens_per_sec:.0f} tok/s)")
        self._log(f"  DeltaCache:  {deltacache_result.total_time_ms:.1f}ms ({deltacache_result.throughput_tokens_per_sec:.0f} tok/s)")
        self._log(f"\nDeltaCache Performance:")
        self._log(f"  Speedup vs baseline: {speedup_vs_baseline:.2f}x")
        self._log(f"  Speedup vs HF cache: {speedup_vs_hf:.2f}x")
        self._log(f"  Cache hit rate: {deltacache_result.cache_hit_rate:.1%}")
        self._log(f"  Token reuse rate: {deltacache_result.token_reuse_rate:.1%}")

        return result

    def run_all_scenarios(self) -> Dict[str, ComparisonResult]:
        """Run all predefined scenarios.

        Returns:
            Dictionary mapping scenario names to results
        """
        results = {}
        for name in SCENARIOS:
            results[name] = self.run_scenario(name)
        return results

    def save_results(
        self,
        results: Dict[str, ComparisonResult],
        output_path: Path,
    ) -> None:
        """Save benchmark results to JSON file.

        Args:
            results: Dictionary of comparison results
            output_path: Path to output JSON file
        """
        output = {}
        for name, result in results.items():
            output[name] = {
                "baseline": asdict(result.baseline),
                "hf_cache": asdict(result.hf_cache),
                "deltacache": asdict(result.deltacache),
                "speedup_vs_baseline": result.speedup_vs_baseline,
                "speedup_vs_hf_cache": result.speedup_vs_hf_cache,
                "token_savings": result.token_savings,
            }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        self._log(f"\nResults saved to {output_path}")


def main():
    """Run benchmark from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="DeltaCache HuggingFace Benchmark")
    parser.add_argument("--model", default="gpt2", help="Model name")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device")
    parser.add_argument("--scenario", default="all", help="Scenario to run (or 'all')")
    parser.add_argument("--output", type=Path, help="Output JSON file")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    args = parser.parse_args()

    benchmark = HFBenchmark(
        model_name=args.model,
        device=args.device,
        verbose=not args.quiet,
    )

    if args.scenario == "all":
        results = benchmark.run_all_scenarios()
    else:
        results = {args.scenario: benchmark.run_scenario(args.scenario)}

    if args.output:
        benchmark.save_results(results, args.output)

    # Print summary table
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"{'Scenario':<30} {'Baseline':<12} {'DeltaCache':<12} {'Speedup':<10} {'Reuse':<10}")
    print("-"*80)
    for name, result in results.items():
        print(f"{name:<30} {result.baseline.total_time_ms:>8.0f}ms  {result.deltacache.total_time_ms:>8.0f}ms  {result.speedup_vs_baseline:>7.2f}x  {result.token_savings:>7.1%}")


if __name__ == "__main__":
    main()
