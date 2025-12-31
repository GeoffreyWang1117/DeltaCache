"""Real model benchmarks for DeltaCache paper experiments.

This module runs actual experiments with real LLM models (not mocks),
measuring true latency, memory usage, and correctness.

Supported models (for 2x RTX 3090):
- TinyLlama-1.1B: Fast iteration, correctness testing
- Qwen2-1.5B: Small model validation
- Mistral-7B: Main benchmark model
- Llama-2-7B: Important baseline
"""

import os
import gc
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple, Any

import torch
import numpy as np
from tqdm import tqdm

# Set up paths
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from deltacache import DeltaCacheManager, DeltaCacheConfig
from deltacache.hf_integration import LlamaStyleAdapter


RESULTS_DIR = Path(__file__).parent / "results" / "paper"


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    model_name: str
    device: str = "cuda"
    dtype: str = "float16"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    gpu_memory_limit_gb: float = 8.0
    eviction_policy: str = "tiered"


@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""
    experiment_name: str
    model_name: str
    num_requests: int
    total_tokens: int
    cached_tokens: int
    computed_tokens: int
    cache_hit_rate: float
    token_reuse_rate: float
    total_time_ms: float
    avg_latency_ms: float
    ttft_ms: float  # Time to first token (prompt processing)
    throughput_tokens_per_sec: float
    memory_used_mb: float
    extra: Dict = field(default_factory=dict)


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def clear_gpu_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class RealModelBenchmark:
    """Benchmark runner with real LLM models."""

    def __init__(
        self,
        config: ExperimentConfig,
        verbose: bool = True,
    ):
        self.config = config
        self.verbose = verbose
        self._adapter: Optional[LlamaStyleAdapter] = None
        self._manager: Optional[DeltaCacheManager] = None

    def log(self, msg: str):
        if self.verbose:
            print(msg)

    def load_model(self):
        """Load model and create DeltaCache manager."""
        self.log(f"Loading model: {self.config.model_name}")

        dtype = torch.float16 if self.config.dtype == "float16" else torch.float32

        self._adapter = LlamaStyleAdapter.from_pretrained(
            self.config.model_name,
            device=self.config.device,
            dtype=dtype,
            load_in_8bit=self.config.load_in_8bit,
            load_in_4bit=self.config.load_in_4bit,
        )

        # Create DeltaCache manager
        cache_config = self._adapter.config
        cache_config.gpu_memory_limit = int(self.config.gpu_memory_limit_gb * 1024 * 1024 * 1024)
        cache_config.eviction_policy = self.config.eviction_policy

        self._manager = DeltaCacheManager(cache_config)

        self.log(f"Model loaded. Config: {cache_config.num_layers} layers, "
                 f"{cache_config.num_heads} heads, {cache_config.head_dim} head_dim")
        self.log(f"GPU memory after loading: {get_gpu_memory_mb():.0f} MB")

    def reset_cache(self):
        """Reset DeltaCache manager."""
        if self._manager:
            self._manager.clear()

    def run_correctness_test(
        self,
        test_prompts: List[str],
        tolerance: float = 1e-3,
    ) -> Dict:
        """Run correctness validation tests.

        Verifies that DeltaCache produces same outputs as non-cached.
        """
        self.log("\n" + "="*60)
        self.log("CORRECTNESS VALIDATION")
        self.log("="*60)

        results = []
        all_correct = True

        for i, prompt in enumerate(tqdm(test_prompts, desc="Testing correctness")):
            tokens = self._adapter.tokenize(prompt)[0].tolist()

            # Reset cache for fair comparison
            self.reset_cache()

            result = self._adapter.verify_cache_correctness(
                tokens,
                self._manager,
                tolerance=tolerance,
            )

            results.append(result)

            if not result["is_correct"]:
                all_correct = False
                self.log(f"  FAIL at prompt {i}: max_diff={result['kv_max_diff']:.6f}")

        # Aggregate results
        summary = {
            "num_tests": len(results),
            "all_correct": all_correct,
            "avg_kv_max_diff": np.mean([r["kv_max_diff"] for r in results]),
            "avg_kv_mean_diff": np.mean([r["kv_mean_diff"] for r in results]),
            "tokens_match_rate": np.mean([r["tokens_match"] for r in results]),
            "tolerance": tolerance,
        }

        self.log(f"\nCorrectness Summary:")
        self.log(f"  Tests: {summary['num_tests']}")
        self.log(f"  All Correct: {summary['all_correct']}")
        self.log(f"  Avg KV Max Diff: {summary['avg_kv_max_diff']:.6f}")
        self.log(f"  Token Match Rate: {summary['tokens_match_rate']:.1%}")

        return summary

    def run_system_prompt_experiment(
        self,
        system_prompt: str,
        user_queries: List[str],
    ) -> BenchmarkResult:
        """Run system prompt caching experiment.

        All queries share the same system prompt prefix.
        """
        self.log("\n" + "="*60)
        self.log("SYSTEM PROMPT CACHING EXPERIMENT")
        self.log("="*60)

        self.reset_cache()

        system_tokens = self._adapter.tokenize(system_prompt)[0].tolist()
        self.log(f"System prompt: {len(system_tokens)} tokens")
        self.log(f"User queries: {len(user_queries)}")

        total_time = 0.0
        latencies = []
        ttft_times = []
        memory_samples = []

        for query in tqdm(user_queries, desc="Processing queries"):
            query_tokens = self._adapter.tokenize(query)[0].tolist()
            full_tokens = system_tokens + query_tokens

            # Measure time
            start = time.perf_counter()
            result = self._manager.compute_incremental(full_tokens, self._adapter)
            elapsed = (time.perf_counter() - start) * 1000

            total_time += elapsed
            latencies.append(elapsed)
            ttft_times.append(elapsed)  # For prompt processing, TTFT = total time
            memory_samples.append(get_gpu_memory_mb())

        stats = self._manager.get_stats()

        result = BenchmarkResult(
            experiment_name="system_prompt",
            model_name=self.config.model_name,
            num_requests=len(user_queries),
            total_tokens=stats["total_tokens"],
            cached_tokens=stats["cached_tokens"],
            computed_tokens=stats["computed_tokens"],
            cache_hit_rate=stats["hit_rate"],
            token_reuse_rate=stats["token_reuse_rate"],
            total_time_ms=total_time,
            avg_latency_ms=np.mean(latencies),
            ttft_ms=np.mean(ttft_times),
            throughput_tokens_per_sec=stats["total_tokens"] / (total_time / 1000),
            memory_used_mb=np.mean(memory_samples),
            extra={
                "system_prompt_length": len(system_tokens),
                "latency_std": np.std(latencies),
                "latency_p50": np.percentile(latencies, 50),
                "latency_p95": np.percentile(latencies, 95),
                "latency_p99": np.percentile(latencies, 99),
            }
        )

        self.log(f"\nResults:")
        self.log(f"  Cache Hit Rate: {result.cache_hit_rate:.1%}")
        self.log(f"  Token Reuse Rate: {result.token_reuse_rate:.1%}")
        self.log(f"  Avg Latency: {result.avg_latency_ms:.1f}ms")
        self.log(f"  Throughput: {result.throughput_tokens_per_sec:.0f} tok/s")

        return result

    def run_baseline_comparison(
        self,
        prompts: List[str],
    ) -> Tuple[BenchmarkResult, BenchmarkResult]:
        """Compare DeltaCache vs baseline (no caching).

        Returns (baseline_result, deltacache_result).
        """
        self.log("\n" + "="*60)
        self.log("BASELINE COMPARISON")
        self.log("="*60)

        # Tokenize all prompts
        tokenized = [self._adapter.tokenize(p)[0].tolist() for p in prompts]

        # Run baseline (no cache - full compute each time)
        self.log("\nRunning baseline (no cache)...")
        baseline_time = 0.0
        baseline_latencies = []
        total_baseline_tokens = 0

        for tokens in tqdm(tokenized, desc="Baseline"):
            input_ids = torch.tensor([tokens], device=self.config.device)
            position_ids = self._adapter.get_position_ids(len(tokens))

            start = time.perf_counter()
            with torch.no_grad():
                _ = self._adapter.model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    use_cache=True,
                    return_dict=True,
                )
            elapsed = (time.perf_counter() - start) * 1000

            baseline_time += elapsed
            baseline_latencies.append(elapsed)
            total_baseline_tokens += len(tokens)

        baseline_result = BenchmarkResult(
            experiment_name="baseline",
            model_name=self.config.model_name,
            num_requests=len(prompts),
            total_tokens=total_baseline_tokens,
            cached_tokens=0,
            computed_tokens=total_baseline_tokens,
            cache_hit_rate=0.0,
            token_reuse_rate=0.0,
            total_time_ms=baseline_time,
            avg_latency_ms=np.mean(baseline_latencies),
            ttft_ms=np.mean(baseline_latencies),
            throughput_tokens_per_sec=total_baseline_tokens / (baseline_time / 1000),
            memory_used_mb=get_gpu_memory_mb(),
        )

        # Run with DeltaCache
        self.log("\nRunning with DeltaCache...")
        self.reset_cache()

        delta_time = 0.0
        delta_latencies = []

        for tokens in tqdm(tokenized, desc="DeltaCache"):
            start = time.perf_counter()
            result = self._manager.compute_incremental(tokens, self._adapter)
            elapsed = (time.perf_counter() - start) * 1000

            delta_time += elapsed
            delta_latencies.append(elapsed)

        stats = self._manager.get_stats()

        delta_result = BenchmarkResult(
            experiment_name="deltacache",
            model_name=self.config.model_name,
            num_requests=len(prompts),
            total_tokens=stats["total_tokens"],
            cached_tokens=stats["cached_tokens"],
            computed_tokens=stats["computed_tokens"],
            cache_hit_rate=stats["hit_rate"],
            token_reuse_rate=stats["token_reuse_rate"],
            total_time_ms=delta_time,
            avg_latency_ms=np.mean(delta_latencies),
            ttft_ms=np.mean(delta_latencies),
            throughput_tokens_per_sec=stats["total_tokens"] / (delta_time / 1000),
            memory_used_mb=get_gpu_memory_mb(),
            extra={
                "speedup": baseline_time / delta_time if delta_time > 0 else 0,
                "token_savings": stats["token_reuse_rate"],
            }
        )

        self.log(f"\nComparison:")
        self.log(f"  Baseline: {baseline_time:.0f}ms ({baseline_result.throughput_tokens_per_sec:.0f} tok/s)")
        self.log(f"  DeltaCache: {delta_time:.0f}ms ({delta_result.throughput_tokens_per_sec:.0f} tok/s)")
        self.log(f"  Speedup: {delta_result.extra['speedup']:.2f}x")
        self.log(f"  Token Reuse: {delta_result.token_reuse_rate:.1%}")

        return baseline_result, delta_result

    def run_prompt_length_sweep(
        self,
        base_prompt: str,
        query_template: str,
        prompt_lengths: List[int],
        queries_per_length: int = 20,
    ) -> List[BenchmarkResult]:
        """Sweep over different system prompt lengths."""
        self.log("\n" + "="*60)
        self.log("PROMPT LENGTH SWEEP")
        self.log("="*60)

        results = []

        for target_len in prompt_lengths:
            self.log(f"\nTesting prompt length: {target_len} tokens")

            # Create prompt of approximately target length
            prompt_tokens = self._adapter.tokenize(base_prompt)[0].tolist()
            # Repeat to reach target length
            while len(prompt_tokens) < target_len:
                prompt_tokens = prompt_tokens + prompt_tokens
            prompt_tokens = prompt_tokens[:target_len]

            # Create queries
            queries = [query_template.format(i=i) for i in range(queries_per_length)]

            self.reset_cache()

            latencies = []
            for query in queries:
                query_tokens = self._adapter.tokenize(query)[0].tolist()
                full_tokens = prompt_tokens + query_tokens

                start = time.perf_counter()
                result = self._manager.compute_incremental(full_tokens, self._adapter)
                elapsed = (time.perf_counter() - start) * 1000
                latencies.append(elapsed)

            stats = self._manager.get_stats()

            result = BenchmarkResult(
                experiment_name=f"prompt_len_{target_len}",
                model_name=self.config.model_name,
                num_requests=queries_per_length,
                total_tokens=stats["total_tokens"],
                cached_tokens=stats["cached_tokens"],
                computed_tokens=stats["computed_tokens"],
                cache_hit_rate=stats["hit_rate"],
                token_reuse_rate=stats["token_reuse_rate"],
                total_time_ms=sum(latencies),
                avg_latency_ms=np.mean(latencies),
                ttft_ms=np.mean(latencies),
                throughput_tokens_per_sec=stats["total_tokens"] / (sum(latencies) / 1000),
                memory_used_mb=get_gpu_memory_mb(),
                extra={"prompt_length": target_len}
            )
            results.append(result)

            self.log(f"  Reuse: {result.token_reuse_rate:.1%}, Latency: {result.avg_latency_ms:.1f}ms")

        return results

    def run_eviction_policy_comparison(
        self,
        prompts: List[str],
        policies: List[str] = ["lru", "lfu", "composite", "tiered", "adaptive"],
        memory_limit_mb: float = 500,
    ) -> Dict[str, BenchmarkResult]:
        """Compare different eviction policies."""
        self.log("\n" + "="*60)
        self.log("EVICTION POLICY COMPARISON")
        self.log("="*60)

        tokenized = [self._adapter.tokenize(p)[0].tolist() for p in prompts]
        results = {}

        for policy in policies:
            self.log(f"\nTesting policy: {policy}")

            # Create new manager with this policy
            config = self._adapter.config
            config.gpu_memory_limit = int(memory_limit_mb * 1024 * 1024)
            config.eviction_policy = policy

            manager = DeltaCacheManager(config)

            latencies = []
            for tokens in tqdm(tokenized, desc=f"  {policy}", leave=False):
                start = time.perf_counter()
                result = manager.compute_incremental(tokens, self._adapter)
                elapsed = (time.perf_counter() - start) * 1000
                latencies.append(elapsed)

            stats = manager.get_stats()

            result = BenchmarkResult(
                experiment_name=f"eviction_{policy}",
                model_name=self.config.model_name,
                num_requests=len(prompts),
                total_tokens=stats["total_tokens"],
                cached_tokens=stats["cached_tokens"],
                computed_tokens=stats["computed_tokens"],
                cache_hit_rate=stats["hit_rate"],
                token_reuse_rate=stats["token_reuse_rate"],
                total_time_ms=sum(latencies),
                avg_latency_ms=np.mean(latencies),
                ttft_ms=np.mean(latencies),
                throughput_tokens_per_sec=stats["total_tokens"] / (sum(latencies) / 1000),
                memory_used_mb=memory_limit_mb,
                extra={"policy": policy}
            )
            results[policy] = result

            self.log(f"    Hit Rate: {result.cache_hit_rate:.1%}, "
                     f"Reuse: {result.token_reuse_rate:.1%}, "
                     f"Latency: {result.avg_latency_ms:.1f}ms")

        return results


def create_test_prompts() -> Dict[str, List[str]]:
    """Create test prompts for experiments."""
    # System prompts of varying complexity
    system_prompts = {
        "short": "You are a helpful assistant.",
        "medium": """You are a helpful AI assistant. You provide accurate and concise answers.
Always be respectful and professional in your responses.""",
        "long": """You are an expert AI assistant with deep knowledge in multiple domains.
Your responses should be accurate, well-structured, and educational.
When answering questions:
1. Start with a clear, concise summary
2. Provide detailed explanations with examples
3. Mention related concepts when relevant
4. Suggest further reading when appropriate
Always maintain intellectual honesty and acknowledge uncertainty when present.""",
    }

    # User queries
    user_queries = [
        "What is machine learning?",
        "Explain neural networks.",
        "How does gradient descent work?",
        "What is backpropagation?",
        "Describe transformers architecture.",
        "What is attention mechanism?",
        "Explain BERT model.",
        "What is GPT?",
        "How do LLMs work?",
        "What is fine-tuning?",
        "Explain transfer learning.",
        "What is prompt engineering?",
        "Describe RAG systems.",
        "What is vector embedding?",
        "How does tokenization work?",
        "What is self-attention?",
        "Explain multi-head attention.",
        "What is layer normalization?",
        "Describe positional encoding.",
        "What is beam search?",
    ]

    # Correctness test prompts
    correctness_prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language.",
        "The capital of France is Paris.",
        "Water boils at 100 degrees Celsius.",
    ]

    return {
        "system_prompts": system_prompts,
        "user_queries": user_queries,
        "correctness_prompts": correctness_prompts,
    }


def main():
    parser = argparse.ArgumentParser(description="DeltaCache Real Model Benchmarks")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="Model name")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--load-8bit", action="store_true", help="Load in 8-bit")
    parser.add_argument("--load-4bit", action="store_true", help="Load in 4-bit")
    parser.add_argument("--memory-limit", type=float, default=8.0,
                        help="GPU memory limit for cache (GB)")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "benchmark_results.json")
    parser.add_argument("--experiment", default="all",
                        choices=["all", "correctness", "system_prompt", "baseline",
                                 "prompt_sweep", "eviction"])

    args = parser.parse_args()

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Config
    config = ExperimentConfig(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        load_in_8bit=args.load_8bit,
        load_in_4bit=args.load_4bit,
        gpu_memory_limit_gb=args.memory_limit,
    )

    # Create benchmark runner
    benchmark = RealModelBenchmark(config)
    benchmark.load_model()

    # Get test data
    test_data = create_test_prompts()

    all_results = {"config": asdict(config), "experiments": {}}

    # Run experiments
    if args.experiment in ["all", "correctness"]:
        correctness = benchmark.run_correctness_test(test_data["correctness_prompts"])
        all_results["experiments"]["correctness"] = correctness

    if args.experiment in ["all", "system_prompt"]:
        for name, prompt in test_data["system_prompts"].items():
            result = benchmark.run_system_prompt_experiment(
                prompt, test_data["user_queries"]
            )
            all_results["experiments"][f"system_prompt_{name}"] = asdict(result)

    if args.experiment in ["all", "baseline"]:
        # Create prompts with shared prefix
        system = test_data["system_prompts"]["medium"]
        prompts = [f"{system}\n\nUser: {q}\nAssistant:" for q in test_data["user_queries"]]
        baseline, delta = benchmark.run_baseline_comparison(prompts)
        all_results["experiments"]["baseline_comparison"] = {
            "baseline": asdict(baseline),
            "deltacache": asdict(delta),
            "speedup": delta.extra.get("speedup", 0),
        }

    if args.experiment in ["all", "prompt_sweep"]:
        results = benchmark.run_prompt_length_sweep(
            base_prompt=test_data["system_prompts"]["long"],
            query_template="Question {i}: What is the meaning of life?",
            prompt_lengths=[50, 100, 200, 500, 1000],
            queries_per_length=10,
        )
        all_results["experiments"]["prompt_length_sweep"] = [asdict(r) for r in results]

    if args.experiment in ["all", "eviction"]:
        # Create diverse prompts for eviction testing
        prompts = []
        for i in range(50):
            prompts.append(f"Topic {i}: Tell me about subject number {i} in detail.")
        results = benchmark.run_eviction_policy_comparison(prompts)
        all_results["experiments"]["eviction_policies"] = {
            k: asdict(v) for k, v in results.items()
        }

    # Save results
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\nResults saved to: {args.output}")

    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)

    if "correctness" in all_results["experiments"]:
        c = all_results["experiments"]["correctness"]
        print(f"Correctness: {'PASS' if c['all_correct'] else 'FAIL'}")

    if "baseline_comparison" in all_results["experiments"]:
        bc = all_results["experiments"]["baseline_comparison"]
        print(f"Speedup vs Baseline: {bc['speedup']:.2f}x")
        print(f"Token Reuse: {bc['deltacache']['token_reuse_rate']:.1%}")


if __name__ == "__main__":
    main()
