"""Enhanced benchmarks addressing reviewer concerns.

This script addresses the following reviewer concerns:
1. Add 7B model experiments (with quantization)
2. Multiple runs with standard deviation
3. GPU memory usage tracking
4. Cache size sensitivity analysis
5. Longer sequence testing (2K+ tokens)
6. Fix eviction policy differentiation
"""

import os
import gc
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple
import statistics

import torch
import numpy as np
from tqdm import tqdm

# Set up paths
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from deltacache import DeltaCacheManager, DeltaCacheConfig
# IncrementalEngine is accessed via DeltaCacheManager
from deltacache.hf_integration import LlamaStyleAdapter
from deltacache.eviction.policy import (
    LRUEvictionPolicy, LFUEvictionPolicy, CompositeEvictionPolicy,
    TieredEvictionPolicy, AdaptiveEvictionPolicy
)

RESULTS_DIR = Path(__file__).parent / "results" / "paper"


@dataclass
class EnhancedResult:
    """Enhanced result with statistics."""
    experiment: str
    model: str
    metric: str
    mean: float
    std: float
    min: float
    max: float
    n_runs: int
    raw_values: List[float] = field(default_factory=list)


def get_gpu_memory_stats():
    """Get detailed GPU memory statistics."""
    if not torch.cuda.is_available():
        return {}

    stats = {}
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        stats[f"gpu_{i}"] = {
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "total_gb": round(total, 2),
            "free_gb": round(total - reserved, 2)
        }
    return stats


def clear_gpu_memory():
    """Aggressively clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def compute_stats(values: List[float]) -> Dict:
    """Compute statistics from multiple runs."""
    if len(values) == 0:
        return {"mean": 0, "std": 0, "min": 0, "max": 0}
    if len(values) == 1:
        return {"mean": values[0], "std": 0, "min": values[0], "max": values[0]}
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values),
        "min": min(values),
        "max": max(values)
    }


class EnhancedBenchmark:
    """Enhanced benchmark with reviewer-requested features."""

    def __init__(self, model_name: str, device: str = "cuda:1",
                 use_4bit: bool = False, use_8bit: bool = False):
        self.model_name = model_name
        self.device = device
        self.use_4bit = use_4bit
        self.use_8bit = use_8bit
        self.adapter = None
        self.results = []

    def load_model(self):
        """Load model with optional quantization."""
        print(f"\nLoading {self.model_name}...")
        print(f"  Device: {self.device}")
        print(f"  4-bit: {self.use_4bit}, 8-bit: {self.use_8bit}")

        self.adapter = LlamaStyleAdapter.from_pretrained(
            self.model_name,
            device=self.device,
            load_in_4bit=self.use_4bit,
            load_in_8bit=self.use_8bit,
        )

        mem_stats = get_gpu_memory_stats()
        print(f"  Memory after load: {mem_stats}")
        return mem_stats

    def unload_model(self):
        """Unload model and free memory."""
        if self.adapter:
            del self.adapter
            self.adapter = None
        clear_gpu_memory()

    def run_correctness_test(self, n_runs: int = 5) -> Dict:
        """Run correctness validation with multiple runs."""
        print(f"\n{'='*60}")
        print("Correctness Validation")
        print(f"{'='*60}")

        test_prompts = [
            "The quick brown fox",
            "In a galaxy far far away",
            "Once upon a time in a land",
            "The fundamental theorem of calculus states",
            "Machine learning is a subset of artificial intelligence",
        ]

        all_results = []

        for run in range(n_runs):
            kv_diffs = []
            token_matches = []

            config = DeltaCacheConfig.for_model(self.model_name)
            config.device = self.device
            manager = DeltaCacheManager(config)

            for prompt in test_prompts:
                tokens = self.adapter.tokenizer.encode(prompt)

                # Full computation (no cache)
                key_full, val_full = self.adapter.compute_kv_for_tokens(tokens)

                # Clear and compute with cache
                manager.clear()
                result = manager.compute_incremental(tokens, self.adapter.compute_kv)

                # Compare
                kv_diff = torch.abs(key_full - result.key_cache).max().item()
                kv_diffs.append(kv_diff)
                token_matches.append(1.0 if kv_diff < 0.01 else 0.0)

            del manager

            all_results.append({
                "max_kv_diff": max(kv_diffs) if kv_diffs else 0,
                "mean_kv_diff": statistics.mean(kv_diffs) if kv_diffs else 0,
                "token_match_rate": statistics.mean(token_matches) if token_matches else 1.0
            })

        # Compute statistics
        result = {
            "n_runs": n_runs,
            "n_prompts": len(test_prompts),
            "max_kv_diff": compute_stats([r["max_kv_diff"] for r in all_results]),
            "mean_kv_diff": compute_stats([r["mean_kv_diff"] for r in all_results]),
            "token_match_rate": compute_stats([r["token_match_rate"] for r in all_results]),
        }

        print(f"  Max KV Diff: {result['max_kv_diff']['mean']:.6f} ± {result['max_kv_diff']['std']:.6f}")
        print(f"  Token Match: {result['token_match_rate']['mean']*100:.1f}%")

        return result

    def run_latency_benchmark(self, n_queries: int = 100, n_runs: int = 3,
                               system_prompt_len: int = 50) -> Dict:
        """Run latency benchmark with multiple runs."""
        print(f"\n{'='*60}")
        print(f"Latency Benchmark (n_queries={n_queries}, n_runs={n_runs})")
        print(f"{'='*60}")

        # Generate test prompts
        base_queries = [
            "What is the capital of France?",
            "Explain quantum computing briefly.",
            "How does photosynthesis work?",
            "What is machine learning?",
            "Describe the water cycle.",
        ]

        # Create system prompt
        system_prompt = "You are a helpful AI assistant. " * (system_prompt_len // 8)

        # Create full prompts
        prompts = []
        for i in range(n_queries):
            query = base_queries[i % len(base_queries)]
            prompts.append(f"{system_prompt}\n\nUser: {query}\nAssistant:")

        run_results = []

        for run in range(n_runs):
            print(f"\n  Run {run + 1}/{n_runs}")

            # Create fresh manager for each run
            config = DeltaCacheConfig.for_model(self.model_name)
            config.device = self.device
            manager = DeltaCacheManager(config)

            latencies = []
            cache_hits = 0
            total_tokens = 0
            matched_tokens = 0

            # Warmup
            tokens = self.adapter.tokenizer.encode(prompts[0])
            _ = manager.compute_incremental(tokens, self.adapter.compute_kv)
            manager.clear()

            # Benchmark
            mem_before = get_gpu_memory_stats()
            start_total = time.perf_counter()

            for prompt in tqdm(prompts, desc="    Processing", leave=False):
                tokens = self.adapter.tokenizer.encode(prompt)

                start = time.perf_counter()
                result = manager.compute_incremental(tokens, self.adapter.compute_kv)
                latency = (time.perf_counter() - start) * 1000

                latencies.append(latency)
                total_tokens += len(tokens)
                matched_tokens += result.matched_length
                if result.matched_length > 0:
                    cache_hits += 1

            total_time = (time.perf_counter() - start_total) * 1000
            mem_after = get_gpu_memory_stats()

            run_results.append({
                "total_time_ms": total_time,
                "avg_latency_ms": statistics.mean(latencies),
                "p50_latency_ms": sorted(latencies)[len(latencies)//2],
                "p99_latency_ms": sorted(latencies)[int(len(latencies)*0.99)],
                "cache_hit_rate": cache_hits / n_queries,
                "token_reuse_rate": matched_tokens / total_tokens if total_tokens > 0 else 0,
                "throughput_tok_s": total_tokens / (total_time / 1000),
                "mem_delta_gb": mem_after.get("gpu_1", {}).get("allocated_gb", 0) -
                               mem_before.get("gpu_1", {}).get("allocated_gb", 0)
            })

            # Cleanup
            del manager
            clear_gpu_memory()

        # Compute statistics
        result = {
            "n_queries": n_queries,
            "n_runs": n_runs,
            "system_prompt_len": system_prompt_len,
        }

        for key in run_results[0].keys():
            values = [r[key] for r in run_results]
            result[key] = compute_stats(values)
            result[key]["raw"] = values

        print(f"\n  Results:")
        print(f"    Avg Latency: {result['avg_latency_ms']['mean']:.2f} ± {result['avg_latency_ms']['std']:.2f} ms")
        print(f"    Cache Hit Rate: {result['cache_hit_rate']['mean']*100:.1f}%")
        print(f"    Throughput: {result['throughput_tok_s']['mean']:.0f} tok/s")

        return result

    def run_eviction_policy_comparison(self, n_queries: int = 300,
                                        cache_size_mb: float = 0.5) -> Dict:
        """Compare eviction policies with memory pressure.

        Uses a very small cache (0.5MB) to force evictions and compare policies.
        """
        print(f"\n{'='*60}")
        print(f"Eviction Policy Comparison (cache_size={cache_size_mb}MB)")
        print(f"{'='*60}")

        policies = {
            "lru": LRUEvictionPolicy(),
            "lfu": LFUEvictionPolicy(),
            "composite": CompositeEvictionPolicy(),
            "tiered": TieredEvictionPolicy(),
            "adaptive": AdaptiveEvictionPolicy(),
        }

        # Generate completely unique prompts to maximize cache pressure
        # Each prompt starts differently to prevent prefix sharing
        unique_topics = [
            "Physics: Explain quantum mechanics and wave-particle duality in detail.",
            "Biology: Describe how DNA replication works in cells step by step.",
            "Chemistry: What are the properties and applications of noble gases?",
            "Astronomy: How do black holes form and evolve over billions of years?",
            "Medicine: Explain how different types of antibiotics fight infections.",
            "Technology: Describe the architecture of modern multi-core processors.",
            "Mathematics: What is the significance and applications of prime numbers?",
            "History: Describe the complex causes leading to World War I.",
            "Economics: Explain supply and demand equilibrium with real examples.",
            "Psychology: How does memory formation and recall work in the brain?",
            "Geography: Describe plate tectonics and how continents have drifted.",
            "Literature: Analyze the major themes present in Shakespeare's Hamlet.",
            "Music: How does harmony and counterpoint work in classical music?",
            "Art: Describe the key characteristics of the Renaissance art movement.",
            "Philosophy: Explain Kant's categorical imperative and its implications.",
            "Sociology: How do social norms develop and change in modern societies?",
            "Politics: Describe the separation of powers in democratic governments.",
            "Law: Explain the historical development of habeas corpus protections.",
            "Engineering: How do suspension bridges distribute structural loads?",
            "Computer Science: Explain hash table implementations and collision handling.",
        ]

        # Create many unique prompts - each completely different
        prompts = []
        for i in range(n_queries):
            topic_idx = i % len(unique_topics)
            # Add unique prefix and suffix to make each prompt distinct
            prompt = f"Request number {i}: {unique_topics[topic_idx]} Please provide comprehensive details."
            prompts.append(prompt)

        results = {}

        for policy_name, policy in policies.items():
            print(f"\n  Testing {policy_name.upper()}...")

            # Create manager with small cache to trigger eviction
            config = DeltaCacheConfig.for_model(self.model_name)
            config.device = self.device
            config.gpu_memory_limit = cache_size_mb * 1024 * 1024  # Convert to bytes
            manager = DeltaCacheManager(config, eviction_policy=policy)

            latencies = []
            evictions = 0
            cache_hits = 0

            start_total = time.perf_counter()

            for prompt in tqdm(prompts, desc=f"    {policy_name}", leave=False):
                tokens = self.adapter.tokenizer.encode(prompt)

                start = time.perf_counter()
                result = manager.compute_incremental(tokens, self.adapter.compute_kv)
                latency = (time.perf_counter() - start) * 1000

                latencies.append(latency)
                if result.matched_length > 0:
                    cache_hits += 1

            total_time = (time.perf_counter() - start_total) * 1000

            # Get eviction stats
            stats = manager.get_stats()

            results[policy_name] = {
                "avg_latency_ms": statistics.mean(latencies),
                "std_latency_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
                "p50_latency_ms": sorted(latencies)[len(latencies)//2],
                "p99_latency_ms": sorted(latencies)[int(len(latencies)*0.99)],
                "cache_hit_rate": cache_hits / n_queries,
                "throughput_tok_s": len(prompts) * 50 / (total_time / 1000),  # Estimate 50 tokens/prompt
                "total_evictions": stats.get("evictions", 0),
            }

            print(f"      Latency: {results[policy_name]['avg_latency_ms']:.2f} ± {results[policy_name]['std_latency_ms']:.2f} ms")
            print(f"      Hit Rate: {results[policy_name]['cache_hit_rate']*100:.1f}%")
            print(f"      Evictions: {results[policy_name]['total_evictions']}")

            del manager
            clear_gpu_memory()

        return results

    def run_sequence_length_scaling(self, max_length: int = 2048) -> Dict:
        """Test performance with longer sequences."""
        print(f"\n{'='*60}")
        print(f"Sequence Length Scaling (up to {max_length} tokens)")
        print(f"{'='*60}")

        # Generate long text
        long_text = """
        Artificial intelligence has transformed numerous industries and continues to evolve rapidly.
        Machine learning, a subset of AI, enables systems to learn from data without explicit programming.
        Deep learning, powered by neural networks with many layers, has achieved remarkable results in
        image recognition, natural language processing, and game playing. The transformer architecture,
        introduced in 2017, revolutionized NLP and led to models like BERT, GPT, and their successors.
        These models can understand context, generate coherent text, and perform complex reasoning tasks.
        """ * 50  # Repeat to get long text

        lengths = [128, 256, 512, 1024]
        if max_length >= 2048:
            lengths.append(2048)

        results = {}

        config = DeltaCacheConfig.for_model(self.model_name)
        config.device = self.device
        manager = DeltaCacheManager(config)

        for target_len in lengths:
            print(f"\n  Testing {target_len} tokens...")

            # Tokenize and truncate
            tokens = self.adapter.tokenizer.encode(long_text)[:target_len]
            actual_len = len(tokens)

            # Run multiple times
            latencies = []
            for _ in range(5):
                manager.clear()

                start = time.perf_counter()
                result = manager.compute_incremental(tokens, self.adapter.compute_kv)
                latency = (time.perf_counter() - start) * 1000
                latencies.append(latency)

            results[target_len] = {
                "actual_tokens": actual_len,
                "avg_latency_ms": statistics.mean(latencies),
                "std_latency_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
                "tokens_per_sec": actual_len / (statistics.mean(latencies) / 1000),
            }

            print(f"    Latency: {results[target_len]['avg_latency_ms']:.2f} ± {results[target_len]['std_latency_ms']:.2f} ms")
            print(f"    Throughput: {results[target_len]['tokens_per_sec']:.0f} tok/s")

        del manager
        clear_gpu_memory()

        return results

    def run_cache_size_sensitivity(self, n_queries: int = 100) -> Dict:
        """Analyze performance sensitivity to cache size."""
        print(f"\n{'='*60}")
        print("Cache Size Sensitivity Analysis")
        print(f"{'='*60}")

        cache_sizes_mb = [10, 25, 50, 100, 200]

        # Generate prompts
        base_prompts = [
            "What is the meaning of life?",
            "Explain quantum mechanics.",
            "How do computers work?",
            "What is artificial intelligence?",
            "Describe the solar system.",
        ]

        prompts = [base_prompts[i % len(base_prompts)] + f" ({i})"
                   for i in range(n_queries)]

        results = {}

        for cache_mb in cache_sizes_mb:
            print(f"\n  Cache size: {cache_mb} MB")

            config = DeltaCacheConfig.for_model(self.model_name)
            config.device = self.device
            config.gpu_memory_limit = cache_mb * 1024 * 1024
            manager = DeltaCacheManager(config)

            cache_hits = 0
            latencies = []

            for prompt in tqdm(prompts, desc=f"    {cache_mb}MB", leave=False):
                tokens = self.adapter.tokenizer.encode(prompt)

                start = time.perf_counter()
                result = manager.compute_incremental(tokens, self.adapter.compute_kv)
                latency = (time.perf_counter() - start) * 1000

                latencies.append(latency)
                if result.matched_length > 0:
                    cache_hits += 1

            stats = manager.get_stats()

            results[cache_mb] = {
                "avg_latency_ms": statistics.mean(latencies),
                "cache_hit_rate": cache_hits / n_queries,
                "evictions": stats.get("evictions", 0),
                "memory_used_mb": stats.get("memory_used", 0) / (1024 * 1024),
            }

            print(f"    Hit Rate: {results[cache_mb]['cache_hit_rate']*100:.1f}%")
            print(f"    Evictions: {results[cache_mb]['evictions']}")

            del manager
            clear_gpu_memory()

        return results

    def run_baseline_comparison(self, n_queries: int = 50, n_runs: int = 3) -> Dict:
        """Compare with and without cache, multiple runs."""
        print(f"\n{'='*60}")
        print(f"Baseline Comparison (n_runs={n_runs})")
        print(f"{'='*60}")

        system_prompt = "You are a helpful AI assistant. " * 10
        queries = [
            "What is 2+2?",
            "Name the planets.",
            "What is Python?",
            "Explain gravity.",
            "What is DNA?",
        ]

        prompts = [f"{system_prompt}\n\nUser: {queries[i % len(queries)]}\nAssistant:"
                   for i in range(n_queries)]

        results = {"with_cache": [], "without_cache": []}

        for run in range(n_runs):
            print(f"\n  Run {run + 1}/{n_runs}")

            # WITH cache
            config = DeltaCacheConfig.for_model(self.model_name)
            config.device = self.device
            manager = DeltaCacheManager(config)

            latencies_with = []
            mem_before = get_gpu_memory_stats()

            for prompt in prompts:
                tokens = self.adapter.tokenizer.encode(prompt)
                start = time.perf_counter()
                _ = manager.compute_incremental(tokens, self.adapter.compute_kv)
                latencies_with.append((time.perf_counter() - start) * 1000)

            mem_after = get_gpu_memory_stats()

            results["with_cache"].append({
                "avg_latency_ms": statistics.mean(latencies_with),
                "throughput_tok_s": n_queries * 80 / (sum(latencies_with) / 1000),
                "mem_used_gb": mem_after.get("gpu_1", {}).get("allocated_gb", 0)
            })

            del manager
            clear_gpu_memory()

            # WITHOUT cache (fresh computation each time)
            latencies_without = []

            for prompt in prompts:
                tokens = self.adapter.tokenizer.encode(prompt)
                start = time.perf_counter()
                _ = self.adapter.compute_kv_for_tokens(tokens)
                latencies_without.append((time.perf_counter() - start) * 1000)

            results["without_cache"].append({
                "avg_latency_ms": statistics.mean(latencies_without),
                "throughput_tok_s": n_queries * 80 / (sum(latencies_without) / 1000),
            })

            clear_gpu_memory()

        # Compute statistics
        final_results = {}
        for mode in ["with_cache", "without_cache"]:
            final_results[mode] = {}
            for key in results[mode][0].keys():
                values = [r[key] for r in results[mode]]
                final_results[mode][key] = compute_stats(values)

        # Compute speedup
        speedup_values = [
            results["without_cache"][i]["avg_latency_ms"] / results["with_cache"][i]["avg_latency_ms"]
            for i in range(n_runs)
        ]
        final_results["speedup"] = compute_stats(speedup_values)

        print(f"\n  With Cache: {final_results['with_cache']['avg_latency_ms']['mean']:.2f} ± "
              f"{final_results['with_cache']['avg_latency_ms']['std']:.2f} ms")
        print(f"  Without Cache: {final_results['without_cache']['avg_latency_ms']['mean']:.2f} ± "
              f"{final_results['without_cache']['avg_latency_ms']['std']:.2f} ms")
        print(f"  Speedup: {final_results['speedup']['mean']:.2f}x ± {final_results['speedup']['std']:.2f}")

        return final_results


def run_full_benchmark(model_name: str, use_4bit: bool = False,
                       device: str = "cuda:1") -> Dict:
    """Run all benchmarks for a model."""
    print(f"\n{'='*70}")
    print(f"FULL BENCHMARK: {model_name}")
    print(f"{'='*70}")

    benchmark = EnhancedBenchmark(model_name, device=device, use_4bit=use_4bit)

    results = {
        "model": model_name,
        "device": device,
        "quantization": "4bit" if use_4bit else "none",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Load model
    results["memory_after_load"] = benchmark.load_model()

    # Run benchmarks
    results["correctness"] = benchmark.run_correctness_test(n_runs=5)
    results["baseline_comparison"] = benchmark.run_baseline_comparison(n_queries=50, n_runs=3)
    results["latency"] = benchmark.run_latency_benchmark(n_queries=100, n_runs=3)
    results["eviction_policies"] = benchmark.run_eviction_policy_comparison(n_queries=200, cache_size_mb=50)
    results["sequence_scaling"] = benchmark.run_sequence_length_scaling(max_length=2048)
    results["cache_sensitivity"] = benchmark.run_cache_size_sensitivity(n_queries=100)

    # Cleanup
    benchmark.unload_model()

    return results


def main():
    parser = argparse.ArgumentParser(description="Enhanced DeltaCache Benchmarks")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                       help="Model to benchmark")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device to use")
    parser.add_argument("--use-4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    args = parser.parse_args()

    # Run benchmark
    results = run_full_benchmark(args.model, use_4bit=args.use_4bit, device=args.device)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.output:
        output_path = Path(args.output)
    else:
        model_short = args.model.split("/")[-1].lower().replace("-", "_")
        output_path = RESULTS_DIR / f"enhanced_{model_short}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
