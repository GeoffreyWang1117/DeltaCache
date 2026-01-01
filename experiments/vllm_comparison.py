"""Compare DeltaCache with vLLM prefix caching and HuggingFace baseline.

This benchmark provides comprehensive comparison for academic evaluation:
- vLLM: PagedAttention without prefix caching (baseline)
- vLLM + Prefix: PagedAttention with prefix caching enabled
- HuggingFace: Standard transformers without any caching
- DeltaCache: Incremental KV computation with prefix tree

Metrics collected:
- Time-to-First-Token (TTFT) / Prefill latency
- P50, P95, P99 latency
- Throughput (tokens/second)
- Memory usage
"""

import json
import time
import statistics
import torch
import gc
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM

# vLLM imports
try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    print("WARNING: vLLM not available. Install with: pip install vllm")

# DeltaCache imports
from deltacache import DeltaCacheConfig, DeltaCacheManager
from deltacache.hf_integration import LlamaStyleAdapter

RESULTS_DIR = Path(__file__).parent / "results" / "paper"


@dataclass
class LatencyStats:
    """Latency statistics."""
    mean: float
    std: float
    p50: float
    p95: float
    p99: float
    min: float
    max: float

    @classmethod
    def from_latencies(cls, latencies: List[float]) -> "LatencyStats":
        if not latencies:
            return cls(0, 0, 0, 0, 0, 0, 0)
        sorted_lat = sorted(latencies)
        n = len(sorted_lat)
        return cls(
            mean=statistics.mean(latencies),
            std=statistics.stdev(latencies) if n > 1 else 0,
            p50=sorted_lat[n // 2],
            p95=sorted_lat[int(n * 0.95)] if n >= 20 else sorted_lat[-1],
            p99=sorted_lat[int(n * 0.99)] if n >= 100 else sorted_lat[-1],
            min=sorted_lat[0],
            max=sorted_lat[-1],
        )


@dataclass
class BenchmarkResult:
    """Comprehensive benchmark result."""
    system: str
    model_name: str
    num_queries: int
    total_tokens: int
    total_time_ms: float
    latency_stats: LatencyStats
    throughput_tok_s: float
    memory_peak_mb: float = 0.0
    cache_hit_rate: float = 0.0
    tokens_reused: int = 0
    extra_info: Dict[str, Any] = field(default_factory=dict)


def cleanup_gpu():
    """Clean up GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def get_gpu_memory_peak_mb() -> float:
    """Get peak GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def reset_gpu_memory_stats():
    """Reset GPU memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# =============================================================================
# Test Prompts - Comprehensive scenarios
# =============================================================================

SYSTEM_PROMPTS = {
    "assistant": """You are a helpful AI assistant. You provide accurate, concise answers.
You always think step by step and explain your reasoning clearly.
You are friendly and professional in your responses.
When asked about technical topics, you provide detailed explanations.""",

    "coder": """You are an expert programmer and software engineer.
You write clean, efficient, and well-documented code.
You follow best practices and design patterns.
You can explain complex technical concepts simply.""",

    "analyst": """You are a data analyst specializing in business intelligence.
You interpret data patterns and provide actionable insights.
You create clear visualizations and reports.
You communicate findings to both technical and non-technical audiences.""",
}

USER_QUERIES = [
    # Science & Technology
    "What is the capital of France?",
    "How does photosynthesis work?",
    "Explain the theory of relativity in simple terms.",
    "What are prime numbers and why are they important?",
    "How do neural networks learn?",
    "What is quantum entanglement?",
    "Explain how vaccines work.",
    "What causes climate change?",
    "How do black holes form?",
    "What is CRISPR gene editing?",
    # Programming
    "What is the difference between Python lists and tuples?",
    "Explain object-oriented programming.",
    "What is a hash table and when should I use it?",
    "How does garbage collection work?",
    "What is the difference between TCP and UDP?",
    "Explain the concept of recursion.",
    "What are design patterns in software engineering?",
    "How does HTTPS encryption work?",
    "What is a REST API?",
    "Explain microservices architecture.",
    # General Knowledge
    "Who wrote Romeo and Juliet?",
    "What is the speed of light?",
    "How do airplanes fly?",
    "What causes earthquakes?",
    "How does the stock market work?",
    "What is the significance of the Renaissance?",
    "How do antibiotics work?",
    "What is dark matter?",
    "Explain the water cycle.",
    "What are the main causes of inflation?",
]


def create_prompts(system_prompt: str, queries: List[str]) -> List[str]:
    """Create full prompts with system prompt prefix."""
    return [f"{system_prompt}\n\nUser: {q}\nAssistant:" for q in queries]


# =============================================================================
# HuggingFace Baseline Benchmark
# =============================================================================

def benchmark_huggingface_baseline(
    model_name: str,
    prompts: List[str],
    device: str = "cuda",
) -> BenchmarkResult:
    """Benchmark HuggingFace without any caching (full recomputation each time)."""
    print(f"\n{'='*60}")
    print(f"HuggingFace Baseline (No Cache)")
    print(f"Model: {model_name}")
    print(f"Prompts: {len(prompts)}")
    print(f"{'='*60}")

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    # Warmup
    print("Warming up...")
    with torch.no_grad():
        inputs = tokenizer(prompts[0], return_tensors="pt").to(device)
        _ = model(**inputs, use_cache=False)

    cleanup_gpu()
    reset_gpu_memory_stats()

    # Benchmark - No caching, full forward pass each time
    print("Running benchmark...")
    latencies = []
    total_tokens = 0

    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            num_tokens = inputs["input_ids"].shape[1]
            total_tokens += num_tokens

            torch.cuda.synchronize()
            start = time.perf_counter()

            # Full forward pass without caching
            _ = model(**inputs, use_cache=False)

            torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)

    total_time = sum(latencies)
    memory_peak = get_gpu_memory_peak_mb()

    latency_stats = LatencyStats.from_latencies(latencies)
    throughput = total_tokens / (total_time / 1000)

    result = BenchmarkResult(
        system="hf_baseline",
        model_name=model_name,
        num_queries=len(prompts),
        total_tokens=total_tokens,
        total_time_ms=total_time,
        latency_stats=latency_stats,
        throughput_tok_s=throughput,
        memory_peak_mb=memory_peak,
    )

    print(f"Total time: {total_time:.2f}ms")
    print(f"Mean latency: {latency_stats.mean:.2f}ms")
    print(f"P50/P95/P99: {latency_stats.p50:.2f}/{latency_stats.p95:.2f}/{latency_stats.p99:.2f}ms")
    print(f"Throughput: {throughput:.0f} tok/s")
    print(f"Memory peak: {memory_peak:.0f} MB")

    # Cleanup
    del model, tokenizer
    cleanup_gpu()

    return result


# =============================================================================
# vLLM Benchmark
# =============================================================================

def benchmark_vllm(
    model_name: str,
    prompts: List[str],
    enable_prefix_caching: bool = False,
) -> Optional[BenchmarkResult]:
    """Benchmark vLLM with or without prefix caching."""
    if not HAS_VLLM:
        print("vLLM not available, skipping...")
        return None

    print(f"\n{'='*60}")
    print(f"vLLM Benchmark (prefix_caching={enable_prefix_caching})")
    print(f"Model: {model_name}")
    print(f"Prompts: {len(prompts)}")
    print(f"{'='*60}")

    # Disable distributed for single-GPU
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    try:
        # Initialize vLLM
        print("Loading model...")
        llm = LLM(
            model=model_name,
            enable_prefix_caching=enable_prefix_caching,
            gpu_memory_utilization=0.4,
            max_model_len=512,
            enforce_eager=True,
        )

        sampling_params = SamplingParams(
            max_tokens=1,  # Just measure prefill
            temperature=0.0,
        )

        tokenizer = llm.get_tokenizer()

        # Warmup
        print("Warming up...")
        _ = llm.generate(prompts[:1], sampling_params)

        reset_gpu_memory_stats()

        # Benchmark - process one at a time to measure individual latencies
        print("Running benchmark...")
        latencies = []
        total_tokens = 0

        for prompt in prompts:
            num_tokens = len(tokenizer.encode(prompt))
            total_tokens += num_tokens

            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = llm.generate([prompt], sampling_params)
            torch.cuda.synchronize()

            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)

        total_time = sum(latencies)
        memory_peak = get_gpu_memory_peak_mb()

        latency_stats = LatencyStats.from_latencies(latencies)
        throughput = total_tokens / (total_time / 1000)

        system_name = "vllm_prefix" if enable_prefix_caching else "vllm_baseline"

        result = BenchmarkResult(
            system=system_name,
            model_name=model_name,
            num_queries=len(prompts),
            total_tokens=total_tokens,
            total_time_ms=total_time,
            latency_stats=latency_stats,
            throughput_tok_s=throughput,
            memory_peak_mb=memory_peak,
        )

        print(f"Total time: {total_time:.2f}ms")
        print(f"Mean latency: {latency_stats.mean:.2f}ms")
        print(f"P50/P95/P99: {latency_stats.p50:.2f}/{latency_stats.p95:.2f}/{latency_stats.p99:.2f}ms")
        print(f"Throughput: {throughput:.0f} tok/s")
        print(f"Memory peak: {memory_peak:.0f} MB")

        del llm
        cleanup_gpu()

        return result

    except Exception as e:
        print(f"vLLM benchmark failed: {e}")
        cleanup_gpu()
        return None


# =============================================================================
# DeltaCache Benchmark
# =============================================================================

def benchmark_deltacache(
    model_name: str,
    prompts: List[str],
    device: str = "cuda",
    use_quantization: bool = False,
) -> BenchmarkResult:
    """Benchmark DeltaCache with incremental KV computation."""
    print(f"\n{'='*60}")
    print(f"DeltaCache Benchmark")
    print(f"Model: {model_name}")
    print(f"Prompts: {len(prompts)}")
    print(f"Quantization: {use_quantization}")
    print(f"{'='*60}")

    # Initialize adapter and manager
    print("Loading model...")
    adapter = LlamaStyleAdapter(
        model_name,
        device=device,
        use_quantization=use_quantization,
    )
    config = DeltaCacheConfig.for_model(model_name)
    manager = DeltaCacheManager(config)

    tokenizer = adapter.tokenizer

    # Warmup
    print("Warming up...")
    tokens = tokenizer.encode(prompts[0])
    _ = manager.compute_incremental(tokens, adapter.compute_kv_for_tokens)
    manager.clear()

    cleanup_gpu()
    reset_gpu_memory_stats()

    # Benchmark
    print("Running benchmark...")
    latencies = []
    total_tokens = 0
    total_reused = 0

    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        num_tokens = len(tokens)
        total_tokens += num_tokens

        torch.cuda.synchronize()
        start = time.perf_counter()

        result = manager.compute_incremental(tokens, adapter.compute_kv_for_tokens)

        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)

        total_reused += result.matched_length

    total_time = sum(latencies)
    memory_peak = get_gpu_memory_peak_mb()

    latency_stats = LatencyStats.from_latencies(latencies)
    throughput = total_tokens / (total_time / 1000)
    stats = manager.get_stats()
    hit_rate = stats.get("hit_rate", 0)

    result = BenchmarkResult(
        system="deltacache",
        model_name=model_name,
        num_queries=len(prompts),
        total_tokens=total_tokens,
        total_time_ms=total_time,
        latency_stats=latency_stats,
        throughput_tok_s=throughput,
        memory_peak_mb=memory_peak,
        cache_hit_rate=hit_rate,
        tokens_reused=total_reused,
    )

    print(f"Total time: {total_time:.2f}ms")
    print(f"Mean latency: {latency_stats.mean:.2f}ms")
    print(f"P50/P95/P99: {latency_stats.p50:.2f}/{latency_stats.p95:.2f}/{latency_stats.p99:.2f}ms")
    print(f"Throughput: {throughput:.0f} tok/s")
    print(f"Memory peak: {memory_peak:.0f} MB")
    print(f"Cache hit rate: {hit_rate:.1%}")
    print(f"Tokens reused: {total_reused}/{total_tokens} ({100*total_reused/total_tokens:.1f}%)")

    # Cleanup
    del manager, adapter
    cleanup_gpu()

    return result


# =============================================================================
# Main Comparison
# =============================================================================

def run_full_comparison(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    num_queries: int = 30,
    system_prompt_key: str = "assistant",
    skip_vllm: bool = False,
):
    """Run comprehensive comparison across all systems."""
    print("\n" + "="*70)
    print("COMPREHENSIVE PREFIX CACHING COMPARISON")
    print("="*70)

    # Create prompts
    system_prompt = SYSTEM_PROMPTS[system_prompt_key]
    queries = USER_QUERIES[:num_queries]
    prompts = create_prompts(system_prompt, queries)

    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  System prompt: {system_prompt_key} ({len(system_prompt)} chars)")
    print(f"  Queries: {len(queries)}")
    print(f"  Total prompts: {len(prompts)}")

    results = []

    # 1. HuggingFace baseline (no cache)
    try:
        result = benchmark_huggingface_baseline(model_name, prompts)
        results.append(result)
    except Exception as e:
        print(f"HuggingFace baseline failed: {e}")

    # 2. vLLM without prefix caching
    if not skip_vllm and HAS_VLLM:
        result = benchmark_vllm(model_name, prompts, enable_prefix_caching=False)
        if result:
            results.append(result)

    # 3. vLLM with prefix caching
    if not skip_vllm and HAS_VLLM:
        result = benchmark_vllm(model_name, prompts, enable_prefix_caching=True)
        if result:
            results.append(result)

    # 4. DeltaCache
    try:
        result = benchmark_deltacache(model_name, prompts)
        results.append(result)
    except Exception as e:
        print(f"DeltaCache failed: {e}")

    # Print summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)

    print(f"\n{'System':<18} {'Mean (ms)':<12} {'P50 (ms)':<12} {'P99 (ms)':<12} {'Throughput':<12} {'Memory (MB)':<12}")
    print("-"*78)

    for r in results:
        print(f"{r.system:<18} {r.latency_stats.mean:<12.2f} {r.latency_stats.p50:<12.2f} "
              f"{r.latency_stats.p99:<12.2f} {r.throughput_tok_s:<12.0f} {r.memory_peak_mb:<12.0f}")

    # Calculate speedups relative to HuggingFace baseline
    hf_baseline = next((r for r in results if r.system == "hf_baseline"), None)

    if hf_baseline:
        print("\nSpeedups vs HuggingFace Baseline:")
        for r in results:
            if r.system != "hf_baseline":
                speedup = hf_baseline.latency_stats.mean / r.latency_stats.mean
                print(f"  {r.system}: {speedup:.2f}x faster")

    # Calculate speedups between caching systems
    vllm_prefix = next((r for r in results if r.system == "vllm_prefix"), None)
    deltacache = next((r for r in results if r.system == "deltacache"), None)

    if vllm_prefix and deltacache:
        speedup = vllm_prefix.latency_stats.mean / deltacache.latency_stats.mean
        print(f"\nDeltaCache vs vLLM Prefix: {speedup:.2f}x")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results_dict = {
        "configuration": {
            "model": model_name,
            "num_queries": len(prompts),
            "system_prompt_length": len(system_prompt.split()),
        },
        "results": [
            {
                "system": r.system,
                "total_time_ms": r.total_time_ms,
                "latency_mean_ms": r.latency_stats.mean,
                "latency_std_ms": r.latency_stats.std,
                "latency_p50_ms": r.latency_stats.p50,
                "latency_p95_ms": r.latency_stats.p95,
                "latency_p99_ms": r.latency_stats.p99,
                "throughput_tok_s": r.throughput_tok_s,
                "memory_peak_mb": r.memory_peak_mb,
                "cache_hit_rate": r.cache_hit_rate,
                "tokens_reused": r.tokens_reused,
            }
            for r in results
        ],
    }

    output_path = RESULTS_DIR / "vllm_comparison.json"
    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


def run_scaling_analysis(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    query_counts: List[int] = [5, 10, 20, 30],
    skip_vllm: bool = False,
):
    """Analyze how performance scales with number of queries."""
    print("\n" + "="*70)
    print("SCALING ANALYSIS: Performance vs Query Count")
    print("="*70)

    system_prompt = SYSTEM_PROMPTS["assistant"]
    all_results = []

    for n in query_counts:
        print(f"\n--- Testing with {n} queries ---")

        queries = USER_QUERIES[:n]
        prompts = create_prompts(system_prompt, queries)

        # DeltaCache
        try:
            result = benchmark_deltacache(model_name, prompts)
            all_results.append({"n": n, **result.__dict__})
        except Exception as e:
            print(f"DeltaCache failed: {e}")

        # vLLM prefix (if available)
        if not skip_vllm and HAS_VLLM:
            result = benchmark_vllm(model_name, prompts, enable_prefix_caching=True)
            if result:
                all_results.append({"n": n, **result.__dict__})

    # Save results
    output_path = RESULTS_DIR / "scaling_analysis.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Convert LatencyStats to dict for JSON serialization
    for r in all_results:
        if "latency_stats" in r and hasattr(r["latency_stats"], "__dict__"):
            r["latency_stats"] = r["latency_stats"].__dict__

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return all_results


def run_system_prompt_comparison(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    num_queries: int = 20,
):
    """Compare performance across different system prompts."""
    print("\n" + "="*70)
    print("SYSTEM PROMPT COMPARISON")
    print("="*70)

    results = []

    for prompt_name, system_prompt in SYSTEM_PROMPTS.items():
        print(f"\n--- System prompt: {prompt_name} ---")

        queries = USER_QUERIES[:num_queries]
        prompts = create_prompts(system_prompt, queries)

        result = benchmark_deltacache(model_name, prompts)
        results.append({
            "system_prompt": prompt_name,
            "prompt_length": len(system_prompt.split()),
            **result.__dict__,
        })

    # Print comparison
    print("\n" + "="*60)
    print("SYSTEM PROMPT COMPARISON SUMMARY")
    print("="*60)

    print(f"\n{'Prompt':<12} {'Length':<10} {'Mean (ms)':<12} {'Reuse Rate':<12}")
    print("-"*46)

    for r in results:
        reuse_rate = r["tokens_reused"] / r["total_tokens"] * 100
        # Handle both dict and LatencyStats object
        if isinstance(r.get("latency_stats"), dict):
            mean = r["latency_stats"]["mean"]
        else:
            mean = r["latency_stats"].mean
        print(f"{r['system_prompt']:<12} {r['prompt_length']:<10} {mean:<12.2f} {reuse_rate:<12.1f}%")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare DeltaCache with vLLM and HuggingFace")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="Model to benchmark")
    parser.add_argument("--num-queries", type=int, default=30, help="Number of queries")
    parser.add_argument("--skip-vllm", action="store_true", help="Skip vLLM benchmarks")
    parser.add_argument("--scaling", action="store_true", help="Run scaling analysis")
    parser.add_argument("--prompts", action="store_true", help="Run system prompt comparison")

    args = parser.parse_args()

    if args.scaling:
        run_scaling_analysis(args.model, skip_vllm=args.skip_vllm)
    elif args.prompts:
        run_system_prompt_comparison(args.model, args.num_queries)
    else:
        run_full_comparison(
            args.model,
            args.num_queries,
            skip_vllm=args.skip_vllm,
        )
