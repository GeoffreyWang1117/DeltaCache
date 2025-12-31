"""Compare DeltaCache with vLLM prefix caching.

NOTE: vLLM comparison requires specific environment setup:
- vLLM 0.12+ installed
- Dedicated GPU without other processes
- NCCL network configuration

If vLLM fails with socket timeout issues, use the HuggingFace baseline
comparison (baseline_comparison in experiment results) instead.
"""

import json
import time
import torch
import gc
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
from transformers import AutoTokenizer

# vLLM imports
try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    print("WARNING: vLLM not available")

# DeltaCache imports
from deltacache import DeltaCacheConfig, DeltaCacheManager
from deltacache.engine.incremental import IncrementalEngine
from deltacache.hf_integration import LlamaStyleAdapter

RESULTS_DIR = Path(__file__).parent / "results" / "paper"


@dataclass
class ComparisonResult:
    """Result from comparison benchmark."""
    system: str  # 'vllm', 'vllm_prefix', 'deltacache'
    model_name: str
    num_queries: int
    total_time_ms: float
    avg_latency_ms: float
    throughput_tok_s: float
    prefill_time_ms: Optional[float] = None


def cleanup_gpu():
    """Clean up GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def benchmark_vllm(
    model_name: str,
    prompts: List[str],
    enable_prefix_caching: bool = False,
) -> ComparisonResult:
    """Benchmark vLLM with or without prefix caching."""
    import os
    # Disable distributed for simple single-GPU usage
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    print(f"\n{'='*60}")
    print(f"vLLM Benchmark (prefix_caching={enable_prefix_caching})")
    print(f"Model: {model_name}")
    print(f"Prompts: {len(prompts)}")
    print(f"{'='*60}")

    # Initialize vLLM with simpler settings
    llm = LLM(
        model=model_name,
        enable_prefix_caching=enable_prefix_caching,
        gpu_memory_utilization=0.3,  # Leave room for other processes
        max_model_len=256,  # Smaller context
        enforce_eager=True,  # Disable CUDA graphs for simplicity
    )

    sampling_params = SamplingParams(
        max_tokens=1,  # Just measure prefill
        temperature=0.0,
    )

    # Warmup
    print("Warming up...")
    _ = llm.generate(prompts[:1], sampling_params)

    # Benchmark
    print("Running benchmark...")
    start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    total_time = (time.perf_counter() - start_time) * 1000  # ms

    # Calculate metrics
    tokenizer = llm.get_tokenizer()
    total_tokens = sum(len(tokenizer.encode(p)) for p in prompts)

    avg_latency = total_time / len(prompts)
    throughput = total_tokens / (total_time / 1000)

    system_name = "vllm_prefix" if enable_prefix_caching else "vllm"

    result = ComparisonResult(
        system=system_name,
        model_name=model_name,
        num_queries=len(prompts),
        total_time_ms=total_time,
        avg_latency_ms=avg_latency,
        throughput_tok_s=throughput,
    )

    print(f"Total time: {total_time:.2f}ms")
    print(f"Avg latency: {avg_latency:.2f}ms")
    print(f"Throughput: {throughput:.0f} tok/s")

    # Cleanup
    del llm
    cleanup_gpu()

    return result


def benchmark_deltacache(
    model_name: str,
    prompts: List[str],
) -> ComparisonResult:
    """Benchmark DeltaCache."""
    print(f"\n{'='*60}")
    print(f"DeltaCache Benchmark")
    print(f"Model: {model_name}")
    print(f"Prompts: {len(prompts)}")
    print(f"{'='*60}")

    # Initialize DeltaCache
    adapter = LlamaStyleAdapter(model_name)
    config = DeltaCacheConfig.for_model(model_name)
    manager = DeltaCacheManager(config)
    engine = IncrementalEngine(adapter, manager)

    tokenizer = adapter.tokenizer

    # Warmup
    print("Warming up...")
    tokens = tokenizer.encode(prompts[0])
    _ = engine.compute_incremental(tokens)
    manager.clear()

    # Benchmark
    print("Running benchmark...")
    total_tokens = 0
    start_time = time.perf_counter()

    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        result = engine.compute_incremental(tokens)
        total_tokens += len(tokens)

    total_time = (time.perf_counter() - start_time) * 1000

    avg_latency = total_time / len(prompts)
    throughput = total_tokens / (total_time / 1000)

    result = ComparisonResult(
        system="deltacache",
        model_name=model_name,
        num_queries=len(prompts),
        total_time_ms=total_time,
        avg_latency_ms=avg_latency,
        throughput_tok_s=throughput,
    )

    print(f"Total time: {total_time:.2f}ms")
    print(f"Avg latency: {avg_latency:.2f}ms")
    print(f"Throughput: {throughput:.0f} tok/s")

    # Cleanup
    del engine, manager, adapter
    cleanup_gpu()

    return result


def create_system_prompt_test(
    system_prompt: str,
    user_queries: List[str],
) -> List[str]:
    """Create prompts with shared system prompt."""
    return [f"{system_prompt}\n\nUser: {q}\nAssistant:" for q in user_queries]


def run_comparison(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """Run full comparison between vLLM and DeltaCache."""
    print("\n" + "="*60)
    print("vLLM vs DeltaCache Comparison")
    print("="*60)

    # Create test prompts
    system_prompt = """You are a helpful AI assistant. You provide accurate, concise answers.
You always think step by step and explain your reasoning clearly.
You are friendly and professional in your responses."""

    user_queries = [
        "What is the capital of France?",
        "How does photosynthesis work?",
        "Explain the theory of relativity.",
        "What are prime numbers?",
        "Who wrote Romeo and Juliet?",
        "What is machine learning?",
        "How do airplanes fly?",
        "What causes the seasons?",
        "Explain quantum computing.",
        "What is the speed of light?",
    ]

    prompts = create_system_prompt_test(system_prompt, user_queries)

    results = []

    # Test 1: vLLM without prefix caching
    try:
        result = benchmark_vllm(model_name, prompts, enable_prefix_caching=False)
        results.append(result)
    except Exception as e:
        print(f"vLLM (no prefix) failed: {e}")

    # Test 2: vLLM with prefix caching
    try:
        result = benchmark_vllm(model_name, prompts, enable_prefix_caching=True)
        results.append(result)
    except Exception as e:
        print(f"vLLM (prefix) failed: {e}")

    # Test 3: DeltaCache
    try:
        result = benchmark_deltacache(model_name, prompts)
        results.append(result)
    except Exception as e:
        print(f"DeltaCache failed: {e}")

    # Print summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)

    print(f"\n{'System':<20} {'Latency (ms)':<15} {'Throughput (tok/s)':<20}")
    print("-"*55)

    for r in results:
        print(f"{r.system:<20} {r.avg_latency_ms:<15.2f} {r.throughput_tok_s:<20.0f}")

    # Calculate speedups
    if len(results) >= 2:
        vllm_base = next((r for r in results if r.system == "vllm"), None)
        vllm_prefix = next((r for r in results if r.system == "vllm_prefix"), None)
        deltacache = next((r for r in results if r.system == "deltacache"), None)

        print("\nSpeedups:")
        if vllm_base and vllm_prefix:
            speedup = vllm_base.avg_latency_ms / vllm_prefix.avg_latency_ms
            print(f"  vLLM prefix vs vLLM base: {speedup:.2f}x")
        if vllm_base and deltacache:
            speedup = vllm_base.avg_latency_ms / deltacache.avg_latency_ms
            print(f"  DeltaCache vs vLLM base: {speedup:.2f}x")
        if vllm_prefix and deltacache:
            speedup = vllm_prefix.avg_latency_ms / deltacache.avg_latency_ms
            print(f"  DeltaCache vs vLLM prefix: {speedup:.2f}x")

    # Save results
    results_dict = {
        "model": model_name,
        "num_queries": len(prompts),
        "system_prompt_length": len(system_prompt.split()),
        "results": [
            {
                "system": r.system,
                "total_time_ms": r.total_time_ms,
                "avg_latency_ms": r.avg_latency_ms,
                "throughput_tok_s": r.throughput_tok_s,
            }
            for r in results
        ],
    }

    output_path = RESULTS_DIR / "vllm_comparison.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


def run_scaling_comparison(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """Compare scaling behavior with number of queries."""
    print("\n" + "="*60)
    print("Scaling Comparison: vLLM vs DeltaCache")
    print("="*60)

    system_prompt = "You are a helpful AI assistant."

    base_queries = [
        "What is 2+2?",
        "What color is the sky?",
        "Who is Einstein?",
        "What is water?",
        "How many days in a week?",
    ]

    query_counts = [5, 10, 20]
    all_results = []

    for n in query_counts:
        print(f"\n--- {n} queries ---")

        # Repeat queries to get desired count
        queries = (base_queries * ((n // len(base_queries)) + 1))[:n]
        prompts = create_system_prompt_test(system_prompt, queries)

        # vLLM with prefix caching
        try:
            result = benchmark_vllm(model_name, prompts, enable_prefix_caching=True)
            all_results.append({"n": n, "system": "vllm_prefix", **result.__dict__})
        except Exception as e:
            print(f"vLLM failed: {e}")

        # DeltaCache
        try:
            result = benchmark_deltacache(model_name, prompts)
            all_results.append({"n": n, "system": "deltacache", **result.__dict__})
        except Exception as e:
            print(f"DeltaCache failed: {e}")

    # Print scaling summary
    print("\n" + "="*60)
    print("SCALING SUMMARY")
    print("="*60)

    print(f"\n{'Queries':<10} {'System':<15} {'Throughput (tok/s)':<20}")
    print("-"*45)

    for r in all_results:
        print(f"{r['n']:<10} {r['system']:<15} {r['throughput_tok_s']:<20.0f}")

    # Save results
    output_path = RESULTS_DIR / "vllm_scaling.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return all_results


if __name__ == "__main__":
    if not HAS_VLLM:
        print("vLLM is required for this comparison")
        exit(1)

    # Run comparison
    run_comparison()

    # Run scaling test
    # run_scaling_comparison()
