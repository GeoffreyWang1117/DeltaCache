"""Fixed vLLM comparison benchmark for ICML 2026 submission.

This script properly compares:
- HuggingFace baseline (no caching)
- vLLM without prefix caching
- vLLM with prefix caching
- DeltaCache with prefix tree

Key fixes:
1. Proper vLLM initialization with isolated GPU
2. Correct measurement methodology
3. Multiple runs with warm-up
"""

import os
import gc
import json
import time
import statistics
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

import torch
from tqdm import tqdm

# Ensure single GPU usage for fair comparison
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from deltacache import DeltaCacheManager, DeltaCacheConfig
from deltacache.hf_integration import LlamaStyleAdapter

# Check vLLM availability
try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    print("WARNING: vLLM not available")

RESULTS_DIR = Path(__file__).parent / "results" / "paper"


def clear_gpu():
    """Clear GPU memory aggressively."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_gpu_mem_mb():
    """Get current GPU memory in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0


@dataclass
class BenchResult:
    """Benchmark result."""
    system: str
    model: str
    n_queries: int
    latency_mean_ms: float
    latency_std_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    throughput_tok_s: float
    memory_mb: float
    cache_hit_rate: float = 0.0
    token_reuse_rate: float = 0.0


def compute_percentile(data: List[float], p: float) -> float:
    """Compute percentile."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p / 100)
    return sorted_data[min(idx, len(sorted_data) - 1)]


# =============================================================================
# Test Prompts - System prompt + diverse queries
# =============================================================================

SYSTEM_PROMPT = """You are a helpful, accurate, and concise AI assistant.
You provide clear explanations and think step by step when needed.
Always be professional and informative in your responses."""

QUERIES = [
    "What is the capital of France?",
    "Explain how photosynthesis works.",
    "What is machine learning?",
    "Describe the water cycle.",
    "How do computers store data?",
    "What causes the seasons?",
    "Explain the theory of relativity simply.",
    "What are prime numbers?",
    "How does the internet work?",
    "What is DNA?",
    "Explain quantum computing basics.",
    "How do vaccines work?",
    "What is climate change?",
    "Describe how airplanes fly.",
    "What is a black hole?",
    "How does encryption work?",
    "What causes earthquakes?",
    "Explain neural networks.",
    "What is the Big Bang theory?",
    "How do batteries store energy?",
    "What is artificial intelligence?",
    "Explain the greenhouse effect.",
    "How does GPS work?",
    "What is dark matter?",
    "Explain how the brain processes information.",
    "What causes thunder and lightning?",
    "How do search engines work?",
    "What is blockchain technology?",
    "Explain the concept of infinity.",
    "How does 3D printing work?",
]


def create_prompts(n_queries: int) -> List[str]:
    """Create prompts with shared system prefix."""
    queries = (QUERIES * ((n_queries // len(QUERIES)) + 1))[:n_queries]
    return [f"{SYSTEM_PROMPT}\n\nUser: {q}\nAssistant:" for q in queries]


# =============================================================================
# Benchmark Functions
# =============================================================================

def benchmark_hf_baseline(
    model_name: str,
    prompts: List[str],
    n_warmup: int = 3,
    device: str = "cuda",
) -> BenchResult:
    """Benchmark HuggingFace without caching."""
    print(f"\n{'='*60}")
    print("HuggingFace Baseline (No Cache)")
    print(f"{'='*60}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    # Warmup
    print(f"Warming up ({n_warmup} iterations)...")
    for i in range(n_warmup):
        with torch.no_grad():
            inputs = tokenizer(prompts[i % len(prompts)], return_tensors="pt").to(device)
            _ = model(**inputs, use_cache=False)

    clear_gpu()
    torch.cuda.reset_peak_memory_stats()

    # Benchmark
    print("Running benchmark...")
    latencies = []
    total_tokens = 0

    for prompt in tqdm(prompts, desc="  HF Baseline"):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        n_tokens = inputs["input_ids"].shape[1]
        total_tokens += n_tokens

        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            _ = model(**inputs, use_cache=False)

        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

    mem_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    total_time_s = sum(latencies) / 1000

    result = BenchResult(
        system="hf_baseline",
        model=model_name,
        n_queries=len(prompts),
        latency_mean_ms=statistics.mean(latencies),
        latency_std_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0,
        latency_p50_ms=compute_percentile(latencies, 50),
        latency_p95_ms=compute_percentile(latencies, 95),
        throughput_tok_s=total_tokens / total_time_s,
        memory_mb=mem_peak,
    )

    print(f"  Mean latency: {result.latency_mean_ms:.2f} ms")
    print(f"  Throughput: {result.throughput_tok_s:.0f} tok/s")

    del model, tokenizer
    clear_gpu()

    return result


def benchmark_vllm(
    model_name: str,
    prompts: List[str],
    enable_prefix_caching: bool,
    n_warmup: int = 3,
) -> Optional[BenchResult]:
    """Benchmark vLLM with or without prefix caching."""
    if not HAS_VLLM:
        print("vLLM not available, skipping...")
        return None

    mode = "with prefix caching" if enable_prefix_caching else "without prefix caching"
    print(f"\n{'='*60}")
    print(f"vLLM ({mode})")
    print(f"{'='*60}")

    try:
        # Force single GPU and disable multiprocessing issues
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        llm = LLM(
            model=model_name,
            enable_prefix_caching=enable_prefix_caching,
            gpu_memory_utilization=0.5,
            max_model_len=1024,
            enforce_eager=True,
            trust_remote_code=True,
        )

        sampling_params = SamplingParams(max_tokens=1, temperature=0.0)
        tokenizer = llm.get_tokenizer()

        # Warmup
        print(f"Warming up ({n_warmup} iterations)...")
        for i in range(n_warmup):
            _ = llm.generate([prompts[i % len(prompts)]], sampling_params)

        clear_gpu()
        torch.cuda.reset_peak_memory_stats()

        # Benchmark - measure individual latencies
        print("Running benchmark...")
        latencies = []
        total_tokens = 0

        for prompt in tqdm(prompts, desc=f"  vLLM {'prefix' if enable_prefix_caching else 'base'}"):
            n_tokens = len(tokenizer.encode(prompt))
            total_tokens += n_tokens

            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = llm.generate([prompt], sampling_params)
            torch.cuda.synchronize()

            latencies.append((time.perf_counter() - start) * 1000)

        mem_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        total_time_s = sum(latencies) / 1000

        system_name = "vllm_prefix" if enable_prefix_caching else "vllm_baseline"

        result = BenchResult(
            system=system_name,
            model=model_name,
            n_queries=len(prompts),
            latency_mean_ms=statistics.mean(latencies),
            latency_std_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0,
            latency_p50_ms=compute_percentile(latencies, 50),
            latency_p95_ms=compute_percentile(latencies, 95),
            throughput_tok_s=total_tokens / total_time_s,
            memory_mb=mem_peak,
        )

        print(f"  Mean latency: {result.latency_mean_ms:.2f} ms")
        print(f"  Throughput: {result.throughput_tok_s:.0f} tok/s")

        del llm
        clear_gpu()

        return result

    except Exception as e:
        print(f"  vLLM benchmark failed: {e}")
        clear_gpu()
        return None


def benchmark_deltacache(
    model_name: str,
    prompts: List[str],
    n_warmup: int = 3,
    device: str = "cuda",
    use_quantization: bool = False,
) -> BenchResult:
    """Benchmark DeltaCache with prefix caching."""
    print(f"\n{'='*60}")
    print("DeltaCache (Incremental KV Computation)")
    print(f"{'='*60}")

    # Load model
    adapter = LlamaStyleAdapter.from_pretrained(
        model_name,
        device=device,
        load_in_8bit=use_quantization,
    )

    config = DeltaCacheConfig.for_model(model_name)
    config.device = device

    tokenizer = adapter.tokenizer

    # Warmup
    print(f"Warming up ({n_warmup} iterations)...")
    manager = DeltaCacheManager(config)
    for i in range(n_warmup):
        tokens = tokenizer.encode(prompts[i % len(prompts)])
        _ = manager.compute_incremental(tokens, adapter.compute_kv)
    manager.clear()
    del manager

    clear_gpu()
    torch.cuda.reset_peak_memory_stats()

    # Benchmark
    print("Running benchmark...")
    manager = DeltaCacheManager(config)

    latencies = []
    total_tokens = 0
    matched_tokens = 0
    cache_hits = 0

    for prompt in tqdm(prompts, desc="  DeltaCache"):
        tokens = tokenizer.encode(prompt)
        n_tokens = len(tokens)
        total_tokens += n_tokens

        torch.cuda.synchronize()
        start = time.perf_counter()

        result = manager.compute_incremental(tokens, adapter.compute_kv)

        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

        matched_tokens += result.matched_length
        if result.matched_length > 0:
            cache_hits += 1

    mem_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    total_time_s = sum(latencies) / 1000

    result = BenchResult(
        system="deltacache",
        model=model_name,
        n_queries=len(prompts),
        latency_mean_ms=statistics.mean(latencies),
        latency_std_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0,
        latency_p50_ms=compute_percentile(latencies, 50),
        latency_p95_ms=compute_percentile(latencies, 95),
        throughput_tok_s=total_tokens / total_time_s,
        memory_mb=mem_peak,
        cache_hit_rate=cache_hits / len(prompts),
        token_reuse_rate=matched_tokens / total_tokens,
    )

    print(f"  Mean latency: {result.latency_mean_ms:.2f} ms")
    print(f"  Throughput: {result.throughput_tok_s:.0f} tok/s")
    print(f"  Cache hit rate: {result.cache_hit_rate:.1%}")
    print(f"  Token reuse: {result.token_reuse_rate:.1%}")

    del manager, adapter
    clear_gpu()

    return result


def run_comparison(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    n_queries: int = 30,
    n_runs: int = 3,
    skip_vllm: bool = False,
) -> Dict:
    """Run full comparison across all systems."""
    print("\n" + "#" * 70)
    print("# VLLM COMPARISON BENCHMARK FOR ICML 2026")
    print(f"# Model: {model_name}")
    print(f"# Queries: {n_queries}, Runs: {n_runs}")
    print(f"# Timestamp: {datetime.now().isoformat()}")
    print("#" * 70)

    prompts = create_prompts(n_queries)

    all_results = {
        "metadata": {
            "model": model_name,
            "n_queries": n_queries,
            "n_runs": n_runs,
            "system_prompt_tokens": len(SYSTEM_PROMPT.split()),
            "timestamp": datetime.now().isoformat(),
        },
        "systems": {},
        "summary": {},
    }

    # Run benchmarks multiple times
    for run_idx in range(n_runs):
        print(f"\n{'='*70}")
        print(f"RUN {run_idx + 1} / {n_runs}")
        print(f"{'='*70}")

        # 1. HuggingFace baseline
        hf_result = benchmark_hf_baseline(model_name, prompts)
        if "hf_baseline" not in all_results["systems"]:
            all_results["systems"]["hf_baseline"] = []
        all_results["systems"]["hf_baseline"].append(asdict(hf_result))

        # 2. DeltaCache
        dc_result = benchmark_deltacache(model_name, prompts)
        if "deltacache" not in all_results["systems"]:
            all_results["systems"]["deltacache"] = []
        all_results["systems"]["deltacache"].append(asdict(dc_result))

        # 3. vLLM (if available and not skipped)
        if not skip_vllm and HAS_VLLM:
            # vLLM without prefix caching
            vllm_base = benchmark_vllm(model_name, prompts, enable_prefix_caching=False)
            if vllm_base:
                if "vllm_baseline" not in all_results["systems"]:
                    all_results["systems"]["vllm_baseline"] = []
                all_results["systems"]["vllm_baseline"].append(asdict(vllm_base))

            # vLLM with prefix caching
            vllm_prefix = benchmark_vllm(model_name, prompts, enable_prefix_caching=True)
            if vllm_prefix:
                if "vllm_prefix" not in all_results["systems"]:
                    all_results["systems"]["vllm_prefix"] = []
                all_results["systems"]["vllm_prefix"].append(asdict(vllm_prefix))

    # Aggregate results
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for system, runs in all_results["systems"].items():
        latencies = [r["latency_mean_ms"] for r in runs]
        throughputs = [r["throughput_tok_s"] for r in runs]

        all_results["summary"][system] = {
            "latency_mean_ms": statistics.mean(latencies),
            "latency_std_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            "throughput_tok_s": statistics.mean(throughputs),
        }

        print(f"\n{system}:")
        print(f"  Latency: {all_results['summary'][system]['latency_mean_ms']:.2f} "
              f"± {all_results['summary'][system]['latency_std_ms']:.2f} ms")
        print(f"  Throughput: {all_results['summary'][system]['throughput_tok_s']:.0f} tok/s")

    # Calculate speedups
    if "hf_baseline" in all_results["summary"]:
        hf_lat = all_results["summary"]["hf_baseline"]["latency_mean_ms"]

        print("\nSpeedups vs HuggingFace Baseline:")
        for system, stats in all_results["summary"].items():
            if system != "hf_baseline":
                speedup = hf_lat / stats["latency_mean_ms"]
                print(f"  {system}: {speedup:.2f}x")
                all_results["summary"][system]["speedup_vs_hf"] = speedup

    # Compare DeltaCache vs vLLM prefix
    if "deltacache" in all_results["summary"] and "vllm_prefix" in all_results["summary"]:
        dc_lat = all_results["summary"]["deltacache"]["latency_mean_ms"]
        vllm_lat = all_results["summary"]["vllm_prefix"]["latency_mean_ms"]

        if dc_lat < vllm_lat:
            print(f"\nDeltaCache is {vllm_lat/dc_lat:.2f}x faster than vLLM prefix caching")
        else:
            print(f"\nvLLM prefix caching is {dc_lat/vllm_lat:.2f}x faster than DeltaCache")

    return all_results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="vLLM Comparison Benchmark")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--n-queries", type=int, default=30)
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--skip-vllm", action="store_true")
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    results = run_comparison(
        model_name=args.model,
        n_queries=args.n_queries,
        n_runs=args.n_runs,
        skip_vllm=args.skip_vllm,
    )

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.output:
        output_path = Path(args.output)
    else:
        model_short = args.model.split("/")[-1].lower().replace("-", "_")
        output_path = RESULTS_DIR / f"vllm_comparison_{model_short}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
