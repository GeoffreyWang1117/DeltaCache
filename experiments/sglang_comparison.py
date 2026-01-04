"""SGLang RadixAttention Comparison for ICML 2026.

This script compares DeltaCache with SGLang's RadixAttention prefix caching.

SGLang uses a radix tree structure for prefix caching, similar in concept to
DeltaCache's approach but implemented at the serving layer.

Note: SGLang requires running a separate server process, so we need to:
1. Start SGLang server
2. Send requests and measure latency
3. Compare with DeltaCache
"""

import os
import gc
import json
import time
import statistics
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import requests

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from deltacache import DeltaCacheManager, DeltaCacheConfig
from deltacache.hf_integration import LlamaStyleAdapter

RESULTS_DIR = Path(__file__).parent / "results" / "paper"


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# =============================================================================
# Test Data
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
]


def create_prompts(n_queries: int) -> List[str]:
    """Create prompts with shared system prefix."""
    queries = (QUERIES * ((n_queries // len(QUERIES)) + 1))[:n_queries]
    return [f"{SYSTEM_PROMPT}\n\nUser: {q}\nAssistant:" for q in queries]


# =============================================================================
# SGLang Benchmark (requires server)
# =============================================================================

def check_sglang_server(port: int = 30000) -> bool:
    """Check if SGLang server is running."""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def benchmark_sglang_api(
    prompts: List[str],
    port: int = 30000,
    n_warmup: int = 3,
) -> Optional[Dict]:
    """Benchmark SGLang via HTTP API."""
    if not check_sglang_server(port):
        print("  SGLang server not running. Please start it first:")
        print(f"  python -m sglang.launch_server --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 --port {port}")
        return None

    print(f"\n{'='*60}")
    print("SGLang RadixAttention Benchmark")
    print(f"{'='*60}")

    url = f"http://localhost:{port}/generate"

    # Warmup
    print(f"Warming up ({n_warmup} iterations)...")
    for i in range(n_warmup):
        payload = {
            "text": prompts[i % len(prompts)],
            "sampling_params": {"max_new_tokens": 1, "temperature": 0},
        }
        requests.post(url, json=payload)

    # Benchmark
    print("Running benchmark...")
    latencies = []

    for prompt in tqdm(prompts, desc="  SGLang"):
        payload = {
            "text": prompt,
            "sampling_params": {"max_new_tokens": 1, "temperature": 0},
        }

        start = time.perf_counter()
        response = requests.post(url, json=payload)
        latencies.append((time.perf_counter() - start) * 1000)

    result = {
        "system": "sglang_radix",
        "n_queries": len(prompts),
        "latency_mean_ms": statistics.mean(latencies),
        "latency_std_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        "latency_p50_ms": sorted(latencies)[len(latencies)//2],
        "latency_p95_ms": sorted(latencies)[int(len(latencies)*0.95)],
    }

    print(f"  Mean latency: {result['latency_mean_ms']:.2f} ms")

    return result


# =============================================================================
# DeltaCache Benchmark (for comparison)
# =============================================================================

def benchmark_deltacache(
    prompts: List[str],
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device: str = "cuda",
    n_warmup: int = 3,
) -> Dict:
    """Benchmark DeltaCache for comparison."""
    print(f"\n{'='*60}")
    print("DeltaCache Benchmark")
    print(f"{'='*60}")

    adapter = LlamaStyleAdapter.from_pretrained(model_name, device=device)
    tokenizer = adapter.tokenizer
    config = DeltaCacheConfig.for_model(model_name)
    config.device = device

    # Warmup
    print(f"Warming up ({n_warmup} iterations)...")
    manager = DeltaCacheManager(config)
    for i in range(n_warmup):
        tokens = tokenizer.encode(prompts[i % len(prompts)])
        _ = manager.compute_incremental(tokens, adapter.compute_kv)
    manager.clear()
    del manager

    clear_gpu()

    # Benchmark
    print("Running benchmark...")
    manager = DeltaCacheManager(config)
    latencies = []
    total_tokens = 0
    matched_tokens = 0

    for prompt in tqdm(prompts, desc="  DeltaCache"):
        tokens = tokenizer.encode(prompt)
        total_tokens += len(tokens)

        torch.cuda.synchronize()
        start = time.perf_counter()
        result = manager.compute_incremental(tokens, adapter.compute_kv)
        torch.cuda.synchronize()

        latencies.append((time.perf_counter() - start) * 1000)
        matched_tokens += result.matched_length

    result = {
        "system": "deltacache",
        "n_queries": len(prompts),
        "latency_mean_ms": statistics.mean(latencies),
        "latency_std_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        "latency_p50_ms": sorted(latencies)[len(latencies)//2],
        "latency_p95_ms": sorted(latencies)[int(len(latencies)*0.95)],
        "token_reuse_rate": matched_tokens / total_tokens,
    }

    print(f"  Mean latency: {result['latency_mean_ms']:.2f} ms")
    print(f"  Token reuse: {result['token_reuse_rate']:.1%}")

    del manager, adapter
    clear_gpu()

    return result


# =============================================================================
# HuggingFace Baseline
# =============================================================================

def benchmark_hf_baseline(
    prompts: List[str],
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device: str = "cuda",
    n_warmup: int = 3,
) -> Dict:
    """Benchmark HuggingFace baseline."""
    print(f"\n{'='*60}")
    print("HuggingFace Baseline")
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

    # Benchmark
    print("Running benchmark...")
    latencies = []

    for prompt in tqdm(prompts, desc="  HF Baseline"):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(**inputs, use_cache=False)
        torch.cuda.synchronize()

        latencies.append((time.perf_counter() - start) * 1000)

    result = {
        "system": "hf_baseline",
        "n_queries": len(prompts),
        "latency_mean_ms": statistics.mean(latencies),
        "latency_std_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        "latency_p50_ms": sorted(latencies)[len(latencies)//2],
        "latency_p95_ms": sorted(latencies)[int(len(latencies)*0.95)],
    }

    print(f"  Mean latency: {result['latency_mean_ms']:.2f} ms")

    del model, tokenizer
    clear_gpu()

    return result


# =============================================================================
# Main Comparison
# =============================================================================

def run_comparison(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    n_queries: int = 30,
    n_runs: int = 3,
    sglang_port: int = 30000,
    device: str = "cuda",
) -> Dict:
    """Run full comparison."""
    print("\n" + "#" * 70)
    print("# SGLang RadixAttention vs DeltaCache Comparison")
    print(f"# Model: {model_name}")
    print(f"# Queries: {n_queries}, Runs: {n_runs}")
    print(f"# Timestamp: {datetime.now().isoformat()}")
    print("#" * 70)

    prompts = create_prompts(n_queries)

    results = {
        "metadata": {
            "model": model_name,
            "n_queries": n_queries,
            "n_runs": n_runs,
            "timestamp": datetime.now().isoformat(),
        },
        "systems": {},
        "summary": {},
    }

    all_results = {"hf_baseline": [], "deltacache": [], "sglang_radix": []}

    for run_idx in range(n_runs):
        print(f"\n{'='*70}")
        print(f"RUN {run_idx + 1} / {n_runs}")
        print(f"{'='*70}")

        # HuggingFace baseline
        hf_result = benchmark_hf_baseline(prompts, model_name, device)
        all_results["hf_baseline"].append(hf_result)

        # DeltaCache
        dc_result = benchmark_deltacache(prompts, model_name, device)
        all_results["deltacache"].append(dc_result)

        # SGLang (if server is running)
        sglang_result = benchmark_sglang_api(prompts, sglang_port)
        if sglang_result:
            all_results["sglang_radix"].append(sglang_result)

    # Aggregate results
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for system, runs in all_results.items():
        if not runs:
            continue

        latencies = [r["latency_mean_ms"] for r in runs]
        results["systems"][system] = {
            "latency_mean_ms": statistics.mean(latencies),
            "latency_std_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        }

        print(f"\n{system}:")
        print(f"  Latency: {results['systems'][system]['latency_mean_ms']:.2f} "
              f"± {results['systems'][system]['latency_std_ms']:.2f} ms")

    # Calculate speedups
    if "hf_baseline" in results["systems"]:
        hf_lat = results["systems"]["hf_baseline"]["latency_mean_ms"]

        print("\nSpeedups vs HuggingFace Baseline:")
        results["speedups"] = {}
        for system, stats in results["systems"].items():
            if system != "hf_baseline":
                speedup = hf_lat / stats["latency_mean_ms"]
                results["speedups"][system] = speedup
                print(f"  {system}: {speedup:.2f}x")

    # Compare DeltaCache vs SGLang
    if "deltacache" in results["systems"] and "sglang_radix" in results["systems"]:
        dc_lat = results["systems"]["deltacache"]["latency_mean_ms"]
        sg_lat = results["systems"]["sglang_radix"]["latency_mean_ms"]

        if dc_lat < sg_lat:
            print(f"\nDeltaCache is {sg_lat/dc_lat:.2f}x faster than SGLang RadixAttention")
        else:
            print(f"\nSGLang RadixAttention is {dc_lat/sg_lat:.2f}x faster than DeltaCache")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SGLang Comparison")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--n-queries", type=int, default=30)
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--sglang-port", type=int, default=30000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-sglang", action="store_true", help="Skip SGLang benchmark")
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    if args.skip_sglang:
        # Just run DeltaCache vs HF comparison
        prompts = create_prompts(args.n_queries)
        results = {
            "metadata": {
                "model": args.model,
                "n_queries": args.n_queries,
                "timestamp": datetime.now().isoformat(),
                "note": "SGLang skipped",
            },
            "systems": {},
        }

        hf_result = benchmark_hf_baseline(prompts, args.model, args.device)
        results["systems"]["hf_baseline"] = hf_result

        dc_result = benchmark_deltacache(prompts, args.model, args.device)
        results["systems"]["deltacache"] = dc_result

        speedup = hf_result["latency_mean_ms"] / dc_result["latency_mean_ms"]
        results["speedups"] = {"deltacache": speedup}

        print(f"\nDeltaCache Speedup: {speedup:.2f}x")

    else:
        results = run_comparison(
            model_name=args.model,
            n_queries=args.n_queries,
            n_runs=args.n_runs,
            sglang_port=args.sglang_port,
            device=args.device,
        )

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = RESULTS_DIR / "sglang_comparison.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
