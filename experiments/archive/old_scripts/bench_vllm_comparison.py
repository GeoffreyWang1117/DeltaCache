"""DeltaCache vs vLLM Automatic Prefix Caching comparison.

Fair comparison on same hardware, same model, same prompts.
Uses TinyLlama-1.1B-Chat since both systems can run it on RTX 3090.

Run: CUDA_VISIBLE_DEVICES=1 python experiments/vllm_apc_comparison.py
"""

import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path(__file__).parent / "results" / "paper"

# Scenarios with fixed prefix + varying suffix
SCENARIOS = {
    "rag_short": {
        "prefix": (
            "You are a helpful assistant. Based on the following document, answer the question.\n\n"
            "Document: Python is a high-level programming language created by Guido van Rossum. "
            "It supports multiple paradigms including object-oriented, functional, and procedural "
            "programming. Python is known for its readable syntax and comprehensive standard library. "
            "Key features include dynamic typing, garbage collection, and support for modules and packages.\n\n"
        ),
        "suffixes": [
            "Question: Who created Python?\nAnswer:",
            "Question: What programming paradigms does Python support?\nAnswer:",
            "Question: What is Python known for?\nAnswer:",
            "Question: What are Python's key features?\nAnswer:",
            "Question: Is Python statically or dynamically typed?\nAnswer:",
        ],
    },
    "rag_long": {
        "prefix": (
            "You are an expert programming assistant. Based on the comprehensive documentation below, "
            "provide a detailed and accurate answer to the question.\n\n"
            "# Python Programming Complete Guide\n\n"
            "## Chapter 1: Introduction\n"
            "Python is a high-level, interpreted programming language created by Guido van Rossum. "
            "First released in 1991, Python emphasizes code readability with significant indentation. "
            "It is dynamically typed and garbage-collected, supporting multiple paradigms including "
            "structured, object-oriented, and functional programming.\n\n"
            "## Chapter 2: Data Types\n"
            "Python has several built-in data types: integers, floats, strings, lists, tuples, sets, "
            "and dictionaries. Variables don't need explicit declaration. Python supports complex numbers, "
            "bytes, bytearrays, and memoryviews. Type hints were added in Python 3.5 via PEP 484.\n\n"
            "## Chapter 3: Control Flow\n"
            "Python uses if/elif/else for conditional execution, for and while loops for iteration, "
            "and try/except/finally for exception handling. List comprehensions provide concise "
            "iteration syntax. The match/case statement was added in Python 3.10.\n\n"
            "## Chapter 4: Functions and Modules\n"
            "Functions are defined with the def keyword. Python supports default arguments, variable "
            "arguments (*args, **kwargs), lambda functions, and decorators. Modules organize code into "
            "reusable files. The import system supports absolute and relative imports.\n\n"
            "## Chapter 5: Object-Oriented Programming\n"
            "Python supports classes with inheritance, encapsulation, and polymorphism. Special methods "
            "(__init__, __str__, __repr__) customize object behavior. Multiple inheritance is supported "
            "with Method Resolution Order (MRO). Abstract base classes from abc module define interfaces.\n\n"
            "## Chapter 6: Standard Library Highlights\n"
            "os and sys for system interaction, json for serialization, re for regular expressions, "
            "collections for advanced data structures, itertools for efficient iteration, functools "
            "for higher-order functions, pathlib for path manipulation, typing for type annotations, "
            "dataclasses for data containers, and asyncio for asynchronous programming.\n\n"
        ),
        "suffixes": [
            "Question: When was Python first released and who created it?\nAnswer:",
            "Question: What data types does Python support?\nAnswer:",
            "Question: How does exception handling work in Python?\nAnswer:",
            "Question: Explain decorators in Python.\nAnswer:",
            "Question: What is the Method Resolution Order?\nAnswer:",
            "Question: Name five modules from Python's standard library.\nAnswer:",
            "Question: What was added in Python 3.10?\nAnswer:",
            "Question: How do list comprehensions work?\nAnswer:",
        ],
    },
    "system_prompt": {
        "prefix": (
            "You are Claude, an AI assistant made by Anthropic. You are helpful, harmless, and honest. "
            "You should follow these guidelines:\n"
            "1. Always be truthful and acknowledge uncertainty when you don't know something.\n"
            "2. Be concise but thorough in your responses.\n"
            "3. If asked to do something harmful, politely decline and explain why.\n"
            "4. Use markdown formatting when appropriate.\n"
            "5. Cite sources when making factual claims.\n\n"
        ),
        "suffixes": [
            "User: What is the capital of France?\nAssistant:",
            "User: Explain quantum computing in simple terms.\nAssistant:",
            "User: Write a haiku about programming.\nAssistant:",
            "User: What are the benefits of exercise?\nAssistant:",
            "User: How does photosynthesis work?\nAssistant:",
        ],
    },
}


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()



def benchmark_deltacache(model_name: str, scenarios: dict) -> dict:
    """Benchmark DeltaCache on same scenarios."""
    from deltacache import DeltaCacheManager, DeltaCacheConfig
    from deltacache.hf_integration import LlamaStyleAdapter

    print("\n  --- DeltaCache ---")

    clear_gpu()

    # Load model via from_pretrained (correct API)
    adapter = LlamaStyleAdapter.from_pretrained(model_name, dtype=torch.float16)
    config = DeltaCacheConfig.for_model(
        "llama-7b" if "llama" in model_name.lower() else "mistral-7b",
        device="cuda",
        eviction_policy="tiered",
    )
    manager = DeltaCacheManager(config)

    results = {}

    for scenario_name, scenario in scenarios.items():
        prefix = scenario["prefix"]
        suffixes = scenario["suffixes"]

        # Tokenize
        tokenizer = adapter.tokenizer
        prefix_tokens = tokenizer.encode(prefix)
        prompts_tokens = [
            tokenizer.encode(prefix + suffix) for suffix in suffixes
        ]

        # Cold run: first query, no cache
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = manager.compute_incremental(
            prompts_tokens[0], adapter.compute_kv
        )
        torch.cuda.synchronize()
        cold_time = time.perf_counter() - t0

        # Warm runs: subsequent queries reuse cached prefix
        warm_times = []
        for tokens in prompts_tokens[1:]:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = manager.compute_incremental(tokens, adapter.compute_kv)
            torch.cuda.synchronize()
            warm_times.append(time.perf_counter() - t0)

        avg_warm = sum(warm_times) / len(warm_times) if warm_times else 0
        cache_stats = manager.get_stats()

        print(f"    {scenario_name}: cold={cold_time*1000:.1f}ms, "
              f"warm_avg={avg_warm*1000:.1f}ms, "
              f"hit_rate={cache_stats.get('hit_rate', 0):.1%}, "
              f"token_reuse={cache_stats.get('token_reuse_rate', 0):.1%}")

        if cold_time > 0 and avg_warm > 0:
            speedup = cold_time / avg_warm
        else:
            speedup = 0

        results[scenario_name] = {
            "cold_ms": cold_time * 1000,
            "warm_avg_ms": avg_warm * 1000,
            "warm_times_ms": [t * 1000 for t in warm_times],
            "speedup_vs_cold": speedup,
            "prefix_tokens": len(prefix_tokens),
            "cache_stats": cache_stats,
        }

        # Reset for next scenario
        manager.clear()

    del adapter, manager
    clear_gpu()

    return results


def compute_comparison(vllm_results: dict, dc_results: dict) -> dict:
    """Compute comparison metrics between vLLM and DeltaCache."""
    comparison = {}

    vllm_apc = vllm_results.get("apc_on", {})
    vllm_no_apc = vllm_results.get("apc_off", {})

    for scenario in dc_results:
        dc = dc_results[scenario]
        vllm_on = vllm_apc.get(scenario, {})
        vllm_off = vllm_no_apc.get(scenario, {})

        comp = {
            "deltacache_warm_ms": dc.get("warm_avg_ms", 0),
            "vllm_apc_warm_ms": vllm_on.get("warm_avg_ms", 0),
            "vllm_no_apc_warm_ms": vllm_off.get("warm_avg_ms", 0),
        }

        # DeltaCache vs vLLM APC
        if comp["vllm_apc_warm_ms"] > 0 and comp["deltacache_warm_ms"] > 0:
            comp["dc_vs_vllm_apc"] = comp["vllm_apc_warm_ms"] / comp["deltacache_warm_ms"]

        # DeltaCache vs vLLM without APC
        if comp["vllm_no_apc_warm_ms"] > 0 and comp["deltacache_warm_ms"] > 0:
            comp["dc_vs_vllm_no_apc"] = comp["vllm_no_apc_warm_ms"] / comp["deltacache_warm_ms"]

        comparison[scenario] = comp

    return comparison


def _run_vllm_single(model_name: str, apc_enabled: bool, scenarios: dict) -> dict:
    """Run a single vLLM benchmark (designed for subprocess isolation)."""
    from vllm import LLM, SamplingParams

    mode = "apc_on" if apc_enabled else "apc_off"
    print(f"\n  --- vLLM {mode} ---")

    llm = LLM(
        model=model_name,
        enable_prefix_caching=apc_enabled,
        max_model_len=2048,
        gpu_memory_utilization=0.60,
        dtype="float16",
        enforce_eager=True,
    )
    sampling_params = SamplingParams(max_tokens=32, temperature=0.0)

    mode_results = {}

    for scenario_name, scenario in scenarios.items():
        prefix = scenario["prefix"]
        suffixes = scenario["suffixes"]
        prompts = [prefix + suffix for suffix in suffixes]

        # Warmup
        _ = llm.generate(prompts[:1], sampling_params)

        # Cold run
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = llm.generate(prompts[:1], sampling_params)
        torch.cuda.synchronize()
        cold_time = time.perf_counter() - t0

        # Warm runs
        warm_times = []
        for prompt in prompts[1:]:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = llm.generate([prompt], sampling_params)
            torch.cuda.synchronize()
            warm_times.append(time.perf_counter() - t0)

        # Batch run
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = llm.generate(prompts, sampling_params)
        torch.cuda.synchronize()
        batch_time = time.perf_counter() - t0

        avg_warm = sum(warm_times) / len(warm_times) if warm_times else 0
        print(f"    {scenario_name}: cold={cold_time*1000:.1f}ms, "
              f"warm_avg={avg_warm*1000:.1f}ms, batch={batch_time*1000:.1f}ms")

        mode_results[scenario_name] = {
            "cold_ms": cold_time * 1000,
            "warm_avg_ms": avg_warm * 1000,
            "warm_times_ms": [t * 1000 for t in warm_times],
            "batch_ms": batch_time * 1000,
            "num_prompts": len(prompts),
        }

    return mode_results


def _subprocess_runner(func_name: str, args: tuple, result_path: str):
    """Generic subprocess runner that saves results to a JSON file."""
    import subprocess as _sp

    # Build a Python command that imports this module and runs the function
    script = f"""
import sys, json
sys.path.insert(0, '{Path(__file__).parent.parent}')
sys.path.insert(0, '{Path(__file__).parent}')
from vllm_apc_comparison import {func_name}, SCENARIOS
import torch
result = {func_name}({', '.join(repr(a) for a in args)}, SCENARIOS)
with open('{result_path}', 'w') as f:
    json.dump(result, f, indent=2, default=str)
"""
    env = os.environ.copy()
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env["VLLM_HOST_IP"] = "127.0.0.1"

    proc = _sp.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, env=env, timeout=300,
    )
    if proc.returncode != 0:
        print(f"  Subprocess stderr:\n{proc.stderr[-500:]}")
        raise RuntimeError(f"Subprocess failed (exit {proc.returncode})")
    print(proc.stdout, end="")


def main():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    print(f"=== DeltaCache vs vLLM APC Comparison ===")
    print(f"Model: {model_name}")
    print(f"GPU: {torch.cuda.get_device_name()}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {
        "model": model_name,
        "gpu": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
    }

    vllm_results = {}

    # Run vLLM APC off in subprocess (prevents GPU memory leak)
    print("\n[1/4] Running vLLM (APC off) in subprocess...")
    tmp_apc_off = RESULTS_DIR / "_tmp_vllm_apc_off.json"
    try:
        _subprocess_runner("_run_vllm_single", (model_name, False), str(tmp_apc_off))
        with open(tmp_apc_off) as f:
            vllm_results["apc_off"] = json.load(f)
        tmp_apc_off.unlink(missing_ok=True)
    except Exception as e:
        print(f"  vLLM (APC off) failed: {e}")
        vllm_results["apc_off"] = {"error": str(e)}

    # Run vLLM APC on in subprocess
    print("\n[2/4] Running vLLM (APC on) in subprocess...")
    tmp_apc_on = RESULTS_DIR / "_tmp_vllm_apc_on.json"
    try:
        _subprocess_runner("_run_vllm_single", (model_name, True), str(tmp_apc_on))
        with open(tmp_apc_on) as f:
            vllm_results["apc_on"] = json.load(f)
        tmp_apc_on.unlink(missing_ok=True)
    except Exception as e:
        print(f"  vLLM (APC on) failed: {e}")
        vllm_results["apc_on"] = {"error": str(e)}

    all_results["vllm"] = vllm_results

    # Run DeltaCache benchmarks (in-process, lightweight)
    print("\n[3/4] Running DeltaCache benchmarks...")
    try:
        dc_results = benchmark_deltacache(model_name, SCENARIOS)
        all_results["deltacache"] = dc_results
    except Exception as e:
        print(f"  DeltaCache benchmark failed: {e}")
        all_results["deltacache"] = {"error": str(e)}
        dc_results = {}

    # Comparison
    print("\n[4/4] Computing comparison...")
    if vllm_results and dc_results and not isinstance(vllm_results.get("apc_on"), dict) or "error" not in vllm_results.get("apc_on", {}):
        comparison = compute_comparison(vllm_results, dc_results)
        all_results["comparison"] = comparison

        print("\n=== Comparison Summary ===")
        print(f"{'Scenario':<20} {'DC warm(ms)':<14} {'vLLM APC(ms)':<14} {'vLLM noAPC(ms)':<16} {'DC vs APC':<12}")
        print("-" * 76)
        for scenario, comp in comparison.items():
            dc_ms = comp.get("deltacache_warm_ms", 0)
            apc_ms = comp.get("vllm_apc_warm_ms", 0)
            no_apc_ms = comp.get("vllm_no_apc_warm_ms", 0)
            ratio = comp.get("dc_vs_vllm_apc", 0)
            print(f"{scenario:<20} {dc_ms:<14.1f} {apc_ms:<14.1f} {no_apc_ms:<16.1f} {ratio:<12.2f}x")

    # Save
    output_path = RESULTS_DIR / "vllm_apc_comparison.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
