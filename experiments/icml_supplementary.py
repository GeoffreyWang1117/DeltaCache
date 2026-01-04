"""ICML 2026 Supplementary Experiments.

This script addresses P0 tasks from ICML_TODO.md:
1. Full precision (fp16) Mistral-7B correctness verification
2. Optimized multi-turn experiments with longer system prompts
3. vLLM comparison with NCCL fixes
"""

import os
import gc
import json
import time
import statistics
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from datetime import datetime

import torch
from tqdm import tqdm

# Set NCCL environment variables to avoid distributed communication issues
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from deltacache import DeltaCacheManager, DeltaCacheConfig
from deltacache.hf_integration import LlamaStyleAdapter

RESULTS_DIR = Path(__file__).parent / "results" / "paper"


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_gpu_mem():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 3)
    return 0


# =============================================================================
# P0-2: Full Precision Correctness Verification
# =============================================================================

def run_fp16_correctness_experiment(
    model_name: str = "mistralai/Mistral-7B-v0.1",
    device: str = "cuda",
    n_samples: int = 100,
) -> Dict:
    """Verify 100% correctness with full precision (fp16) model.

    This addresses the issue where 8-bit quantized model only achieved 10% accuracy.
    Full precision should achieve 100% correctness with fp16 numerical tolerance.
    """
    print("\n" + "=" * 70)
    print("P0-2: Full Precision (fp16) Correctness Verification")
    print(f"Model: {model_name}")
    print("=" * 70)

    # Load model in fp16 (no quantization)
    print(f"\nLoading {model_name} in fp16...")
    adapter = LlamaStyleAdapter.from_pretrained(
        model_name,
        device=device,
        load_in_8bit=False,  # Full precision
    )
    tokenizer = adapter.tokenizer
    print(f"Model loaded. GPU memory: {get_gpu_mem():.2f} GB")

    config = DeltaCacheConfig.for_model(model_name)
    config.device = device

    # Test prompts with varying lengths
    test_prompts = [
        # Short prompts
        "The capital of France is",
        "Machine learning is",
        "The theory of relativity states",
        "Python programming language was",
        "Artificial intelligence enables",
        # Medium prompts
        "Deep learning has revolutionized many fields including computer vision and natural language processing. The key innovation is",
        "The transformer architecture, introduced in 2017, has become the foundation of modern language models. Its key component is",
        "Quantum computing harnesses quantum mechanical phenomena to process information in ways that classical computers cannot. The basic unit is",
        # Longer prompts
        "In the field of machine learning, neural networks are computational systems inspired by biological neural networks. They consist of layers of interconnected nodes that process information. The training process involves",
        "The development of large language models has been one of the most significant advances in artificial intelligence. These models are trained on vast amounts of text data and can perform tasks such as",
    ]

    manager = DeltaCacheManager(config)
    results = {
        "model": model_name,
        "precision": "fp16",
        "n_samples": n_samples,
        "samples": [],
        "summary": {}
    }

    n_correct = 0
    max_key_diff = 0
    max_val_diff = 0

    # Expand prompts to reach n_samples
    prompts_expanded = (test_prompts * (n_samples // len(test_prompts) + 1))[:n_samples]

    for i, prompt in enumerate(tqdm(prompts_expanded, desc="  Verifying correctness")):
        tokens = tokenizer.encode(prompt)

        # Get KV via DeltaCache (uses caching)
        dc_result = manager.compute_incremental(tokens, adapter.compute_kv)

        # Get KV via direct computation (ground truth)
        kv_ground_truth = adapter.compute_kv_for_tokens(tokens)

        # Compare with fp16 tolerance
        key_diff = (dc_result.key_cache - kv_ground_truth[0]).abs().max().item()
        val_diff = (dc_result.value_cache - kv_ground_truth[1]).abs().max().item()

        # fp16 tolerance - should be very small for identical computation
        tolerance = 1e-3
        is_correct = key_diff < tolerance and val_diff < tolerance

        if is_correct:
            n_correct += 1

        max_key_diff = max(max_key_diff, key_diff)
        max_val_diff = max(max_val_diff, val_diff)

        results["samples"].append({
            "prompt_len": len(tokens),
            "key_max_diff": key_diff,
            "val_max_diff": val_diff,
            "is_correct": is_correct,
        })

        # Clear cache periodically to test fresh computation
        if (i + 1) % 20 == 0:
            manager.clear()

    del manager
    del adapter
    clear_gpu()

    accuracy = n_correct / len(results["samples"])
    results["summary"] = {
        "total_samples": len(results["samples"]),
        "correct_samples": n_correct,
        "accuracy": accuracy,
        "max_key_diff": max_key_diff,
        "max_val_diff": max_val_diff,
    }

    print(f"\n  Results:")
    print(f"  Accuracy: {accuracy*100:.1f}% ({n_correct}/{len(results['samples'])})")
    print(f"  Max Key Diff: {max_key_diff:.2e}")
    print(f"  Max Val Diff: {max_val_diff:.2e}")

    if accuracy >= 0.99:
        print(f"  ✓ PASS: Correctness verified at fp16 precision!")
    else:
        print(f"  ✗ FAIL: Unexpected errors in fp16 mode")

    return results


# =============================================================================
# P0-3: Optimized Multi-turn Experiment
# =============================================================================

# Longer system prompts for better cache utilization
LONG_SYSTEM_PROMPTS = {
    "assistant": """You are a highly capable AI assistant with expertise in a wide range of topics including science, technology, mathematics, history, arts, and current events. Your responses should be:

1. Accurate and well-researched, drawing on comprehensive knowledge
2. Clear and well-organized, using appropriate formatting when helpful
3. Thoughtful and nuanced, acknowledging complexity and multiple perspectives
4. Engaging and accessible, adapting your communication style to the context
5. Ethical and responsible, declining requests that could cause harm

When answering questions, think step by step and provide thorough explanations. If you're uncertain about something, acknowledge it honestly. Always strive to be helpful while maintaining accuracy and integrity.

You have been trained on data up to 2024 and can discuss a wide range of topics with expertise. You excel at explaining complex concepts in understandable terms.""",

    "coding": """You are an expert software engineer and programming assistant with deep knowledge of:

- Multiple programming languages (Python, JavaScript, TypeScript, Java, C++, Rust, Go, etc.)
- Software architecture and design patterns
- Data structures and algorithms
- Web development (frontend and backend)
- Database design and optimization
- DevOps and cloud infrastructure
- Testing and debugging practices
- Code review and best practices

When helping with code:
1. Write clean, readable, and maintainable code
2. Follow language-specific conventions and best practices
3. Include appropriate error handling
4. Add helpful comments for complex logic
5. Consider performance implications
6. Suggest tests when appropriate
7. Explain your reasoning and trade-offs

You should provide complete, working solutions whenever possible, and explain how to integrate them into existing codebases.""",

    "science": """You are a knowledgeable science tutor with expertise across multiple disciplines:

- Physics: Classical mechanics, thermodynamics, electromagnetism, quantum mechanics, relativity
- Chemistry: Organic, inorganic, physical, biochemistry
- Biology: Molecular biology, genetics, ecology, evolution, physiology
- Earth Science: Geology, meteorology, oceanography, environmental science
- Astronomy: Solar system, stellar evolution, cosmology, observational techniques
- Mathematics: Calculus, linear algebra, statistics, number theory

Your teaching approach:
1. Start with fundamentals before advanced concepts
2. Use analogies and real-world examples
3. Provide step-by-step derivations for mathematical problems
4. Connect topics to broader scientific principles
5. Encourage curiosity and critical thinking
6. Correct misconceptions gently but thoroughly

You make complex scientific concepts accessible while maintaining accuracy and rigor.""",
}

def run_optimized_multiturn_experiment(
    adapter,
    tokenizer,
    config,
    n_runs: int = 3,
) -> Dict:
    """Optimized multi-turn experiment with longer system prompts.

    Key improvements:
    1. Much longer system prompts (300+ tokens) for better cache hit rate
    2. More turns to accumulate context
    3. Realistic conversation patterns
    """
    print("\n" + "=" * 70)
    print("P0-3: Optimized Multi-turn Conversation")
    print("=" * 70)

    # Conversations with long system prompts
    conversations = [
        {
            "system": LONG_SYSTEM_PROMPTS["assistant"],
            "turns": [
                {"user": "What is machine learning?", "assistant": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without explicit programming."},
                {"user": "What are the main types?", "assistant": "The three main types are: supervised learning (labeled data), unsupervised learning (pattern discovery), and reinforcement learning (reward-based)."},
                {"user": "Can you explain neural networks?", "assistant": "Neural networks are computing systems inspired by biological brains. They consist of layers of interconnected nodes (neurons) that process and transmit information."},
                {"user": "How does deep learning relate to this?", "assistant": "Deep learning uses neural networks with many layers (hence 'deep'). This allows them to learn hierarchical representations of data."},
                {"user": "What are some applications?", "assistant": "Applications include image recognition, natural language processing, autonomous vehicles, drug discovery, and recommendation systems."},
                {"user": "What about the future of AI?", "assistant": "The future includes advances in multimodal AI, more efficient models, better reasoning capabilities, and broader real-world applications."},
            ]
        },
        {
            "system": LONG_SYSTEM_PROMPTS["coding"],
            "turns": [
                {"user": "How do I read a file in Python?", "assistant": "Use the open() function with a context manager: with open('file.txt', 'r') as f: content = f.read()"},
                {"user": "What about writing to files?", "assistant": "For writing, use 'w' mode: with open('file.txt', 'w') as f: f.write('content'). Use 'a' to append."},
                {"user": "How do I handle errors?", "assistant": "Use try-except blocks: try: risky_code() except FileNotFoundError: handle_error()"},
                {"user": "Can you show me a complete example?", "assistant": "Here's a complete example with error handling and both read/write operations."},
                {"user": "How about working with JSON?", "assistant": "Use the json module: import json; data = json.load(f) to read, json.dump(data, f) to write."},
                {"user": "What are best practices?", "assistant": "Always use context managers, handle exceptions, use appropriate modes, and consider encoding for text files."},
            ]
        },
        {
            "system": LONG_SYSTEM_PROMPTS["science"],
            "turns": [
                {"user": "Explain photosynthesis.", "assistant": "Photosynthesis is the process by which plants convert light energy into chemical energy (glucose) using CO2 and water."},
                {"user": "What is the chemical equation?", "assistant": "6CO2 + 6H2O + light energy → C6H12O6 + 6O2. Carbon dioxide and water yield glucose and oxygen."},
                {"user": "What are the light reactions?", "assistant": "Light reactions occur in thylakoids, where chlorophyll absorbs light, water is split, and ATP and NADPH are produced."},
                {"user": "And the Calvin cycle?", "assistant": "The Calvin cycle occurs in the stroma, using ATP and NADPH to fix CO2 into glucose through a series of enzyme-catalyzed reactions."},
                {"user": "Why is this important for life?", "assistant": "Photosynthesis produces oxygen and is the primary source of organic compounds and energy for most ecosystems on Earth."},
            ]
        },
    ]

    results = {
        "conversations": [],
        "summary": {}
    }

    all_speedups = []
    all_reuse_rates = []
    all_ttft_speedups = []

    for conv_idx, conv in enumerate(conversations):
        print(f"\n  Conversation {conv_idx + 1}/{len(conversations)}")
        system_tokens = len(tokenizer.encode(conv["system"]))
        print(f"  System prompt: {system_tokens} tokens")

        conv_speedups = []
        conv_reuse_rates = []
        conv_ttft_speedups = []

        for run in range(n_runs):
            # WITH DeltaCache
            manager = DeltaCacheManager(config)
            context = conv["system"]
            dc_latencies = []
            dc_first_latency = None
            total_tokens = 0
            matched_tokens = 0

            for turn_idx, turn in enumerate(conv["turns"]):
                context += f"\n\nUser: {turn['user']}\nAssistant:"
                tokens = tokenizer.encode(context)

                torch.cuda.synchronize()
                start = time.perf_counter()
                result = manager.compute_incremental(tokens, adapter.compute_kv)
                torch.cuda.synchronize()

                latency = (time.perf_counter() - start) * 1000
                dc_latencies.append(latency)

                if turn_idx == 0:
                    dc_first_latency = latency

                total_tokens += len(tokens)
                matched_tokens += result.matched_length

                context += f" {turn['assistant']}"

            del manager
            clear_gpu()

            # WITHOUT cache (baseline)
            context = conv["system"]
            hf_latencies = []
            hf_first_latency = None

            for turn_idx, turn in enumerate(conv["turns"]):
                context += f"\n\nUser: {turn['user']}\nAssistant:"
                tokens = tokenizer.encode(context)

                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = adapter.compute_kv_for_tokens(tokens)
                torch.cuda.synchronize()

                latency = (time.perf_counter() - start) * 1000
                hf_latencies.append(latency)

                if turn_idx == 0:
                    hf_first_latency = latency

                context += f" {turn['assistant']}"

            clear_gpu()

            # Compute metrics
            speedup = sum(hf_latencies) / sum(dc_latencies)
            reuse_rate = matched_tokens / total_tokens
            ttft_speedup = hf_first_latency / dc_first_latency if dc_first_latency > 0 else 1.0

            conv_speedups.append(speedup)
            conv_reuse_rates.append(reuse_rate)
            conv_ttft_speedups.append(ttft_speedup)

        results["conversations"].append({
            "system_prompt_tokens": system_tokens,
            "n_turns": len(conv["turns"]),
            "speedup_mean": statistics.mean(conv_speedups),
            "speedup_std": statistics.stdev(conv_speedups) if len(conv_speedups) > 1 else 0,
            "token_reuse": statistics.mean(conv_reuse_rates),
            "ttft_speedup": statistics.mean(conv_ttft_speedups),
        })

        all_speedups.extend(conv_speedups)
        all_reuse_rates.extend(conv_reuse_rates)
        all_ttft_speedups.extend(conv_ttft_speedups)

        print(f"    Turns: {len(conv['turns'])}, Speedup: {statistics.mean(conv_speedups):.2f}x, "
              f"Reuse: {statistics.mean(conv_reuse_rates)*100:.1f}%, TTFT: {statistics.mean(conv_ttft_speedups):.1f}x")

    results["summary"] = {
        "mean_speedup": statistics.mean(all_speedups),
        "mean_token_reuse": statistics.mean(all_reuse_rates),
        "mean_ttft_speedup": statistics.mean(all_ttft_speedups),
    }

    print(f"\n  Summary:")
    print(f"  Mean Speedup: {results['summary']['mean_speedup']:.2f}x")
    print(f"  Mean Token Reuse: {results['summary']['mean_token_reuse']*100:.1f}%")
    print(f"  Mean TTFT Speedup: {results['summary']['mean_ttft_speedup']:.1f}x")

    return results


# =============================================================================
# P0-1: vLLM Comparison with NCCL Fixes
# =============================================================================

def run_vllm_comparison_offline(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    n_queries: int = 30,
    n_runs: int = 3,
    device: str = "cuda",
) -> Dict:
    """Run vLLM comparison in offline mode to avoid NCCL issues.

    Approach: Test each system separately, then compare results.
    """
    print("\n" + "=" * 70)
    print("P0-1: vLLM Comparison (Offline Mode)")
    print(f"Model: {model_name}")
    print("=" * 70)

    # Check vLLM availability
    try:
        from vllm import LLM, SamplingParams
        HAS_VLLM = True
    except ImportError:
        HAS_VLLM = False
        print("WARNING: vLLM not available, will only run DeltaCache benchmark")

    # Test prompts with shared prefix
    system_prompt = """You are a helpful, accurate, and concise AI assistant.
You provide clear explanations and think step by step when needed.
Always be professional and informative in your responses."""

    queries = [
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
    ] * (n_queries // 10 + 1)
    queries = queries[:n_queries]

    prompts = [f"{system_prompt}\n\nUser: {q}\nAssistant:" for q in queries]

    results = {
        "metadata": {
            "model": model_name,
            "n_queries": n_queries,
            "n_runs": n_runs,
            "system_prompt_tokens": len(system_prompt.split()),
            "timestamp": datetime.now().isoformat(),
        },
        "systems": {},
    }

    # 1. DeltaCache benchmark
    print("\n  [1/3] DeltaCache...")
    adapter = LlamaStyleAdapter.from_pretrained(model_name, device=device)
    tokenizer = adapter.tokenizer
    config = DeltaCacheConfig.for_model(model_name)
    config.device = device

    dc_latencies_all = []
    for run in range(n_runs):
        manager = DeltaCacheManager(config)
        dc_latencies = []
        total_tokens = 0
        matched_tokens = 0

        for prompt in prompts:
            tokens = tokenizer.encode(prompt)
            total_tokens += len(tokens)

            torch.cuda.synchronize()
            start = time.perf_counter()
            result = manager.compute_incremental(tokens, adapter.compute_kv)
            torch.cuda.synchronize()

            dc_latencies.append((time.perf_counter() - start) * 1000)
            matched_tokens += result.matched_length

        dc_latencies_all.extend(dc_latencies)
        del manager
        clear_gpu()

    results["systems"]["deltacache"] = {
        "latency_mean_ms": statistics.mean(dc_latencies_all),
        "latency_std_ms": statistics.stdev(dc_latencies_all),
        "token_reuse_rate": matched_tokens / total_tokens,
    }
    print(f"    Latency: {results['systems']['deltacache']['latency_mean_ms']:.2f} ms")

    del adapter
    clear_gpu()

    # 2. HuggingFace baseline
    print("\n  [2/3] HuggingFace baseline...")
    from transformers import AutoModelForCausalLM, AutoTokenizer as HFTokenizer

    hf_tokenizer = HFTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    hf_model.eval()

    hf_latencies_all = []
    for run in range(n_runs):
        hf_latencies = []
        for prompt in prompts:
            inputs = hf_tokenizer(prompt, return_tensors="pt").to(device)

            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                _ = hf_model(**inputs, use_cache=False)
            torch.cuda.synchronize()

            hf_latencies.append((time.perf_counter() - start) * 1000)

        hf_latencies_all.extend(hf_latencies)
        clear_gpu()

    results["systems"]["hf_baseline"] = {
        "latency_mean_ms": statistics.mean(hf_latencies_all),
        "latency_std_ms": statistics.stdev(hf_latencies_all),
    }
    print(f"    Latency: {results['systems']['hf_baseline']['latency_mean_ms']:.2f} ms")

    del hf_model, hf_tokenizer
    clear_gpu()

    # 3. vLLM (if available)
    if HAS_VLLM:
        print("\n  [3/3] vLLM with prefix caching...")
        try:
            llm = LLM(
                model=model_name,
                enable_prefix_caching=True,
                gpu_memory_utilization=0.5,
                max_model_len=1024,
                enforce_eager=True,
                trust_remote_code=True,
            )

            sampling_params = SamplingParams(max_tokens=1, temperature=0.0)

            # Warmup
            for i in range(3):
                _ = llm.generate([prompts[i]], sampling_params)

            vllm_latencies_all = []
            for run in range(n_runs):
                vllm_latencies = []
                for prompt in prompts:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    _ = llm.generate([prompt], sampling_params)
                    torch.cuda.synchronize()

                    vllm_latencies.append((time.perf_counter() - start) * 1000)

                vllm_latencies_all.extend(vllm_latencies)

            results["systems"]["vllm_prefix"] = {
                "latency_mean_ms": statistics.mean(vllm_latencies_all),
                "latency_std_ms": statistics.stdev(vllm_latencies_all),
            }
            print(f"    Latency: {results['systems']['vllm_prefix']['latency_mean_ms']:.2f} ms")

            del llm
            clear_gpu()

        except Exception as e:
            print(f"    vLLM failed: {e}")
            results["systems"]["vllm_prefix"] = {"error": str(e)}

    # Calculate speedups
    hf_lat = results["systems"]["hf_baseline"]["latency_mean_ms"]

    results["speedups"] = {}
    for system, data in results["systems"].items():
        if system != "hf_baseline" and "latency_mean_ms" in data:
            results["speedups"][system] = hf_lat / data["latency_mean_ms"]

    print("\n  Speedups vs HuggingFace:")
    for system, speedup in results["speedups"].items():
        print(f"    {system}: {speedup:.2f}x")

    return results


# =============================================================================
# Main Runner
# =============================================================================

def run_all_supplementary(
    device: str = "cuda",
    skip_fp16_mistral: bool = False,
    skip_vllm: bool = False,
) -> Dict:
    """Run all supplementary experiments."""
    print("\n" + "#" * 70)
    print("# ICML 2026 SUPPLEMENTARY EXPERIMENTS")
    print(f"# Timestamp: {datetime.now().isoformat()}")
    print("#" * 70)

    results = {
        "timestamp": datetime.now().isoformat(),
        "experiments": {}
    }

    # P0-2: fp16 Correctness
    if not skip_fp16_mistral:
        print("\n\n" + "=" * 70)
        print("Running fp16 Mistral-7B correctness verification...")
        print("(This requires ~14GB GPU memory)")
        print("=" * 70)

        try:
            results["experiments"]["fp16_correctness"] = run_fp16_correctness_experiment(
                model_name="mistralai/Mistral-7B-v0.1",
                device=device,
                n_samples=100,
            )
        except Exception as e:
            print(f"fp16 correctness experiment failed: {e}")
            results["experiments"]["fp16_correctness"] = {"error": str(e)}

        clear_gpu()

    # P0-3: Optimized Multi-turn (use TinyLlama first for quick test)
    print("\n\n" + "=" * 70)
    print("Running optimized multi-turn experiment...")
    print("=" * 70)

    try:
        adapter = LlamaStyleAdapter.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            device=device,
        )
        tokenizer = adapter.tokenizer
        config = DeltaCacheConfig.for_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        config.device = device

        results["experiments"]["optimized_multiturn_tinyllama"] = run_optimized_multiturn_experiment(
            adapter, tokenizer, config, n_runs=3
        )

        del adapter
        clear_gpu()
    except Exception as e:
        print(f"Optimized multi-turn experiment failed: {e}")
        results["experiments"]["optimized_multiturn_tinyllama"] = {"error": str(e)}

    # P0-1: vLLM Comparison
    if not skip_vllm:
        print("\n\n" + "=" * 70)
        print("Running vLLM comparison...")
        print("=" * 70)

        try:
            results["experiments"]["vllm_comparison"] = run_vllm_comparison_offline(
                model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                n_queries=30,
                n_runs=3,
                device=device,
            )
        except Exception as e:
            print(f"vLLM comparison failed: {e}")
            results["experiments"]["vllm_comparison"] = {"error": str(e)}

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"icml_supplementary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to: {output_path}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ICML 2026 Supplementary Experiments")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-fp16", action="store_true", help="Skip fp16 Mistral-7B")
    parser.add_argument("--skip-vllm", action="store_true", help="Skip vLLM comparison")
    parser.add_argument("--fp16-only", action="store_true", help="Only run fp16 correctness")
    parser.add_argument("--multiturn-only", action="store_true", help="Only run multi-turn")
    parser.add_argument("--vllm-only", action="store_true", help="Only run vLLM comparison")

    args = parser.parse_args()

    if args.fp16_only:
        results = run_fp16_correctness_experiment(device=args.device)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / "fp16_correctness_mistral.json", "w") as f:
            json.dump(results, f, indent=2)
    elif args.multiturn_only:
        adapter = LlamaStyleAdapter.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0", device=args.device)
        config = DeltaCacheConfig.for_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        config.device = args.device
        results = run_optimized_multiturn_experiment(adapter, adapter.tokenizer, config)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / "optimized_multiturn_tinyllama.json", "w") as f:
            json.dump(results, f, indent=2)
    elif args.vllm_only:
        results = run_vllm_comparison_offline(device=args.device)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / "vllm_comparison_offline.json", "w") as f:
            json.dump(results, f, indent=2)
    else:
        run_all_supplementary(
            device=args.device,
            skip_fp16_mistral=args.skip_fp16,
            skip_vllm=args.skip_vllm,
        )


if __name__ == "__main__":
    main()
