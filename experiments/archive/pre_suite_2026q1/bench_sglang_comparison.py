#!/usr/bin/env python3
"""Comparison benchmark: DeltaCache LayerBudget vs SGLang vs HF baseline.

Three benchmark modes:
  - SGLang: HTTP API to external SGLang server (/v1/completions)
  - DeltaCache: In-process HF model + LayerBudgetCache
  - HF baseline: Plain model.generate()

Workloads focus on shared-prefix scenarios where cache compression matters:
  - shared_prefix: same system prompt + diverse user queries
  - document_qa: long shared document + short questions

Usage:
  python bench_sglang_comparison.py --skip-sglang            # DeltaCache + HF only
  python bench_sglang_comparison.py --sglang-url http://localhost:30000
  python bench_sglang_comparison.py --model mistralai/Mistral-7B-Instruct-v0.2 --load-in-4bit
"""

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from bench_e2e_real import load_model, clear_gpu, gpu_mem_mb


# ── Workloads ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful AI assistant specializing in science and technology. "
    "You provide accurate, well-structured answers based on established "
    "scientific knowledge. When discussing complex topics, break them down "
    "into clear, understandable parts. Always cite relevant principles and "
    "theories. If you're uncertain about something, acknowledge the limits "
    "of your knowledge. Aim for responses that are both informative and "
    "accessible to a general audience interested in science."
)

USER_QUERIES = [
    "Explain the difference between nuclear fission and fusion.",
    "How does CRISPR gene editing work?",
    "What causes the aurora borealis?",
    "Describe how a quantum computer differs from a classical computer.",
    "What is the significance of the Higgs boson discovery?",
    "How do vaccines train the immune system?",
    "Explain the concept of entropy in thermodynamics.",
    "What are gravitational waves and how were they detected?",
    "How does photosynthesis convert light into energy?",
    "What is dark matter and why do scientists believe it exists?",
    "Explain the greenhouse effect and its role in climate change.",
    "How does machine learning differ from traditional programming?",
    "What is the uncertainty principle in quantum mechanics?",
    "How do neurons transmit signals in the brain?",
    "Explain the theory of plate tectonics.",
]

DOCUMENT = (
    "The transformer architecture, introduced in 'Attention Is All You Need' "
    "(Vaswani et al., 2017), has become the foundation of modern natural "
    "language processing. At its core, the transformer uses self-attention "
    "mechanisms to process sequences in parallel, unlike recurrent neural "
    "networks which process tokens sequentially. The key innovation is the "
    "scaled dot-product attention, where queries, keys, and values are "
    "computed from input embeddings. Multi-head attention allows the model "
    "to jointly attend to information from different representation "
    "subspaces. The transformer consists of an encoder-decoder structure, "
    "though many modern variants use only the decoder (like GPT) or only "
    "the encoder (like BERT). Recent advances include efficient attention "
    "mechanisms such as FlashAttention, grouped-query attention (GQA), "
    "and various forms of KV cache optimization for inference. The KV "
    "cache stores key and value projections from previous tokens to avoid "
    "recomputation during autoregressive generation. As context lengths "
    "grow, KV cache memory becomes a bottleneck, motivating techniques "
    "like quantization (KIVI), token eviction (H2O, SnapKV), and hybrid "
    "approaches like LayerBudget that jointly optimize token retention "
    "and precision per layer."
)

DOC_QUESTIONS = [
    "What is the key innovation of the transformer architecture?",
    "How does multi-head attention work?",
    "What is the difference between GPT and BERT architectures?",
    "Why is KV cache optimization important for inference?",
    "What are the main approaches to reducing KV cache memory?",
    "How does FlashAttention improve efficiency?",
    "What is grouped-query attention?",
    "Explain the encoder-decoder structure of transformers.",
]


def create_workloads() -> Dict[str, List[str]]:
    """Generate workloads with different prefix-sharing patterns."""
    workloads = {}

    # Shared prefix: same system prompt + different user queries
    workloads["shared_prefix"] = [
        f"{SYSTEM_PROMPT}\n\nUser: {q}\nAssistant:"
        for q in USER_QUERIES
    ]

    # Document QA: long shared document + short questions
    workloads["document_qa"] = [
        f"Document:\n{DOCUMENT}\n\nQuestion: {q}\nAnswer:"
        for q in DOC_QUESTIONS
    ]

    # Diverse: no shared prefix (worst case for caching)
    workloads["diverse"] = [
        f"User: {q}\nAssistant:" for q in USER_QUERIES[:8]
    ]

    return workloads


# ── Result dataclass ─────────────────────────────────────────────────────

@dataclass
class BenchResult:
    system: str
    workload: str
    n_queries: int
    latency_mean_ms: float
    latency_std_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    ttft_mean_ms: float
    throughput_rps: float
    peak_gpu_mb: float
    extra: Dict = field(default_factory=dict)


# ── SGLang benchmark ─────────────────────────────────────────────────────

def bench_sglang(prompts: List[str], sglang_url: str,
                 max_new_tokens: int = 64) -> BenchResult:
    """Benchmark SGLang via OpenAI-compatible API."""
    import requests

    latencies = []
    ttfts = []

    # Warmup
    try:
        requests.post(
            f"{sglang_url}/v1/completions",
            json={"model": "default", "prompt": "Hello", "max_tokens": 4},
            timeout=30,
        )
    except Exception as e:
        print(f"    SGLang warmup failed: {e}")
        return BenchResult(
            system="sglang", workload="error", n_queries=0,
            latency_mean_ms=0, latency_std_ms=0, latency_p50_ms=0,
            latency_p95_ms=0, ttft_mean_ms=0, throughput_rps=0,
            peak_gpu_mb=0, extra={"error": str(e)},
        )

    t_total_start = time.perf_counter()

    for prompt in prompts:
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                f"{sglang_url}/v1/completions",
                json={
                    "model": "default",
                    "prompt": prompt,
                    "max_tokens": max_new_tokens,
                    "temperature": 0,
                },
                timeout=120,
            )
            resp.raise_for_status()
            latency = (time.perf_counter() - t0) * 1000
            latencies.append(latency)

            # TTFT approximation: SGLang doesn't expose this directly
            # Use usage.prompt_tokens ratio as proxy
            data = resp.json()
            prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)
            total_tokens = data.get("usage", {}).get("total_tokens", 0)
            gen_tokens = total_tokens - prompt_tokens
            if gen_tokens > 0:
                ttft_est = latency * (1 / (gen_tokens + 1))
                ttfts.append(ttft_est)
        except Exception as e:
            print(f"    SGLang request failed: {e}")
            continue

    t_total = time.perf_counter() - t_total_start

    if not latencies:
        return BenchResult(
            system="sglang", workload="error", n_queries=0,
            latency_mean_ms=0, latency_std_ms=0, latency_p50_ms=0,
            latency_p95_ms=0, ttft_mean_ms=0, throughput_rps=0,
            peak_gpu_mb=0, extra={"error": "no successful requests"},
        )

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]

    return BenchResult(
        system="sglang",
        workload="",
        n_queries=len(latencies),
        latency_mean_ms=round(statistics.mean(latencies), 1),
        latency_std_ms=round(statistics.stdev(latencies) if len(latencies) > 1 else 0, 1),
        latency_p50_ms=round(p50, 1),
        latency_p95_ms=round(p95, 1),
        ttft_mean_ms=round(statistics.mean(ttfts) if ttfts else 0, 1),
        throughput_rps=round(len(latencies) / t_total, 2),
        peak_gpu_mb=0,  # Can't measure SGLang's GPU usage from client
    )


# ── DeltaCache LayerBudget benchmark ─────────────────────────────────────

def bench_deltacache(model, tokenizer, prompts: List[str],
                     budget_mb: float, max_new_tokens: int = 64) -> BenchResult:
    """Benchmark DeltaCache with LayerBudgetCache."""
    from deltacache.integrations.hf_cache import LayerBudgetCache

    device = next(model.parameters()).device
    latencies = []
    ttfts = []
    compressions = []

    gen_kw = dict(max_new_tokens=max_new_tokens, do_sample=False,
                  pad_token_id=tokenizer.eos_token_id)

    # Warmup
    input_ids = tokenizer.encode("Hello", return_tensors="pt").to(device)
    with torch.no_grad():
        model.generate(input_ids, max_new_tokens=4, pad_token_id=tokenizer.eos_token_id)
    clear_gpu()

    torch.cuda.reset_peak_memory_stats()
    mem_start = gpu_mem_mb()

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        clear_gpu()
        cache = LayerBudgetCache(max_memory_mb=budget_mb, compression_ratio=0.25)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(input_ids, past_key_values=cache, **gen_kw)
        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)

        gen_len = out.shape[1] - input_ids.shape[1]
        if gen_len > 0:
            ttfts.append(latency / (gen_len + 1))

        compressions.append(cache._compressed)

    peak_gpu = torch.cuda.max_memory_allocated() / 1024 ** 2 - mem_start

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]

    return BenchResult(
        system="deltacache_layerbudget",
        workload="",
        n_queries=len(latencies),
        latency_mean_ms=round(statistics.mean(latencies), 1),
        latency_std_ms=round(statistics.stdev(latencies) if len(latencies) > 1 else 0, 1),
        latency_p50_ms=round(p50, 1),
        latency_p95_ms=round(p95, 1),
        ttft_mean_ms=round(statistics.mean(ttfts) if ttfts else 0, 1),
        throughput_rps=round(len(latencies) / (sum(latencies) / 1000), 2),
        peak_gpu_mb=round(peak_gpu, 1),
        extra={
            "budget_mb": budget_mb,
            "compressed_count": sum(compressions),
            "total_queries": len(compressions),
        },
    )


# ── HF baseline benchmark ───────────────────────────────────────────────

def bench_hf_baseline(model, tokenizer, prompts: List[str],
                      max_new_tokens: int = 64) -> BenchResult:
    """HF generate baseline (no caching optimization)."""
    device = next(model.parameters()).device
    latencies = []
    ttfts = []

    gen_kw = dict(max_new_tokens=max_new_tokens, do_sample=False,
                  pad_token_id=tokenizer.eos_token_id)

    # Warmup
    input_ids = tokenizer.encode("Hello", return_tensors="pt").to(device)
    with torch.no_grad():
        model.generate(input_ids, max_new_tokens=4, pad_token_id=tokenizer.eos_token_id)
    clear_gpu()

    torch.cuda.reset_peak_memory_stats()
    mem_start = gpu_mem_mb()

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        clear_gpu()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(input_ids, **gen_kw)
        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)

        gen_len = out.shape[1] - input_ids.shape[1]
        if gen_len > 0:
            ttfts.append(latency / (gen_len + 1))

    peak_gpu = torch.cuda.max_memory_allocated() / 1024 ** 2 - mem_start

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]

    return BenchResult(
        system="hf_baseline",
        workload="",
        n_queries=len(latencies),
        latency_mean_ms=round(statistics.mean(latencies), 1),
        latency_std_ms=round(statistics.stdev(latencies) if len(latencies) > 1 else 0, 1),
        latency_p50_ms=round(p50, 1),
        latency_p95_ms=round(p95, 1),
        ttft_mean_ms=round(statistics.mean(ttfts) if ttfts else 0, 1),
        throughput_rps=round(len(latencies) / (sum(latencies) / 1000), 2),
        peak_gpu_mb=round(peak_gpu, 1),
    )


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DeltaCache vs SGLang comparison")
    parser.add_argument("--model", default="Qwen/Qwen2-0.5B")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--budget-mb", type=float, default=None,
                        help="LayerBudget memory budget (auto if omitted)")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--sglang-url", default="http://localhost:30000")
    parser.add_argument("--skip-sglang", action="store_true",
                        help="Skip SGLang benchmark (DeltaCache + HF only)")
    parser.add_argument("--workloads", nargs="+",
                        default=["shared_prefix", "document_qa", "diverse"],
                        choices=["shared_prefix", "document_qa", "diverse"])
    args = parser.parse_args()

    RESULTS_DIR = Path(__file__).parent / "results" / "e2e"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load model for DeltaCache and HF baseline
    model, tokenizer = load_model(args.model, args.load_in_4bit)
    nl = model.config.num_hidden_layers
    nh = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    hd = model.config.hidden_size // model.config.num_attention_heads

    # Auto budget
    if args.budget_mb is None:
        from bench_e2e_real import auto_budget
        budgets = auto_budget(nl, nh, hd)
        budget_mb = budgets.dropin_budget_mb
    else:
        budget_mb = args.budget_mb

    workloads = create_workloads()
    all_results = []

    for wl_name in args.workloads:
        prompts = workloads[wl_name]

        print(f"\n{'='*70}")
        print(f"  Workload: {wl_name} ({len(prompts)} queries)")
        print(f"  Prefix len: ~{len(tokenizer.encode(prompts[0].split('User:')[0] if 'User:' in prompts[0] else prompts[0][:200]))} tokens")
        print(f"{'='*70}")

        # HF Baseline
        print(f"\n  [HF Baseline]")
        hf_result = bench_hf_baseline(model, tokenizer, prompts, args.max_new_tokens)
        hf_result.workload = wl_name
        all_results.append(hf_result)
        print(f"    latency={hf_result.latency_mean_ms:.0f}ms p95={hf_result.latency_p95_ms:.0f}ms "
              f"rps={hf_result.throughput_rps:.2f} gpu={hf_result.peak_gpu_mb:.0f}MB")

        # DeltaCache LayerBudget
        print(f"\n  [DeltaCache LayerBudget] budget={budget_mb:.1f}MB")
        dc_result = bench_deltacache(model, tokenizer, prompts, budget_mb, args.max_new_tokens)
        dc_result.workload = wl_name
        all_results.append(dc_result)
        compressed = dc_result.extra.get("compressed_count", 0)
        total = dc_result.extra.get("total_queries", 0)
        print(f"    latency={dc_result.latency_mean_ms:.0f}ms p95={dc_result.latency_p95_ms:.0f}ms "
              f"rps={dc_result.throughput_rps:.2f} gpu={dc_result.peak_gpu_mb:.0f}MB "
              f"compressed={compressed}/{total}")

        # SGLang
        if not args.skip_sglang:
            print(f"\n  [SGLang] url={args.sglang_url}")
            sg_result = bench_sglang(prompts, args.sglang_url, args.max_new_tokens)
            sg_result.workload = wl_name
            all_results.append(sg_result)
            if sg_result.n_queries > 0:
                print(f"    latency={sg_result.latency_mean_ms:.0f}ms p95={sg_result.latency_p95_ms:.0f}ms "
                      f"rps={sg_result.throughput_rps:.2f}")
            else:
                print(f"    FAILED: {sg_result.extra.get('error', 'unknown')}")
        else:
            print(f"\n  [SGLang] skipped")

    # Summary table
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'System':<25s} {'Workload':<16s} {'Mean(ms)':>9s} {'P95(ms)':>8s} "
          f"{'RPS':>6s} {'GPU(MB)':>8s}")
    print(f"  {'-'*75}")
    for r in all_results:
        print(f"  {r.system:<25s} {r.workload:<16s} {r.latency_mean_ms:>8.0f}ms "
              f"{r.latency_p95_ms:>7.0f}ms {r.throughput_rps:>5.2f} {r.peak_gpu_mb:>7.0f}MB")

    # Save
    output = {
        "metadata": {
            "model": args.model,
            "budget_mb": budget_mb,
            "max_new_tokens": args.max_new_tokens,
            "sglang_url": args.sglang_url if not args.skip_sglang else "skipped",
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "timestamp": datetime.now().isoformat(),
        },
        "results": [asdict(r) for r in all_results],
    }

    safe = args.model.replace("/", "_").replace("-", "_").lower()
    path = RESULTS_DIR / f"sglang_cmp_{safe}_{datetime.now():%Y%m%d_%H%M}.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
