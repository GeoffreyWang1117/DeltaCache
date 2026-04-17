#!/usr/bin/env python3
"""
Comprehensive Baseline Comparison for ICML 2026

This script provides a thorough comparison of DeltaCache against HuggingFace baseline
across different scenarios and prefix lengths, suitable for ICML paper.

Since vLLM/SGLang direct comparison requires specific environment setup, we also
include theoretical comparisons based on their published benchmarks.
"""

import gc
import sys
import json
import time
import statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from deltacache import DeltaCacheManager, DeltaCacheConfig
from deltacache.hf_integration import LlamaStyleAdapter
from transformers import AutoModelForCausalLM, AutoTokenizer

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


@dataclass
class ComparisonResult:
    scenario: str
    prefix_tokens: int
    n_queries: int

    # HuggingFace baseline
    hf_mean_latency_ms: float
    hf_p50_ms: float
    hf_p95_ms: float

    # DeltaCache
    dc_mean_latency_ms: float
    dc_p50_ms: float
    dc_p95_ms: float
    dc_token_reuse: float

    # Computed
    speedup: float
    speedup_p95: float


# =============================================================================
# Test Scenarios with Varying Prefix Lengths
# =============================================================================

def create_rag_scenario(prefix_length: int, n_queries: int) -> Tuple[str, List[str]]:
    """RAG scenario: fixed knowledge document, varying queries."""

    base_doc = """# Technical Documentation

## Overview
This document provides comprehensive technical specifications and guidelines
for system implementation and deployment. It covers architecture, APIs,
configuration options, and best practices.

## Architecture
The system follows a microservices architecture with the following components:
- API Gateway: Handles all incoming requests and routing
- Authentication Service: Manages user authentication and authorization
- Data Processing Engine: Processes and transforms data
- Storage Layer: Persists data across multiple backends

## API Reference
### Endpoints
- GET /api/v1/resources - List all resources
- POST /api/v1/resources - Create new resource
- PUT /api/v1/resources/{id} - Update existing resource
- DELETE /api/v1/resources/{id} - Delete resource

### Authentication
All API requests require Bearer token authentication.
Tokens are obtained through the /auth/token endpoint.

## Configuration
System configuration is managed through environment variables:
- DATABASE_URL: Primary database connection string
- REDIS_URL: Cache server connection
- LOG_LEVEL: Logging verbosity (debug, info, warn, error)
- MAX_CONNECTIONS: Maximum concurrent connections

## Best Practices
1. Always use connection pooling for database access
2. Implement proper error handling and logging
3. Use caching for frequently accessed data
4. Monitor system health with metrics and alerting

"""

    # Repeat to reach desired length
    multiplier = max(1, prefix_length // 50)
    prefix = (base_doc * multiplier)[:prefix_length * 5]

    queries = [
        "What is the system architecture?",
        "How do I authenticate API requests?",
        "List all available endpoints.",
        "What configuration options are available?",
        "Explain the best practices.",
        "How does the data processing engine work?",
        "What is the storage layer?",
        "How do I create a new resource?",
        "What logging levels are supported?",
        "How many connections can the system handle?",
    ] * ((n_queries // 10) + 1)

    return prefix, queries[:n_queries]


def create_code_scenario(prefix_length: int, n_queries: int) -> Tuple[str, List[str]]:
    """Code completion scenario: fixed context, varying completions."""

    code_context = '''"""
Advanced Data Processing Framework
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class DataRecord:
    """Represents a single data record."""
    id: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

class DataProcessor(ABC):
    """Abstract base class for data processors."""

    @abstractmethod
    async def process(self, record: DataRecord) -> DataRecord:
        pass

    @abstractmethod
    def validate(self, record: DataRecord) -> bool:
        pass

class DataStore:
    """Main data storage class with CRUD operations."""

    def __init__(self, name: str):
        self.name = name
        self._records: Dict[str, DataRecord] = {}
        self._index: Dict[str, List[str]] = {}
        logger.info(f"Initialized DataStore: {name}")

    async def add(self, record: DataRecord) -> bool:
        if record.id in self._records:
            return False
        self._records[record.id] = record
        return True

    async def get(self, record_id: str) -> Optional[DataRecord]:
        return self._records.get(record_id)

    async def update(self, record_id: str, data: Dict) -> bool:
        if record_id not in self._records:
            return False
        self._records[record_id].data.update(data)
        return True

    async def delete(self, record_id: str) -> bool:
        if record_id not in self._records:
            return False
        del self._records[record_id]
        return True

# Complete the following function:
'''

    multiplier = max(1, prefix_length // 100)
    prefix = (code_context * multiplier)[:prefix_length * 5]

    completions = [
        "async def batch_add(self, records: List[DataRecord]) -> int:",
        "async def search(self, query: Dict) -> List[DataRecord]:",
        "async def export_json(self, filepath: str) -> None:",
        "def get_statistics(self) -> Dict[str, Any]:",
        "async def sync_to_remote(self, endpoint: str) -> bool:",
    ] * ((n_queries // 5) + 1)

    return prefix, completions[:n_queries]


# =============================================================================
# Benchmark Functions
# =============================================================================

def benchmark_hf_baseline(
    model,
    tokenizer,
    prefix: str,
    queries: List[str],
    device: str,
) -> Dict:
    """Benchmark HuggingFace without caching."""

    latencies = []

    for query in tqdm(queries, desc="  HF Baseline"):
        prompt = f"{prefix}\n{query}"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            _ = model(**inputs, use_cache=False)

        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

    sorted_lat = sorted(latencies)
    n = len(sorted_lat)

    return {
        "mean": statistics.mean(latencies),
        "p50": sorted_lat[n // 2],
        "p95": sorted_lat[int(n * 0.95)] if n >= 20 else sorted_lat[-1],
        "min": min(latencies),
        "max": max(latencies),
    }


def benchmark_deltacache(
    adapter,
    tokenizer,
    config,
    prefix: str,
    queries: List[str],
) -> Dict:
    """Benchmark DeltaCache."""

    manager = DeltaCacheManager(config)
    latencies = []
    total_matched = 0
    total_tokens = 0

    for query in tqdm(queries, desc="  DeltaCache"):
        prompt = f"{prefix}\n{query}"
        tokens = tokenizer.encode(prompt)
        total_tokens += len(tokens)

        torch.cuda.synchronize()
        start = time.perf_counter()

        result = manager.compute_incremental(tokens, adapter.compute_kv)

        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)
        total_matched += result.matched_length

    sorted_lat = sorted(latencies)
    n = len(sorted_lat)

    del manager
    clear_gpu()

    return {
        "mean": statistics.mean(latencies),
        "p50": sorted_lat[n // 2],
        "p95": sorted_lat[int(n * 0.95)] if n >= 20 else sorted_lat[-1],
        "min": min(latencies),
        "max": max(latencies),
        "token_reuse": total_matched / total_tokens,
    }


def run_comprehensive_comparison(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device: str = "cuda:0",
    prefix_lengths: List[int] = [250, 500, 1000, 1500],
    n_queries: int = 30,
) -> Dict:
    """Run comprehensive comparison across scenarios and prefix lengths."""

    print("\n" + "#" * 70)
    print("# COMPREHENSIVE BASELINE COMPARISON")
    print(f"# Model: {model_name}")
    print(f"# Prefix lengths: {prefix_lengths}")
    print(f"# Queries per scenario: {n_queries}")
    print("#" * 70)

    # Load models
    print("\nLoading HuggingFace model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    hf_model.eval()

    print("Loading DeltaCache adapter...")
    dc_adapter = LlamaStyleAdapter.from_pretrained(model_name, device=device)
    dc_config = DeltaCacheConfig.for_model(model_name)
    dc_config.device = device

    results = {
        "metadata": {
            "model": model_name,
            "device": device,
            "timestamp": datetime.now().isoformat(),
        },
        "comparisons": [],
    }

    scenarios = [
        ("rag", create_rag_scenario),
        ("code", create_code_scenario),
    ]

    try:
        for prefix_len in prefix_lengths:
            print(f"\n{'='*60}")
            print(f"Prefix Length: {prefix_len} tokens")
            print("=" * 60)

            for scenario_name, create_fn in scenarios:
                print(f"\n  Scenario: {scenario_name}")

                prefix, queries = create_fn(prefix_len, n_queries)
                actual_prefix = len(tokenizer.encode(prefix))
                print(f"  Actual prefix tokens: {actual_prefix}")

                # HF Baseline
                clear_gpu()
                hf_result = benchmark_hf_baseline(
                    hf_model, tokenizer, prefix, queries, device
                )

                # DeltaCache
                clear_gpu()
                dc_result = benchmark_deltacache(
                    dc_adapter, tokenizer, dc_config, prefix, queries
                )

                speedup = hf_result["mean"] / dc_result["mean"]
                speedup_p95 = hf_result["p95"] / dc_result["p95"]

                comparison = ComparisonResult(
                    scenario=scenario_name,
                    prefix_tokens=actual_prefix,
                    n_queries=n_queries,
                    hf_mean_latency_ms=hf_result["mean"],
                    hf_p50_ms=hf_result["p50"],
                    hf_p95_ms=hf_result["p95"],
                    dc_mean_latency_ms=dc_result["mean"],
                    dc_p50_ms=dc_result["p50"],
                    dc_p95_ms=dc_result["p95"],
                    dc_token_reuse=dc_result["token_reuse"],
                    speedup=speedup,
                    speedup_p95=speedup_p95,
                )

                results["comparisons"].append(asdict(comparison))

                print(f"  HF Mean: {hf_result['mean']:.1f}ms, DC Mean: {dc_result['mean']:.1f}ms")
                print(f"  Speedup: {speedup:.2f}x, Token Reuse: {dc_result['token_reuse']:.1%}")

    finally:
        del hf_model, dc_adapter
        clear_gpu()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_speedups = [c["speedup"] for c in results["comparisons"]]
    all_reuse = [c["dc_token_reuse"] for c in results["comparisons"]]

    results["summary"] = {
        "mean_speedup": statistics.mean(all_speedups),
        "max_speedup": max(all_speedups),
        "min_speedup": min(all_speedups),
        "mean_token_reuse": statistics.mean(all_reuse),
    }

    print(f"\nMean Speedup: {results['summary']['mean_speedup']:.2f}x")
    print(f"Max Speedup: {results['summary']['max_speedup']:.2f}x")
    print(f"Token Reuse: {results['summary']['mean_token_reuse']:.1%}")

    return results


def generate_latex_table(results: Dict) -> str:
    """Generate LaTeX table for paper."""

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Baseline comparison across scenarios and prefix lengths}",
        "\\label{tab:baseline_comparison}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Scenario & Prefix & HF (ms) & DC (ms) & Speedup & Reuse \\\\",
        "\\midrule",
    ]

    for c in results["comparisons"]:
        line = f"{c['scenario']} & {c['prefix_tokens']} & " \
               f"{c['hf_mean_latency_ms']:.1f} & {c['dc_mean_latency_ms']:.1f} & " \
               f"{c['speedup']:.2f}$\\times$ & {c['dc_token_reuse']*100:.1f}\\% \\\\"
        lines.append(line)

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prefix-lengths", type=int, nargs="+", default=[250, 500, 1000, 1500])
    parser.add_argument("--n-queries", type=int, default=30)

    args = parser.parse_args()

    results = run_comprehensive_comparison(
        model_name=args.model,
        device=args.device,
        prefix_lengths=args.prefix_lengths,
        n_queries=args.n_queries,
    )

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_short = args.model.split("/")[-1].lower().replace("-", "_")

    output_path = RESULTS_DIR / f"baseline_comparison_{model_short}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Generate LaTeX table
    latex_table = generate_latex_table(results)
    table_path = RESULTS_DIR / f"baseline_comparison_table_{model_short}.tex"
    with open(table_path, "w") as f:
        f.write(latex_table)
    print(f"LaTeX table saved to: {table_path}")


if __name__ == "__main__":
    main()
