#!/usr/bin/env python3
"""
Long Context Experiments for ICML 2026 - P1

Tests DeltaCache performance with longer context lengths (2K-8K tokens)
to verify the quadratic speedup benefits for long-context applications.

Key insight: Speedup should increase with context length since:
- Attention computation is O(n²)
- Cache hit saves more computation for longer sequences
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

RESULTS_DIR = Path(__file__).parent / "results" / "paper"


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def create_long_context(target_tokens: int, tokenizer) -> str:
    """Create a document with approximately target_tokens tokens."""

    # Base document content (will be repeated)
    base_content = """
# Comprehensive Technical Documentation

## Section 1: System Architecture Overview

The distributed computing system follows a microservices architecture designed for
high availability and horizontal scalability. Each service operates independently,
communicating through well-defined APIs and message queues.

### 1.1 Core Components

The system consists of the following primary components:

1. **API Gateway**: Serves as the single entry point for all client requests.
   It handles authentication, rate limiting, request routing, and load balancing.

2. **Service Registry**: Maintains a dynamic registry of all available services,
   their locations, and health status. Services register on startup and
   periodically send heartbeats.

3. **Configuration Server**: Centralized configuration management for all services.
   Supports dynamic configuration updates without service restarts.

4. **Message Broker**: Enables asynchronous communication between services.
   Supports multiple messaging patterns including pub/sub and request/reply.

### 1.2 Data Flow

Data flows through the system in the following manner:
- Client requests arrive at the API Gateway
- Gateway authenticates and routes to appropriate service
- Service processes request, potentially calling other services
- Responses are aggregated and returned to client

## Section 2: API Reference

### 2.1 RESTful Endpoints

All endpoints follow REST conventions with proper HTTP methods:

```
GET    /api/v1/resources          - List resources
POST   /api/v1/resources          - Create resource
GET    /api/v1/resources/{id}     - Get specific resource
PUT    /api/v1/resources/{id}     - Update resource
DELETE /api/v1/resources/{id}     - Delete resource
```

### 2.2 Authentication

Authentication uses OAuth 2.0 with JWT tokens:
- Access tokens expire after 1 hour
- Refresh tokens valid for 30 days
- All tokens are signed with RS256

### 2.3 Rate Limiting

Rate limits are applied per client:
- Standard tier: 100 requests/minute
- Premium tier: 1000 requests/minute
- Enterprise tier: Custom limits

## Section 3: Configuration Guide

### 3.1 Environment Variables

Required environment variables for deployment:
- DATABASE_URL: PostgreSQL connection string
- REDIS_URL: Redis cache connection
- JWT_SECRET: Secret for token signing
- LOG_LEVEL: Logging verbosity (debug|info|warn|error)

### 3.2 Service Configuration

Each service can be configured through YAML files:

```yaml
server:
  port: 8080
  timeout: 30s
  max_connections: 1000

database:
  pool_size: 20
  idle_timeout: 300s

cache:
  enabled: true
  ttl: 3600s
```

## Section 4: Deployment Guide

### 4.1 Docker Deployment

Build and run using Docker:

```bash
docker build -t myservice .
docker run -p 8080:8080 myservice
```

### 4.2 Kubernetes Deployment

For production Kubernetes deployments:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myservice
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myservice
```

## Section 5: Monitoring and Observability

### 5.1 Metrics

Key metrics collected include:
- Request latency (p50, p95, p99)
- Error rates by type
- Resource utilization (CPU, memory)
- Queue depths and processing times

### 5.2 Logging

Structured logging in JSON format:
- Timestamp in ISO 8601 format
- Request ID for distributed tracing
- Service name and version
- Log level and message

### 5.3 Alerting

Alerts configured for:
- High error rates (>1%)
- Latency degradation
- Service unavailability
- Resource exhaustion

"""

    # Calculate how many repetitions needed
    base_tokens = len(tokenizer.encode(base_content))
    repetitions = max(1, (target_tokens // base_tokens) + 1)

    # Create document with unique section numbering
    full_content = ""
    for i in range(repetitions):
        section_content = base_content.replace("Section 1", f"Section {i*5 + 1}")
        section_content = section_content.replace("Section 2", f"Section {i*5 + 2}")
        section_content = section_content.replace("Section 3", f"Section {i*5 + 3}")
        section_content = section_content.replace("Section 4", f"Section {i*5 + 4}")
        section_content = section_content.replace("Section 5", f"Section {i*5 + 5}")
        full_content += section_content

    # Truncate to approximate target
    tokens = tokenizer.encode(full_content)
    if len(tokens) > target_tokens:
        # Decode back to string at target length
        full_content = tokenizer.decode(tokens[:target_tokens])

    return full_content


@dataclass
class LongContextResult:
    """Result for a single context length experiment."""
    target_tokens: int
    actual_tokens: int
    n_queries: int

    baseline_mean_ms: float
    baseline_std_ms: float
    deltacache_mean_ms: float
    deltacache_std_ms: float
    deltacache_first_ms: float  # First query (cache miss)
    deltacache_cached_ms: float  # Subsequent queries (cache hit)

    speedup_overall: float
    speedup_cached: float
    token_reuse_rate: float


def run_long_context_experiment(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device: str = "cuda:0",
    context_lengths: List[int] = [500, 1000, 2000, 3000, 4000],
    n_queries: int = 15,
) -> Dict:
    """Run experiment across different context lengths."""

    print("\n" + "#" * 70)
    print("# LONG CONTEXT EXPERIMENTS")
    print(f"# Model: {model_name}")
    print(f"# Context lengths: {context_lengths}")
    print(f"# Queries per length: {n_queries}")
    print("#" * 70)

    # Load model
    print("\nLoading model...")
    adapter = LlamaStyleAdapter.from_pretrained(model_name, device=device)
    tokenizer = adapter.tokenizer
    config = DeltaCacheConfig.for_model(model_name)
    config.device = device

    results = {
        "metadata": {
            "model": model_name,
            "device": device,
            "context_lengths": context_lengths,
            "n_queries": n_queries,
            "timestamp": datetime.now().isoformat(),
        },
        "experiments": [],
    }

    queries = [
        "What is the system architecture?",
        "Describe the API endpoints.",
        "How does authentication work?",
        "What environment variables are needed?",
        "Explain the deployment process.",
        "What metrics are collected?",
        "How is logging configured?",
        "What alerts are set up?",
        "Describe the data flow.",
        "What are the rate limits?",
        "How does service registration work?",
        "What is the configuration format?",
        "Explain Kubernetes deployment.",
        "How are errors handled?",
        "What is the message broker for?",
    ]

    try:
        for target_tokens in context_lengths:
            print(f"\n{'='*60}")
            print(f"Context Length: {target_tokens} tokens")
            print("=" * 60)

            # Create context
            context = create_long_context(target_tokens, tokenizer)
            actual_tokens = len(tokenizer.encode(context))
            print(f"Actual tokens: {actual_tokens}")

            # Baseline measurement
            print("\nRunning baseline...")
            baseline_latencies = []

            for query in tqdm(queries[:n_queries], desc="  Baseline"):
                prompt = f"{context}\n\nQuestion: {query}\nAnswer:"
                tokens = tokenizer.encode(prompt)

                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = adapter.compute_kv_for_tokens(tokens)
                torch.cuda.synchronize()

                baseline_latencies.append((time.perf_counter() - start) * 1000)
                clear_gpu()

            # DeltaCache measurement
            print("Running DeltaCache...")
            clear_gpu()
            manager = DeltaCacheManager(config)

            dc_latencies = []
            total_matched = 0
            total_tokens_processed = 0

            for i, query in enumerate(tqdm(queries[:n_queries], desc="  DeltaCache")):
                prompt = f"{context}\n\nQuestion: {query}\nAnswer:"
                tokens = tokenizer.encode(prompt)
                total_tokens_processed += len(tokens)

                torch.cuda.synchronize()
                start = time.perf_counter()
                result = manager.compute_incremental(tokens, adapter.compute_kv)
                torch.cuda.synchronize()

                dc_latencies.append((time.perf_counter() - start) * 1000)
                total_matched += result.matched_length

            del manager
            clear_gpu()

            # Calculate metrics
            baseline_mean = statistics.mean(baseline_latencies)
            dc_mean = statistics.mean(dc_latencies)
            dc_first = dc_latencies[0]
            dc_cached = statistics.mean(dc_latencies[1:]) if len(dc_latencies) > 1 else dc_first

            experiment_result = LongContextResult(
                target_tokens=target_tokens,
                actual_tokens=actual_tokens,
                n_queries=n_queries,
                baseline_mean_ms=baseline_mean,
                baseline_std_ms=statistics.stdev(baseline_latencies) if len(baseline_latencies) > 1 else 0,
                deltacache_mean_ms=dc_mean,
                deltacache_std_ms=statistics.stdev(dc_latencies) if len(dc_latencies) > 1 else 0,
                deltacache_first_ms=dc_first,
                deltacache_cached_ms=dc_cached,
                speedup_overall=baseline_mean / dc_mean,
                speedup_cached=baseline_mean / dc_cached if dc_cached > 0 else 0,
                token_reuse_rate=total_matched / total_tokens_processed,
            )

            results["experiments"].append(asdict(experiment_result))

            print(f"\n  Baseline: {baseline_mean:.1f} ms")
            print(f"  DeltaCache: {dc_mean:.1f} ms (first: {dc_first:.1f}, cached: {dc_cached:.1f})")
            print(f"  Overall Speedup: {experiment_result.speedup_overall:.2f}x")
            print(f"  Cached Speedup: {experiment_result.speedup_cached:.2f}x")
            print(f"  Token Reuse: {experiment_result.token_reuse_rate:.1%}")

    finally:
        del adapter
        clear_gpu()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Context Length Scaling")
    print("=" * 70)

    print(f"\n{'Tokens':<12} {'Baseline':<12} {'DC Mean':<12} {'DC Cached':<12} {'Speedup':<12} {'Cached ×':<12}")
    print("-" * 72)

    for exp in results["experiments"]:
        print(f"{exp['actual_tokens']:<12} {exp['baseline_mean_ms']:<12.1f} "
              f"{exp['deltacache_mean_ms']:<12.1f} {exp['deltacache_cached_ms']:<12.1f} "
              f"{exp['speedup_overall']:<12.2f}× {exp['speedup_cached']:<12.2f}×")

    # Calculate scaling trend
    if len(results["experiments"]) >= 2:
        tokens = [e["actual_tokens"] for e in results["experiments"]]
        speedups = [e["speedup_cached"] for e in results["experiments"]]

        results["summary"] = {
            "min_tokens": min(tokens),
            "max_tokens": max(tokens),
            "min_speedup": min(speedups),
            "max_speedup": max(speedups),
            "speedup_per_1k_tokens": (max(speedups) - min(speedups)) / ((max(tokens) - min(tokens)) / 1000),
        }

        print(f"\nSpeedup scaling: {results['summary']['speedup_per_1k_tokens']:.2f}× per 1K tokens")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Long Context Experiments")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[500, 1000, 1500, 2000])
    parser.add_argument("--n-queries", type=int, default=15)

    args = parser.parse_args()

    results = run_long_context_experiment(
        model_name=args.model,
        device=args.device,
        context_lengths=args.context_lengths,
        n_queries=args.n_queries,
    )

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_short = args.model.split("/")[-1].lower().replace("-", "_")

    output_path = RESULTS_DIR / f"long_context_{model_short}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
