# DeltaCache Engineering Design Document

**Date**: 2026-02-21
**Status**: Active
**Target**: Production-quality AI infra middleware for consumer GPUs

---

## 1. Positioning

DeltaCache is a **lightweight KV cache middleware** for LLM inference on consumer-grade GPUs (RTX 3090/4090, 24GB VRAM). Unlike full serving engines (vLLM, SGLang), it operates as an embeddable library (`pip install deltacache`) that integrates with existing HuggingFace workflows.

### 1.1 Competitive Landscape

| System | Type | Data Structure | Eviction | Tiered Storage | Scale |
|--------|------|----------------|----------|----------------|-------|
| **vLLM APC** | Serving engine | Hash table (block-level, 16 tok) | LRU | Via LMCache connector | 70B+ on H100 |
| **SGLang** | Serving engine | Radix tree (token-level) | LRU/LFU | HiCache (GPU/CPU/Disk) | 70B+ on H100 |
| **LMCache** | Middleware | Token DB + chunks (256 tok) | LRU | GPU/CPU/Disk/Redis | Integrates w/ vLLM |
| **DeltaCache** | Library | Prefix tree (token-level) | 5 policies + layer-aware | GPU/CPU (planned) | 7B on RTX 3090 |

### 1.2 DeltaCache's Niche

- **No engine replacement needed**: Works with vanilla HuggingFace `model.generate()`
- **Consumer GPU optimized**: Memory-constrained environments (24GB VRAM)
- **Pluggable eviction**: Layer-aware, attention-aware, adaptive policies
- **Minimal overhead**: Library-level, no scheduler/server infrastructure

---

## 2. Architecture

### 2.1 Current Architecture

```
┌──────────────────────────────────────────────┐
│                User Application              │
├──────────────────────────────────────────────┤
│              DeltaCacheManager               │
│  ┌──────────┐ ┌──────────┐ ┌──────────────┐ │
│  │ PrefixTree│ │MemoryPool│ │EvictionPolicy│ │
│  └──────────┘ └──────────┘ └──────────────┘ │
│  ┌──────────────────┐ ┌──────────────────┐   │
│  │IncrementalEngine │ │   RoPEHandler    │   │
│  └──────────────────┘ └──────────────────┘   │
├──────────────────────────────────────────────┤
│           HF Integration Adapters            │
│  ┌───────┐ ┌──────────┐ ┌───────┐           │
│  │ Llama │ │ Mistral  │ │ GPT-2 │           │
│  └───────┘ └──────────┘ └───────┘           │
└──────────────────────────────────────────────┘
```

### 2.2 Target Architecture (v0.2.0)

```
┌──────────────────────────────────────────────────────┐
│                   User Application                    │
├──────────────────────────────────────────────────────┤
│                 DeltaCacheManager                     │
│  ┌────────────┐ ┌─────────────────┐ ┌──────────────┐ │
│  │ PrefixTree │ │  TieredMemPool  │ │EvictionPolicy│ │
│  └────────────┘ │  ┌───────────┐  │ │  + Registry  │ │
│                 │  │ GPU Pool  │  │ └──────────────┘ │
│  ┌────────────┐ │  ├───────────┤  │ ┌──────────────┐ │
│  │ Incremental│ │  │ CPU Pool  │  │ │MemoryMonitor │ │
│  │   Engine   │ │  │ (pinned)  │  │ │ (watermarks) │ │
│  └────────────┘ │  ├───────────┤  │ └──────────────┘ │
│                 │  │ Quantizer │  │ ┌──────────────┐ │
│  ┌────────────┐ │  │ (INT8/4)  │  │ │   Metrics    │ │
│  │RoPEHandler │ │  └───────────┘  │ │ (prometheus) │ │
│  └────────────┘ └─────────────────┘ └──────────────┘ │
├──────────────────────────────────────────────────────┤
│               HF Integration Adapters                 │
├──────────────────────────────────────────────────────┤
│                   CLI / Benchmark                     │
│  deltacache bench | deltacache stats | deltacache serve│
└──────────────────────────────────────────────────────┘
```

---

## 3. Key Engineering Components

### 3.1 GPU Memory Monitor (NEW)

**Problem**: Current memory management uses static limits. Real GPU memory is shared with the model, PyTorch allocator, and other processes. OOM crashes are possible.

**Design**: Proactive watermark-based monitoring using `torch.cuda.mem_get_info()`.

```
GPU Memory Usage
100% ─────────────── OOM
 95% ─────────────── CRITICAL: emergency eviction
 85% ─────────────── HIGH_WATERMARK: start eviction
 70% ─────────────── LOW_WATERMARK: eviction target
  0% ─────────────── Empty
```

**Key decisions**:
- Poll interval: 100ms (configurable)
- Use `torch.cuda.mem_get_info()` (reads actual GPU state, not just PyTorch allocations)
- High watermark default: 85% → triggers eviction
- Low watermark default: 70% → eviction stops here
- Critical threshold: 95% → emergency: evict aggressively, skip CPU offload
- Thread-safe: monitoring runs in background daemon thread

**Integration**: `MemoryMonitor` replaces the static `gpu_limit` check in `MemoryPool`. The existing `_eviction_callback` mechanism is preserved.

### 3.2 Tiered Cache: GPU → CPU with Async Transfer (NEW)

**Problem**: When GPU memory fills, current behavior either evicts completely (LRU/LFU) or calls `block.to_cpu()` synchronously (blocking inference).

**Design**: Async tiered storage with pinned CPU memory and CUDA streams.

**Key decisions based on research**:

1. **Pre-allocated pinned CPU buffer pool**:
   - Allocate pinned memory once at init via `torch.empty(..., pin_memory=True)`
   - Do NOT call `.pin_memory()` on existing tensors (1.2x slower per PyTorch docs)
   - Size: configurable, default = 2x GPU cache budget

2. **CUDA stream isolation**:
   - Dedicated `torch.cuda.Stream` for offload transfers (GPU→CPU)
   - Dedicated `torch.cuda.Stream` for prefetch transfers (CPU→GPU)
   - Both separate from default compute stream → true overlap

3. **Transfer safety rules** (from PyTorch docs):
   - Never mutate pinned tensors after initiating non-blocking H2D transfer
   - Call `tensor.record_stream(s)` to prevent allocator reuse before stream completes
   - Synchronize before reading D2H results on CPU side

4. **Bandwidth budget** (PCIe 3.0 x16 on RTX 3090):
   - Effective: ~12 GB/s per direction
   - 1GB cache block transfer: ~83ms
   - Must overlap with compute to hide latency

5. **Prefetch strategy**:
   - On `lookup()`, if result is CPU-resident, immediately start async prefetch
   - Track outstanding prefetches; synchronize only when data is actually needed
   - Best-effort: if not ready in time, fall back to synchronous transfer

### 3.3 KV Cache Quantization (NEW)

**Problem**: CPU-tier KV cache consumes lots of host memory. 64GB CPU limits effective cache capacity.

**Design**: KIVI-style asymmetric quantization for the CPU tier.

**Key decisions based on research** (KIVI, ICML 2024):

1. **Asymmetric quantization**:
   - **Keys**: Per-channel quantization (channels have consistent outlier patterns)
   - **Values**: Per-token quantization (attention sparsity isolates per-token error)
   - Formula: `Q(X) = round((X - min) / scale)`, `scale = (max - min) / (2^B - 1)`

2. **Precision levels**:
   - INT8: ~2x memory savings, <0.5% perplexity impact → **default for CPU tier**
   - INT4: ~2.5x savings, <2% impact → optional aggressive mode
   - No INT2: >20% quality loss, not worth it

3. **Residual window**: Keep last 128 tokens in FP16 (matches KIVI recommendation)

4. **When to quantize**: On GPU→CPU offload. Dequantize on CPU→GPU prefetch.

### 3.4 Eviction Policy Registry (Enhancement)

**Problem**: Users can't easily add custom eviction policies.

**Design**: Plugin-based registry with standard scoring interface.

```python
@deltacache.register_eviction_policy("my_policy")
class MyPolicy(EvictionPolicy):
    def select_victims(self, tree, pool, required) -> List[EvictionCandidate]:
        ...
```

**Integration**: Extend `create_eviction_policy()` to check registry before built-in policies. Add `EvictionContext` dataclass to provide richer context (memory_pressure, access_history, layer_stats).

### 3.5 Prometheus Metrics (NEW)

**Metrics to expose**:

| Metric | Type | Description |
|--------|------|-------------|
| `deltacache_lookups_total` | Counter | Total cache lookups |
| `deltacache_hits_total` | Counter | Cache hits |
| `deltacache_misses_total` | Counter | Cache misses |
| `deltacache_hit_rate` | Gauge | Current hit rate |
| `deltacache_gpu_memory_bytes` | Gauge | GPU memory used by cache |
| `deltacache_cpu_memory_bytes` | Gauge | CPU memory used by cache |
| `deltacache_gpu_utilization` | Gauge | GPU memory utilization ratio |
| `deltacache_evictions_total` | Counter | Total evictions |
| `deltacache_offloads_total` | Counter | GPU→CPU offloads |
| `deltacache_prefetches_total` | Counter | CPU→GPU prefetches |
| `deltacache_prefetch_hit_rate` | Gauge | Prefetch success rate |
| `deltacache_lookup_duration_seconds` | Histogram | Lookup latency |
| `deltacache_eviction_duration_seconds` | Histogram | Eviction latency |
| `deltacache_transfer_duration_seconds` | Histogram | GPU↔CPU transfer latency |
| `deltacache_cached_sequences` | Gauge | Number of cached sequences |
| `deltacache_cached_tokens` | Gauge | Total cached tokens |

**Implementation**: Optional dependency on `prometheus_client`. Metrics disabled when library not installed. No runtime overhead when disabled.

---

## 4. Implementation Milestones

### M1: Engineering Foundation (Week 1-2)
- [x] Design document
- [ ] CI/CD (GitHub Actions: pytest, ruff, mypy)
- [ ] pyproject.toml: add `[monitoring]`, `[quantize]` optional deps
- [ ] ruff.toml + mypy config
- [ ] GPU Memory Monitor (`deltacache/core/memory_monitor.py`)
- [ ] Integrate monitor into DeltaCacheManager

### M2: Tiered Cache (Week 3-4)
- [ ] Pinned CPU memory pool (`deltacache/core/pinned_pool.py`)
- [ ] Async transfer manager (`deltacache/core/async_transfer.py`)
- [ ] Integrate into MemoryPool (GPU→CPU offload path)
- [ ] Prefetch on cache lookup (CPU→GPU path)
- [ ] KV quantization for CPU tier (`deltacache/core/kv_quantizer.py`)

### M3: Observability & Benchmarks (Week 5-6)
- [ ] Prometheus metrics module (`deltacache/metrics.py`)
- [ ] Structured logging with `structlog`
- [ ] Standardized benchmark suite (`deltacache bench` CLI)
- [ ] vLLM APC comparison benchmark
- [ ] Eviction policy registry

---

## 5. Key Technical References

### vLLM APC
- Hash-based block matching (16-token blocks), LRU eviction
- V1 engine: <1% overhead at 0% hit rate (pre-allocated block pool, O(1) eviction)
- Frequency-aware eviction RFC in progress (#23641)
- RTX 3090: works but must set `--max-model-len`, TP hangs without NVLink

### SGLang RadixAttention + HiCache
- Radix tree (compressed trie) with token-level granularity
- Cache-aware scheduling (LPM): 96% optimal hit rate, 1.9x throughput
- HiCache: layer-wise overlapping (load layer N+1 while computing layer N)
- GPU-assisted I/O kernels: 3x transfer throughput vs cudaMemcpyAsync
- Memory layout decoupling: layer-first on GPU, page-first on host (2x throughput)

### GPU↔CPU Transfer
- Pinned memory: allocate directly via `torch.empty(pin_memory=True)`, NOT `.pin_memory()`
- PCIe 3.0 x16: ~12 GB/s effective
- Double buffering with 2 CUDA streams for overlap
- `record_stream()` to prevent allocator reuse race conditions

### KV Quantization
- KIVI (ICML 2024): per-channel keys + per-token values, 2-bit viable with 128-token FP16 residual
- INT4: 2.5x savings, <2% perplexity impact (recommended balance)
- INT8: 2x savings, <0.5% impact (safe default)
- HuggingFace `quanto` backend: production-ready but 3x slower with weight quantization
