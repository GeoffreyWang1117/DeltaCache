# Research Plan: Dual-Adaptive KV Cache Management

**Start date**: March 2026
**Target venue**: Workshop / short paper (MLSys, NeurIPS Workshop, AAAI)

---

## Core Idea

Current KV cache compression methods apply **uniform** policies across all transformer layers:
- H2O evicts the same fraction of tokens from every layer
- KIVI quantizes all layers to the same bit-width

But layers have dramatically different characteristics:
- **Attention sparsity** (H2O Gini): ranges from 0.431 (Layer 0, uniform) to 0.925 (Layer 4, very sparse)
- **Semantic importance** (DeltaCache layer weight): ranges from 0.182 (Layer 0) to 0.955 (Layer 20)
- **Correlation = 0.194** (weak) → these signals are complementary, not redundant

**Dual-Adaptive KV Management** jointly optimizes per-layer:
1. **Eviction budget**: high-sparsity layers keep fewer tokens, low-sparsity layers keep more
2. **Quantization precision**: less important layers use INT2/INT4, important layers use INT8/FP16

Total memory stays the same as uniform baseline, but quality improves.

---

## Key Data Already Available

### H2O Reproduction (completed, on RTX 3090)

| Layer | Gini (sparsity) | Top-20% Capture | DeltaCache Weight |
|-------|:---:|:---:|:---:|
| 0 | 0.431 | 46.8% | 0.182 |
| 4 | **0.925** | **93.5%** | 0.356 |
| 8 | 0.887 | 90.2% | 0.579 |
| 12 | 0.705 | 73.2% | 0.773 |
| 16 | 0.738 | 76.7% | 0.894 |
| 20 | 0.794 | 82.0% | **0.955** |

- H2O at 20% budget retains 73.4% attention mass (TinyLlama-1.1B)
- Cross-layer stability ~60% → Heavy Hitters vary by layer → per-layer budget justified
- Early layers: high sparsity → aggressive eviction + low-bit quant
- Late layers: low sparsity, high importance → conservative eviction + high-bit quant

---

## Phase 1: Per-Layer Adaptive Eviction Budget (Week 1-2)

### 1.1 Implement Gini-based budget allocation
- Formula: `budget[l] = base × (1 - α × normalized_gini[l])`
- High Gini (Layer 4: 0.925) → smaller budget (fewer tokens needed)
- Low Gini (Layer 0: 0.431) → larger budget (attention is spread out)
- Constraint: total tokens across all layers = same as uniform baseline

### 1.2 Sweep α parameter
- α = 0: uniform baseline (standard H2O)
- α = 1: fully adaptive (maximum differentiation)
- Measure perplexity and downstream accuracy at each α

### 1.3 Compare with PyramidKV
- PyramidKV also does per-layer budget but uses a different signal (attention sink pattern)
- Our signal: Gini coefficient of attention distribution
- Need to articulate why Gini is better/different

### Deliverable
- Per-layer budget allocation code (modify H2O)
- Perplexity vs memory Pareto curve (Gini-adaptive vs uniform)

---

## Phase 2: Per-Layer Mixed-Precision Quantization (Week 2-3)

### 2.1 Implement layer-adaptive KIVI
- KIVI baseline: uniform INT2 for all layers
- Ours: assign bit-width per layer based on importance signal
  - Layer weight < 0.3 → INT2
  - Layer weight 0.3–0.7 → INT4
  - Layer weight > 0.7 → INT8
- Constraint: average bits/layer = same as uniform INT2 (or INT4)

### 2.2 Compare importance signals for quantization
- DeltaCache layer weight (sigmoid-based)
- H2O Gini coefficient
- Attention entropy
- Random (control)

### 2.3 Evaluate quantization quality
- Per-layer cosine similarity (key and value caches)
- End-to-end perplexity (WikiText-2, C4)
- Downstream task accuracy (MMLU subset, HellaSwag)

### Deliverable
- Mixed-precision quantization implementation
- Signal comparison table

---

## Phase 3: Joint Optimization Framework (Week 3-4)

### 3.1 Joint eviction + quantization
- Each layer gets a tuple: (eviction_ratio, quant_bits)
- Objective: minimize perplexity degradation subject to memory budget
- Simple approach: grid search over (α_eviction, α_quant)
- Advanced: use Gini for eviction, layer weight for quant (different signals)

### 3.2 Full comparison with baselines

| Baseline | Per-layer eviction? | Per-layer quant? | Source |
|----------|:---:|:---:|--------|
| H2O | No (uniform budget) | N/A | NeurIPS 2023 |
| PyramidKV | Yes (pyramid shape) | No | arXiv 2024 |
| KIVI | N/A | No (uniform bits) | ICML 2024 |
| SqueezeAttention | No | Yes (by sensitivity) | arXiv 2024 |
| KVTuner | No | Yes (by difficulty) | arXiv 2024 |
| **Ours** | **Yes (Gini-based)** | **Yes (importance-based)** | — |

### 3.3 Ablation study
- Eviction-only vs quant-only vs joint
- Gini signal vs layer-weight signal vs combined
- Budget levels: 10%, 20%, 50% of full KV cache

### Deliverable
- Joint optimization framework
- Complete comparison table
- Paper draft

---

## Overlap Risks & Differentiation

### vs PyramidKV
- They: fixed pyramid shape (more budget at bottom, less at top)
- Us: data-driven Gini-based allocation (adapts to actual attention distribution)
- Key difference: our allocation is model-specific and input-adaptive

### vs SqueezeAttention / KVTuner
- They: per-layer quantization sensitivity analysis
- Us: combine quantization with eviction in a unified framework
- Key difference: dual-adaptive (both dimensions), not just one

### vs H2O
- They: uniform budget across layers
- Us: per-layer budget based on attention sparsity statistics
- Key difference: same eviction mechanism, better budget allocation

---

## Hardware Plan

- **Primary**: Single RTX 3090 (24GB)
- **Models**: TinyLlama-1.1B (fast iteration), Llama-2-7B (main results), Mistral-7B
- **Context lengths**: 2K, 4K, 8K tokens
- **Benchmarks**: WikiText-2 (perplexity), LongBench (downstream), RULER (needle-in-haystack)

## Papers to Download

Already have (in `papers/`):
- H2O (NeurIPS 2023) — baseline eviction
- ShadowKV (2024) — long-context KV compression
- Sirius (NeurIPS 2024) — speculative decoding + KV

Need to download:
- KIVI (ICML 2024) — baseline quantization
- PyramidKV (arXiv 2024) — per-layer budget comparison
- SqueezeAttention (arXiv 2024) — per-layer quant sensitivity
- KVTuner (arXiv 2024) — per-layer quant difficulty
- MiKV (arXiv 2024) — mixed-precision KV cache

## Risks

| Risk | Mitigation |
|------|-----------|
| PyramidKV already solves per-layer budget | Differentiate via Gini signal + joint optimization with quant |
| Uniform quant is already good enough | Show per-layer quality metrics differ significantly |
| 7B models too small to show effect | Most KV papers evaluate on 7B; focus on quality, not scale |
| Two mechanisms don't compose well | Ablation showing joint > sum of parts; if not, report honestly |
