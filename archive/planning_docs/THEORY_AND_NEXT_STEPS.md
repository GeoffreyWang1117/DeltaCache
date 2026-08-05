# Theory Gaps, Competitor Update & Ecosystem Integration

**Date**: 2026-03-28 (updated with scout + ecosystem results)

---

## 1. Theoretical Status

| Theory Component | Status | Evidence |
|-----------------|:------:|---------|
| Coverage model f^(1-g) | ✅ Validated | MAE=3.9% on Mistral-7B, conservative (safe). Added to paper. |
| Fidelity constants | ✅ Corrected | F(8)=0.9999, F(4)=0.9964 (was 0.98/0.92). Paper updated. |
| Gini heterogeneity | ✅ Confirmed | Range 0.70-0.99 on Mistral-7B (1.4× spread) |
| Sigmoid importance | ⚠️ Weak | Ablation: sparsity-only sometimes better than combined. Keep as heuristic. |
| Greedy optimality | ❌ Unproven | Need exhaustive search on TinyLlama for empirical bound |
| "Eviction is free" | ⚠️ Unexplained | Need 10x+ compression to find failure point |

### Remaining theory actions
1. **P0**: Run 10x/15x compression to find eviction failure point (~30 min)
2. **P1**: Exhaustive search on TinyLlama to bound greedy gap (~20 min)
3. Paper already updated with coverage validation + corrected fidelity constants

---

## 2. New Competitors (Scout, last 6 months)

### Must-cite (added to paper)
| Paper | Venue | Relation |
|-------|-------|----------|
| **KVmix** | AAAI 2026 | Gradient-based per-layer mixed-precision KV quant. Direct competitor. ✅ Added |

### Should-cite (for revision)
| Paper | Venue | Relation |
|-------|-------|----------|
| LookaheadKV | arXiv 2026 | Lookahead eviction without draft generation |
| ForesightKV | arXiv 2026 | Long-term contribution learning for reasoning |
| ARKV | arXiv 2026 | Adaptive KV cache under memory budget |
| ContiguousKV | arXiv 2026 | Granularity-aligned prefix KV management |
| "One Size Does Not Fit All" | arXiv 2026 | Token-wise adaptive compression |

### Gap confirmed
No new work does **per-layer joint (token, bits) online optimization**. LayerBudget's gap still holds.

---

## 3. Open-Source Ecosystem Integration

### Framework Landscape

| Framework | Stars | Per-Layer Budget | Joint Quant+Evict | Extension API | Integration Path |
|-----------|------:|:----------------:|:-----------------:|:-------------:|-----------------|
| **NVIDIA KVPress** | ~1K | Yes (experimental) | No (separate) | Yes (ScorerPress) | **Primary target** |
| **HF Transformers** | — | Subclassable | Subclassable | Yes (Cache base) | **Research target** |
| vLLM | 67K | No | No | No | Baseline only |
| SGLang | 25K | No | No | Storage only | Baseline only |
| KVCache-Factory | 1.3K | PyramidKV only | No | No | Related work |
| LMCache | 7.8K | N/A | N/A | Storage plugins | Complementary |

### Integration Strategy

**1. KVPress integration (highest impact)**
- Create `LayerBudgetPress(BasePress)` that overrides `compress()`
- KVPress already has `PerLayerCompressionPress` for per-layer eviction ratios
- LayerBudget would be the **first KVPress method to jointly optimize eviction + quantization**
- Clean API: inherit `BasePress`, implement scoring/compression logic

**2. HuggingFace DynamicCache subclass (easiest)**
- Create `LayerBudgetCache(DynamicCache)` that overrides `update()`
- Our codebase already uses `DynamicCache` everywhere
- Immediately usable by anyone with `transformers>=4.36`

**3. Paper positioning statement**
> "LayerBudget is the first method to jointly optimize per-layer token eviction and mixed-precision quantization. While NVIDIA KVPress supports per-layer eviction ratios via PerLayerCompressionPress, and HuggingFace supports uniform QuantizedCache, no existing framework provides joint per-layer optimization of both dimensions."

### What to include in code release
- `deltacache/` core library (already MIT licensed)
- `experiments/baselines/` — all 13 baseline implementations
- `experiments/bench_unified_quality.py` — unified evaluation pipeline
- KVPress wrapper (if time permits)
- HF DynamicCache subclass

---

## 4. Priority Order for Remaining Work

| # | Task | Type | Time | Impact |
|---|------|------|------|--------|
| 1 | 10x/15x compression experiment | Experiment | 30min | High — validates "eviction is free" claim |
| 2 | Greedy vs exhaustive on TinyLlama | Theory | 20min | Medium — bounds approximation quality |
| 3 | KVPress `LayerBudgetPress` wrapper | Code | 2hr | High — ecosystem integration |
| 4 | HF `LayerBudgetCache` class | Code | 1hr | Medium — ease of use |
| 5 | Expanded MMLU (50+ questions) | Experiment | 1hr | Medium — statistical reliability |
| 6 | Cross-layer Jaccard analysis | Analysis | 10min | Low — curiosity |
