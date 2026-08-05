# DeltaCache/LayerBudget Competitive Analysis

**Date:** 2026-03-27
**Status:** Track 1 (Prefix Caching) abandoned; Track 2 (LayerBudget) is the active direction

---

## 1. Track 1 (Prefix Caching) — Post-mortem

### ICLR SPOT Workshop — REJECTED (scores: 6/7/4)
- **R1 (6):** Missing prefill detail, small eval (15 queries), baselines not in main tables
- **R2 (7):** Method slows down multi-turn + short prompts; limited to one model family; no vLLM/sglang integration
- **R3 (4):** Core argument muddled (recomputation is layer-agnostic); undefined hit rate; missing e2e speedup under pressure; weak ablation; thin eval (2 runs, 15 queries, no variance)

### Why abandon:
- CMU InfiniAI lab (Beidi Chen) actively competing in this space (ShadowKV → ICML 2025 Spotlight)
- Fundamental weakness: layer-aware eviction for prefix caching doesn't hold up — recomputation cost is layer-agnostic
- Fails on common workloads (multi-turn, short prompts)

---

## 2. Track 2 (LayerBudget) — Competitive Landscape

### 2.1 LayerBudget's Unique Position

**No existing work simultaneously does:**
1. Per-layer token budget allocation ✓
2. Per-layer quantization precision selection ✓
3. Joint optimization under unified memory constraint ✓
4. Online (training-free, no offline calibration) ✓

### 2.2 Closest Competitors

| Paper | Venue | What it does | What LayerBudget adds |
|-------|-------|-------------|----------------------|
| **CAKE** | ICLR 2025 | Per-layer eviction (entropy × variance) | + quantization dimension |
| **KVTuner** | ICML 2025 | Per-layer mixed-precision quant | + eviction dimension |
| **EvolKV** | EMNLP 2025 | Evolutionary per-layer budget search | + quantization + analytical allocation (vs search) |
| **EVICPRESS** | arXiv Dec 2025 | Joint evict+quant | System-level, not per-layer model-intrinsic |
| **MiniKV** | ACL 2025 | 2-bit + pyramid budget | Fixed pyramid + uniform bits vs adaptive |
| **D2O** | ICLR 2025 | Layer+token dynamic allocation | + quantization dimension |

### 2.3 All Relevant Baselines (by category)

#### Per-Layer Token Eviction (no quantization)
| Method | Venue | Signal | Per-layer? | Open Source |
|--------|-------|--------|:---:|:---:|
| H2O | NeurIPS 2023 | Cumulative attention | No (uniform) | ✓ FMInference/H2O |
| SnapKV | NeurIPS 2024 | Observation window voting | No (uniform) | ✓ FasterDecoding/SnapKV |
| PyramidKV | COLM 2025 | Fixed pyramid shape | Yes (heuristic) | ✓ KVCache-Factory |
| CAKE | ICLR 2025 | Entropy × temporal variance | Yes (adaptive) | ✓ antgroup/cakekv |
| D2O | ICLR 2025 | Attention diversity | Yes (adaptive) | ✓ AIoT-MLSys-Lab/D2O |
| SqueezeAttention | ICLR 2025 | Cosine sim before/after attn | Yes (binary) | ✓ hetailang/SqueezeAttention |
| DynamicKV | EMNLP 2025 Findings | Attention score variance | Yes (adaptive) | ✓ DreamMr/DynamicKV |
| LAVa | EMNLP 2025 Findings | Residual stream info loss | Yes (adaptive) | ✗ reproduce |
| EvolKV | EMNLP 2025 Findings | Evolutionary search | Yes (searched) | ✗ reproduce |
| Ada-KV | NeurIPS 2025 | Per-head entropy | Head-wise | ✓ FFY0/AdaKV |
| Lethe | AAAI 2026 | Forgetting score | Yes + temporal | ✗ reproduce |
| SpindleKV | ACL 2025 | Depth-dependent strategy | Yes (zoned) | ? |

#### Per-Layer / Mixed-Precision Quantization (no eviction)
| Method | Venue | Strategy | Per-layer? | Open Source |
|--------|-------|---------|:---:|:---:|
| KIVI | ICML 2024 | Asymmetric INT2/INT4 | No (uniform) | ✓ jy-yuan/KIVI |
| KVTuner | ICML 2025 | Sensitivity-based mixed-precision | Yes | ✓ cmd2001/KVTuner |
| XQuant | EMNLP 2025 | Cross-layer, sub-1.4 bit | Yes | ✓ brinenick511/XQuant |
| TurboQuant | ICLR 2026 (Google) | 3-bit zero-loss | No (uniform) | ? |
| KVTC | ICLR 2026 (NVIDIA) | PCA + adaptive quant | Per-layer | ? |
| Kitty | MLSys 2026 | Channel-level 2-bit | Per-channel | ? |

#### Joint Approaches
| Method | Venue | Per-layer both? | Online? |
|--------|-------|:---:|:---:|
| "More Tokens Lower Precision" | EMNLP 2025 | No (uniform) | Yes |
| MiniKV | ACL 2025 | Partial (pyramid+uniform 2-bit) | Yes |
| EVICPRESS | arXiv Dec 2025 | System-level (not model-intrinsic) | Yes |
| **LayerBudget (ours)** | — | **Yes (Gini + importance)** | **Yes** |

### 2.4 Confirmed Venue Corrections
- PyramidKV → **COLM 2025** (not ICLR as previously assumed)
- SqueezeAttention → **ICLR 2025**
- ShadowKV → **ICML 2025 Spotlight**
- Ada-KV → **NeurIPS 2025**
- Twilight → **NeurIPS 2025 Spotlight**

---

## 3. Venue Strategy

| Venue | Deadline | Fit | Notes |
|-------|----------|-----|-------|
| **EMNLP 2026 (ARR May)** | May 25 | ⭐⭐⭐⭐⭐ | Many KV cache papers published here (DynamicKV, LAVa, EvolKV, XQuant) |
| **NeurIPS 2026** | ~May mid | ⭐⭐⭐⭐ | Ada-KV, Twilight published here. Tight timeline |
| **MLSys 2027** | ~Oct | ⭐⭐⭐⭐⭐ | Best systems fit. Time for larger models |
| **ACL 2026 (ARR)** | ARR cycle | ⭐⭐⭐⭐ | SpindleKV, MiniKV, MoQAE published here |
| ICML Workshop | ~Apr | ⭐⭐⭐ | Wastes experiment volume for 4-page paper |

---

## 4. Key Risks

| Risk | Impact | Mitigation |
|------|--------|-----------|
| KVTuner results look too good in our eval | Unfair comparison | Verify reproduction against their reported numbers |
| EvolKV's searched allocation beats analytical allocation | Weakens contribution | Show analytical is faster (1ms vs minutes) + comparable quality |
| TurboQuant (ICLR 2026) zero-loss at 3-bit | Strong uniform baseline | Our advantage is at higher compression (4-6x) where uniform fails |
| Reviewer asks for 13B+ model | Weak eval | Plan for Llama-2-13B if time/GPU permits |
