# Experiment Gap Analysis

**Date**: 2026-03-28
**Goal**: Identify all missing experiments needed for a complete paper submission

---

## 1. COMPLETED EXPERIMENTS ✅

### 1.1 Main Quality (PPL) — Unified 12-Baseline Comparison
| Config | File | Status |
|--------|------|:------:|
| Llama-2-7B @ 512 tok | `unified_quality_512tok_llama2_7b.json` | ✅ |
| Llama-2-7B @ 1024 tok | `unified_quality_1024tok_llama_2_7b_chat_hf.json` | ✅ |
| Llama-2-7B @ 2048 tok | `unified_quality_2048tok_llama2_7b.json` | ✅ |
| Mistral-7B @ 512 tok | `unified_quality_512tok_mistral_7b_instruct_v0.2.json` | ✅ |
| Mistral-7B @ 1024 tok | `unified_quality_1024tok_mistral_7b_instruct_v0.2.json` | ✅ |
| Mistral-7B @ 2048 tok | `unified_quality_2048tok_mistral_7b_instruct_v0.2.json` | ✅ |
| TinyLlama @ 1024 tok | `unified_quality_1024tok_tinyllama.json` | ✅ |

### 1.2 Ablation Study (from OLD experiments, only CAKE+KVTuner baselines)
| Config | File | Status |
|--------|------|:------:|
| TinyLlama ablation | `ablation_tinyllama_1.1b_chat_v1.0.json` | ✅ (old) |
| Mistral-7B ablation | `ablation_mistral_7b_instruct_v0.2.json` | ✅ (old) |

### 1.3 Profiling Overhead (OLD, only TinyLlama)
| Config | File | Status |
|--------|------|:------:|
| TinyLlama overhead | `profiling_overhead_tinyllama_1.1b_chat_v1.0.json` | ✅ (old) |

### 1.4 Downstream Tasks (OLD, only 5 methods)
| Config | File | Status |
|--------|------|:------:|
| TinyLlama MMLU | `downstream_tinyllama_1.1b_chat_v1.0.json` | ✅ (old) |
| Llama-2-7B MMLU | `downstream_llama_2_7b_chat_hf.json` | ✅ (old) |

### 1.5 Figures
| Figure | File | Status |
|--------|------|:------:|
| PPL vs CR (all configs) | `ppl_vs_compression_all.pdf` | ✅ |
| Method heatmap comparison | `method_heatmap_comparison.pdf` | ✅ |
| Win rate chart | `win_rate_chart.pdf` | ✅ |
| Allocation heatmap | `allocation_heatmap.pdf` | ✅ (old) |
| Signal curves | `signal_curves.pdf` | ✅ (old) |
| Ablation bars | `ablation_bars.pdf` | ✅ (old) |

---

## 2. MISSING EXPERIMENTS (Must Fix)

### 2.1 🔴 P0: Ablation on Llama-2-7B (paper claims it but data is from old pipeline)
**Problem**: Paper Table 5 shows ablation on "TinyLlama" and "Mistral-7B", but:
- The ablation data is from the OLD `bench_layer_budget_quality.py` with only 5 baselines
- The old data used different eval texts (hardcoded prompts, not WikiText-2)
- Need to re-run ablations with the unified pipeline for consistency

**Action**: Run 5 ablations on Llama-2-7B @ 1024 tokens with unified pipeline:
1. Component: eviction-only vs quant-only vs joint
2. Signal: sparsity-only vs importance-only vs combined
3. Precision: {4,16} vs {4,8,16}
4. Token selection: H2O vs random
5. Profile stability: per-input vs fixed

**Effort**: ~30 min GPU time

### 2.2 🔴 P0: Profiling Overhead on 7B Models
**Problem**: Table 6 (overhead) only has TinyLlama data. Reviewers will ask "what about 7B?"

**Action**: Run `bench_profiling_overhead.py` (or equivalent) on:
- Llama-2-7B @ {512, 1024, 2048} tokens
- Mistral-7B @ {512, 1024, 2048} tokens

**Effort**: ~20 min GPU time

### 2.3 🔴 P0: Downstream Task Eval with All Baselines
**Problem**: Paper mentions MMLU results (line 333) but data is from OLD pipeline with only 5 methods. Need to run with all 12 baselines for consistency.

**Action**: Run MMLU 5-shot on:
- Llama-2-7B @ 3x and 6x compression, all 12 baselines
- Mistral-7B @ 3x and 6x compression, all 12 baselines

**Effort**: ~1-2 hours GPU time (MMLU evaluation is slow)

### 2.4 🟡 P1: Cosine Similarity with Meaningful Metric
**Problem**: Current cosine sim = 1.000 for all eviction methods (compares retained tokens to themselves). Need a metric that captures information loss from eviction.

**Options**:
- (a) Compare full-sequence reconstruction (zero-pad evicted positions) — shows position-aware loss
- (b) Use PPL ratio as the primary metric (already done) and drop cosine sim from paper
- (c) Report "attention mass retained" instead of cosine sim

**Action**: Option (b) is simplest — PPL ratio is the real quality metric. Remove cosine sim claims or clarify it only applies to quantization methods.

**Effort**: Paper editing only, no GPU time

### 2.5 🟡 P1: Longer Context (4096+ tokens) to Stress-Test Eviction
**Problem**: At 512-2048 tokens, all eviction methods are lossless. The hypothesis that eviction differentiation matters at longer contexts is untested.

**Action**: Run at least one config at 4096 tokens:
- Mistral-7B @ 4096 tokens (model supports up to 32K)
- May require `load_in_4bit` + gradient checkpointing to fit attention matrix in VRAM

**Effort**: ~30 min GPU time, may need memory optimization

### 2.6 🟡 P1: XQuant Baseline Memory Reporting Fix
**Problem**: XQuant reports `memory_bytes` as dequantized FP16 size, not actual quantized size. Shows `actual_cr = 1.0x` which is incorrect.

**Action**: Fix `xquant.py` `memory_bytes()` to compute actual quantized size based on per-layer bit assignments.

**Effort**: 10 min code fix

---

## 3. NICE-TO-HAVE EXPERIMENTS (If Time Permits)

### 3.1 EvolKV in Unified Comparison
**Status**: Implemented but excluded from experiments (too slow — ~100s per prompt)
**Action**: Run with reduced generations (already set to 15) on 1-2 configs for completeness
**Effort**: ~1 hour

### 3.2 Llama-3.1-8B / Qwen3-8B
**Status**: Models cached locally, not tested
**Action**: Run unified quality on one seq length to show generalization
**Effort**: ~30 min per model

### 3.3 Throughput / Latency Measurement
**Status**: Not measured. Only PPL quality is reported.
**Action**: Measure end-to-end inference latency with compressed vs full KV
**Effort**: ~1 hour

### 3.4 Attention Mass Retained Metric
**Status**: Not computed
**Action**: For each eviction method, compute `sum(attn[retained_positions]) / sum(attn[all_positions])` to show how much attention mass different methods capture
**Effort**: ~30 min

---

## 4. PRIORITY ORDER

| Priority | Task | GPU Time | Impact |
|----------|------|----------|--------|
| **P0** | Ablation on Llama-2-7B (unified) | 30 min | High — consistency |
| **P0** | Profiling overhead on 7B models | 20 min | High — reviewer question |
| **P0** | Fix XQuant memory reporting | 0 min | Medium — correctness |
| **P0** | Downstream MMLU (all baselines) | 1-2 hrs | High — completeness |
| P1 | Paper edit: cosine sim clarification | 0 min | Medium — clarity |
| P1 | 4096-token experiment | 30 min | Medium — stronger claim |
| P2 | EvolKV inclusion | 1 hr | Low — one more baseline |
| P2 | Llama-3.1 / Qwen3 | 30 min | Low — generalization |
| P2 | Throughput measurement | 1 hr | Medium — practical impact |

**Total P0 effort: ~2-3 hours GPU time**
**Total P0+P1 effort: ~3-4 hours GPU time**
