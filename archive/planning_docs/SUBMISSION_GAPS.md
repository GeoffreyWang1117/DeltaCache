# Submission Gap Analysis

**Date**: 2026-03-28
**Paper**: 8 pages, ICML 2026 format, clean compile
**Target venues**: EMNLP 2026 (ARR May 25), NeurIPS 2026 (~May mid), MLSys 2027 (~Oct)

---

## A. PAPER CONTENT ISSUES

### 🔴 A1: Unreferenced tables and figures
- `tab:positioning` (Table 1): defined but never `\ref`'d in text
- `tab:memory` (Table 3): defined but never `\ref`'d in text
- `fig:pareto` (Figure 1): defined but never `\ref`'d in text
- **Fix**: Add `\ref` in appropriate paragraphs. ~5 min.

### 🔴 A2: Old figures need regeneration
- `allocation_heatmap.pdf` and `signal_curves.pdf` are from OLD experiments (TinyLlama, old pipeline)
- They don't match the new unified results on 7B models
- **Fix**: Regenerate from unified data on Llama-2-7B / Mistral-7B. ~30 min code + GPU.

### 🔴 A3: No figure for new results
- The paper has `method_heatmap_comparison.pdf` and `win_rate_chart.pdf` generated but NOT referenced
- The main results table (Table 2) is very wide (17 columns) — may overflow in 2-column format
- **Fix**: Consider splitting Table 2 into 2 tables (one per model), or use the heatmap figure instead.

### 🟡 A4: Abstract mentions "12 baselines" but Table 1 shows 9 rows
- The positioning table groups some methods (e.g., "H2O / SnapKV", "CAKE / LAVa / EvolKV")
- Some baselines in the paper (Ada-KV) have "per-head" which doesn't fit the binary columns
- **Fix**: Clarify in caption or expand table.

### 🟡 A5: Fidelity constants inconsistency
- Method section says F(8)=0.9999, F(4)=0.9964 (corrected)
- But the allocator code still uses F(8)=0.98, F(4)=0.92 (old values)
- This means ALL experiments used the OLD fidelity constants
- **Fix**: Either (a) re-run experiments with corrected constants, or (b) note that the allocator uses approximate constants and the real fidelity is higher. Option (b) is honest and sufficient.

### 🟡 A6: "Eviction is free" claim needs stronger evidence
- Current evidence: PPL ratio = 1.000 for all eviction methods at 512-4096 tokens, up to 20x
- But PPL evaluation uses zero-padding for evicted positions (positions filled from full KV in `build_cache`)
- This means we're actually evaluating with FULL KV as fallback for evicted positions — NOT truly testing eviction impact
- **This is a critical methodological issue** that could invalidate the "eviction is free" finding
- **Fix**: Need to verify — either pad with zeros (no fallback) or clarify the evaluation methodology in paper.

---

## B. EXPERIMENTAL GAPS

### 🔴 B1: Verify evaluation methodology (A6 above)
The `build_cache()` function fills evicted positions from `full_keys/full_values`:
```python
if full_keys is not None:
    k_full[0] = full_keys[layer_idx].to(device)
    v_full[0] = full_values[layer_idx].to(device)
```
This means evicted tokens are REPLACED with original values, not zeroed out. The model effectively sees the full KV cache for eviction methods — which explains why PPL ratio = 1.000.

**This is the #1 issue.** Must fix and re-evaluate.

### 🟡 B2: No long-context benchmark (LongBench, RULER)
- All evaluations use WikiText-2 perplexity and simplified MMLU
- Reviewers will ask for LongBench (standard for KV cache compression)
- **Fix**: Run LongBench on at least one model. ~2-4 hours.

### 🟡 B3: No throughput/latency measurement
- Paper only reports profiling overhead (Table 5)
- No end-to-end generation throughput comparison
- **Fix**: Measure tokens/second with compressed vs full KV. ~1 hour.

### 🟡 B4: Only 4-bit model weights
- All 7B experiments use 4-bit weight quantization (BitsAndBytes)
- This adds noise to the evaluation — weight quantization errors compound with KV quantization
- Ideally should also test FP16 weights (but needs 2x GPU memory)
- **Fix**: At minimum, acknowledge as limitation. Or use Mistral-7B FP16 on a single GPU (fits in 24GB without output_attentions).

### 🟡 B5: No standard deviation / confidence intervals
- Most PPL ratio values are averaged over 4-6 texts without reporting std
- At ratio ~1.000, differences of 0.001-0.005 may not be statistically significant
- **Fix**: Report std in tables, run significance tests between methods.

### ⬜ B6: Missing baselines
- EvolKV (EMNLP 2025) — implemented but excluded (too slow). Should include with pre-calibrated allocation.
- XQuant results have memory reporting issues (memory_bytes was buggy, now fixed but data is from pre-fix runs)
- **Fix**: Re-run XQuant with fixed memory reporting. Run EvolKV on 1-2 configs.

---

## C. THEORETICAL GAPS

### 🟡 C1: Sigmoid importance questionable
- Ablation shows sparsity_only (0.996) beats combined (0.998) on Llama-2-7B
- The importance signal may be actively hurting rather than helping
- **Fix**: Either remove importance from the method, or reframe it as "importance provides memory efficiency, not quality improvement" (which the ablation supports via lower memory for combined vs sparsity-only).

### 🟡 C2: Coverage model validated but not used in paper figures
- We validated MAE=3.9% but don't show the validation plot
- **Fix**: Add a supplementary figure or inline text showing predicted vs actual.

### ⬜ C3: No theoretical analysis of when eviction matters
- The paper empirically shows eviction is free but doesn't explain WHY
- Missing: analysis of attention mass retained at different CRs, relationship between sink/recent token count and quality preservation

---

## D. PRESENTATION GAPS

### 🟡 D1: Paper is dense — 8 pages with 5 tables + 3 figures
- ICML allows 9 pages. Could use the extra page for clearer presentation.
- Consider moving overhead table to appendix.

### 🟡 D2: No supplementary material
- Should include: full per-layer allocation visualizations, extended results tables, code availability statement

### 🟡 D3: Missing error analysis
- When LayerBudget loses to KVTuner (4 of 32 configs), what happens? Analysis of failure modes.

### ⬜ D4: No related work comparison table for baselines
- Should show which baselines have open-source code, which we reimplemented, and any known differences from original

---

## E. PRIORITY RANKING

| # | Issue | Impact | Effort | Risk if ignored |
|---|-------|--------|--------|-----------------|
| **B1** | Evaluation methodology — eviction fills from full KV | 🔴 Critical | 2hr | Paper could be rejected on this basis |
| **A2** | Old figures from TinyLlama | 🔴 Must fix | 30min | Inconsistency with new data |
| **A1** | Unreferenced tables/figures | 🔴 Easy fix | 5min | Looks careless |
| **B2** | No LongBench | 🟡 High | 4hr | Major reviewer concern |
| **A5** | Fidelity constants mismatch | 🟡 Medium | 10min | Needs explanation |
| **B5** | No std/CI in results | 🟡 Medium | 30min | Questioned statistical significance |
| **B3** | No throughput measurement | 🟡 Medium | 1hr | "Only quality, no speed" criticism |
| **A3** | Table 2 too wide | 🟡 Layout | 20min | May not render in 2-col |
| **C1** | Importance signal questionable | 🟡 Theoretical | 20min | Reviewer may point out |
| **B4** | Only 4-bit weights | ⬜ Low | Note it | Minor limitation |
| **B6** | Missing EvolKV/XQuant | ⬜ Low | 1hr | Completeness |
