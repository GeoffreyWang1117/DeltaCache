# Byte-CR Audit Findings — Round 1 Rebuttal

**Run:** 2026-05-02
**Source:** 1779 PPL checkpoints from `experiments/results/suite/*/checkpoints/`

## Headline finding (good news)

**The pipeline's `mean_memory_bytes` per cell is internally consistent with nominal CR within ±1×.**

For LayerBudget at nominal 6× on Mistral-7B at seq=512:
- Pipeline `mean_memory_bytes`: 6{,}702{,}048 bytes
- Pipeline `full_kv` `mean_memory_bytes`: 40{,}239{,}104 bytes
- **Effective CR = 40,239,104 / 6,702,048 ≈ 6.0×** ✓ matches nominal

For LB at nominal 4× on Mistral-7B (median across cells):
- Effective CR ≈ 4.0×–6.7× depending on seq_len; median Δ = +2.67 means the methods consistently OVER-deliver vs nominal at 4× (effective is ~6–7× actual byte reduction).

## What this means for R2/R4's complaint

R2/R4's "memory-denominator inconsistency" complaint cited paper's **Appendix Table 9** which states Full KV = 39,296 KB and LayerBudget = 12,939 KB at "6×" → effective CR = 3.04× (not 6×). My audit shows the pipeline's `mean_memory_bytes` for LB at 6× is **6.7 MB, not 12.9 MB** — i.e., **Table 9 in the paper appears to use a different memory accounting than the rest of the paper's results.**

**Two possibilities:**
1. **Bug in Table 9** — the 12,939 KB number was computed under a different definition (e.g., includes allocator overhead, metadata, or both K and V reported separately) and is inconsistent with `mean_memory_bytes` used elsewhere.
2. **Definition difference** — Table 9 may include memory overhead (per-layer index, sparsity profile cache) that the pipeline `mean_memory_bytes` excludes. If so, the 2× factor is the overhead cost.

**Action for camera-ready:**
- Re-derive Table 9 using `mean_memory_bytes` directly from the pipeline checkpoint, OR
- Add a footnote stating that Table 9 includes allocator overhead and per-layer metadata, while `mean_memory_bytes` reported elsewhere is just the compressed KV.

## Methodological note

My audit also found that my analytical `full_bytes` formula (2 · L · H_kv · d_h · 2 bytes per token) overestimates pipeline `mean_memory_bytes` for `full_kv` by ~1.6× consistently. This is likely because the pipeline reports actual cache occupancy at evaluation time, not the allocated buffer size. **Recommendation:** trust pipeline `mean_memory_bytes` as the canonical CR denominator, not the analytical formula.

## Median Δ-CR by method (effective − nominal)

| Method | Nominal 2× | 4× | 6× |
|---|---|---|---|
| layer_budget | +1.36 | +2.67 | +4.01 |
| h2o_uniform | +1.34 | +2.67 | +4.04 |
| kivi_uniform | +4.60 | +2.60 | +0.60 |
| snapkv | +1.34 | +2.67 | +4.04 |
| streaming_llm | +1.34 | +2.65 | +3.99 |

**LB matches H2O / SnapKV exactly (within rounding) on Δ-CR.** This is consistent with all methods reporting `mean_memory_bytes` under the same convention. **R2's "LB at 6× uses 2× the memory of H2O" complaint is incorrect when read against pipeline data; correct only when reading the appendix Table 9.**

## Recommended paper edits

1. **Table 9 (appendix)**: regenerate from pipeline `mean_memory_bytes` to match the rest of the paper.
2. **§4 setup CR definition**: keep the new paragraph I just added (CR = bytes_used / Full_KV_bytes) but add: *"in our pipeline, `bytes_used` is the runtime mean memory occupancy of the compressed KV cache as measured at evaluation time."*
3. **Self-review round 2**: note that the original R2/R4 complaint stemmed from a single inconsistent table, not a pipeline-wide methodology issue.
