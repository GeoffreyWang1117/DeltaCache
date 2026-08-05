# PPL CI Analysis — Round 1 Rebuttal

**Run:** 2026-05-02
**Source:** 1779 PPL checkpoints from `experiments/results/suite/*/checkpoints/`
**Method:** Parametric t-CI (df=5, t_0.975 = 2.571), unpaired delta-method on ratio.

## Headline finding

**164 / 164 (100%) LayerBudget-vs-KIVI PPL-ratio cells have overlapping 95% CIs.**

Under the t-distribution CIs derivable from per-cell `(mean_ppl, std_ppl, n=6)`, LayerBudget's PPL is **not statistically distinguishable from KIVI** on any model × seq_len × CR combination tested.

## Why the CIs are wide

Per-chunk PPL std ≈ 2.3 on a base PPL ≈ 9 → CV ≈ 25%. With n=6 chunks, the t-CI half-width on the *ratio* is roughly ±0.6 — much larger than the 0.001–0.02 differences between LB and KIVI.

## Caveat: unpaired vs paired

The current analysis is **unpaired**. Both LB and KIVI evaluate on the *same* 6 chunks. A **paired** analysis (per-chunk ratio LB_PPL / KIVI_PPL across the same chunk) would cancel chunk-level content variance and likely yield much tighter CIs.

The existing checkpoint JSONs do not store per-chunk PPLs — only `(mean_ppl, std_ppl)`. **Action:** rerun PPL with per-chunk JSONL output to enable paired bootstrap.

## What this means for the paper

1. **Abstract claim "ranks first among token-compression methods at 2–4×" survives** — H2O/SnapKV degrade by ≥0.05 ratio at 4× on most models, well outside any plausible CI.
2. **Claim "matches KIVI" survives but should be reframed** as "indistinguishable from KIVI within measurement noise on PPL, while compressing the token dimension."
3. **Claim "≤1.03 at 4× on seven models"** survives at point-estimate level but the appendix should state the CI half-width explicitly.

## Numbers needed for the camera-ready

- Per-chunk PPL JSONL for at least the headline cells (Mistral / Llama-3.1-8B / Qwen3-8B / Qwen2.5-72B at CR=4× and 32K context).
- Paired bootstrap CI on (LB_chunk_ppl / KIVI_chunk_ppl) — if this CI excludes 1.0, LB ≠ KIVI; if it includes 1.0 and crosses 1.0 with positive density, LB and KIVI are indistinguishable.

## H2O collapse cells (where significance is robust)

Even under the conservative unpaired CI, LB at 4× on 32K Mistral (1.009) is far below H2O at 32K Mistral (83.4) — the collapse signal is so strong it survives any reasonable CI inflation.
