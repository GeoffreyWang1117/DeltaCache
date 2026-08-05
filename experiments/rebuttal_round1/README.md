# Rebuttal Round 1 — Supplementary Experiments

Six supplementary experiments addressing the round-1 self-review.
Ranked by leverage (impact on review score) ÷ cost (GPU-hours).

| # | Experiment | Addresses | GPU-hours (est.) | Leverage |
|---|---|---|---|---|
| 1 | `f_b_calibration.py` | R1: F(b) single-model calibration | ~0.5 | high |
| 2 | `ppl_bootstrap_ci.py` | R2: no error bars | 0 (re-analyzes existing JSONs) | high |
| 3 | `byte_cr_audit.py` | R2/R4: CR denominator inconsistency | 0 (analytical) | high |
| 4 | `loo_second_model.py` | R3: inverted-importance one-model | ~3 | very high |
| 5 | `mean_fill_kl_more_models.py` | R4: 96% KL one-model | ~1 | medium |
| 6 | `h2o_sanity_official.py` | R4: H2O MMLU 36.5% anomaly | ~2 | high |

## Run order (recommended)

```bash
# 0 GPU-hours (analytical / data analysis) — do these first
python ppl_bootstrap_ci.py
python byte_cr_audit.py

# Then run the cheap GPU experiments
python f_b_calibration.py        # ~0.5 h: F(b) cosine sim per model
python mean_fill_kl_more_models.py  # ~1 h: mean-fill KL on Llama-2/3.1-7/8B

# Finally the expensive ones
python h2o_sanity_official.py    # ~2 h: clone H2O artifact, rerun MMLU
python loo_second_model.py       # ~3 h: leave-one-out on Llama-2-13B + Qwen2.5-14B
```

Output JSONs land in `results/`; each script also writes a `_summary.md`.

## What each experiment is supposed to settle

1. **F(b) calibration**: confirms or invalidates transferring Mistral's INT4/INT8 fidelity constants to the other six models. Expected outcome: F(8) within 0.0005 of Mistral's, F(4) within 0.005. If divergence is larger, the cost model needs per-model F(b).

2. **Bootstrap PPL CI**: produces 95% CIs for every cell in Table 1, 18, 19. Likely outcome: LB vs KIVI differences at 2–4× are not statistically distinguishable; LB vs H2O at 6× and at 32K are large effects. Reframes the "matches KIVI" claim into "indistinguishable from KIVI within noise."

3. **Byte-CR audit**: recomputes effective CR (= Full_KV_bytes / method_bytes) for every method × CR cell. Populates Appendix C's memory-matched comparison table. Outcome: the "ranks 1st" claim either survives memory-matching (camera-ready) or needs reframing.

4. **LOO on second model**: critical scientific test of the inverted-importance generalization. Run on Llama-2-13B (40L MHA) and Qwen2.5-14B (48L GQA) at 6×. If early layers dominate eviction sensitivity on at least one of these, the principle is general. If late layers dominate on both, the inverted-importance is Mistral-specific and the paper should scope down.

5. **Mean-fill KL on Llama-2/Llama-3.1**: checks whether 96% KL-reduction generalizes. Single-shot.

6. **H2O sanity**: clone https://github.com/FMInference/H2O at the published commit, run MMLU on Llama-2-7B at heavy_ratio=0.1, recent_ratio=0.1 (their default for this CR). Compare to our 36.5% reimpl. Plausible outcomes: (a) reimpl bug → fix; (b) harness mismatch → document; (c) repro 36.5% → strengthens our claim and provides a citable note.
