# Archived: pre-suite one-off runners (2026 Q1)

These scripts ran between 2025-12 and 2026-04, **before** `experiments/suite/` became the
unified runner. They are superseded for future work but are kept because **they are the
provenance for numbers that appear in `paper/neurips2026/main.tex`**. Do not delete them;
without them several published figures become untraceable.

**Do not trust any number these produced without re-deriving it.** Several
were found to be wrong or to rest on n=2 samples.

## Script → results directory → what it produced

| Script | Writes to `experiments/results/` | Feeds which paper claim | Audit status |
|---|---|---|---|
| `run_profile_and_optimality.py` | `profile_optimality/` | "greedy achieves 118–119% of 500-trial random search" | **WRONG.** Source data is 115.6–120.9%, n=2 prompts. See CLAIM_VERIFY §D1 |
| `run_fill_and_sweep.py` | `fill_and_sweep/` | mean-fill vs zero-fill vs median/random ablation | n=2 texts |
| `run_corrected_ablation_v2.py` | `ablation_v2/` | importance-direction ablation (inverted / sigmoid / U-shaped) | Note: several paper tables labelled "LayerBudget" carry the *sigmoid* row, not the stated `inverted` default |
| `run_signal_analysis.py` | `signal_analysis/` | Gini ⟂ importance correlation (ρ=0.194) | measured on TinyLlama-1.1B, which is in no results table |
| `run_system_benchmarks.py` | `system/` | TPOT / peak-memory table | Shows LayerBudget **slower** at identical peak memory (41.0→43.0 ms/token, 282.3 MB both) |
| `bench_e2e_real.py` | `e2e/` | end-to-end memory savings | — |
| `bench_vllm_layerbudget.py` | `e2e/` | "end-to-end vLLM integration confirms 24–72% memory savings" | **UNSUPPORTED.** Every output file reports `mean_freed_pct: 0.0`. See CLAIM_VERIFY §D3 |
| `bench_sglang_comparison.py` | `e2e/` | SGLang comparison | `sglang_url: "skipped"` in all outputs |
| `run_main_results_v2.py`, `run_1024_all_models.py`, `run_1024_and_mmlu.py` | `main_v2/` | early main-results tables | superseded by `suite/` |
| `run_corrected_matrix.py`, `run_corrected_ablation_mmlu.py` | `paper/` | ICML-era tables | superseded |
| `run_longbench.py` | `longbench/` | LongBench | harness is broken at the implementation level; scores unusable |
| `run_ruler.py` | `ruler/` | RULER | superseded by `suite/tasks/ruler.py` |
| `run_mmlu_expanded.py` | `mmlu/` | MMLU | prompts too short to discriminate compression |
| `run_allocator_fixes.py` | `allocator_fixes/` | allocator debugging | — |
| `diagnose_a2.py` | `diagnose/` | A2 ablation diagnosis | — |
| `generate_neurips_figures.py`, `generate_corrected_figures.py` | `figures/` | paper figures | Figure 1 (right) contradicts its own caption; Figs 2–3 drawn on TinyLlama |
| `run_neurips_experiments.sh` | — | driver for the April NeurIPS batch | superseded by `suite/run_all.py` |

## Moved elsewhere, not archived

`run_layerwise_and_distortion.py` → **`experiments/faithfulness/measure_output_divergence.py`**.
It is the only script here that is *forward-looking*: it measures output divergence between a
compressed-cache run and an uncompressed reference, which is the measurement the current
research direction is built on. See `experiments/faithfulness/README.md`.

## Planning docs

`EXPERIMENT_PLAN.md` (2025-12) and `EXPERIMENT_MATRIX.md` (2026-03) describe the prefix-caching
and early-LayerBudget plans. Both directions are closed. Kept for history only.
