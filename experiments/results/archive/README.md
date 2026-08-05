# Archived Experiment Results

## pre_correction/
Results from the unified 13-baseline pipeline BEFORE the build_cache bug was fixed.
These used full KV fallback for evicted positions, making all eviction methods
appear lossless (PPL ratio = 1.000). **DO NOT USE for paper.**

## original_pipeline/
Results from the original bench_layer_budget*.py scripts (before the 13-baseline
unified comparison was implemented). Different eval texts, different methodology.

## Other archived results
- Simulated / mock results from early development
- Old TinyLlama / Qwen2 / vLLM comparison runs

## Current results (in `experiments/results/paper/`):
- `corrected_*` — Zero-fill evaluation (corrected build_cache)
- `meanfill_*` — Mean-fill evaluation (**final paper results**)
