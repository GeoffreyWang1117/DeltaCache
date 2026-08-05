# Archived Experiment Scripts

These scripts are from earlier iterations of the experiment pipeline.

**IMPORTANT**: Most of these use the OLD `build_cache()` that fills evicted
positions from the full KV cache — making eviction appear lossless. Results
from these scripts are INVALID for eviction method comparisons.

## Current pipeline (use these instead):
- `experiments/run_corrected_matrix.py` — Main quality experiments (zero-fill + mean-fill)
- `experiments/run_corrected_ablation_mmlu.py` — Ablation + MMLU (zero-fill)
- `experiments/generate_corrected_figures.py` — Figure generation
- `experiments/baselines/` — All 13 baseline implementations (unified interface)
