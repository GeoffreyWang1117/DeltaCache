# H2O Reimpl Sanity Check vs. Official Artifact — Plan

**Issue:** our H2O reimpl reports MMLU = 36.5% on Llama-2-7B at CR=4× (Table 15).
The official H2O paper (Zhang et al. 2024) claims <2% MMLU degradation at comparable CR — should be ~46% for Llama-2-7B (Full KV ≈ 47%).

A 10pp gap between our reimpl and the paper's claim is a credibility risk.

## Two scenarios

1. **Reimpl is correct, paper claim is wrong** — H2O genuinely collapses on MMLU at CR=4×. Strengthens our paper's "ranks 1st" claim and is a publishable observation in its own right.
2. **Reimpl has a tuning / harness bug** — our 36.5% is wrong. Fixing the reimpl raises H2O closer to KIVI's ~47%, and our 21pp MMLU advantage shrinks to ~1pp. Materially weakens the abstract.

## Procedure

### 1. Set up the official artifact

```bash
cd ~/Projects/DeltaCache/baselines
git clone https://github.com/FMInference/H2O h2o_official
cd h2o_official
git checkout <version-tag-from-paper>  # try main first, then v1.0 if exists
pip install -r requirements.txt
```

### 2. Run their MMLU script with CR=4×

The official H2O uses `heavy_ratio` and `recent_ratio`. For nominal CR=4× their docs typically suggest:
- `heavy_ratio=0.1, recent_ratio=0.1` (top 10% heavy + last 10% recent = 20% kept = CR=5×, close enough)

```bash
python h2o_hf/run_lm_eval_harness.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --task mmlu \
    --heavy_ratio 0.1 --recent_ratio 0.1 \
    --output official_h2o_llama2_7b_mmlu_cr4.json
```

### 3. Run our reimpl with matched settings

Our `experiments/suite` has H2O via `--method h2o_uniform --cr 4.0`. Confirm `heavy_ratio` and `recent_ratio` parameters match the official artifact (read both source files; document the mapping).

```bash
python -m experiments.suite.runner \
    --model llama2_7b --task mmlu --method h2o_uniform --cr 4.0 \
    --output ours_h2o_llama2_7b_mmlu_cr4.json
```

### 4. Compare

| Setup | MMLU |
|---|---|
| Full KV | ~47% |
| Official H2O reimpl | ? |
| Our H2O reimpl | 36.5% |

**If official H2O reproduces ~46%:** our reimpl has a bug. Search for diff in:
- token selection logic (heavy hitters: cumulative attention threshold? top-k? per-layer or global?)
- recent window inclusion (last K vs last K%)
- attention score normalization (raw scores? softmax-normalized?)
- whether MMLU prompt format triggers different tokenization

**If official H2O also drops to ~36%:** our reimpl is correct, the original paper's "<2%" claim was on a different evaluation (different MMLU subset, different prompt format, different CR definition). Document the discrepancy and cite our reproduction.

### 5. Time and cost

- Cloning + setup: 30 min
- Two MMLU runs (1140 questions × 57 subjects): ~1.5 h on A100, ~3 h on RTX 3090
- Comparison + writeup: 30 min

**Total: ~2-4 GPU hours.**

## Why this is high-leverage

The 21pp MMLU advantage is the single most striking number in the downstream-task analysis (§6 of the paper). If it shrinks to 1pp, the abstract's "matches or exceeds quantization-only quality on retrieval benchmarks" framing has to drop "exceeds." If it holds, the paper has a citable reproduction artifact that prior H2O critiques can use.

Either outcome is a paper-quality improvement.

## Expected outcome (priors)

Honest prior: **30% the official artifact reproduces ~46%, our reimpl has a bug; 60% the official artifact also drops to ~36% on MMLU at CR=4×; 10% other.** The H2O paper's <2% claim was at a *higher* token retention budget (CR ~2×), and the original MMLU evaluation may have been on a smaller subject subset. Our CR=4× is more aggressive than what most H2O reports use.

Either way, the camera-ready needs this resolved.
