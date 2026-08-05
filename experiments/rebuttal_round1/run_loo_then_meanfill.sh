#!/bin/bash
# Sequential runner: LOO on second model → mean-fill KL across models.
# Run on GPU 1 (free). Total est: ~4h.
set -euo pipefail
cd /home/coder-gw/Projects/DeltaCache/experiments/rebuttal_round1
export CUDA_VISIBLE_DEVICES=1

echo "=== [$(date)] Starting LOO on second model ==="
python3 loo_second_model.py 2>&1 | tee results/loo_second_model.log

echo "=== [$(date)] Starting mean-fill KL across models ==="
python3 mean_fill_kl_more_models.py 2>&1 | tee results/mean_fill_kl.log

echo "=== [$(date)] All sequential experiments complete ==="
