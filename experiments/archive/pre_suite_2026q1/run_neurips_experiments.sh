#!/bin/bash
# Run all experiments needed for NeurIPS 2026 submission.
# Fills Table 1 gaps and diagnoses Llama-3.1 6x issue.
#
# Expected GPU time: ~4-5 hours total on RTX 3090
# Run from: experiments/ directory

set -e
cd "$(dirname "$0")"

# Use GPU 1 (more free memory)
export CUDA_VISIBLE_DEVICES=1

echo "=============================================="
echo " NeurIPS 2026 Experiment Pipeline"
echo " $(date)"
echo "=============================================="

# ── Step 1: Mistral-7B @ 1024 tokens (~1.5 hrs) ──
echo ""
echo ">>> Step 1/3: Mistral-7B @ 1024tok"
python run_1024_and_mmlu.py \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --model-short Mistral-7B \
    --exps 27

# ── Step 2: Llama-3.1-8B @ 1024 tokens (~1.5 hrs) ──
echo ""
echo ">>> Step 2/3: Llama-3.1-8B @ 1024tok"
python run_1024_and_mmlu.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --model-short Llama-3.1-8B \
    --exps 27

# ── Step 3: Llama-3.1 diagnostic (6x degradation) (~1 hr) ──
echo ""
echo ">>> Step 3/3: Llama-3.1-8B diagnostic (6x issue)"
python diagnose_a2.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --model-short Llama-3.1-8B \
    --seq-len 512 \
    --cr 6.0 \
    --exps b c

echo ""
echo "=============================================="
echo " All experiments complete!"
echo " Results in: experiments/results/main_v2/"
echo "             experiments/results/diagnose/"
echo "=============================================="
