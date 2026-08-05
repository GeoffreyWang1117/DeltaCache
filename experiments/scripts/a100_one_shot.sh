#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
#  A100 One-Shot Experiment Script — DeltaCache / LayerBudget
# ═══════════════════════════════════════════════════════════════════
#
#  Modes:
#    bash a100_one_shot.sh 7b       # Session 1: 7B + Exp A/B (~10h, $16)
#    bash a100_one_shot.sh 13b      # Session 2: 13B (~13h, $21)
#    bash a100_one_shot.sh all      # Everything (~24h, $38)
#    bash a100_one_shot.sh quick    # Critical only: 7B PPL+MMLU (~4h, $6)
#
#  Prerequisites:
#    pip install -e ".[all]" && pip install datasets bitsandbytes accelerate
#    huggingface-cli login
#
#  What this produces:
#    results/suite/<run_id>/         — Full benchmark data
#    results/exp_a/*.json            — Layer mechanism analysis
#    results/exp_b/*.json            — Quantize-first optimality
#    ~/.cache/duoattention_profiles/ — DuoAttention head profiles
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

MODE="${1:-7b}"  # default: 7b session

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$EXP_DIR")"

cd "$EXP_DIR"

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] WARN:${NC} $*"; }
err()  { echo -e "${RED}[$(date +%H:%M:%S)] ERROR:${NC} $*"; }

log "Mode: $MODE"
case "$MODE" in
    7b|13b|all|quick) ;;
    *) err "Unknown mode: $MODE. Use: 7b, 13b, all, quick"; exit 1 ;;
esac

# ── Pre-flight checks ───────────────────────────────────────────
log "Pre-flight checks..."

if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    err "CUDA not available. Aborting."
    exit 1
fi

GPU_INFO=$(python3 -c "
import torch
name = torch.cuda.get_device_name(0)
props = torch.cuda.get_device_properties(0)
mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
print(f'{name} ({mem/1e9:.0f}GB)')
" 2>/dev/null || echo "unknown GPU")
log "GPU: $GPU_INFO"

# Verify key packages
python3 -c "import transformers, datasets, bitsandbytes" 2>&1 || {
    warn "Installing missing packages..."
    pip install -q transformers datasets bitsandbytes accelerate
}

# Verify deltacache is importable
python3 -c "import deltacache" 2>&1 || {
    warn "Installing deltacache..."
    cd "$PROJECT_ROOT" && pip install -e ".[all]" -q && cd "$EXP_DIR"
}

# Verify uni-layer (for Exp A)
python3 -c "import uni_layer" 2>/dev/null || {
    warn "uni-layer not found. Exp A will be skipped."
    SKIP_EXP_A=1
}
SKIP_EXP_A=${SKIP_EXP_A:-0}

log "All checks passed."
echo ""

# ── Timer ────────────────────────────────────────────────────────
TOTAL_START=$(date +%s)

phase_timer() {
    local start=$1 label=$2
    local elapsed=$(( $(date +%s) - start ))
    local mins=$(( elapsed / 60 ))
    log "$label completed in ${mins}m"
}

# ── Helper: profile DuoAttention for a list of models ────────────
profile_duo() {
    local models=("$@")
    for MODEL in "${models[@]}"; do
        SHORT=$(python3 -c "
import sys; sys.path.insert(0,'.')
from suite.config import MODEL_ZOO
print(MODEL_ZOO['$MODEL'].short_name)
" 2>/dev/null)

        PROFILE="$HOME/.cache/duoattention_profiles/${SHORT}.pt"
        if [ -f "$PROFILE" ]; then
            log "  $MODEL profile exists, skipping"
        else
            log "  Profiling $MODEL..."
            python3 scripts/profile_duo_attention.py \
                --model "$MODEL" \
                --calib-samples 8 --calib-len 1024 --epochs 4 \
                --device cuda || warn "  $MODEL profile failed (non-fatal)"
        fi
    done
}

# ═════════════════════════════════════════════════════════════════
#  QUICK MODE: Critical experiments only (~4h)
#  PPL + MMLU + MATH on 7B, single Exp A, single Exp B
# ═════════════════════════════════════════════════════════════════
if [ "$MODE" = "quick" ]; then
    log "═══ QUICK MODE: Critical 7B experiments ═══"
    P_START=$(date +%s)

    profile_duo llama3.1-8b mistral-7b llama2-7b qwen3-8b

    python3 -m suite.run_all \
        --tier paper \
        --model-size 7b \
        --hardware a100_80g \
        --gpu-speedup 2.5 \
        --tasks ppl,mmlu,math

    if [ "$SKIP_EXP_A" = "0" ]; then
        log "  Exp A: llama3.1-8b"
        python3 scripts/exp_a_layer_mechanism.py \
            --model llama3.1-8b \
            --n-samples 10 --seq-len 2048 \
            --device cuda || warn "  Exp A failed (non-fatal)"
    fi

    log "  Exp B: llama3.1-8b"
    python3 scripts/exp_b_quantize_first.py \
        --model llama3.1-8b \
        --seq-len 2048 \
        --device cuda || warn "  Exp B failed (non-fatal)"

    phase_timer $P_START "Quick mode"
fi

# ═════════════════════════════════════════════════════════════════
#  7B MODE: Full 7B benchmark + Exp A/B (~10h)
# ═════════════════════════════════════════════════════════════════
if [ "$MODE" = "7b" ] || [ "$MODE" = "all" ]; then
    # Phase 0: DuoAttention profiles
    log "═══ PHASE 0: DuoAttention Profiles (7B) ═══"
    P0_START=$(date +%s)
    profile_duo llama2-7b mistral-7b llama3.1-8b qwen3-8b
    phase_timer $P0_START "Phase 0 (profiles)"
    echo ""

    # Phase 1: Full 7B benchmark (auto-resumes if interrupted)
    log "═══ PHASE 1: 7B Full Benchmark ═══"
    P1_START=$(date +%s)
    python3 -m suite.run_all \
        --tier paper \
        --model-size 7b \
        --hardware a100_80g \
        --gpu-speedup 2.5 \
        --resume latest
    phase_timer $P1_START "Phase 1 (7B benchmark)"
    echo ""

    # Phase 2: Exp A
    if [ "$SKIP_EXP_A" = "0" ]; then
        log "═══ PHASE 2: Exp A — Layer Mechanism ═══"
        P2_START=$(date +%s)
        for MODEL in llama3.1-8b mistral-7b llama2-7b; do
            log "  Exp A: $MODEL"
            python3 scripts/exp_a_layer_mechanism.py \
                --model "$MODEL" \
                --n-samples 10 --seq-len 2048 \
                --device cuda || warn "  Exp A $MODEL failed (non-fatal)"
        done
        phase_timer $P2_START "Phase 2 (Exp A)"
    else
        warn "Skipping Exp A (uni-layer not installed)"
    fi
    echo ""

    # Phase 3: Exp B
    log "═══ PHASE 3: Exp B — Quantize-First ═══"
    P3_START=$(date +%s)
    for MODEL in llama3.1-8b mistral-7b llama2-7b; do
        log "  Exp B: $MODEL"
        python3 scripts/exp_b_quantize_first.py \
            --model "$MODEL" \
            --seq-len 2048 \
            --device cuda || warn "  Exp B $MODEL failed (non-fatal)"
    done
    phase_timer $P3_START "Phase 3 (Exp B)"
    echo ""
fi

# ═════════════════════════════════════════════════════════════════
#  13B MODE: Full 13B benchmark (~13h)
# ═════════════════════════════════════════════════════════════════
if [ "$MODE" = "13b" ] || [ "$MODE" = "all" ]; then
    log "═══ PHASE 4: 13B Benchmark ═══"
    P4_START=$(date +%s)

    profile_duo llama2-13b qwen2.5-14b

    python3 -m suite.run_all \
        --tier paper \
        --model-size 13b \
        --hardware a100_80g \
        --gpu-speedup 2.5 \
        --resume latest
    phase_timer $P4_START "Phase 4 (13B benchmark)"
    echo ""
fi

# ═════════════════════════════════════════════════════════════════
#  PHASE 5: Aggregate all results
# ═════════════════════════════════════════════════════════════════
log "═══ PHASE 5: Aggregate Results ═══"

# Find all run IDs and aggregate
for RUN_DIR in results/suite/*/; do
    RUN_ID=$(basename "$RUN_DIR")
    log "  Aggregating $RUN_ID..."
    python3 -m suite.aggregate --run-id "$RUN_ID" 2>&1 | tail -5 || true
done

# Summary of Exp A/B
echo ""
log "═══ Exp A Summaries ═══"
for f in results/exp_a/*_summary.txt; do
    [ -f "$f" ] && cat "$f" && echo ""
done

log "═══ Exp B Summaries ═══"
for f in results/exp_b/*_summary.txt; do
    [ -f "$f" ] && cat "$f" && echo ""
done

# ═════════════════════════════════════════════════════════════════
#  DONE
# ═════════════════════════════════════════════════════════════════
TOTAL_ELAPSED=$(( $(date +%s) - TOTAL_START ))
TOTAL_HOURS=$(echo "scale=1; $TOTAL_ELAPSED / 3600" | bc)
TOTAL_COST=$(echo "scale=2; $TOTAL_HOURS * 1.60" | bc)

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ALL PHASES COMPLETE"
echo "  Total time: ${TOTAL_HOURS}h"
echo "  Estimated cost: \$${TOTAL_COST}"
echo ""
echo "  Results:"
echo "    Suite runs:  results/suite/*/"
echo "    Exp A:       results/exp_a/*.json"
echo "    Exp B:       results/exp_b/*.json"
echo "    Profiles:    ~/.cache/duoattention_profiles/"
echo ""
echo "  Next: copy results back to local machine:"
echo "    scp -r results/ local:DeltaCache/experiments/results/"
echo "═══════════════════════════════════════════════════════════════"
