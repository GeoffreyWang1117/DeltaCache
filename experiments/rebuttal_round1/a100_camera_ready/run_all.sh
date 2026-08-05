#!/usr/bin/env bash
# Camera-ready A100 runner. Resumable, checkpointed, smoke-mode supported.
#
# Usage:
#   bash run_all.sh                      # full run (~75 min A100 80GB)
#   DC_SMOKE=1 bash run_all.sh           # 5-min smoke test before committing
#   bash run_all.sh niah                 # only NIAH at 4K
#   bash run_all.sh long                 # only 16K long-context
#   bash run_all.sh download <ssh-host>  # scp results back to local

set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] WARN:${NC} $*"; }
err()  { echo -e "${RED}[$(date +%H:%M:%S)] ERR:${NC} $*"; }

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

MODE="${1:-all}"
case "$MODE" in
    all|niah|long|smoke) ;;
    download)
        SSH_HOST="${2:?usage: bash run_all.sh download <ssh-host>}"
        log "Downloading results from $SSH_HOST..."
        scp -r "$SSH_HOST:$HERE/results" "./results_from_${SSH_HOST}_$(date +%Y%m%d_%H%M%S)" || err "scp failed"
        exit 0
        ;;
    *) err "unknown mode: $MODE"; exit 1 ;;
esac

# Pre-flight
if [ ! -f "preflight.sh" ]; then err "preflight.sh missing"; exit 1; fi
log "Running pre-flight checks..."
bash preflight.sh || { err "Pre-flight failed"; exit 1; }

if [ "$MODE" = "smoke" ]; then
    log "SMOKE MODE: 5-minute sanity check"
    DC_SMOKE=1 python3 -u niah_lb_kv_4k.py 2>&1 | tee niah_smoke.log
    DC_SMOKE=1 python3 -u lb_kv_long_context_16k.py 2>&1 | tee long_smoke.log
    log "Smoke run complete. Inspect logs and rerun without DC_SMOKE for full sweep."
    exit 0
fi

if [ "$MODE" = "all" ] || [ "$MODE" = "niah" ]; then
    log "=== [1/2] NIAH at 4K on LB-KV ==="
    python3 -u niah_lb_kv_4k.py 2>&1 | tee niah_lb_kv_4k.log
    log "Phase 1 done."
fi

if [ "$MODE" = "all" ] || [ "$MODE" = "long" ]; then
    log "=== [2/2] LB-KV at 16K long context ==="
    python3 -u lb_kv_long_context_16k.py 2>&1 | tee lb_kv_long_context_16k.log
    log "Phase 2 done."
fi

log ""
log "=== Done. Results in: $HERE/results/ ==="
ls -la "$HERE/results/" 2>/dev/null || true
log ""
log "To download from local machine:"
log "  scp -r vast:$HERE/results ./results_from_vast"
