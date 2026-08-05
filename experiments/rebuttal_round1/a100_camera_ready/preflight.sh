#!/usr/bin/env bash
# Pre-flight checks for vast.ai A100 instance.
# Run this FIRST after SSH'ing in. Must pass before launching the long jobs.

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'
log()  { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[ERR]${NC} $*"; exit 1; }

echo "=== DeltaCache A100 Pre-flight ==="

# 1. GPU
python3 -c "
import torch, sys
if not torch.cuda.is_available():
    sys.exit('CUDA not available')
props = torch.cuda.get_device_properties(0)
mem_gb = props.total_memory / 1e9
name = props.name
print(f'GPU: {name} ({mem_gb:.0f} GB)')
if mem_gb < 70:
    sys.exit(f'GPU has only {mem_gb:.0f}GB; need 80GB A100. Eager-attention at 16K will OOM on smaller cards.')
" || err "GPU check failed"
log "GPU verified (A100 80GB or larger)"

# 2. Disk
DISK_FREE_GB=$(df -BG /root 2>/dev/null | awk 'NR==2{gsub(/G/,"",$4); print int($4)}' || df -BG / | awk 'NR==2{gsub(/G/,"",$4); print int($4)}')
if [ "${DISK_FREE_GB:-0}" -lt 50 ]; then
    err "Disk free is ${DISK_FREE_GB}GB; need ≥50GB for model downloads + results"
fi
log "Disk free: ${DISK_FREE_GB}GB"

# 3. Python packages
python3 -c "import torch, transformers, datasets, bitsandbytes" 2>&1 || {
    warn "Installing core ML stack..."
    pip install -q --upgrade pip
    pip install -q torch transformers datasets bitsandbytes accelerate
}
log "Python packages OK"

# 4. deltacache importable
python3 -c "
import sys
sys.path.insert(0, '/root/DeltaCache')
import deltacache
" 2>&1 || {
    err "deltacache not importable — clone the repo to /root/DeltaCache and run 'pip install -e .'"
}
log "deltacache importable"

# 5. layer_budget_kv (the new variant)
python3 -c "
import sys
sys.path.insert(0, '/root/DeltaCache')
sys.path.insert(0, '/root/DeltaCache/experiments')
sys.path.insert(0, '/root/DeltaCache/experiments/rebuttal_round1')
import layer_budget_kv
from baselines.base import REGISTRY
assert 'layer_budget_kv' in REGISTRY, 'layer_budget_kv not registered'
" 2>&1 || err "layer_budget_kv not loadable — verify experiments/rebuttal_round1/layer_budget_kv.py exists"
log "layer_budget_kv registered"

# 6. HF auth (Llama-2 is gated)
python3 -c "
from huggingface_hub import HfApi
try:
    HfApi().whoami()
    print('HF auth OK')
except Exception as e:
    raise SystemExit(f'HF auth failed: {e}. Run: huggingface-cli login')
" 2>&1 || err "HuggingFace not authenticated (Llama-2 download will fail)"
log "HuggingFace auth OK"

# 7. Test that we can actually pull a tiny gated model file (smoke test for HF cache + network)
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='meta-llama/Llama-2-7b-chat-hf', filename='config.json', cache_dir='/root/.cache/huggingface')
" 2>&1 || err "Cannot pull Llama-2-7b config — check HF token has llama-2 access"
log "Llama-2 access OK"

echo ""
echo "=== Pre-flight passed. Launch with:"
echo "    nohup bash run_all.sh > run.log 2>&1 &"
echo "    tail -f run.log"
