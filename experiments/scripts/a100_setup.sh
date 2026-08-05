#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
#  A100 Quick Setup — run this FIRST on a fresh rented machine
# ═══════════════════════════════════════════════════════════════════
#
#  Usage (on the rented A100):
#    curl -sL <raw_url>/a100_setup.sh | bash
#    # or:
#    git clone <repo> DeltaCache && cd DeltaCache
#    bash experiments/scripts/a100_setup.sh
#
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

echo "═══ DeltaCache A100 Setup ═══"

# 1. System packages (if needed)
if ! command -v nvcc &>/dev/null; then
    echo "WARN: nvcc not found. Assuming CUDA is pre-installed via driver."
fi

# 2. Python environment
echo "Installing Python packages..."
pip install -q --upgrade pip setuptools wheel

# Core ML stack
pip install -q \
    torch \
    transformers \
    tokenizers \
    datasets \
    accelerate \
    bitsandbytes \
    scipy \
    tqdm \
    click

# Install deltacache itself
if [ -f "pyproject.toml" ]; then
    pip install -q -e ".[all]"
    echo "deltacache installed from local source"
elif [ -d "../DeltaCache" ]; then
    cd ../DeltaCache && pip install -q -e ".[all]" && cd -
else
    echo "ERROR: Cannot find DeltaCache project root"
    exit 1
fi

# 3. uni-layer (for Exp A layer mechanism analysis)
# Install from the Engineering repo if available, otherwise skip
if python3 -c "import uni_layer" 2>/dev/null; then
    echo "uni-layer already installed"
else
    echo "WARN: uni-layer not installed. Install it manually if you want Exp A:"
    echo "  pip install uni-layer  # or install from local source"
fi

# 4. HuggingFace login check
if ! python3 -c "from huggingface_hub import HfApi; HfApi().whoami()" 2>/dev/null; then
    echo ""
    echo "═══ HuggingFace Login Required ═══"
    echo "Llama models need HF authentication."
    echo "Run: huggingface-cli login"
    echo ""
fi

# 5. GPU check
echo ""
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'GPU {i}: {props.name} ({props.total_mem/1e9:.0f}GB)')
"

# 6. Quick import test
python3 -c "
import deltacache
import transformers
import datasets
import bitsandbytes
print('All imports OK')
"

echo ""
echo "═══ Setup Complete ═══"
echo ""
echo "Next steps:"
echo "  1. huggingface-cli login  (if not already done)"
echo "  2. cd experiments"
echo "  3. bash scripts/a100_one_shot.sh 2>&1 | tee a100_run.log"
echo ""
echo "Estimated runtime: 10-12h"
echo "Estimated cost: ~\$16-20 (A100 80GB @ \$1.60/hr)"
