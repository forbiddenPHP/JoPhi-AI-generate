#!/usr/bin/env bash

# ── RVC Worker — Start API Server ────────────────────────────────────────────
# Starts the rvc-python API server on localhost:5100
# ─────────────────────────────────────────────────────────────────────────────

CONDA_BIN="/opt/miniconda3/bin/conda"
ENV_NAME="rvc"
PORT="${1:-5100}"
MODELS_DIR="$(cd "$(dirname "$0")" && pwd)/models"

if ! "$CONDA_BIN" env list 2>/dev/null | grep -q "^${ENV_NAME} "; then
    echo "ERROR: '$ENV_NAME' env not found. Run: ./worker/rvc/install.sh"
    exit 1
fi

echo "Starting RVC API server on port $PORT …"
echo "  Stop with Ctrl+C"
echo ""

eval "$($CONDA_BIN shell.bash hook)"
conda activate "$ENV_NAME"

# faiss-cpu 1.7.3 segfaults on arm64 — disable MPS to avoid further issues
export PYTORCH_MPS_DISABLE=1

python -c "
import torch
torch.backends.mps.is_available = lambda: False
from rvc_python.__main__ import main
import sys
sys.argv = ['rvc_python', 'api', '-p', '$PORT', '-md', '$MODELS_DIR']
main()
"
