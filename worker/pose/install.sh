#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONDA_BIN="${CONDA_BIN:-$(cat ~/.ai-conda-path 2>/dev/null || command -v conda 2>/dev/null || echo conda)}"
ENV_NAME="openpose"
WHEELS_DIR="$SCRIPT_DIR/wheels"
LOCKFILE="$SCRIPT_DIR/requirements.lock"

echo "── Pose Worker (DWPose) ──"

if "$CONDA_BIN" env list 2>/dev/null | grep -q "^${ENV_NAME} " || [ -d "$(dirname "$(dirname "$CONDA_BIN")")/envs/$ENV_NAME" ]; then
    echo "  Removing old '$ENV_NAME' env …"
    "$CONDA_BIN" env remove -y -n "$ENV_NAME" > /dev/null 2>&1
    rm -rf "$(dirname "$(dirname "$CONDA_BIN")")/envs/$ENV_NAME" 2>/dev/null || true
fi

echo "  Creating env: $ENV_NAME (Python 3.12) …"
"$CONDA_BIN" create -y -q -n "$ENV_NAME" python=3.12 > /dev/null 2>&1

if [ -d "$WHEELS_DIR" ] && [ "$(ls -A "$WHEELS_DIR"/*.whl 2>/dev/null)" ]; then
    echo "  Using cached wheels (offline) …"
    "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
        pip install -q --no-index --find-links "$WHEELS_DIR" -r "$LOCKFILE"
elif [ -f "$LOCKFILE" ]; then
    echo "  Using pinned versions from requirements.lock …"
    "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
        pip install -q -r "$LOCKFILE"
else
    echo "  No lockfile or wheels found, installing from PyPI …"
    "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
        pip install -q dwpose torch onnxruntime opencv-python numpy
    echo "  Generating requirements.lock …"
    "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
        pip freeze > "$LOCKFILE" 2>/dev/null
fi

# ── Verify ──
if "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python -c "
from dwpose.wholebody import Wholebody
import torch
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'  DWPose OK (torch {torch.__version__}, device: {device})')
" 2>/dev/null; then
    echo "✓ openpose env ready"
else
    echo "✗ openpose env verification failed"
    exit 1
fi
