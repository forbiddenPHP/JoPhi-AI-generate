#!/usr/bin/env bash
set -euo pipefail

CONDA_BIN="${CONDA_BIN:-/opt/miniconda3/bin/conda}"
ENV_NAME="openpose"

echo "── Pose Worker (DWPose) ──"

if "$CONDA_BIN" env list 2>/dev/null | grep -q "^${ENV_NAME} " || [ -d "/opt/miniconda3/envs/$ENV_NAME" ]; then
    echo "  Removing old '$ENV_NAME' env ..."
    "$CONDA_BIN" env remove -y -n "$ENV_NAME" > /dev/null 2>&1
    rm -rf "/opt/miniconda3/envs/$ENV_NAME" 2>/dev/null || true
fi

echo "  Creating env: $ENV_NAME (Python 3.12) ..."
"$CONDA_BIN" create -y -q -n "$ENV_NAME" python=3.12 > /dev/null 2>&1

echo "  Installing dwpose ..."
"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" pip install -q dwpose

echo "✓ openpose env ready"
