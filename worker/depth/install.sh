#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONDA_BIN="${CONDA_BIN:-/opt/miniconda3/bin/conda}"
ENV_NAME="depth"
WHEELS_DIR="$SCRIPT_DIR/wheels"

echo "── Depth Worker (Depth Anything V2) ──"

if "$CONDA_BIN" env list 2>/dev/null | grep -q "^${ENV_NAME} " || [ -d "/opt/miniconda3/envs/$ENV_NAME" ]; then
    echo "  Removing old '$ENV_NAME' env …"
    "$CONDA_BIN" env remove -y -n "$ENV_NAME" > /dev/null 2>&1
    rm -rf "/opt/miniconda3/envs/$ENV_NAME" 2>/dev/null || true
fi

echo "  Creating env: $ENV_NAME (Python 3.12) …"
"$CONDA_BIN" create -y -q -n "$ENV_NAME" python=3.12 > /dev/null 2>&1

if [ -d "$WHEELS_DIR" ] && [ "$(ls -A "$WHEELS_DIR"/*.whl 2>/dev/null)" ]; then
    echo "  Using cached wheels (offline) …"
    "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" pip install -q --no-index --find-links "$WHEELS_DIR" \
        transformers torch pillow
else
    echo "  Installing Depth Anything V2 dependencies from PyPI …"
    "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" pip install -q \
        transformers torch pillow
fi

echo "✓ depth env ready"
