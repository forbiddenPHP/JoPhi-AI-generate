#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LTX_CORE_DIR="$SCRIPT_DIR/ltx-core"
LTX_PIPELINES_DIR="$SCRIPT_DIR/ltx-pipelines"
CONDA_BIN="${CONDA_BIN:-$(cat ~/.ai-conda-path 2>/dev/null || command -v conda 2>/dev/null || echo conda)}"
ENV_NAME="ltx2"
WHEELS_DIR="$SCRIPT_DIR/wheels"

echo "── Video Worker (LTX-2.3) ──"

if "$CONDA_BIN" env list 2>/dev/null | grep -q "^${ENV_NAME} " || [ -d "$(dirname "$(dirname "$CONDA_BIN")")/envs/$ENV_NAME" ]; then
    echo "  Removing old '$ENV_NAME' env …"
    "$CONDA_BIN" env remove -y -n "$ENV_NAME" > /dev/null 2>&1
    rm -rf "$(dirname "$(dirname "$CONDA_BIN")")/envs/$ENV_NAME" 2>/dev/null || true
fi

echo "  Creating env: $ENV_NAME (Python 3.12) …"
"$CONDA_BIN" create -y -q -n "$ENV_NAME" python=3.12 > /dev/null 2>&1

if [ -d "$WHEELS_DIR" ] && [ "$(ls -A "$WHEELS_DIR"/*.whl 2>/dev/null)" ]; then
    echo "  Using cached wheels (offline) …"
    "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" pip install -q --no-index --find-links "$WHEELS_DIR" -e "$LTX_CORE_DIR" -e "$LTX_PIPELINES_DIR"
else
    echo "  Installing LTX-2.3 dependencies from PyPI …"
    "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" pip install -q -e "$LTX_CORE_DIR"
    "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" pip install -q -e "$LTX_PIPELINES_DIR"
fi

echo "✓ ltx2 env ready"
