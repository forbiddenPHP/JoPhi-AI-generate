#!/usr/bin/env bash
# set -e  # removed: let setup continue on errors

# ── SFX Worker Installer ──────────────────────────────────────────────────
# Creates a dedicated conda env for EzAudio (text-to-audio diffusion).
# Generates sound effects from text prompts.
# ──────────────────────────────────────────────────────────────────────────

CONDA_BIN="${CONDA_BIN:-$(cat ~/.ai-conda-path 2>/dev/null || command -v conda 2>/dev/null || echo conda)}"
ENV_NAME="ezaudio"

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  SFX Worker — Installer (EzAudio)"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ── Check conda ──────────────────────────────────────────────────────────

if [ ! -f "$CONDA_BIN" ]; then
    echo -e "${RED}ERROR: conda not found at $CONDA_BIN${NC}"
    echo "  Install miniconda: brew install --cask miniconda"
    # exit 1  # warn only, do not abort setup
fi

# ── Create env ───────────────────────────────────────────────────────────

if "$CONDA_BIN" env list 2>/dev/null | grep -q "^${ENV_NAME} " || [ -d "$(dirname "$(dirname "$CONDA_BIN")")/envs/$ENV_NAME" ]; then
    echo "  Removing old '$ENV_NAME' env …"
    "$CONDA_BIN" env remove -y -n "$ENV_NAME" > /dev/null 2>&1
    rm -rf "$(dirname "$(dirname "$CONDA_BIN")")/envs/$ENV_NAME" 2>/dev/null || true
fi

echo "  Creating env: $ENV_NAME (Python 3.10) …"
"$CONDA_BIN" create -y -q -n "$ENV_NAME" python=3.10 > /dev/null 2>&1
echo -e "${GREEN}✓${NC} Env created"

# ── Install EzAudio dependencies ────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCKFILE="$SCRIPT_DIR/requirements.lock"
WHEELS_DIR="$SCRIPT_DIR/wheels"

echo "  Installing EzAudio dependencies …"

if [ -d "$WHEELS_DIR" ] && [ "$(ls -A "$WHEELS_DIR"/*.whl 2>/dev/null)" ]; then
    # Offline install from cached wheels
    echo "  Using cached wheels (offline) …"
    "$CONDA_BIN" run -n "$ENV_NAME" pip install --no-index --find-links="$WHEELS_DIR" -r "$LOCKFILE" 2>&1 | \
        grep -E "^(Successfully|Installing|ERROR)" | head -5
elif [ -f "$LOCKFILE" ]; then
    # Online install with pinned versions
    echo "  Using pinned versions from requirements.lock …"
    "$CONDA_BIN" run -n "$ENV_NAME" pip install -r "$LOCKFILE" 2>&1 | \
        grep -E "^(Successfully|Installing|Downloading|ERROR)" | head -5
else
    # Fallback: install from PyPI
    echo "  No lockfile or wheels found, installing from PyPI …"
    "$CONDA_BIN" run -n "$ENV_NAME" pip install -r "$SCRIPT_DIR/requirements.txt" 2>&1 | \
        grep -E "^(Successfully|Installing|Downloading|ERROR)" | head -5

    # Generate lockfile for future installs
    echo "  Generating requirements.lock …"
    "$CONDA_BIN" run -n "$ENV_NAME" pip freeze > "$LOCKFILE" 2>/dev/null
    echo -e "${GREEN}✓${NC} Lockfile saved"
fi

# ── Verify ───────────────────────────────────────────────────────────────

echo ""
echo "── Verification ──"

if "$CONDA_BIN" run -n "$ENV_NAME" python -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from api.ezaudio import EzAudio
print('  OK EzAudio API importable')
" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} EzAudio installed"
else
    echo -e "${RED}WARNING: EzAudio installation failed${NC}"
    echo "  Try manually:"
    echo "    conda activate $ENV_NAME"
    echo "    pip install -r $SCRIPT_DIR/requirements.txt"
fi

"$CONDA_BIN" run -n "$ENV_NAME" python -c "
import torch
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'  PyTorch {torch.__version__} (device: {device})')
"

echo ""
echo -e "${GREEN}✓${NC} SFX Worker ready"
echo ""
