#!/usr/bin/env bash
# set -e  # removed: let setup continue on errors

# ── Separate Worker Installer ────────────────────────────────────────────────
# Creates a dedicated conda env for demucs (Meta/Facebook) audio source
# separation. Splits audio into stems: vocals, drums, bass, other.
# ──────────────────────────────────────────────────────────────────────────────

CONDA_BIN="/opt/miniconda3/bin/conda"
ENV_NAME="separate"

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Separate Worker — Installer (demucs)"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ── Check conda ──────────────────────────────────────────────────────────────

if [ ! -f "$CONDA_BIN" ]; then
    echo -e "${RED}ERROR: conda not found at $CONDA_BIN${NC}"
    echo "  Install miniconda: brew install --cask miniconda"
    # exit 1  # warn only, do not abort setup
fi

# ── Create env ───────────────────────────────────────────────────────────────

if "$CONDA_BIN" env list 2>/dev/null | grep -q "^${ENV_NAME} " || [ -d "/opt/miniconda3/envs/$ENV_NAME" ]; then
    echo "  Removing old '$ENV_NAME' env ..."
    "$CONDA_BIN" env remove -y -n "$ENV_NAME" > /dev/null 2>&1
    rm -rf "/opt/miniconda3/envs/$ENV_NAME" 2>/dev/null || true
fi

echo "  Creating env: $ENV_NAME (Python 3.10) ..."
"$CONDA_BIN" create -y -q -n "$ENV_NAME" python=3.10 > /dev/null 2>&1
echo -e "${GREEN}✓${NC} Env created"

# ── Install demucs ───────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCKFILE="$SCRIPT_DIR/requirements.lock"
WHEELS_DIR="$SCRIPT_DIR/wheels"

echo "  Installing demucs (this may take several minutes) ..."

if [ -d "$WHEELS_DIR" ] && [ "$(ls -A "$WHEELS_DIR"/*.whl 2>/dev/null)" ]; then
    # Offline install from cached wheels
    echo "  Using cached wheels (offline) ..."
    "$CONDA_BIN" run -n "$ENV_NAME" pip install --no-index --find-links="$WHEELS_DIR" -r "$LOCKFILE" 2>&1 | \
        grep -E "^(Successfully|Installing|ERROR)" | head -5
elif [ -f "$LOCKFILE" ]; then
    # Online install with pinned versions
    echo "  Using pinned versions from requirements.lock ..."
    "$CONDA_BIN" run -n "$ENV_NAME" pip install -r "$LOCKFILE" 2>&1 | \
        grep -E "^(Successfully|Installing|Downloading|ERROR)" | head -5
else
    # Fallback: install from PyPI
    echo "  No lockfile or wheels found, installing from PyPI ..."

    "$CONDA_BIN" run -n "$ENV_NAME" pip install \
        "torch>=2.1.1" \
        "torchaudio>=2.1.1" \
        "soundfile" \
        2>&1 | grep -E "^(Successfully|ERROR)" | head -3

    "$CONDA_BIN" run -n "$ENV_NAME" pip install \
        "demucs" \
        2>&1 | grep -E "^(Successfully|ERROR)" | head -3

    # Generate lockfile for future installs
    echo "  Generating requirements.lock ..."
    "$CONDA_BIN" run -n "$ENV_NAME" pip freeze > "$LOCKFILE" 2>/dev/null
    echo -e "${GREEN}✓${NC} Lockfile saved"
fi

# ── Verify ───────────────────────────────────────────────────────────────────

echo ""
echo "── Verification ──"

if "$CONDA_BIN" run -n "$ENV_NAME" python -c "
from demucs.pretrained import get_model
from demucs.apply import apply_model
print('  OK demucs imported')
" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} demucs installed"
else
    echo -e "${RED}WARNING: demucs installation failed${NC}"
    echo "  Try manually:"
    echo "    conda activate separate"
    echo "    pip install demucs"
fi

"$CONDA_BIN" run -n "$ENV_NAME" python -c "
import torch
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'  PyTorch {torch.__version__} (device: {device})')
"

echo ""
echo -e "${GREEN}✓${NC} Separate Worker ready"
echo ""
