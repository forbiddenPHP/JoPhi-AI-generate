#!/usr/bin/env bash
set -e

# ── HeartMuLa Music Worker Installer ──────────────────────────────────────
# Creates a dedicated conda env for HeartMuLa music generation.
# Separate env because HeartMuLa needs torch>=2.4, transformers==4.57.0,
# torchtune==0.4.0, bitsandbytes==0.49.0 — all of which conflict with
# RVC's old torch/omegaconf/fairseq and enhance's numpy versions.
#
# Downloads ~several GB of model checkpoints from HuggingFace.
# snapshot_download is resumable — interrupted downloads continue on retry.
# ──────────────────────────────────────────────────────────────────────────

CONDA_BIN="/opt/miniconda3/bin/conda"
ENV_NAME="heartmula"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CKPT_DIR="$PROJECT_DIR/music_models/ckpt"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  HeartMuLa Music Worker — Installer"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ── Check conda ──────────────────────────────────────────────────────────

if [ ! -f "$CONDA_BIN" ]; then
    echo -e "${RED}ERROR: conda not found at $CONDA_BIN${NC}"
    echo "  Install miniconda: brew install --cask miniconda"
    exit 1
fi

# ── Check git-lfs (needed to download models from HuggingFace) ──────────

if ! git lfs version > /dev/null 2>&1; then
    echo "  Installing git-lfs ..."
    if command -v brew &> /dev/null; then
        HOMEBREW_NO_AUTO_UPDATE=1 brew install git-lfs > /dev/null 2>&1
    else
        echo -e "${RED}ERROR: git-lfs not found. Install with: brew install git-lfs${NC}"
        exit 1
    fi
    git lfs install > /dev/null 2>&1
    echo -e "${GREEN}✓${NC} git-lfs installed"
fi

# ── Check disk space ────────────────────────────────────────────────────

AVAIL_GB=$(df -g "$PROJECT_DIR" 2>/dev/null | awk 'NR==2 {print $4}' || echo "0")
if [ "$AVAIL_GB" -lt 15 ] 2>/dev/null; then
    echo -e "${YELLOW}WARNING: Only ${AVAIL_GB} GB free. HeartMuLa needs ~15 GB (env + checkpoints).${NC}"
    echo "  Continue anyway? (Ctrl+C to abort)"
    sleep 3
fi

# ── Create env ──────────────────────────────────────────────────────────

if "$CONDA_BIN" env list 2>/dev/null | grep -q "^${ENV_NAME} "; then
    echo "  Removing old '$ENV_NAME' env ..."
    "$CONDA_BIN" env remove -y -n "$ENV_NAME" > /dev/null 2>&1
fi

echo "  Creating env: $ENV_NAME (Python 3.10) ..."
"$CONDA_BIN" create -y -q -n "$ENV_NAME" python=3.10 > /dev/null 2>&1
echo -e "${GREEN}✓${NC} Env created"

# ── Clone heartlib (local copy) ────────────────────────────────────────

HEARTLIB_DIR="$SCRIPT_DIR/heartlib"
LOCKFILE="$SCRIPT_DIR/requirements.lock"
WHEELS_DIR="$SCRIPT_DIR/wheels"

if [ ! -d "$HEARTLIB_DIR" ]; then
    echo "  Cloning heartlib ..."
    git clone https://github.com/HeartMuLa/heartlib.git "$HEARTLIB_DIR"
    # Remove .git to avoid nested repos
    rm -rf "$HEARTLIB_DIR/.git"
fi

# ── Install dependencies ──────────────────────────────────────────────

echo "  Installing heartlib (this may take several minutes) ..."

if [ -d "$WHEELS_DIR" ] && [ "$(ls -A "$WHEELS_DIR"/*.whl 2>/dev/null)" ]; then
    # Offline install from cached wheels — works even if PyPI is gone
    echo "  Using cached wheels (offline) ..."
    "$CONDA_BIN" run -n "$ENV_NAME" pip install --no-index --find-links="$WHEELS_DIR" -r "$LOCKFILE" 2>&1 | \
        grep -E "^(Successfully|Installing|ERROR)" | head -5
    # heartlib itself (editable, from local copy)
    "$CONDA_BIN" run -n "$ENV_NAME" pip install --no-deps -e "$HEARTLIB_DIR" 2>&1 | \
        grep -E "^(Successfully|ERROR)" | head -3
elif [ -f "$LOCKFILE" ]; then
    # Online install with pinned versions
    echo "  Using pinned versions from requirements.lock ..."
    "$CONDA_BIN" run -n "$ENV_NAME" pip install -r "$LOCKFILE" 2>&1 | \
        grep -E "^(Successfully|Installing|Downloading|ERROR)" | head -5
    "$CONDA_BIN" run -n "$ENV_NAME" pip install --no-deps -e "$HEARTLIB_DIR" 2>&1 | \
        grep -E "^(Successfully|ERROR)" | head -3
else
    # Fallback: install from PyPI
    echo "  No lockfile or wheels found, installing from PyPI ..."

    if "$CONDA_BIN" run -n "$ENV_NAME" pip install -e "$HEARTLIB_DIR" 2>&1 | \
        grep -E "^(Successfully|ERROR|error:)" | head -5; then
        echo "  heartlib install succeeded."
        "$CONDA_BIN" run -n "$ENV_NAME" pip install torchcodec 2>&1 | tail -1
    else
        echo -e "${YELLOW}  heartlib install had issues, trying without problematic deps ...${NC}"
        "$CONDA_BIN" run -n "$ENV_NAME" pip install --no-deps -e "$HEARTLIB_DIR" 2>&1 | \
            grep -E "^(Successfully|ERROR)" | head -3

        echo "  Installing inference dependencies ..."
        "$CONDA_BIN" run -n "$ENV_NAME" pip install \
            "torch>=2.4" \
            "torchaudio>=2.4" \
            "transformers>=4.40" \
            "numpy==2.0.2" \
            "einops>=0.8" \
            "soundfile" \
            "tqdm>=4.60" \
            "vector-quantize-pytorch>=1.20" \
            "accelerate>=1.0" \
            "tokenizers>=0.20" \
            "huggingface-hub>=0.20" \
            "torchcodec" \
            2>&1 | grep -E "^(Successfully|ERROR)" | head -5
    fi

    # Generate lockfile for future installs
    echo "  Generating requirements.lock ..."
    "$CONDA_BIN" run -n "$ENV_NAME" pip freeze > "$LOCKFILE" 2>/dev/null
    echo -e "${GREEN}✓${NC} Lockfile saved"
fi

# Model downloads handled by setup.sh (centralized at the end)

# ── Verify ─────────────────────────────────────────────────────────────

echo ""
echo "── Verification ──"

if "$CONDA_BIN" run -n "$ENV_NAME" python -c "
from heartlib import HeartMuLaGenPipeline
print('  OK heartlib import ready')
" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} heartlib installed"
else
    echo -e "${RED}ERROR: heartlib installation failed${NC}"
    echo "  Try manually:"
    echo "    conda activate $ENV_NAME"
    echo "    pip install -e music_worker/heartlib"
    exit 1
fi

"$CONDA_BIN" run -n "$ENV_NAME" python -c "
import torch
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'  PyTorch {torch.__version__} (device: {device})')
"

# Checkpoint verification handled by setup.sh after model restore/download

if [ "$device" = "cpu" ] 2>/dev/null; then
    echo ""
    echo -e "${YELLOW}NOTE: No CUDA GPU detected. Music generation will use CPU.${NC}"
    echo "  This is slow (~10-30 min for 4 min of music) but works."
fi

echo ""
echo -e "${GREEN}✓${NC} HeartMuLa Music Worker ready"
echo ""
