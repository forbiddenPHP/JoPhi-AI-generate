#!/usr/bin/env bash
set -e

# ── Enhance Worker Installer ────────────────────────────────────────────────
# Creates a dedicated conda env for resemble-enhance (audio denoising +
# super-resolution). Separate env because resemble-enhance needs
# omegaconf >= 2.3.0 and numpy >= 1.26, which conflict with RVC's pinned
# versions (omegaconf 2.0.6, numpy 1.23.5).
#
# deepspeed is listed as a dependency but only needed for training,
# not inference. On macOS ARM64 it doesn't compile, so we skip it.
# ─────────────────────────────────────────────────────────────────────────────

CONDA_BIN="/opt/miniconda3/bin/conda"
ENV_NAME="enhance"

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Enhance Worker — Installer (resemble-enhance)"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ── Check conda ──────────────────────────────────────────────────────────────

if [ ! -f "$CONDA_BIN" ]; then
    echo -e "${RED}ERROR: conda not found at $CONDA_BIN${NC}"
    echo "  Install miniconda: brew install --cask miniconda"
    exit 1
fi

# ── Check git-lfs (needed to download models from HuggingFace) ──────────────

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

# ── Create env ───────────────────────────────────────────────────────────────

if "$CONDA_BIN" env list 2>/dev/null | grep -q "^${ENV_NAME} "; then
    echo "  Removing old '$ENV_NAME' env ..."
    "$CONDA_BIN" env remove -y -n "$ENV_NAME" > /dev/null 2>&1
fi

echo "  Creating env: $ENV_NAME (Python 3.12) ..."
"$CONDA_BIN" create -y -q -n "$ENV_NAME" python=3.12 > /dev/null 2>&1
echo -e "${GREEN}✓${NC} Env created"

# ── Install resemble-enhance ────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCKFILE="$SCRIPT_DIR/requirements.lock"
WHEELS_DIR="$SCRIPT_DIR/wheels"

echo "  Installing resemble-enhance (this may take several minutes) ..."

if [ -d "$WHEELS_DIR" ] && [ "$(ls -A "$WHEELS_DIR"/*.whl 2>/dev/null)" ]; then
    # Offline install from cached wheels — works even if PyPI is gone
    echo "  Using cached wheels (offline) ..."
    # resemble-enhance has a hard deepspeed dep that doesn't build on macOS.
    # Install it --no-deps first, then install everything else from the lockfile.
    # 1) resemble-enhance --no-deps (skip deepspeed, won't compile on macOS)
    "$CONDA_BIN" run -n "$ENV_NAME" pip install --no-index --no-deps --find-links="$WHEELS_DIR" resemble-enhance 2>&1 | \
        grep -E "^(Successfully|Installing|ERROR)" | head -3
    # 2) deepspeed stub so pip sees it as satisfied
    echo "  Installing deepspeed stubs (inference only) ..."
    "$CONDA_BIN" run -n "$ENV_NAME" python "$SCRIPT_DIR/patch_deepspeed_stub.py"
    # 3) remaining deps from lockfile
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

    # First try: normal pip install (works if deepspeed compiles)
    if "$CONDA_BIN" run -n "$ENV_NAME" pip install resemble-enhance 2>&1 | \
        grep -E "^(Successfully|ERROR|error:)" | head -5; then
        echo "  Full install succeeded."
    else
        echo "  Full install had issues, trying without deepspeed ..."
        "$CONDA_BIN" run -n "$ENV_NAME" pip install --no-deps resemble-enhance 2>&1 | \
            grep -E "^(Successfully|ERROR)" | head -3

        echo "  Installing inference dependencies ..."
        "$CONDA_BIN" run -n "$ENV_NAME" pip install \
            "torch>=2.1.1" \
            "torchaudio>=2.1.1" \
            "librosa>=0.10.1" \
            "numpy>=1.26.2" \
            "omegaconf>=2.3.0" \
            "scipy>=1.11.4" \
            "soundfile>=0.12.1" \
            "rich>=13.7.0" \
            "tqdm>=4.66.1" \
            "resampy>=0.4.2" \
            "tabulate>=0.8.10" \
            2>&1 | grep -E "^(Successfully|ERROR)" | head -5
    fi

    # Generate lockfile for future installs
    echo "  Generating requirements.lock ..."
    "$CONDA_BIN" run -n "$ENV_NAME" pip freeze > "$LOCKFILE" 2>/dev/null
    echo -e "${GREEN}✓${NC} Lockfile saved"
fi

# ── Verify ───────────────────────────────────────────────────────────────────

echo ""
echo "── Verification ──"

if "$CONDA_BIN" run -n "$ENV_NAME" python -c "
from resemble_enhance.enhancer.inference import denoise, enhance
print('  OK resemble-enhance inference ready')
" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} resemble-enhance installed"
else
    echo -e "${RED}ERROR: resemble-enhance installation failed${NC}"
    echo "  Try manually:"
    echo "    conda activate enhance"
    echo "    pip install resemble-enhance"
    exit 1
fi

"$CONDA_BIN" run -n "$ENV_NAME" python -c "
import torch
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'  PyTorch {torch.__version__} (device: {device})')
"

echo ""
echo -e "${GREEN}✓${NC} Enhance Worker ready"
echo ""
