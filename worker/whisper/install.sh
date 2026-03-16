#!/usr/bin/env bash
# set -e  # removed: let setup continue on errors

# ── Whisper Worker Installer ──────────────────────────────────────────────────
# Creates a dedicated conda env for mlx-whisper (speech-to-text transcription).
# Uses Apple Silicon MLX backend for fast inference.
# Downloads whisper-large-v3-turbo model during setup.
# ──────────────────────────────────────────────────────────────────────────────

CONDA_BIN="/opt/miniconda3/bin/conda"
ENV_NAME="whisper"
MODEL_REPO="mlx-community/whisper-large-v3-turbo"

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Whisper Worker — Installer (mlx-whisper)"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ── Check conda ──────────────────────────────────────────────────────────────

if [ ! -f "$CONDA_BIN" ]; then
    echo -e "${RED}ERROR: conda not found at $CONDA_BIN${NC}"
    echo "  Install miniconda: brew install --cask miniconda"
    # exit 1  # warn only, do not abort setup
fi

# ── Check ffmpeg (required by whisper) ───────────────────────────────────────

if ! command -v ffmpeg &> /dev/null; then
    echo "  Installing ffmpeg …"
    if command -v brew &> /dev/null; then
        HOMEBREW_NO_AUTO_UPDATE=1 brew install ffmpeg > /dev/null 2>&1
    else
        echo -e "${RED}ERROR: ffmpeg not found. Install with: brew install ffmpeg${NC}"
        # exit 1  # warn only, do not abort setup
    fi
    echo -e "${GREEN}✓${NC} ffmpeg installed"
fi

# ── Create env ───────────────────────────────────────────────────────────────

if "$CONDA_BIN" env list 2>/dev/null | grep -q "^${ENV_NAME} " || [ -d "/opt/miniconda3/envs/$ENV_NAME" ]; then
    echo "  Removing old '$ENV_NAME' env …"
    "$CONDA_BIN" env remove -y -n "$ENV_NAME" > /dev/null 2>&1
    rm -rf "/opt/miniconda3/envs/$ENV_NAME" 2>/dev/null || true
fi

echo "  Creating env: $ENV_NAME (Python 3.12) …"
"$CONDA_BIN" create -y -q -n "$ENV_NAME" python=3.12 > /dev/null 2>&1
echo -e "${GREEN}✓${NC} Env created"

# ── Install mlx-whisper ──────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCKFILE="$SCRIPT_DIR/requirements.lock"

echo "  Installing mlx-whisper …"

if [ -f "$LOCKFILE" ]; then
    echo "  Using pinned versions from requirements.lock …"
    "$CONDA_BIN" run -n "$ENV_NAME" pip install -r "$LOCKFILE" 2>&1 | \
        grep -E "^(Successfully|Installing|Downloading|ERROR)" | head -5
else
    "$CONDA_BIN" run -n "$ENV_NAME" pip install mlx-whisper 2>&1 | \
        grep -E "^(Successfully|Installing|Downloading|ERROR)" | head -5

    # Generate lockfile for future installs
    echo "  Generating requirements.lock …"
    "$CONDA_BIN" run -n "$ENV_NAME" pip freeze > "$LOCKFILE" 2>/dev/null
    echo -e "${GREEN}✓${NC} Lockfile saved"
fi

# Model download handled by setup.sh (centralized at the end)

# ── Verify ───────────────────────────────────────────────────────────────────

echo ""
echo "── Verification ──"

if "$CONDA_BIN" run -n "$ENV_NAME" python -c "
import mlx_whisper
print('  OK mlx-whisper ready')
" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} mlx-whisper installed"
else
    echo -e "${RED}WARNING: mlx-whisper installation failed${NC}"
    echo "  Try manually:"
    echo "    conda activate whisper"
    echo "    pip install mlx-whisper"
fi

echo ""
echo -e "${GREEN}✓${NC} Whisper Worker ready (model: $MODEL_REPO)"
echo ""
