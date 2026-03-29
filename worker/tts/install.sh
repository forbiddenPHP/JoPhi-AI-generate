#!/usr/bin/env bash
# set -e  # removed: let setup continue on errors

# ── AI-TTS Worker Installer ────────────────────────────────────────────────
# Creates a dedicated conda env for Qwen3-TTS via mlx-audio.
# Apple Silicon optimized (MLX backend). No API key needed.
# ────────────────────────────────────────────────────────────────────────────

CONDA_BIN="${CONDA_BIN:-$(cat ~/.ai-conda-path 2>/dev/null || command -v conda 2>/dev/null || echo conda)}"
ENV_NAME="ai-tts"

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  AI-TTS Worker — Installer (Qwen3-TTS via mlx-audio)"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ── Check conda ──────────────────────────────────────────────────────────────

if [ ! -f "$CONDA_BIN" ]; then
    echo -e "${RED}ERROR: conda not found at $CONDA_BIN${NC}"
    echo "  Install miniconda: brew install --cask miniconda"
    # exit 1  # warn only, do not abort setup
fi

# ── Create env ───────────────────────────────────────────────────────────────

if "$CONDA_BIN" env list 2>/dev/null | grep -q "^${ENV_NAME} " || [ -d "$(dirname "$(dirname "$CONDA_BIN")")/envs/$ENV_NAME" ]; then
    echo "  Removing old '$ENV_NAME' env …"
    "$CONDA_BIN" env remove -y -n "$ENV_NAME" > /dev/null 2>&1
    rm -rf "$(dirname "$(dirname "$CONDA_BIN")")/envs/$ENV_NAME" 2>/dev/null || true
fi

echo "  Creating env: $ENV_NAME (Python 3.11) …"
"$CONDA_BIN" create -y -q -n "$ENV_NAME" python=3.11 > /dev/null 2>&1
echo -e "${GREEN}✓${NC} Env created"

# ── Install mlx-audio + soundfile ────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCKFILE="$SCRIPT_DIR/requirements.lock"
WHEELS_DIR="$SCRIPT_DIR/wheels"

echo "  Installing mlx-audio …"
if [ -d "$WHEELS_DIR" ] && [ "$(ls -A "$WHEELS_DIR"/*.whl 2>/dev/null)" ]; then
    echo "  Using cached wheels (offline) …"
    "$CONDA_BIN" run -n "$ENV_NAME" pip install --no-index --find-links="$WHEELS_DIR" -r "$LOCKFILE" 2>&1 | \
        grep -E "^(Successfully|Installing|ERROR)" | head -5
elif [ -f "$LOCKFILE" ]; then
    echo "  Using pinned versions from requirements.lock …"
    "$CONDA_BIN" run -n "$ENV_NAME" pip install -r "$LOCKFILE" 2>&1 | \
        grep -E "^(Successfully|Installing|Downloading|ERROR)" | head -5
else
    "$CONDA_BIN" run -n "$ENV_NAME" pip install mlx-audio soundfile 2>&1 | \
        grep -E "^(Successfully|Installing|Downloading|ERROR)" | head -5

    # Generate lockfile for future installs
    echo "  Generating requirements.lock …"
    "$CONDA_BIN" run -n "$ENV_NAME" pip freeze > "$LOCKFILE" 2>/dev/null
    echo -e "${GREEN}✓${NC} Lockfile saved"
fi

# ── Remove .git from dependencies ────────────────────────────────────────────

SITE_PACKAGES=$("$CONDA_BIN" run -n "$ENV_NAME" python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)
if [ -n "$SITE_PACKAGES" ]; then
    find "$SITE_PACKAGES" -name ".git" -type d -exec rm -rf {} + 2>/dev/null || true
fi

# Model download handled by setup.sh (centralized at the end)

# ── Verify ───────────────────────────────────────────────────────────────────

echo ""
echo "── Verification ──"

if "$CONDA_BIN" run -n "$ENV_NAME" python -c "
from mlx_audio.tts.utils import load_model
import soundfile
print('  OK mlx-audio + soundfile ready')
" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} mlx-audio installed"
else
    echo -e "${RED}WARNING: mlx-audio installation failed${NC}"
    echo "  Try manually:"
    echo "    conda activate ai-tts"
    echo "    pip install mlx-audio soundfile"
fi

# Patch: lower ICL repetition penalty for voice cloning (1.5 is too aggressive, cuts off text)
QWEN_TTS_PY=$("$CONDA_BIN" run -n "$ENV_NAME" python -c "import mlx_audio; from pathlib import Path; print(Path(mlx_audio.__file__).parent / 'tts/models/qwen3_tts/qwen3_tts.py')" 2>/dev/null)
if [ -f "$QWEN_TTS_PY" ] && grep -q "max(repetition_penalty, 1.5)" "$QWEN_TTS_PY"; then
    sed -i '' 's/max(repetition_penalty, 1.5)/max(repetition_penalty, 1.2)/' "$QWEN_TTS_PY"
    echo "  Patched: ICL rep_penalty 1.5 → 1.2"
fi

echo ""
echo -e "${GREEN}✓${NC} AI-TTS Worker ready"
echo ""
