#!/usr/bin/env bash
# set -e  # removed: let setup continue on errors

# ── Text Worker Installer ─────────────────────────────────────────────────
# Creates a dedicated conda env for LLM inference.
# Uses the ollama Python package for Ollama engine.
# Pillow for image resize, requests for future engines.
# Auto-updates: inference.py checks once/day for ollama package updates.
# ──────────────────────────────────────────────────────────────────────────

CONDA_BIN="${CONDA_BIN:-$(cat ~/.ai-conda-path 2>/dev/null || command -v conda 2>/dev/null || echo conda)}"
ENV_NAME="text"

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Text Worker — Installer (LLM Inference)"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ── Check conda ──────────────────────────────────────────────────────────────

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

echo "  Creating env: $ENV_NAME (Python 3.11) …"
"$CONDA_BIN" create -y -q -n "$ENV_NAME" python=3.11 > /dev/null 2>&1
echo -e "${GREEN}✓${NC} Env created"

# ── Install packages ─────────────────────────────────────────────────────

echo "  Installing ollama + Pillow + requests …"
"$CONDA_BIN" run -n "$ENV_NAME" pip install ollama Pillow requests 2>&1 | \
    grep -E "^(Successfully|Installing|Downloading|ERROR)" | head -5

# ── Remove .git from dependencies ────────────────────────────────────────

SITE_PACKAGES=$("$CONDA_BIN" run -n "$ENV_NAME" python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)
if [ -n "$SITE_PACKAGES" ]; then
    find "$SITE_PACKAGES" -name ".git" -type d -exec rm -rf {} + 2>/dev/null || true
fi

# ── Create models directory ──────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/models/ollama"

# ── Verify ───────────────────────────────────────────────────────────────

echo ""
echo "── Verification ──"

if "$CONDA_BIN" run -n "$ENV_NAME" python -c "
import ollama, requests, PIL
print('  OK ollama + requests + Pillow ready')
" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} All packages installed"
else
    echo -e "${RED}WARNING: installation failed${NC}"
    echo "  Try manually:"
    echo "    conda activate text"
    echo "    pip install ollama Pillow requests"
fi

echo ""
echo -e "${GREEN}✓${NC} Text Worker ready"
echo ""
