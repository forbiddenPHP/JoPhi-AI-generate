#!/usr/bin/env bash
set -e

# ── Revoicer — Model Backup ─────────────────────────────────────────────────
# Collects all model files into ./models/ for offline restore.
# Run after initial setup to create a backup of all downloaded models.
#
# Usage:
#   bash backup-models.sh              Create ./models/ directory
#   bash backup-models.sh --zip        Create ./models/ + compress to models.zip
#
# Restore:
#   Place models/ or models.zip next to setup.sh, then run setup.sh.
# ─────────────────────────────────────────────────────────────────────────────

CONDA_BIN="/opt/miniconda3/bin/conda"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models"

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Revoicer — Model Backup"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Clean previous backup
if [ -d "$MODELS_DIR" ]; then
    echo "  Removing previous backup ..."
    rm -rf "$MODELS_DIR"
fi
mkdir -p "$MODELS_DIR"

# ── RVC voice models ─────────────────────────────────────────────────────────

if [ -d "$SCRIPT_DIR/rvc_models" ] && [ "$(ls -A "$SCRIPT_DIR/rvc_models" 2>/dev/null)" ]; then
    echo "  Backing up RVC models ..."
    cp -a "$SCRIPT_DIR/rvc_models" "$MODELS_DIR/rvc_models"
    echo -e "  ${GREEN}✓${NC} RVC models ($(du -sh "$MODELS_DIR/rvc_models" | cut -f1))"
else
    echo "  ── RVC models: not found, skipping"
fi

# ── HeartMuLa checkpoints ────────────────────────────────────────────────────

if [ -d "$SCRIPT_DIR/music_models/ckpt" ] && [ "$(ls -A "$SCRIPT_DIR/music_models/ckpt" 2>/dev/null)" ]; then
    echo "  Backing up HeartMuLa models ..."
    mkdir -p "$MODELS_DIR/music_models"
    cp -a "$SCRIPT_DIR/music_models/ckpt" "$MODELS_DIR/music_models/ckpt"
    echo -e "  ${GREEN}✓${NC} HeartMuLa models ($(du -sh "$MODELS_DIR/music_models" | cut -f1))"
else
    echo "  ── HeartMuLa models: not found, skipping"
fi

# ── ACE-Step checkpoints ─────────────────────────────────────────────────────

ACE_CKPT="$SCRIPT_DIR/ace_worker/ACE-Step-1.5/checkpoints"
if [ -d "$ACE_CKPT" ] && [ "$(ls -A "$ACE_CKPT" 2>/dev/null)" ]; then
    echo "  Backing up ACE-Step checkpoints ..."
    cp -a "$ACE_CKPT" "$MODELS_DIR/ace_checkpoints"
    echo -e "  ${GREEN}✓${NC} ACE-Step checkpoints ($(du -sh "$MODELS_DIR/ace_checkpoints" | cut -f1))"
else
    echo "  ── ACE-Step checkpoints: not found, skipping"
fi

# ── resemble-enhance model_repo ──────────────────────────────────────────────

ENHANCE_SITE=$("$CONDA_BIN" run -n enhance python -c "import resemble_enhance; print(resemble_enhance.__path__[0])" 2>/dev/null || true)
if [ -n "$ENHANCE_SITE" ] && [ -d "$ENHANCE_SITE/model_repo" ]; then
    echo "  Backing up enhance models ..."
    cp -a "$ENHANCE_SITE/model_repo" "$MODELS_DIR/enhance_model_repo"
    rm -rf "$MODELS_DIR/enhance_model_repo/.git"
    echo -e "  ${GREEN}✓${NC} Enhance models ($(du -sh "$MODELS_DIR/enhance_model_repo" | cut -f1))"
else
    echo "  ── Enhance models: not found, skipping"
fi

# ── HuggingFace models (pyannote, whisper) ────────────────────────────────────

HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"
HF_BACKED=0
mkdir -p "$MODELS_DIR/huggingface"

for pattern in "models--pyannote--*" "models--mlx-community--whisper-*"; do
    for model_dir in "$HF_CACHE"/$pattern; do
        [ -d "$model_dir" ] || continue
        model_name=$(basename "$model_dir")
        echo "  Backing up $model_name ..."
        # -rL: dereference symlinks (HF cache uses symlinks snapshots → blobs)
        cp -rL "$model_dir" "$MODELS_DIR/huggingface/$model_name"
        HF_BACKED=1
    done
done

if [ "$HF_BACKED" -eq 1 ]; then
    echo -e "  ${GREEN}✓${NC} HuggingFace models ($(du -sh "$MODELS_DIR/huggingface" | cut -f1))"
else
    echo "  ── HuggingFace models: not found, skipping"
    rmdir "$MODELS_DIR/huggingface" 2>/dev/null || true
fi

# ── Torch hub (demucs) ───────────────────────────────────────────────────────

TORCH_CKPT="$HOME/.cache/torch/hub/checkpoints"
if [ -d "$TORCH_CKPT" ] && [ "$(ls -A "$TORCH_CKPT" 2>/dev/null)" ]; then
    echo "  Backing up demucs models (torch hub) ..."
    cp -a "$TORCH_CKPT" "$MODELS_DIR/torch_hub"
    echo -e "  ${GREEN}✓${NC} Demucs models ($(du -sh "$MODELS_DIR/torch_hub" | cut -f1))"
else
    echo "  ── Demucs models: not found, skipping"
fi

# ── Summary ──────────────────────────────────────────────────────────────────

echo ""
TOTAL=$(du -sh "$MODELS_DIR" | cut -f1)
echo "═══════════════════════════════════════════════════════════"
echo "  Backup complete: $TOTAL in ./models/"
echo "═══════════════════════════════════════════════════════════"

# ── Optional: compress to zip ────────────────────────────────────────────────

if [ "$1" = "--zip" ]; then
    echo ""
    echo "  Compressing to models.zip ..."
    cd "$SCRIPT_DIR"
    zip -r -q models.zip models/
    ZIP_SIZE=$(du -sh "$SCRIPT_DIR/models.zip" | cut -f1)
    echo -e "  ${GREEN}✓${NC} models.zip ($ZIP_SIZE)"
fi

echo ""
