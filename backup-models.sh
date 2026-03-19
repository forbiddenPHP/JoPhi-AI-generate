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

if [ -d "$SCRIPT_DIR/worker/rvc/models" ] && [ "$(ls -A "$SCRIPT_DIR/worker/rvc/models" 2>/dev/null)" ]; then
    echo "  Backing up RVC models ..."
    cp -a "$SCRIPT_DIR/worker/rvc/models" "$MODELS_DIR/rvc_models"
    echo -e "  ${GREEN}✓${NC} RVC models ($(du -sh "$MODELS_DIR/rvc_models" | cut -f1))"
else
    echo "  ── RVC models: not found, skipping"
fi

# ── HeartMuLa checkpoints ────────────────────────────────────────────────────

if [ -d "$SCRIPT_DIR/worker/music/models/ckpt" ] && [ "$(ls -A "$SCRIPT_DIR/worker/music/models/ckpt" 2>/dev/null)" ]; then
    echo "  Backing up HeartMuLa models ..."
    mkdir -p "$MODELS_DIR/music_models"
    cp -a "$SCRIPT_DIR/worker/music/models/ckpt" "$MODELS_DIR/music_models/ckpt"
    echo -e "  ${GREEN}✓${NC} HeartMuLa models ($(du -sh "$MODELS_DIR/music_models" | cut -f1))"
else
    echo "  ── HeartMuLa models: not found, skipping"
fi

# ── ACE-Step checkpoints ─────────────────────────────────────────────────────

ACE_CKPT="$SCRIPT_DIR/worker/ace/ACE-Step-1.5/checkpoints"
if [ -d "$ACE_CKPT" ] && [ "$(ls -A "$ACE_CKPT" 2>/dev/null)" ]; then
    echo "  Backing up ACE-Step checkpoints ..."
    cp -a "$ACE_CKPT" "$MODELS_DIR/ace_checkpoints"
    echo -e "  ${GREEN}✓${NC} ACE-Step checkpoints ($(du -sh "$MODELS_DIR/ace_checkpoints" | cut -f1))"
else
    echo "  ── ACE-Step checkpoints: not found, skipping"
fi

# ── EzAudio checkpoints ──────────────────────────────────────────────────────

SFX_CKPT="$SCRIPT_DIR/worker/sfx/ckpts"
if [ -d "$SFX_CKPT" ] && [ "$(find "$SFX_CKPT" -name '*.pt' 2>/dev/null | head -1)" ]; then
    echo "  Backing up EzAudio checkpoints ..."
    cp -a "$SFX_CKPT" "$MODELS_DIR/sfx_ckpts"
    echo -e "  ${GREEN}✓${NC} EzAudio checkpoints ($(du -sh "$MODELS_DIR/sfx_ckpts" | cut -f1))"
else
    echo "  ── EzAudio checkpoints: not found, skipping"
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

# ── HuggingFace models (pyannote, whisper, Qwen3-TTS, flan-t5-xl) ────────────

HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"
HF_BACKED=0
mkdir -p "$MODELS_DIR/huggingface"

for pattern in "models--pyannote--*" "models--mlx-community--whisper-*" "models--mlx-community--Qwen3-TTS-*" "models--google--flan-t5-xl"; do
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

# ── FLUX.2 models (worker/image/models/) ─────────────────────────────────────

IMAGE_MODELS="$SCRIPT_DIR/worker/image/models"
if [ -d "$IMAGE_MODELS" ] && [ "$(ls -A "$IMAGE_MODELS" 2>/dev/null)" ]; then
    echo "  Backing up FLUX.2 models ..."
    # -rL: dereference symlinks (HF cache uses symlinks snapshots → blobs)
    cp -rL "$IMAGE_MODELS" "$MODELS_DIR/image_models"
    echo -e "  ${GREEN}✓${NC} FLUX.2 models ($(du -sh "$MODELS_DIR/image_models" | cut -f1))"
else
    echo "  ── FLUX.2 models: not found, skipping"
fi

# ── DWPose models (worker/pose/models/) ──────────────────────────────────────

POSE_MODELS="$SCRIPT_DIR/worker/pose/models"
if [ -d "$POSE_MODELS" ] && [ "$(ls -A "$POSE_MODELS" 2>/dev/null)" ]; then
    echo "  Backing up DWPose models ..."
    cp -rL "$POSE_MODELS" "$MODELS_DIR/pose_models"
    echo -e "  ${GREEN}✓${NC} DWPose models ($(du -sh "$MODELS_DIR/pose_models" | cut -f1))"
else
    echo "  ── DWPose models: not found, skipping"
fi

# ── SD 1.5 models (worker/sd15/models/) ───────────────────────────────────────

SD15_MODELS="$SCRIPT_DIR/worker/sd15/models"
if [ -d "$SD15_MODELS" ] && [ "$(ls -A "$SD15_MODELS" 2>/dev/null)" ]; then
    echo "  Backing up SD 1.5 models …"
    cp -rL "$SD15_MODELS" "$MODELS_DIR/sd15_models"
    echo -e "  ${GREEN}✓${NC} SD 1.5 models ($(du -sh "$MODELS_DIR/sd15_models" | cut -f1))"
else
    echo "  ── SD 1.5 models: not found, skipping"
fi

# ── SD 1.5 LoRAs (worker/sd15/loras/) ────────────────────────────────────────

SD15_LORAS="$SCRIPT_DIR/worker/sd15/loras"
if [ -d "$SD15_LORAS" ] && [ "$(ls -A "$SD15_LORAS" 2>/dev/null)" ]; then
    echo "  Backing up SD 1.5 LoRAs …"
    cp -rL "$SD15_LORAS" "$MODELS_DIR/sd15_loras"
    echo -e "  ${GREEN}✓${NC} SD 1.5 LoRAs ($(du -sh "$MODELS_DIR/sd15_loras" | cut -f1))"
else
    echo "  ── SD 1.5 LoRAs: not found, skipping"
fi

# ── Depth Anything V2 models (worker/depth/models/) ──────────────────────────

DEPTH_MODELS="$SCRIPT_DIR/worker/depth/models"
if [ -d "$DEPTH_MODELS" ] && [ "$(ls -A "$DEPTH_MODELS" 2>/dev/null)" ]; then
    echo "  Backing up Depth Anything V2 models …"
    cp -rL "$DEPTH_MODELS" "$MODELS_DIR/depth_models"
    echo -e "  ${GREEN}✓${NC} Depth Anything V2 models ($(du -sh "$MODELS_DIR/depth_models" | cut -f1))"
else
    echo "  ── Depth Anything V2 models: not found, skipping"
fi

# ── Lineart models (worker/lineart/models/) ──────────────────────────────────

LINEART_MODELS="$SCRIPT_DIR/worker/lineart/models"
if [ -d "$LINEART_MODELS" ] && [ "$(ls -A "$LINEART_MODELS" 2>/dev/null)" ]; then
    echo "  Backing up Lineart models …"
    cp -rL "$LINEART_MODELS" "$MODELS_DIR/lineart_models"
    echo -e "  ${GREEN}✓${NC} Lineart models ($(du -sh "$MODELS_DIR/lineart_models" | cut -f1))"
else
    echo "  ── Lineart models: not found, skipping"
fi

# ── NormalMap models (worker/normalmap/models/) ──────────────────────────────

NORMALMAP_MODELS="$SCRIPT_DIR/worker/normalmap/models"
if [ -d "$NORMALMAP_MODELS" ] && [ "$(ls -A "$NORMALMAP_MODELS" 2>/dev/null)" ]; then
    echo "  Backing up NormalMap models …"
    cp -rL "$NORMALMAP_MODELS" "$MODELS_DIR/normalmap_models"
    echo -e "  ${GREEN}✓${NC} NormalMap models ($(du -sh "$MODELS_DIR/normalmap_models" | cut -f1))"
else
    echo "  ── NormalMap models: not found, skipping"
fi

# ── Sketch models (worker/sketch/models/) ────────────────────────────────────

SKETCH_MODELS="$SCRIPT_DIR/worker/sketch/models"
if [ -d "$SKETCH_MODELS" ] && [ "$(ls -A "$SKETCH_MODELS" 2>/dev/null)" ]; then
    echo "  Backing up Sketch models …"
    cp -rL "$SKETCH_MODELS" "$MODELS_DIR/sketch_models"
    echo -e "  ${GREEN}✓${NC} Sketch models ($(du -sh "$MODELS_DIR/sketch_models" | cut -f1))"
else
    echo "  ── Sketch models: not found, skipping"
fi

# ── Upscale models (worker/upscale/models/) ──────────────────────────────────

UPSCALE_MODELS="$SCRIPT_DIR/worker/upscale/models"
if [ -d "$UPSCALE_MODELS" ] && [ "$(ls -A "$UPSCALE_MODELS" 2>/dev/null)" ]; then
    echo "  Backing up Upscale models …"
    cp -rL "$UPSCALE_MODELS" "$MODELS_DIR/upscale_models"
    echo -e "  ${GREEN}✓${NC} Upscale models ($(du -sh "$MODELS_DIR/upscale_models" | cut -f1))"
else
    echo "  ── Upscale models: not found, skipping"
fi

# ── Text worker configs (LLM default overrides) ─────────────────────────────

TEXT_CONFIGS="$SCRIPT_DIR/worker/text/models"
if [ -d "$TEXT_CONFIGS" ] && [ "$(find "$TEXT_CONFIGS" -name 'config.json' 2>/dev/null | head -1)" ]; then
    echo "  Backing up text worker configs ..."
    cp -a "$TEXT_CONFIGS" "$MODELS_DIR/text_configs"
    echo -e "  ${GREEN}✓${NC} Text configs ($(du -sh "$MODELS_DIR/text_configs" | cut -f1))"
else
    echo "  ── Text worker configs: not found, skipping"
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
