#!/usr/bin/env bash
# ── Revoicer — Master Installer ─────────────────────────────────────────────
# Creates twelve conda environments + one uv project:
#   1. rvc        — Python 3.10 + pip<=23.3 for RVC voice conversion worker
#   2. enhance    — Python 3.12 for resemble-enhance (audio post-processing, MPS)
#   3. heartmula  — Python 3.10 for HeartMuLa music generation
#   4. acestep    — ACE-Step 1.5 music generation (managed by uv, not conda)
#   5. whisper    — Python 3.12 for mlx-whisper transcription (Apple Silicon)
#   6. diarize    — Python 3.10 for speaker diarization (pyannote.audio)
#   7. separate   — Python 3.10 for audio source separation (demucs)
#   8. ai-tts     — Python 3.11 for Qwen3-TTS (mlx-audio, Apple Silicon)
#   9. lang-detect — Python 3.11 for language detection (langdetect)
#  10. ezaudio    — Python 3.10 for EzAudio SFX generation (text-to-audio)
#  11. ltx2       — Python 3.12 for LTX-2.3 video generation (MPS)
#  12. tts-mist   — Python 3.11 for CLI + Web-App
#
# Why separate envs?
#   RVC depends on omegaconf 2.0.6 and fairseq 0.12.2 (broken metadata,
#   need old pip). resemble-enhance needs omegaconf >= 2.3.0 and
#   numpy >= 1.26 (conflicts with RVC). HeartMuLa needs torch>=2.4 and
#   transformers==4.57.0 (conflicts with both). ACE-Step uses uv for its
#   own dependency management. The main app uses modern Python.
#
# Why sudo (Xcode license)?
#   fairseq and pyworld contain C/C++ code that must be compiled.
#   On macOS, the system compiler (clang) requires the Xcode license
#   to be accepted. Run: sudo xcodebuild -license accept
# ─────────────────────────────────────────────────────────────────────────────

# ── Detect conda — resolve once, cache in ~/.ai-conda-path ────────────────
_resolve_conda() {
    # 1. Already cached?
    if [ -f "$HOME/.ai-conda-path" ]; then
        local cached
        cached=$(cat "$HOME/.ai-conda-path")
        if [ -x "$cached" ]; then echo "$cached"; return; fi
    fi
    # 2. conda active in current shell? (e.g. user sees "(base)")
    if command -v conda &>/dev/null; then
        local base
        base=$(conda info --base 2>/dev/null)
        if [ -x "$base/bin/conda" ]; then echo "$base/bin/conda"; return; fi
    fi
    # 3. Search common locations
    for p in \
        /opt/miniconda3/bin/conda \
        /opt/homebrew/Caskroom/miniconda/base/bin/conda \
        "$HOME/miniconda3/bin/conda" \
        "$HOME/anaconda3/bin/conda" \
        /usr/local/Caskroom/miniconda/base/bin/conda; do
        if [ -x "$p" ]; then echo "$p"; return; fi
    done
    return 1
}
CONDA_BIN=$(_resolve_conda)
if [ -n "$CONDA_BIN" ]; then
    echo "$CONDA_BIN" > "$HOME/.ai-conda-path"
    export CONDA_BIN
else
    export CONDA_BIN=""   # will trigger install prompt below
fi
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models"
MODELS_ZIP="$SCRIPT_DIR/models.zip"

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Revoicer — Setup"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ── Extract models.zip if present ─────────────────────────────────────────────

if [ -f "$MODELS_ZIP" ] && [ ! -d "$MODELS_DIR" ]; then
    echo "  Found models.zip — extracting ..."
    unzip -q "$MODELS_ZIP" -d "$SCRIPT_DIR"
    echo -e "${GREEN}✓${NC} models.zip extracted"
    echo ""
fi

# ── Prerequisites ────────────────────────────────────────────────────────────

if [ -z "$CONDA_BIN" ] || [ ! -x "$CONDA_BIN" ]; then
    echo -e "${RED}conda not found${NC}"
    read -p "  Install Miniconda via brew? [Y/n] " ans
    if [[ "$ans" =~ ^[Nn] ]]; then
        echo "  Aborted. Install manually: brew install --cask miniconda"
        exit 1
    fi
    HOMEBREW_NO_AUTO_UPDATE=1 brew install --cask miniconda
    # Re-detect after install and cache
    CONDA_BIN=$(_resolve_conda)
    if [ -z "$CONDA_BIN" ]; then
        echo -e "${RED}ERROR: conda still not found after install${NC}"
        exit 1
    fi
    echo "$CONDA_BIN" > "$HOME/.ai-conda-path"
    export CONDA_BIN
    echo -e "${GREEN}✓${NC} Miniconda installed → $CONDA_BIN"
fi

# Check Xcode CLI tools + license
if ! xcode-select -p > /dev/null 2>&1; then
    echo -e "${RED}Xcode Command Line Tools not found.${NC}"
    read -p "  Install now? [Y/n] " ans
    if [[ "$ans" =~ ^[Nn] ]]; then
        echo "  Aborted. Install manually: xcode-select --install"
        exit 1
    fi
    xcode-select --install
    echo "  Waiting for installation to complete ..."
    until xcode-select -p > /dev/null 2>&1; do sleep 5; done
    echo -e "${GREEN}✓${NC} Xcode CLI tools installed"
fi
if ! /usr/bin/clang --version > /dev/null 2>&1; then
    echo -e "${RED}Xcode license not accepted.${NC}"
    read -p "  Accept now? (requires sudo) [Y/n] " ans
    if [[ "$ans" =~ ^[Nn] ]]; then
        echo "  Aborted. Run manually: sudo xcodebuild -license accept"
        exit 1
    fi
    sudo xcodebuild -license accept
    echo -e "${GREEN}✓${NC} Xcode license accepted"
fi

# Check brew
if ! command -v brew &> /dev/null; then
    echo -e "${RED}brew not found.${NC}"
    read -p "  Install Homebrew? [Y/n] " ans
    if [[ "$ans" =~ ^[Nn] ]]; then
        echo "  Aborted. Install manually: https://brew.sh"
        exit 1
    fi
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    eval "$(/opt/homebrew/bin/brew shellenv)"
    echo -e "${GREEN}✓${NC} brew installed"
fi

# Check ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${RED}ffmpeg not found.${NC}"
    read -p "  Install via brew? [Y/n] " ans
    if [[ "$ans" =~ ^[Nn] ]]; then
        echo "  Aborted. Install manually: brew install ffmpeg"
        exit 1
    fi
    HOMEBREW_NO_AUTO_UPDATE=1 brew install ffmpeg
    echo -e "${GREEN}✓${NC} ffmpeg installed"
fi

# Check git-lfs
if ! command -v git-lfs &> /dev/null; then
    echo -e "${RED}git-lfs not found.${NC}"
    read -p "  Install via brew? [Y/n] " ans
    if [[ "$ans" =~ ^[Nn] ]]; then
        echo "  Aborted. Install manually: brew install git-lfs"
        exit 1
    fi
    HOMEBREW_NO_AUTO_UPDATE=1 brew install git-lfs
    git lfs install
    echo -e "${GREEN}✓${NC} git-lfs installed"
fi

# Ensure LFS files are pulled (wheels, models, etc. may be pointer-only after clone)
if git -C "$SCRIPT_DIR" lfs ls-files 2>/dev/null | grep -q ' - '; then
    echo "  Pulling Git LFS files (wheels, models) ..."
    git -C "$SCRIPT_DIR" lfs pull
    echo -e "${GREEN}✓${NC} LFS files pulled"
fi

# Check macOS + Apple Silicon
if [ "$(uname)" != "Darwin" ]; then
    echo -e "${RED}ERROR: macOS required.${NC}"
    echo "  This project only runs on macOS."
    exit 1
fi
if [ "$(uname -m)" != "arm64" ]; then
    echo -e "${RED}ERROR: Apple Silicon (arm64) required.${NC}"
    echo "  This project only runs on macOS with Apple Silicon (M1/M2/M3/M4)."
    exit 1
fi

# unar — universeller Entpacker für .rar, .7z etc. (Modell-Archive)
if ! command -v unar &> /dev/null; then
    echo "  Installing unar (archive extractor) ..."
    HOMEBREW_NO_AUTO_UPDATE=1 brew install unar > /dev/null 2>&1
    echo -e "${GREEN}✓${NC} unar installed"
fi

# ollama — local LLM inference server (text worker)
if ! command -v ollama &> /dev/null; then
    echo -e "${RED}ollama not found.${NC}"
    read -p "  Install via brew? [Y/n] " yn
    if [[ "$yn" =~ ^[Nn] ]]; then
        echo "  Aborted. Install manually: brew install --cask ollama"
    else
        HOMEBREW_NO_AUTO_UPDATE=1 brew install --cask ollama
        echo -e "${GREEN}✓${NC} ollama installed"
    fi
fi

# uv — package manager for ACE-Step (need >= 0.5 for pyproject.toml support)
UV_MIN_VERSION="0.5"
UV_PATH=""
for _uv_candidate in "$HOME/.local/bin/uv" "$(command -v uv 2>/dev/null)"; do
    if [ -x "$_uv_candidate" ] 2>/dev/null; then
        _uv_ver=$("$_uv_candidate" --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+')
        if [ "$(printf '%s\n' "$UV_MIN_VERSION" "$_uv_ver" | sort -V | head -n1)" = "$UV_MIN_VERSION" ]; then
            UV_PATH="$_uv_candidate"
            break
        fi
    fi
done
if [ -z "$UV_PATH" ]; then
    echo "  Installing uv (package manager for ACE-Step) ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh 2>/dev/null
    UV_PATH="$HOME/.local/bin/uv"
    echo -e "${GREEN}✓${NC} uv installed"
fi
export PATH="$(dirname "$UV_PATH"):$PATH"

echo -e "${GREEN}✓${NC} Prerequisites OK"

# ── Step 1: RVC Worker Env ───────────────────────────────────────────────────

echo ""
echo "── Step 1/22: RVC Worker ──"
bash "$SCRIPT_DIR/worker/rvc/install.sh"

# ── Step 2: Enhance Worker Env ───────────────────────────────────────────────

echo ""
echo "── Step 2/22: Enhance Worker ──"
bash "$SCRIPT_DIR/worker/enhance/install.sh"

# ── Step 3: HeartMuLa Music Worker Env ───────────────────────────────────────

echo ""
echo "── Step 3/22: HeartMuLa Music Worker ──"
bash "$SCRIPT_DIR/worker/music/install.sh"

# ── Step 4: ACE-Step Music Worker (uv) ───────────────────────────────────────

echo ""
echo "── Step 4/22: ACE-Step Music Worker ──"
bash "$SCRIPT_DIR/worker/ace/install.sh"

# ── Step 5: Whisper Worker Env ────────────────────────────────────────────────

echo ""
echo "── Step 5/22: Whisper Worker ──"
bash "$SCRIPT_DIR/worker/whisper/install.sh"

# ── Step 6: Diarize Worker Env ───────────────────────────────────────────────

echo ""
echo "── Step 6/22: Diarize Worker ──"
bash "$SCRIPT_DIR/worker/diarize/install.sh"

# ── Step 7: Separate Worker Env ──────────────────────────────────────────────

echo ""
echo "── Step 7/22: Separate Worker ──"
bash "$SCRIPT_DIR/worker/separate/install.sh"

# ── Step 8: AI-TTS Worker Env ────────────────────────────────────────────────

echo ""
echo "── Step 8/22: AI-TTS Worker ──"
bash "$SCRIPT_DIR/worker/tts/install.sh"

# ── Step 9: Language Detect Worker Env ───────────────────────────────────────

echo ""
echo "── Step 9/22: Language Detect Worker ──"
bash "$SCRIPT_DIR/worker/langdetect/install.sh"

# ── Step 10: SFX Worker Env ─────────────────────────────────────────────────

echo ""
echo "── Step 10/22: SFX Worker (EzAudio) ──"
bash "$SCRIPT_DIR/worker/sfx/install.sh"

# ── Step 11: Text Worker Env ─────────────────────────────────────────────────

echo ""
echo "── Step 11/22: Text Worker ──"
bash "$SCRIPT_DIR/worker/text/install.sh"

# ── Step 12/22: Image Worker (FLUX.2) ─────────────────────────────────────

echo ""
echo "── Step 12/22: Image Worker (FLUX.2) ──"
bash "$SCRIPT_DIR/worker/image/install.sh"

# ── Step 13/22: Pose Worker (DWPose/OpenPose) ─────────────────────────────

echo ""
echo "── Step 13/22: Pose Worker (DWPose) ──"
bash "$SCRIPT_DIR/worker/pose/install.sh"

echo ""
echo "── Step 14/22: SD 1.5 Worker (MatureMaleMix) ──"
bash "$SCRIPT_DIR/worker/sd15/install.sh"

echo ""
echo "── Step 15/22: Depth Worker (Depth Anything V2) ──"
bash "$SCRIPT_DIR/worker/depth/install.sh"

echo ""
echo "── Step 16/22: Lineart Worker (TEED / Canny) ──"
bash "$SCRIPT_DIR/worker/lineart/install.sh"

echo ""
echo "── Step 17/22: NormalMap Worker (Marigold-Normals) ──"
bash "$SCRIPT_DIR/worker/normalmap/install.sh"

echo ""
echo "── Step 18/22: Sketch Worker (HED / OpenCV DNN) ──"
bash "$SCRIPT_DIR/worker/sketch/install.sh"

echo ""
echo "── Step 19/22: Upscale Worker (Real-ESRGAN) ──"
bash "$SCRIPT_DIR/worker/upscale/install.sh"

echo ""
echo "── Step 20/22: Segment Worker (BiRefNet / rembg) ──"
bash "$SCRIPT_DIR/worker/segment/install.sh"

echo ""
echo "── Step 21/22: Video Worker (LTX-2.3) ──"
bash "$SCRIPT_DIR/worker/ltx2/install.sh"

# ── Step 22/22: Main App Env ────────────────────────────────────────────────

echo ""
echo "── Step 22/22: Main App (tts-mist) ──"

ENV_NAME="tts-mist"

CONDA_BASE=$(dirname "$(dirname "$CONDA_BIN")")
if "$CONDA_BIN" env list 2>/dev/null | grep -q "^${ENV_NAME} " || [ -d "$CONDA_BASE/envs/$ENV_NAME" ]; then
    echo "  Removing old '$ENV_NAME' env ..."
    "$CONDA_BIN" env remove -y -n "$ENV_NAME" > /dev/null 2>&1
    rm -rf "$CONDA_BASE/envs/$ENV_NAME" 2>/dev/null || true
fi

echo "  Creating env: $ENV_NAME (Python 3.11) ..."
"$CONDA_BIN" create -y -q -n "$ENV_NAME" python=3.11 > /dev/null 2>&1

echo "  Installing packages ..."
"$CONDA_BIN" run -n "$ENV_NAME" pip install -q -r "$SCRIPT_DIR/requirements.txt"
echo -e "${GREEN}✓${NC} tts-mist env ready"

# ── Models: restore from backup OR download from HuggingFace ─────────────────
# This is the LAST step. All envs are ready at this point.

if [ -d "$MODELS_DIR" ]; then
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  Restoring models from ./models/"
    echo "══════════════════════════════════════════════════════════════"
    echo ""

    # RVC voice models → ./worker/rvc/models/
    if [ -d "$MODELS_DIR/rvc_models" ]; then
        cp -a "$MODELS_DIR/rvc_models/." "$SCRIPT_DIR/worker/rvc/models/"
        echo -e "  ${GREEN}✓${NC} RVC models restored"
    fi

    # HeartMuLa checkpoints → ./worker/music/models/ckpt/
    if [ -d "$MODELS_DIR/music_models" ]; then
        mkdir -p "$SCRIPT_DIR/worker/music/models"
        cp -a "$MODELS_DIR/music_models/." "$SCRIPT_DIR/worker/music/models/"
        echo -e "  ${GREEN}✓${NC} HeartMuLa models restored"
    fi

    # ACE-Step checkpoints → ./worker/ace/ACE-Step-1.5/checkpoints/
    if [ -d "$MODELS_DIR/ace_checkpoints" ]; then
        mkdir -p "$SCRIPT_DIR/worker/ace/ACE-Step-1.5/checkpoints"
        cp -a "$MODELS_DIR/ace_checkpoints/." "$SCRIPT_DIR/worker/ace/ACE-Step-1.5/checkpoints/"
        echo -e "  ${GREEN}✓${NC} ACE-Step checkpoints restored"
    fi

    # resemble-enhance model_repo → inside enhance conda env
    if [ -d "$MODELS_DIR/enhance_model_repo" ]; then
        ENHANCE_SITE=$("$CONDA_BIN" run -n enhance python -c "import resemble_enhance; print(resemble_enhance.__path__[0])" 2>/dev/null)
        if [ -n "$ENHANCE_SITE" ]; then
            cp -a "$MODELS_DIR/enhance_model_repo" "$ENHANCE_SITE/model_repo"
            echo -e "  ${GREEN}✓${NC} Enhance models restored"
        fi
    fi

    # HuggingFace models (pyannote, whisper, Qwen3-TTS, flan-t5-xl, depth-anything) → ~/.cache/huggingface/hub/
    HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"
    if [ -d "$MODELS_DIR/huggingface" ]; then
        mkdir -p "$HF_CACHE"
        for model_dir in "$MODELS_DIR/huggingface"/models--*; do
            [ -d "$model_dir" ] || continue
            model_name=$(basename "$model_dir")
            if [ ! -d "$HF_CACHE/$model_name" ]; then
                cp -a "$model_dir" "$HF_CACHE/$model_name"
            fi
        done
        echo -e "  ${GREEN}✓${NC} HuggingFace models restored (pyannote, whisper, Qwen3-TTS, flan-t5-xl, depth-anything)"
    fi

    # Torch hub (demucs) → ~/.cache/torch/hub/checkpoints/
    if [ -d "$MODELS_DIR/torch_hub" ]; then
        mkdir -p "$HOME/.cache/torch/hub/checkpoints"
        cp -a "$MODELS_DIR/torch_hub/." "$HOME/.cache/torch/hub/checkpoints/"
        echo -e "  ${GREEN}✓${NC} Demucs models restored"
    fi

    # EzAudio checkpoints → worker/sfx/ckpts/
    if [ -d "$MODELS_DIR/sfx_ckpts" ]; then
        mkdir -p "$SCRIPT_DIR/worker/sfx/ckpts"
        cp -a "$MODELS_DIR/sfx_ckpts/." "$SCRIPT_DIR/worker/sfx/ckpts/"
        echo -e "  ${GREEN}✓${NC} EzAudio checkpoints restored"
    fi

    # Text worker configs (LLM default overrides) → worker/text/models/
    if [ -d "$MODELS_DIR/text_configs" ]; then
        mkdir -p "$SCRIPT_DIR/worker/text/models"
        cp -a "$MODELS_DIR/text_configs/." "$SCRIPT_DIR/worker/text/models/"
        echo -e "  ${GREEN}✓${NC} Text worker configs restored"
    fi

    # FLUX.2 models (HF cache) → worker/image/models/
    if [ -d "$MODELS_DIR/image_models" ]; then
        mkdir -p "$SCRIPT_DIR/worker/image/models"
        cp -a "$MODELS_DIR/image_models/." "$SCRIPT_DIR/worker/image/models/"
        echo -e "  ${GREEN}✓${NC} FLUX.2 models restored"
    fi

    # DWPose models → worker/pose/models/
    if [ -d "$MODELS_DIR/pose_models" ]; then
        mkdir -p "$SCRIPT_DIR/worker/pose/models"
        cp -a "$MODELS_DIR/pose_models/." "$SCRIPT_DIR/worker/pose/models/"
        echo -e "  ${GREEN}✓${NC} DWPose models restored"
    fi

    # SD 1.5 models + LoRAs → worker/sd15/
    if [ -d "$MODELS_DIR/sd15_models" ]; then
        mkdir -p "$SCRIPT_DIR/worker/sd15/models"
        cp -a "$MODELS_DIR/sd15_models/." "$SCRIPT_DIR/worker/sd15/models/"
        echo -e "  ${GREEN}✓${NC} SD 1.5 models restored"
    fi
    if [ -d "$MODELS_DIR/sd15_loras" ]; then
        mkdir -p "$SCRIPT_DIR/worker/sd15/loras"
        cp -a "$MODELS_DIR/sd15_loras/." "$SCRIPT_DIR/worker/sd15/loras/"
        echo -e "  ${GREEN}✓${NC} SD 1.5 LoRAs restored"
    fi

    # Depth Pro model → worker/depth/models/
    if [ -d "$MODELS_DIR/depth_models" ]; then
        mkdir -p "$SCRIPT_DIR/worker/depth/models"
        cp -a "$MODELS_DIR/depth_models/." "$SCRIPT_DIR/worker/depth/models/"
        echo -e "  ${GREEN}✓${NC} Depth Anything V2 models restored"
    fi

    # Lineart models → worker/lineart/models/
    if [ -d "$MODELS_DIR/lineart_models" ]; then
        mkdir -p "$SCRIPT_DIR/worker/lineart/models"
        cp -a "$MODELS_DIR/lineart_models/." "$SCRIPT_DIR/worker/lineart/models/"
        echo -e "  ${GREEN}✓${NC} Lineart models restored"
    fi

    # NormalMap models → worker/normalmap/models/
    if [ -d "$MODELS_DIR/normalmap_models" ]; then
        mkdir -p "$SCRIPT_DIR/worker/normalmap/models"
        cp -a "$MODELS_DIR/normalmap_models/." "$SCRIPT_DIR/worker/normalmap/models/"
        echo -e "  ${GREEN}✓${NC} NormalMap models restored"
    fi

    # Sketch models → worker/sketch/models/
    if [ -d "$MODELS_DIR/sketch_models" ]; then
        mkdir -p "$SCRIPT_DIR/worker/sketch/models"
        cp -a "$MODELS_DIR/sketch_models/." "$SCRIPT_DIR/worker/sketch/models/"
        echo -e "  ${GREEN}✓${NC} Sketch models restored"
    fi

    # Upscale models → worker/upscale/models/
    if [ -d "$MODELS_DIR/upscale_models" ]; then
        mkdir -p "$SCRIPT_DIR/worker/upscale/models"
        cp -a "$MODELS_DIR/upscale_models/." "$SCRIPT_DIR/worker/upscale/models/"
        echo -e "  ${GREEN}✓${NC} Upscale models restored"
    fi

    # Segment models (rembg/BiRefNet) → ~/.u2net/
    if [ -d "$MODELS_DIR/segment_u2net" ]; then
        mkdir -p "$HOME/.u2net"
        cp -a "$MODELS_DIR/segment_u2net/." "$HOME/.u2net/"
        echo -e "  ${GREEN}✓${NC} Segment models restored"
    fi

    # LTX-2.3 models → worker/ltx2/models/
    if [ -d "$MODELS_DIR/ltx2_models" ]; then
        mkdir -p "$SCRIPT_DIR/worker/ltx2/models"
        cp -a "$MODELS_DIR/ltx2_models/." "$SCRIPT_DIR/worker/ltx2/models/"
        echo -e "  ${GREEN}✓${NC} LTX-2.3 models restored"
    fi

    echo -e "  ${GREEN}✓${NC} Model restore complete"

else
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  Downloading models (no ./models/ backup found)"
    echo "══════════════════════════════════════════════════════════════"
    echo ""

    HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"

    # ── HeartMuLa checkpoints ────────────────────────────────────────────
    CKPT_DIR="$SCRIPT_DIR/worker/music/models/ckpt"
    mkdir -p "$CKPT_DIR"

    echo "── HeartMuLa checkpoints ──"
    if [ -d "$CKPT_DIR/HeartMuLa-oss-3B" ]; then
        echo "  HeartMuLa checkpoints already present — skipping"
    else
        echo "  [1/4] HeartMuLa/HeartMuLaGen (tokenizer + config) ..."
        "$CONDA_BIN" run -n heartmula python -c "
from huggingface_hub import snapshot_download
snapshot_download('HeartMuLa/HeartMuLaGen', local_dir='$CKPT_DIR')
print('  OK')
"
        echo "  [2/4] HeartMuLa/HeartMuLa-oss-3B-happy-new-year (model weights) ..."
        "$CONDA_BIN" run -n heartmula python -c "
from huggingface_hub import snapshot_download
snapshot_download('HeartMuLa/HeartMuLa-oss-3B-happy-new-year',
                  local_dir='$CKPT_DIR/HeartMuLa-oss-3B')
print('  OK')
"
        echo "  [3/4] HeartMuLa/HeartCodec-oss-20260123 (codec) ..."
        "$CONDA_BIN" run -n heartmula python -c "
from huggingface_hub import snapshot_download
snapshot_download('HeartMuLa/HeartCodec-oss-20260123',
                  local_dir='$CKPT_DIR/HeartCodec-oss')
print('  OK')
"
        echo "  [4/4] HeartMuLa/HeartTranscriptor-oss (lyrics transcription) ..."
        "$CONDA_BIN" run -n heartmula python -c "
from huggingface_hub import snapshot_download
snapshot_download('HeartMuLa/HeartTranscriptor-oss',
                  local_dir='$CKPT_DIR/HeartTranscriptor-oss')
print('  OK')
"
    fi
    echo -e "${GREEN}✓${NC} HeartMuLa checkpoints downloaded"

    # ── ACE-Step DiT models ──────────────────────────────────────────────
    ACESTEP_DIR="$SCRIPT_DIR/worker/ace/ACE-Step-1.5"
    UV_BIN="${UV_PATH:-$(command -v uv)}"

    echo ""
    echo "── ACE-Step models ──"
    for MODEL_NAME in acestep-v15-turbo acestep-v15-sft acestep-v15-base acestep-5Hz-lm-0.6B; do
        if [ -f "$ACESTEP_DIR/checkpoints/$MODEL_NAME/model.safetensors" ]; then
            echo "  $MODEL_NAME already present — skipping"
        else
            echo "  Downloading $MODEL_NAME ..."
            cd "$ACESTEP_DIR"
            "$UV_BIN" run python -c "
from acestep.model_downloader import ensure_dit_model
from pathlib import Path
ok, msg = ensure_dit_model('$MODEL_NAME', Path('$ACESTEP_DIR/checkpoints'))
print(msg)
if not ok:
    raise SystemExit(1)
"
            cd "$SCRIPT_DIR"
        fi
    done
    echo -e "${GREEN}✓${NC} ACE-Step models downloaded"

    # ── Whisper model ────────────────────────────────────────────────────
    echo ""
    echo "── Whisper model ──"
    if [ -d "$HF_CACHE/models--mlx-community--whisper-large-v3-turbo" ]; then
        echo "  Whisper model already present — skipping"
    else
        "$CONDA_BIN" run -n whisper python -c "
from huggingface_hub import snapshot_download
path = snapshot_download('mlx-community/whisper-large-v3-turbo')
print(f'  Model cached at: {path}')
"
    fi
    echo -e "${GREEN}✓${NC} Whisper model downloaded"

    # ── Qwen3-TTS models ─────────────────────────────────────────────
    echo ""
    echo "── Qwen3-TTS models ──"
    if [ -d "$HF_CACHE/models--mlx-community--Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit" ]; then
        echo "  Qwen3-TTS models already present — skipping"
    else
        "$CONDA_BIN" run -n ai-tts python -c "
from huggingface_hub import snapshot_download
path = snapshot_download('mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit')
print(f'  1.7B cached at: {path}')
path = snapshot_download('mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit')
print(f'  0.6B cached at: {path}')
path = snapshot_download('mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit')
print(f'  1.7B Base (voice cloning) cached at: {path}')
"
    fi
    echo -e "${GREEN}✓${NC} Qwen3-TTS models downloaded"

    # ── EzAudio checkpoints ───────────────────────────────────────────
    SFX_CKPT_DIR="$SCRIPT_DIR/worker/sfx/ckpts"
    echo ""
    echo "── EzAudio checkpoints ──"
    mkdir -p "$SFX_CKPT_DIR/s3" "$SFX_CKPT_DIR/vae"

    for CKPT_FILE in "s3/ezaudio_s3_xl.pt" "vae/1m.pt"; do
        LOCAL="$SFX_CKPT_DIR/$CKPT_FILE"
        if [ -f "$LOCAL" ]; then
            echo "  $CKPT_FILE already present — skipping"
        else
            echo "  Downloading $CKPT_FILE ..."
            curl -L -o "$LOCAL" "https://huggingface.co/OpenSound/EzAudio/resolve/main/ckpts/$CKPT_FILE"
        fi
    done
    echo -e "${GREEN}✓${NC} EzAudio checkpoints downloaded"

    # ── EzAudio text encoder (flan-t5-xl) ─────────────────────────────
    echo ""
    echo "── EzAudio text encoder (flan-t5-xl) ──"
    if [ -d "$HF_CACHE/models--google--flan-t5-xl" ]; then
        echo "  flan-t5-xl already present — skipping"
    else
        "$CONDA_BIN" run -n ezaudio python -c "
from transformers import T5Tokenizer, T5EncoderModel
T5Tokenizer.from_pretrained('google/flan-t5-xl')
T5EncoderModel.from_pretrained('google/flan-t5-xl')
print('  flan-t5-xl cached')
"
    fi
    echo -e "${GREEN}✓${NC} EzAudio text encoder downloaded"

    # ── FLUX.2 models ────────────────────────────────────────────────
    IMAGE_MODELS_DIR="$SCRIPT_DIR/worker/image/models"
    echo ""
    echo "── FLUX.2 models ──"
    if [ -d "$IMAGE_MODELS_DIR/hub/models--black-forest-labs--FLUX.2-klein-4B" ]; then
        echo "  FLUX.2 models already present — skipping"
    else
        "$CONDA_BIN" run -n flux2 python -c "
import os
os.environ['HF_HOME'] = '$IMAGE_MODELS_DIR'
from huggingface_hub import hf_hub_download
for repo, fn in [
    ('black-forest-labs/FLUX.2-klein-4B', 'flux-2-klein-4b.safetensors'),
    ('black-forest-labs/FLUX.2-klein-9B', 'flux-2-klein-9b.safetensors'),
    ('black-forest-labs/FLUX.2-klein-base-4B', 'flux-2-klein-base-4b.safetensors'),
    ('black-forest-labs/FLUX.2-klein-base-9B', 'flux-2-klein-base-9b.safetensors'),
    ('black-forest-labs/FLUX.2-dev', 'ae.safetensors'),
]:
    print(f'  Downloading {repo}/{fn} ...')
    hf_hub_download(repo_id=repo, filename=fn, repo_type='model')

# Text encoders
from huggingface_hub import snapshot_download
print('  Downloading Qwen/Qwen3-4B (text encoder for 4B models) ...')
snapshot_download('Qwen/Qwen3-4B')
print('  Downloading Qwen/Qwen3-8B (text encoder for 9B models) ...')
snapshot_download('Qwen/Qwen3-8B')
print('  Done')
"
    fi
    echo -e "${GREEN}✓${NC} FLUX.2 models downloaded"

    # ── OpenPose (DWPose) models ──────────────────────────────────────
    POSE_MODELS_DIR="$SCRIPT_DIR/worker/pose/models"
    echo ""
    echo "── DWPose models ──"
    if [ -d "$POSE_MODELS_DIR/hub/models--yzd-v--DWPose" ]; then
        echo "  DWPose models already present — skipping"
    else
        "$CONDA_BIN" run -n openpose python -c "
import os
os.environ['HF_HOME'] = '$POSE_MODELS_DIR'
from huggingface_hub import hf_hub_download
print('  Downloading yolox_l.onnx (person detector) ...')
hf_hub_download('yzd-v/DWPose', 'yolox_l.onnx')
print('  Downloading dw-ll_ucoco_384.onnx (pose estimator) ...')
hf_hub_download('yzd-v/DWPose', 'dw-ll_ucoco_384.onnx')
print('  Done')
"
    fi
    echo -e "${GREEN}✓${NC} DWPose models downloaded"

    # ── SD 1.5 models (CivitAI) ──────────────────────────────────────
    SD15_MODELS_DIR="$SCRIPT_DIR/worker/sd15/models"
    SD15_LORAS_DIR="$SCRIPT_DIR/worker/sd15/loras"
    mkdir -p "$SD15_MODELS_DIR" "$SD15_LORAS_DIR"

    echo ""
    echo "── SD 1.5 models (CivitAI) ──"

    if [ ! -f "$SD15_MODELS_DIR/maturemalemix_v14.safetensors" ]; then
        echo "  Downloading MatureMaleMix v1.4 …"
        curl -L -o "$SD15_MODELS_DIR/maturemalemix_v14.safetensors" \
            "https://civitai.com/api/download/models/75441?type=Model&format=SafeTensor&size=full&fp=fp16" || {
            echo "  ⚠ Failed to download MatureMaleMix"
        }
    else
        echo "  MatureMaleMix v1.4 already exists"
    fi

    if [ ! -f "$SD15_MODELS_DIR/dreamshaper_8.safetensors" ]; then
        echo "  Downloading DreamShaper 8 …"
        curl -L -o "$SD15_MODELS_DIR/dreamshaper_8.safetensors" \
            "https://civitai.com/api/download/models/128713?type=Model&format=SafeTensor&size=pruned&fp=fp16" || {
            echo "  ⚠ Failed to download DreamShaper"
        }
    else
        echo "  DreamShaper 8 already exists"
    fi

    if [ ! -f "$SD15_LORAS_DIR/add_detail.safetensors" ]; then
        echo "  Downloading Add More Details LoRA …"
        curl -L -o "$SD15_LORAS_DIR/add_detail.safetensors" \
            "https://civitai.com/api/download/models/87153?type=Model&format=SafeTensor" || {
            echo "  ⚠ Failed to download LoRA"
        }
    else
        echo "  Add More Details LoRA already exists"
    fi

    echo -e "${GREEN}✓${NC} SD 1.5 models downloaded"

    # ── Depth Anything V2 models ────────────────────────────────────────
    echo ""
    echo "── Depth Anything V2 models ──"
    if [ -d "$HF_CACHE/models--depth-anything--Depth-Anything-V2-Small-hf" ]; then
        echo "  Depth Anything V2 already present — skipping"
    else
        "$CONDA_BIN" run --no-capture-output -n depth python -c "
import os
os.environ['HF_HOME'] = '$SCRIPT_DIR/worker/depth/models'
from transformers import pipeline
print('  Downloading Depth Anything V2 Small …')
pipeline('depth-estimation', model='depth-anything/Depth-Anything-V2-Small-hf')
print('  Done')
" || {
            echo "  ⚠ Failed to download Depth Anything V2"
        }
    fi
    echo -e "${GREEN}✓${NC} Depth Anything V2 models downloaded"

    # ── LTX-2.3 models ────────────────────────────────────────────────
    LTX2_MODELS_DIR="$SCRIPT_DIR/worker/ltx2/models"
    mkdir -p "$LTX2_MODELS_DIR"
    echo ""
    echo "── LTX-2.3 models ──"
    if [ -f "$LTX2_MODELS_DIR/ltx-2.3-22b-distilled.safetensors" ]; then
        echo "  LTX-2.3 models already present — skipping"
    else
        "$CONDA_BIN" run -n ltx2 python -c "
import os
os.environ['HF_HOME'] = '$LTX2_MODELS_DIR'
from huggingface_hub import hf_hub_download, snapshot_download

for fn in [
    'ltx-2.3-22b-distilled.safetensors',
    'ltx-2.3-22b-dev.safetensors',
    'ltx-2.3-spatial-upscaler-x2-1.1.safetensors',
    'ltx-2.3-22b-distilled-lora-384.safetensors',
]:
    print(f'  Downloading {fn} ...')
    hf_hub_download(repo_id='Lightricks/LTX-2.3', filename=fn, local_dir='$LTX2_MODELS_DIR')

print('  Downloading Gemma 3 12B (text encoder) ...')
snapshot_download('google/gemma-3-12b-it', local_dir='$LTX2_MODELS_DIR/gemma-3-12b-it')
print('  Done')
"
    fi
    echo -e "${GREEN}✓${NC} LTX-2.3 models downloaded"

    echo ""
    echo -e "${GREEN}✓${NC} All model downloads complete"
fi

# ── Done ─────────────────────────────────────────────────────────────────────

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Setup complete!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  Usage:"
echo ""
echo "    # AI Text-to-Speech"
echo "    python generate.py voice ai-tts --text 'Hello world' -o demos/"
echo "    python generate.py voice ai-tts -v Serena -t 'dramatic' --text 'Silence.' -o demos/"
echo ""
echo "    # Start RVC worker"
echo "    python generate.py server start"
echo ""
echo "    # Show available models"
echo "    python generate.py ps"
echo ""
echo "    # Install a voice model"
echo "    python generate.py models --engine rvc search \"neutral male\""
echo "    python generate.py models --engine rvc install <model-id>"
echo ""
echo "    # Voice conversion"
echo "    python generate.py voice rvc --model my-voice input.wav -o output/"
echo ""
echo "    # Generate music"
echo "    python generate.py audio ace-step -l 'In der Disco' -t 'disco,happy' -o music.mp3"
echo ""
