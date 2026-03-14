#!/usr/bin/env bash
set -e

# ── Revoicer — Master Installer ─────────────────────────────────────────────
# Creates nine conda environments + one uv project:
#   1. rvc        — Python 3.10 + pip<=23.3 for RVC voice conversion worker
#   2. enhance    — Python 3.10 for resemble-enhance (audio post-processing)
#   3. heartmula  — Python 3.10 for HeartMuLa music generation
#   4. acestep    — ACE-Step 1.5 music generation (managed by uv, not conda)
#   5. whisper    — Python 3.12 for mlx-whisper transcription (Apple Silicon)
#   6. diarize    — Python 3.10 for speaker diarization (pyannote.audio)
#   7. separate   — Python 3.10 for audio source separation (demucs)
#   8. ai-tts     — Python 3.11 for Qwen3-TTS (mlx-audio, Apple Silicon)
#   9. lang-detect — Python 3.11 for language detection (langdetect)
#  10. tts-mist   — Python 3.11 for CLI + Web-App
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

CONDA_BIN="/opt/miniconda3/bin/conda"
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

if [ ! -f "$CONDA_BIN" ]; then
    echo -e "${RED}ERROR: conda not found at $CONDA_BIN${NC}"
    echo "  Install: brew install --cask miniconda"
    exit 1
fi

# Check Xcode license
if ! /usr/bin/clang --version > /dev/null 2>&1; then
    echo -e "${RED}ERROR: C compiler not available.${NC}"
    echo "  The RVC worker needs to compile C extensions (fairseq, pyworld)."
    echo "  Accept the Xcode license:"
    echo ""
    echo "    sudo xcodebuild -license accept"
    echo ""
    exit 1
fi

# Check brew
if ! command -v brew &> /dev/null; then
    echo -e "${RED}ERROR: brew not found.${NC}"
    echo "  Install: https://brew.sh"
    exit 1
fi

# unar — universeller Entpacker für .rar, .7z etc. (Modell-Archive)
if ! command -v unar &> /dev/null; then
    echo "  Installing unar (archive extractor) ..."
    HOMEBREW_NO_AUTO_UPDATE=1 brew install unar > /dev/null 2>&1
    echo -e "${GREEN}✓${NC} unar installed"
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
echo "── Step 1/10: RVC Worker ──"
bash "$SCRIPT_DIR/rvc_worker/install.sh"

# ── Step 2: Enhance Worker Env ───────────────────────────────────────────────

echo ""
echo "── Step 2/10: Enhance Worker ──"
bash "$SCRIPT_DIR/enhance_worker/install.sh"

# ── Step 3: HeartMuLa Music Worker Env ───────────────────────────────────────

echo ""
echo "── Step 3/10: HeartMuLa Music Worker ──"
bash "$SCRIPT_DIR/music_worker/install.sh"

# ── Step 4: ACE-Step Music Worker (uv) ───────────────────────────────────────

echo ""
echo "── Step 4/10: ACE-Step Music Worker ──"
bash "$SCRIPT_DIR/ace_worker/install.sh"

# ── Step 5: Whisper Worker Env ────────────────────────────────────────────────

echo ""
echo "── Step 5/10: Whisper Worker ──"
bash "$SCRIPT_DIR/whisper_worker/install.sh"

# ── Step 6: Diarize Worker Env ───────────────────────────────────────────────

echo ""
echo "── Step 6/10: Diarize Worker ──"
bash "$SCRIPT_DIR/diarize_worker/install.sh"

# ── Step 7: Separate Worker Env ──────────────────────────────────────────────

echo ""
echo "── Step 7/10: Separate Worker ──"
bash "$SCRIPT_DIR/separate_worker/install.sh"

# ── Step 8: AI-TTS Worker Env ────────────────────────────────────────────────

echo ""
echo "── Step 8/10: AI-TTS Worker ──"
bash "$SCRIPT_DIR/tts_worker/install.sh"

# ── Step 9: Language Detect Worker Env ───────────────────────────────────────

echo ""
echo "── Step 9/10: Language Detect Worker ──"
bash "$SCRIPT_DIR/langdetect_worker/install.sh"

# ── Step 10: Main App Env ────────────────────────────────────────────────────

echo ""
echo "── Step 10/10: Main App (tts-mist) ──"

ENV_NAME="tts-mist"

if "$CONDA_BIN" env list 2>/dev/null | grep -q "^${ENV_NAME} "; then
    echo "  Removing old '$ENV_NAME' env ..."
    "$CONDA_BIN" env remove -y -n "$ENV_NAME" > /dev/null 2>&1
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

    # RVC voice models → ./rvc_models/
    if [ -d "$MODELS_DIR/rvc_models" ]; then
        cp -a "$MODELS_DIR/rvc_models" "$SCRIPT_DIR/rvc_models"
        echo -e "  ${GREEN}✓${NC} RVC models restored"
    fi

    # HeartMuLa checkpoints → ./music_models/ckpt/
    if [ -d "$MODELS_DIR/music_models" ]; then
        mkdir -p "$SCRIPT_DIR/music_models"
        cp -a "$MODELS_DIR/music_models/." "$SCRIPT_DIR/music_models/"
        echo -e "  ${GREEN}✓${NC} HeartMuLa models restored"
    fi

    # ACE-Step checkpoints → ./ace_worker/ACE-Step-1.5/checkpoints/
    if [ -d "$MODELS_DIR/ace_checkpoints" ]; then
        mkdir -p "$SCRIPT_DIR/ace_worker/ACE-Step-1.5/checkpoints"
        cp -a "$MODELS_DIR/ace_checkpoints/." "$SCRIPT_DIR/ace_worker/ACE-Step-1.5/checkpoints/"
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

    # HuggingFace models (pyannote, whisper) → ~/.cache/huggingface/hub/
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
        echo -e "  ${GREEN}✓${NC} HuggingFace models restored (pyannote, whisper, Qwen3-TTS)"
    fi

    # Torch hub (demucs) → ~/.cache/torch/hub/checkpoints/
    if [ -d "$MODELS_DIR/torch_hub" ]; then
        mkdir -p "$HOME/.cache/torch/hub/checkpoints"
        cp -a "$MODELS_DIR/torch_hub/." "$HOME/.cache/torch/hub/checkpoints/"
        echo -e "  ${GREEN}✓${NC} Demucs models restored"
    fi

    echo -e "  ${GREEN}✓${NC} Model restore complete"

else
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  Downloading models (no ./models/ backup found)"
    echo "══════════════════════════════════════════════════════════════"
    echo ""

    # ── HeartMuLa checkpoints ────────────────────────────────────────────
    CKPT_DIR="$SCRIPT_DIR/music_models/ckpt"
    mkdir -p "$CKPT_DIR"

    echo "── HeartMuLa checkpoints ──"
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
    echo -e "${GREEN}✓${NC} HeartMuLa checkpoints downloaded"

    # ── ACE-Step DiT models ──────────────────────────────────────────────
    ACESTEP_DIR="$SCRIPT_DIR/ace_worker/ACE-Step-1.5"
    UV_BIN="${UV_PATH:-$(command -v uv)}"

    echo ""
    echo "── ACE-Step models ──"
    for MODEL_NAME in acestep-v15-sft acestep-v15-base acestep-5Hz-lm-0.6B; do
        if [ -d "$ACESTEP_DIR/checkpoints/$MODEL_NAME" ]; then
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
    "$CONDA_BIN" run -n whisper python -c "
from huggingface_hub import snapshot_download
path = snapshot_download('mlx-community/whisper-large-v3-turbo')
print(f'  Model cached at: {path}')
"
    echo -e "${GREEN}✓${NC} Whisper model downloaded"

    # ── Qwen3-TTS models ─────────────────────────────────────────────
    echo ""
    echo "── Qwen3-TTS models ──"
    "$CONDA_BIN" run -n ai-tts python -c "
from huggingface_hub import snapshot_download
path = snapshot_download('mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit')
print(f'  1.7B cached at: {path}')
path = snapshot_download('mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit')
print(f'  0.6B cached at: {path}')
"
    echo -e "${GREEN}✓${NC} Qwen3-TTS models downloaded"

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
echo "    conda activate tts-mist"
echo ""
echo "    # AI Text-to-Speech"
echo "    python generate.py voice --engine ai-tts --text 'Hello world' -o demos/"
echo "    python generate.py voice --engine ai-tts -v Serena -t 'dramatic' --text 'Silence.' -o demos/"
echo ""
echo "    # Start RVC worker"
echo "    python generate.py server start"
echo ""
echo "    # Show available models"
echo "    python generate.py ps"
echo ""
echo "    # Install a voice model"
echo "    python generate.py models search \"neutral male\""
echo "    python generate.py models install <model-id>"
echo ""
echo "    # Voice conversion"
echo "    python generate.py voice --engine rvc --model my-voice input.wav -o output/"
echo ""
echo "    # Generate music"
echo "    python generate.py audio --engine ace-step -l 'In der Disco' -t 'disco,happy' -o music.mp3"
echo ""
