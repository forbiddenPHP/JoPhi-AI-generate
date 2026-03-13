# Revoicer

CLI + Web-UI for voice conversion, audio enhancement, AI music generation, lyrics transcription, speaker diarization, and audio source separation.

## Quick Setup

```bash
# 1. Install all environments (6 conda envs + 1 uv project)
bash setup.sh

# 2. Activate main env
conda activate tts-mist

# 3. Start RVC worker (background server for voice conversion)
python revoicer.py server start

# 4. Install a voice model
python revoicer.py models search "neutral male"
python revoicer.py models install User/ModelName

# 5. Ready!
python revoicer.py convert input.wav
```

### Requirements

- macOS (Apple Silicon)
- Miniconda (`brew install --cask miniconda`)
- uv (`brew install uv`) — for ACE-Step music generation
- ffmpeg (`brew install ffmpeg`)
- git-lfs (`brew install git-lfs`)
- ~20 GB disk for model checkpoints

### Environments

`setup.sh` creates 7 isolated conda environments + 1 uv project to avoid dependency conflicts:

| Env | Python | Manager | Purpose |
|-----|--------|---------|---------|
| `tts-mist` | 3.11 | conda | Main CLI + Web UI |
| `rvc` | 3.10 | conda | RVC voice conversion worker |
| `enhance` | 3.10 | conda | Audio enhancement (resemble-enhance) |
| `whisper` | 3.12 | conda | Audio transcription (mlx-whisper) |
| `heartmula` | 3.10 | conda | HeartMuLa music generation + lyrics transcription |
| `diarize` | 3.10 | conda | Speaker diarization (pyannote.audio) |
| `separate` | 3.10 | conda | Audio source separation (demucs) |
| `ace_worker` | 3.11+ | uv | ACE-Step 1.5 music generation (default engine) |

### Offline Safety (Wheels Cache)

Each worker has a local `wheels/` directory with cached Python packages and a `requirements.lock` with pinned versions. This is intentional — for several reasons:

- **RVC, resemble-enhance, and heartlib had to be partially rewritten** to run on macOS/Apple Silicon (MPS instead of CUDA, deepspeed workarounds, FAISS fixes, etc.). If the upstream repos introduce breaking changes or disappear, these modifications would be lost.
- **Reproducible builds**: The exact same package versions are installed every time. No "works on my machine".
- **Offline install**: Works without internet access — ideal for air-gapped systems or unreliable connections.

The install scripts check in this order:
1. `wheels/` present → Offline install from local `.whl` files
2. `requirements.lock` present → Online install with pinned versions
3. Neither found → Fallback to PyPI, then generate lockfile

```
rvc_worker/wheels/      — 283 MB (71 packages)
enhance_worker/wheels/  — 216 MB (99 packages)
music_worker/wheels/    — 199 MB (95 packages)
separate_worker/wheels/ —  92 MB (29 packages)
```

### Model Backup (Offline Restore)

All model checkpoints (~39 GB total) can be backed up for offline restore on a fresh system. This ensures that even if HuggingFace, PyTorch Hub, or upstream repos go offline, the project can be fully restored.

```bash
# Create backup after initial setup
bash backup-models.sh          # → ./models/ directory
bash backup-models.sh --zip    # → ./models/ + models.zip

# Restore on fresh system: place models/ or models.zip next to setup.sh
bash setup.sh                  # auto-detects and restores models
```

**What gets backed up:**

| Backup path | Source | Size |
|---|---|---|
| `models/rvc_models/` | Installed RVC voice models | ~2.4 GB |
| `models/music_models/` | HeartMuLa checkpoints | ~24 GB |
| `models/ace_checkpoints/` | ACE-Step 1.5 DiT + LM | ~9.4 GB |
| `models/enhance_model_repo/` | resemble-enhance weights | ~1.5 GB |
| `models/huggingface/` | pyannote + mlx-whisper (HF cache) | ~1.5 GB |
| `models/torch_hub/` | demucs models (torch cache) | ~169 MB |

Both `models/` and `models.zip` are gitignored.

---

## Features

### Voice Conversion (`convert`)

Convert audio to a different voice using RVC with automatic pitch detection.

```bash
# Single file
python revoicer.py convert input.wav

# With specific voice model
python revoicer.py convert input.wav --voice my-model

# Batch conversion
python revoicer.py convert *.wav -o ./output/

# Manual pitch shift (semitones)
python revoicer.py convert input.wav --pitch 12

# Set target pitch directly
python revoicer.py convert input.wav --target-hz 280

# Choose pitch detection algorithm
python revoicer.py convert input.wav --decoder crepe
```

**Options:**
- `-o, --output` — Output directory
- `-v, --voice` — Voice model name
- `--decoder` — Pitch detection: `rmvpe` (default), `crepe`, `harvest`, `pm`
- `--pitch` — Manual pitch shift in semitones (disables auto-pitch)
- `--target-hz` — Target voice pitch in Hz

---

### Audio Enhancement (`enhance`)

Denoise and super-resolve audio using resemble-enhance.

```bash
# Full enhancement (denoise + super-resolution)
python revoicer.py enhance input.wav -o ./enhanced/

# Denoise only (faster)
python revoicer.py enhance input.wav --denoise-only

# Batch
python revoicer.py enhance *.wav -o ./enhanced/
```

**Options:**
- `-o, --output` — Output directory
- `--denoise-only` — Skip super-resolution
- `--enhance-only` — Only super-resolution, skip denoising

---

### Music Generation (`music`)

Generate music from lyrics using ACE-Step 1.5 (default) or HeartMuLa.

```bash
# ACE-Step (default engine)
python revoicer.py music \
  -f lyrics.txt \
  -t "upbeat electronic dance music with synth bass" \
  -s 30 -o song.mp3

# HeartMuLa engine
python revoicer.py music --engine heart \
  -f lyrics.txt \
  -t "disco,happy,synthesizer" \
  -s 30 -o song.mp3

# ACE-Step with advanced params
python revoicer.py music -f lyrics.txt \
  -t "cinematic orchestral" \
  --steps 16 --cfg-scale 10.0 --seed 42 -s 60
```

**Common options (both engines):**
- `--engine {ace,ace-turbo,ace-sft,ace-base,heart}` — Music engine (default: `ace`)
  - `ace` / `ace-turbo` — ACE-Step 1.5 turbo (8 steps, fast)
  - `ace-sft` — ACE-Step 1.5 SFT (50 steps, high quality)
  - `ace-base` — ACE-Step 1.5 base (50 steps, all features)
  - `heart` — HeartMuLa
- `-l, --lyrics` — Inline lyrics text
- `-f, --lyrics-file` — Path to lyrics file
- `-t, --tags` — Style tags or caption (required)
- `-o, --output` — Output path (default: `./music_<timestamp>.mp3`)
- `-s, --seconds` — Max audio length in seconds (default: 20)
- `--duration` — Max audio length in ms (overrides `--seconds`)
- `--seed` — Random seed for reproducibility
- `--topk` — Top-k sampling (heart: 50, ace: 0=off)
- `--temperature` — Sampling temperature (heart: 1.0, ace: 0.85)
- `--cfg-scale` — CFG scale (heart: 1.5, ace: 7.0)
- `--bpm` — Beats per minute (default: auto)
- `--keyscale` — Musical key, e.g. `"C Major"`, `"Am"` (default: auto)
- `--timesignature` — Time signature: `2`=2/4, `3`=3/4, `4`=4/4, `6`=6/8 (default: auto)
- `--timeout` — Generation timeout in seconds (default: 1800)

**ACE-Step specific options:**
- `--steps` — Inference steps (default: 8)
- `--shift` — Timestep shift (default: 3.0)
- `--no-thinking` — Disable LM chain-of-thought
- `--infer-method {ode,sde}` — Inference method (default: ode)
- `--lm-cfg` — LM guidance scale (default: 2.0)
- `--top-p` — Nucleus sampling (default: 0.9)
- `--batch-size` — Parallel samples (default: 1)
- `--instrumental` — Force instrumental output

**Lyrics format:** Use section tags like `[Verse]`, `[Chorus]`, `[Bridge]`, `[Intro]`, `[Outro]`.

---

### Audio Transcription (`transcribe`)

Transcribe audio to text using mlx-whisper (Apple Silicon optimized).

```bash
# Transcribe to JSON (default)
python revoicer.py transcribe audio.wav

# All formats (json, txt, srt, vtt, tsv)
python revoicer.py transcribe audio.wav --format all -o ./output/

# With language hint and word timestamps
python revoicer.py transcribe audio.wav --input-language en --word-timestamps

# Specific model
python revoicer.py transcribe audio.wav --model large-v3

# Batch transcription
python revoicer.py transcribe *.wav --format srt -o ./transcripts/
```

**Options:**
- `--model` — Whisper model: `tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo` (default)
- `--input-language` — Language hint (e.g. `en`, `de`, `ja`)
- `--word-timestamps` — Include word-level timing
- `--format` — Output format: `json` (default), `txt`, `srt`, `vtt`, `tsv`, `all`
- `-o, --output` — Output directory for transcript files
- `--timeout` — Timeout in seconds (default: 600)

---

### Speaker Diarization (`diarize`)

Split dialogue audio into separate tracks per speaker using pyannote.audio (runs on MPS). Each output file has the full length of the original — silence where the speaker is not active.

```bash
# Auto-detect speakers
python revoicer.py diarize interview.wav -o ./speakers/

# Known number of speakers
python revoicer.py diarize podcast.wav -o ./speakers/ --speakers 3

# With statistics (gaps, overlaps, coverage)
python revoicer.py diarize dialog.wav -o ./speakers/ --verify
```

**Output:**
- `dialog_SPEAKER_00.wav` — Full length, only Speaker 0, silence elsewhere
- `dialog_SPEAKER_01.wav` — Full length, only Speaker 1, silence elsewhere
- `dialog_SPEAKER_00_compact.wav` — Active parts only (for transcription)
- `dialog_diarize.json` — Diarization segments (start, end, speaker)
- `dialog_stats.json` — Statistics (with `--verify`)

**Options:**
- `-o, --output` — Output directory
- `--speakers` — Number of speakers (auto-detect if not set)
- `--hf-token` — HuggingFace token (or set `HF_TOKEN` env var)
- `--verify` — Show diarization statistics (segments, coverage, gaps, overlaps)

**Note:** pyannote.audio models are gated on HuggingFace. You need to accept the terms at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0), then set `HF_TOKEN` or run `huggingface-cli login`.

---

### Audio Source Separation (`separate`)

Split audio into stems (vocals, drums, bass, other) using demucs. Each stem has the full length of the original.

```bash
# Default model (htdemucs)
python revoicer.py separate song.mp3 -o ./stems/

# Fine-tuned model (higher quality)
python revoicer.py separate song.mp3 -o ./stems/ --model htdemucs_ft

# Batch separation
python revoicer.py separate *.mp3 -o ./stems/
```

**Output:**
- `song_vocals.wav`
- `song_drums.wav`
- `song_bass.wav`
- `song_other.wav`

**Options:**
- `-o, --output` — Output directory
- `--model` — Demucs model (default: `htdemucs`, alt: `htdemucs_ft`)

---

### Lyrics Transcription (`transcribe-lyrics`)

Extract lyrics from audio using HeartTranscriptor.

```bash
# Print to stdout
python revoicer.py transcribe-lyrics song.mp3

# Save to file
python revoicer.py transcribe-lyrics song.mp3 -o lyrics.txt
```

**Options:**
- `-o, --output` — Output file (default: print to stdout)
- `--timeout` — Timeout in seconds (default: 600)

**Tip:** For best results, separate vocals first (e.g. with demucs) before transcribing.

---

### Model Management (`models`)

```bash
# List installed models
python revoicer.py models list

# Search HuggingFace + voice-models.com
python revoicer.py models search "female singer"
python revoicer.py models search "anime" --limit 50

# Install from HuggingFace
python revoicer.py models install User/ModelRepo
python revoicer.py models install User/MultiModelRepo --file "specific_voice"
python revoicer.py models install User/Repo --name "my-custom-name"

# Install from direct URL
python revoicer.py models install "https://huggingface.co/.../model.zip"

# Remove
python revoicer.py models remove my-model

# Calibrate pitch (auto-detect target F0)
python revoicer.py models calibrate my-model

# Set pitch manually
python revoicer.py models set-pitch my-model 120   # male ~120 Hz
python revoicer.py models set-pitch my-model 220   # female ~220 Hz
python revoicer.py models set-pitch my-model 280   # child ~280 Hz
```

---

### Server Management (`server`)

The RVC worker runs as a background API server on port 5100.

```bash
python revoicer.py server start          # Start (default port 5100)
python revoicer.py server start -p 5200  # Custom port
python revoicer.py server status         # Check status
python revoicer.py server stop           # Stop
```

---

### System Status (`--PS`)

```bash
python revoicer.py --PS
```

Shows installed models, target pitch settings, and server status.

---

### Web UI (`app.py`)

```bash
python app.py
# Open http://localhost:5000
```

Features:
- Dashboard with RVC worker status
- Model browser (search + install from HuggingFace)
- Audio conversion with playback
- Batch conversion

API endpoints:
- `GET /api/status` — Server status
- `GET /api/models` — List models
- `GET /api/models/search?q=query` — Search models
- `POST /api/models/install` — Install model
- `DELETE /api/models/<name>` — Remove model
- `POST /api/convert` — Convert audio
- `GET /api/audio/<filename>` — Download result

---

## TODOs

### app.py Web UI
- [ ] Add music generation page (lyrics input, tag picker, duration slider, generate + play)
- [ ] Add lyrics transcription page (upload audio, show extracted text)
- [ ] Add audio enhancement page (upload, denoise/enhance, download)
- [ ] Add model calibration UI (pitch detection visualization, manual F0 override)
- [ ] Real-time generation progress (WebSocket or SSE for progress bars)
- [ ] Audio waveform visualization
- [ ] Drag & drop file upload

### Pipeline
- [ ] Queue system for long-running generation jobs
- [ ] GPU memory management (load/unload models on demand)
- [ ] Audio output normalization (consistent loudness across engines and tools)

### Music Generation
- [ ] MPS float16/bfloat16 support (currently float32 — autocast warning)
- [ ] HeartCLAP integration (audio-text alignment, when heartlib implements the pipeline)
- [ ] Song structure support (generate full songs with verse/chorus/bridge transitions)
- [x] Vocal separation (demucs) as preprocessing for transcription
