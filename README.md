# generate — Unified Media Generation Toolkit

CLI for voice synthesis, voice conversion, audio enhancement, AI music generation, lyrics transcription, speaker diarization, and audio source separation. macOS Apple Silicon only.

**Entry point:** `generate.py`

## Quick Setup

```bash
# 1. Install all environments (9 conda envs + 1 uv project)
bash setup.sh

# 2. Activate main env
conda activate tts-mist

# 3. Start RVC worker (needed for voice conversion)
python generate.py server start

# 4. Install a voice model
python generate.py models search "neutral male"
python generate.py models install User/ModelName

# 5. Ready!
python generate.py voice --engine rvc --model my-voice input.wav
```

### Requirements

- macOS (Apple Silicon)
- Miniconda (`brew install --cask miniconda`)
- uv (`brew install uv`) — for ACE-Step music generation
- ffmpeg (`brew install ffmpeg`)
- git-lfs (`brew install git-lfs`)
- Xcode CLI tools with accepted license (`sudo xcodebuild -license accept`)
- ~20 GB disk for model checkpoints

---

## ABI Overview

```
generate.py <medium> --engine <backend> [--model <variant>] [input] [options]
```

| Medium | Engine | Purpose |
|--------|--------|---------|
| `voice` | `ai-tts` | Neural TTS via Qwen3-TTS (text → audio) |
| `voice` | `say` | macOS native TTS (text → audio) |
| `voice` | `rvc` | RVC voice conversion (audio → audio) |
| `audio` | `enhance` | Denoise + super-resolution |
| `audio` | `demucs` | Source separation (stems) |
| `audio` | `ace-step` | AI music generation |
| `audio` | `heartmula` | AI music generation (alt engine) |
| `audio` | `diarize` | Speaker diarization |
| `text` | `whisper` | Audio transcription |
| `text` | `heartmula-transcribe` | Lyrics extraction |
| `output` | `audio-concatenate` | Concatenate audio files (with per-clip trim, fades, volume, crossfade) |
| `output` | `audio-mucs` | Mix/overlay audio files in parallel (with per-clip trim, fades, volume, pan) |
| `server` | — | RVC worker management |
| `models` | — | Model install/search/remove |
| `ps` | — | System status |

Future mediums (stubs): `image`, `video`, `vision`, `translation`, `comparison`

---

## Global Options

### `--screen-log-format json`

Switches all status/progress output from human-readable TUI to machine-readable JSON events. Works with every command.

```bash
python generate.py voice --engine rvc --model my-voice input.wav --screen-log-format json
python generate.py ps --screen-log-format json
```

<details>
<summary>JSON event format</summary>

```json
{"type": "stage", "message": "Loading model ...", "ts": 1710000000.0}
{"type": "progress", "message": "45%|████▍     | 45/100", "percent": 45.0, "ts": 1710000001.0}
{"type": "log", "message": "Processing complete", "ts": 1710000002.0}
```

Worker results (JSON) are always printed to stdout, regardless of mode.

</details>

---

## Text Input (unified)

All engines that accept text input use the same flags:

| Flag | Short | Purpose |
|------|-------|---------|
| `--text` / `--lyrics` | `-l` | Inline text |
| `--text-file` / `--lyrics-file` | `-f` | Read text from file |

Both `--lyrics` and `--text` are aliases — they resolve to the same internal field. Use whichever name fits your context (lyrics for music, text for TTS). `--text-file` / `--lyrics-file` reads the file contents and resolves before the engine runs.

---

## Features

### Voice — AI TTS (`voice --engine ai-tts`)

Neural text-to-speech via Qwen3-TTS (mlx-audio). Runs locally on Apple Silicon.

```bash
# Basic
python generate.py voice --engine ai-tts --text "Hello world" -o demos/

# With preset voice
python generate.py voice --engine ai-tts -v Serena --text "Hello" -o demos/

# With style instructions (refine mode)
python generate.py voice --engine ai-tts -v Aiden -t "dramatic, slow" --text "Silence." -o demos/

# Smaller model (faster, less quality)
python generate.py voice --engine ai-tts --tts-model small --text "Quick test" -o demos/

# Auto-detect language
python generate.py voice --engine ai-tts --text "Der Fuchs springt über den Bach" -o demos/

# Explicit language
python generate.py voice --engine ai-tts -v Dylan --language de --text "Hallo Welt" -o demos/

# Text from file
python generate.py voice --engine ai-tts --text-file story.txt -o demos/

# Dialog (multiple voices)
python generate.py voice --engine ai-tts --text "[Aiden] Hi! [Serena] Hello there!" -o demos/

# Dialog with per-segment style instructions
python generate.py voice --engine ai-tts --text "[Aiden: excited, fast] Hi! [Serena: calm, slow] Hello there." -o demos/

# Reproduce from prompt sidecar file
python generate.py voice --engine ai-tts --prompt-file demos/speech.txt -o demos/

# Reproduce with different voice (CLI flags override sidecar values)
python generate.py voice --engine ai-tts --prompt-file demos/speech.txt -v Dylan -o demos/

# List available voices
python generate.py voice --engine ai-tts --list-voices
```

<details>
<summary>Voices</summary>

| Name | Gender |
|------|--------|
| Aiden | Male |
| Dylan | Male |
| Eric | Male |
| Ryan | Male |
| Uncle_Fu | Male |
| Vivian | Female |
| Serena | Female |
| Ono_Anna | Female |
| Sohee | Female |

</details>

<details>
<summary>Languages</summary>

Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian

Language is auto-detected if `--language` is not set (via langdetect).

</details>

<details>
<summary>Options</summary>

- `--text`, `-l` — Inline text to speak
- `--text-file`, `-f` — Path to text file
- `--prompt-file`, `-p` — Load text + params from prompt sidecar (.txt)
- `-v`, `--voice` — Preset voice (Aiden, Serena, ...)
- `-t`, `--tags` — Style instructions for refine mode ("dramatic, slow, whispering")
- `--tts-model` — Model size: `large` (default, 1.7B) or `small` (0.6B)
- `--language` — ISO language code: de, en, fr, ja, ko, zh, ru, pt, es, it (auto-detected if omitted)
- `--list-voices` — Show available voices
- `-o, --output` — Output directory

**Refine mode:** When `--tags` is set, the model uses an instruction-guided generation pipeline for more expressive speech. Without `--tags`, text is synthesized directly.

**Dialog:** Use `[VoiceName]` markers in text to switch between speakers. Each segment is generated separately and concatenated with 0.4s silence gaps. Without markers, the `-v` voice is used for the entire text.

**Per-segment style:** Use `[VoiceName: style instructions]` to set style per segment. Per-segment instructions override global `--tags`. Example: `[Dylan: excited, fast] Wow! [Serena: calm, slow] Indeed.`

**Prompt sidecar:** Every generation saves a `.txt` file alongside the `.wav` with all parameters (voice, language, model, tags) and the full text. Use `--prompt-file` to reload and reproduce a generation. CLI flags override sidecar values.

**Audio format:** WAV at native model sample rate.

</details>

---

### Voice — macOS TTS (`voice --engine say`)

Generate speech from text using macOS `say` command. Optionally pipe through RVC for voice conversion.

```bash
# System default voice
python generate.py voice --engine say --text "Hallo Welt" -o demos/

# Specific macOS voice
python generate.py voice --engine say -v Anna --text "Hallo Welt" -o demos/

# With speaking rate
python generate.py voice --engine say -v Samantha --rate 180 --text "Hello world" -o demos/

# Text from file
python generate.py voice --engine say --text-file story.txt -o demos/

# Say + RVC voice conversion pipeline
python generate.py voice --engine say --model my-voice --text "Hallo Welt" -o demos/
python generate.py voice --engine say -v Anna --model my-voice --text "Hallo" -o demos/
```

<details>
<summary>Options & details</summary>

**Options:**
- `--text`, `-l` — Inline text to speak
- `--text-file`, `-f` — Path to text file
- `-v`, `--voice` — macOS voice name (default: system voice). List with `say -v '?'`
- `--rate` — Speaking rate in words per minute
- `--model` — RVC model for voice conversion post-processing (optional)
- `-o, --output` — Output directory

**Pipeline:**
- Without `--model`: `say` → 44100 Hz WAV → done
- With `--model`: `say` → temp WAV → RVC conversion → output (temp WAV cleaned up)

**Audio format:** Always outputs 44100 Hz, 16-bit WAV (`--data-format=LEI16@44100`).



</details>

---

### Voice — RVC Conversion (`voice --engine rvc`)

Convert audio to a different voice using RVC with automatic pitch detection. Requires running server (`generate.py server start`).

```bash
# Single file (auto-pitch from model config)
python generate.py voice --engine rvc --model my-voice input.wav

# Batch conversion
python generate.py voice --engine rvc --model my-voice *.wav -o ./output/

# Manual pitch shift (semitones, disables auto-pitch)
python generate.py voice --engine rvc --model my-voice input.wav --pitch 12

# Set target pitch directly (Hz)
python generate.py voice --engine rvc --model my-voice input.wav --target-hz 280

# Choose pitch detection algorithm
python generate.py voice --engine rvc --model my-voice input.wav --decoder crepe
```

<details>
<summary>Options & details</summary>

**Options:**
- `--model` — RVC voice model name (required)
- `-o, --output` — Output directory
- `--decoder` — Pitch detection: `rmvpe` (default), `crepe`, `harvest`, `pm`
- `--pitch` — Manual pitch shift in semitones (disables auto-pitch)
- `--target-hz` — Target voice pitch in Hz (overrides model config)

**Auto-Pitch (default behavior):**

When neither `--pitch` nor `--target-hz` is set:
1. Detects input file's fundamental frequency (F0) using `pyworld.harvest()`
2. Reads model's `target_f0` from `rvc_models/<model>/revoicer.json`
3. Computes pitch shift: `semitones = 12 * log2(target_f0 / input_f0)`
4. Applies the shift via RVC's `f0up_key` parameter

If the model has no `target_f0`, run `models set-pitch` first.

**Implicit behaviors:**
- Non-WAV files are auto-converted to WAV (44.1 kHz) via ffmpeg
- Output format is always WAV
- All output paths are printed as JSON array to stdout

</details>

---

### Audio Enhancement (`audio --engine enhance`)

Denoise and super-resolve audio using resemble-enhance.

```bash
python generate.py audio --engine enhance input.wav -o ./enhanced/
python generate.py audio --engine enhance input.wav --denoise-only
python generate.py audio --engine enhance input.wav --enhance-only
python generate.py audio --engine enhance *.wav -o ./enhanced/
```

<details>
<summary>Options</summary>

- `-o, --output` — Output directory
- `--denoise-only` — Skip super-resolution
- `--enhance-only` — Only super-resolution, skip denoising

</details>

---

### Audio Separation (`audio --engine demucs`)

Split audio into stems (vocals, drums, bass, other) using demucs.

```bash
python generate.py audio --engine demucs song.mp3 -o ./stems/
python generate.py audio --engine demucs song.mp3 -o ./stems/ --model htdemucs_ft
```

Output: `song_vocals.wav`, `song_drums.wav`, `song_bass.wav`, `song_other.wav`

<details>
<summary>Options</summary>

- `-o, --output` — Output directory
- `--model` — Demucs model (default: `htdemucs`, alt: `htdemucs_ft`)

</details>

---

### Music Generation (`audio --engine ace-step` / `audio --engine heartmula`)

Generate music from lyrics and style tags.

```bash
# ACE-Step (default turbo model)
python generate.py audio --engine ace-step \
  -l "[Verse] La la la" -t "upbeat disco" -s 30 -o song.mp3

# ACE-Step SFT (50 steps)
python generate.py audio --engine ace-step --model sft \
  -f lyrics.txt -t "cinematic orchestral" -s 60

# HeartMuLa
python generate.py audio --engine heartmula \
  -l "[Verse] La la la" -t "disco,happy" -s 30 -o song.mp3
```

<details>
<summary>Common options (both engines)</summary>

- `--lyrics` / `--text`, `-l` — Inline lyrics
- `--lyrics-file` / `--text-file`, `-f` — Lyrics from file
- `--tags`, `-t` — Style tags or caption (required)
- `-o, --output` — Output path (default: `./music_<timestamp>.mp3`)
- `-s, --seconds` — Duration in seconds (default: 20)
- `--duration` — Duration in ms (overrides `--seconds`)
- `--seed` — Random seed
- `--topk` — Top-k sampling
- `--temperature` — Sampling temperature
- `--cfg-scale` — CFG scale

ACE-Step `--model` variants: `turbo` (default if omitted, 8 steps), `sft` (50 steps), `base` (50 steps)

</details>

<details>
<summary>ACE-Step specific options</summary>

- `--steps` — Inference steps (default: 8)
- `--shift` — Timestep shift (default: 3.0)
- `--no-thinking` — Disable LM chain-of-thought
- `--infer-method {ode,sde}` — Inference method (default: ode)
- `--lm-cfg` — LM guidance scale (default: 2.0)
- `--top-p` — Nucleus sampling (default: 0.9)
- `--batch-size` — Parallel samples (default: 1)
- `--instrumental` — Force instrumental output
- `--bpm` — Beats per minute
- `--keyscale` — Musical key (e.g. `"C Major"`, `"Am"`)
- `--timesignature` — Time signature: `2`=2/4, `3`=3/4, `4`=4/4, `6`=6/8

</details>

**Lyrics format:** Use section tags like `[Verse]`, `[Chorus]`, `[Bridge]`, `[Intro]`, `[Outro]`.

**Prompt guides:** See [prompt-guides/ACE-Step.md](prompt-guides/ACE-Step.md) and [prompt-guides/HeartMuLa.md](prompt-guides/HeartMuLa.md).

---

### Speaker Diarization (`audio --engine diarize`)

Split dialogue audio into separate tracks per speaker using pyannote.audio (MPS).

```bash
python generate.py audio --engine diarize interview.wav -o ./speakers/
python generate.py audio --engine diarize podcast.wav -o ./speakers/ --speakers 3
python generate.py audio --engine diarize dialog.wav -o ./speakers/ --verify
```

<details>
<summary>Options</summary>

- `-o, --output` — Output directory
- `--speakers` — Number of speakers (auto-detect if not set)
- `--hf-token` — HuggingFace token (or `HF_TOKEN` env var)
- `--verify` — Transcribe each segment to verify diarization quality

**Note:** Requires accepting pyannote model terms on HuggingFace and setting `HF_TOKEN`.

</details>

---

### Transcription (`text --engine whisper`)

Transcribe audio to text using mlx-whisper (Apple Silicon optimized).

```bash
python generate.py text --engine whisper audio.wav
python generate.py text --engine whisper audio.wav --format srt -o ./transcripts/
python generate.py text --engine whisper audio.wav --input-language de --word-timestamps
python generate.py text --engine whisper audio.wav --model large-v3
```

<details>
<summary>Options</summary>

- `--model` — Whisper model: `tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo` (default)
- `--input-language` — Language hint (e.g. `en`, `de`, `ja`)
- `--word-timestamps` — Word-level timing
- `--format` — Output: `json` (default), `txt`, `srt`, `vtt`, `tsv`, `all`
- `-o, --output` — Output directory
- `--timeout` — Timeout in seconds (default: 600)

</details>

---

### Lyrics Extraction (`text --engine heartmula-transcribe`)

Extract lyrics from audio using HeartTranscriptor.

```bash
python generate.py text --engine heartmula-transcribe song.mp3
python generate.py text --engine heartmula-transcribe song.mp3 -o lyrics.txt
```

---

### Audio Concatenation (`output --engine audio-concatenate`)

Concatenate multiple audio files into one. Supports per-clip trim, volume, fades, and crossfade via `--clip`.

```bash
# Simple concatenation
python generate.py output --engine audio-concatenate a.wav b.wav c.mp3 -o out.wav

# With bitrate (for compressed formats)
python generate.py output --engine audio-concatenate a.wav b.wav -o out.mp3 --output-bitrate 128k

# Per-clip options via --clip INDEX:key=val,key=val
python generate.py output --engine audio-concatenate \
  intro.wav speech.wav outro.wav background.mp3 \
  --clip 0:fade-in=0.3 \
  --clip 1:crossfade=0.5,volume=1.2 \
  --clip 3:start=0,end=6.9,volume=0.5,fade-out=0.5 \
  -o final.wav
```

<details>
<summary>Options</summary>

- `-o, --output` — Output file path (format from extension: .wav, .mp3, .ogg, .m4a, .opus, .aiff)
- `--output-bitrate` — Audio bitrate for compressed formats (e.g. `128k`, `320k`)
- `--clip INDEX:key=val,key=val` — Per-clip options (repeatable)

**Per-clip keys:**

| Key | Type | Description |
|-----|------|-------------|
| `fade-in` | float (s) | Fade in from silence at clip start |
| `fade-out` | float (s) | Fade out to silence at clip end |
| `crossfade` | float (s) | Crossfade from previous clip (ignored on first clip) |
| `volume` | float | Volume factor (0.5 = half, 2.0 = double) |
| `start` | float (s) | Trim: start position |
| `end` | float (s) | Trim: end position |
| `pan` | float (-1..+1) | Stereo panning: -1 = left, 0 = center (default), +1 = right |

Clips without `--clip` entry are used as-is. All inputs are normalized to 44100 Hz stereo before processing.

</details>

---

### Audio Mixing (`output --engine audio-mucs`)

Mix multiple audio files in parallel (overlay). Supports per-clip trim, volume, pan, and fades via `--clip`.

```bash
# Simple mix (all tracks overlaid)
python generate.py output --engine audio-mucs track1.wav track2.wav -o mix.wav

# Stereo remix with per-clip pan and volume
python generate.py output --engine audio-mucs \
  vocals.wav drums.wav bass.wav guitars.wav \
  --clip 0:pan=-0.2,volume=0.8 \
  --clip 1:pan=0.3,volume=0.6 \
  --clip 2:pan=0.0,volume=0.7 \
  --clip 3:pan=-0.5,volume=0.5 \
  -o remix.wav
```

<details>
<summary>Options</summary>

- `-o, --output` — Output file path
- `--output-bitrate` — Audio bitrate for compressed formats (e.g. `128k`, `320k`)
- `--clip INDEX:key=val,key=val` — Per-clip options (repeatable)

**Per-clip keys:** Same as `audio-concatenate` (`fade-in`, `fade-out`, `volume`, `start`, `end`, `pan`). `crossfade` is ignored (not applicable for parallel mix).

All inputs are normalized to 44100 Hz stereo. Output is passed through `alimiter` to prevent clipping.

</details>

---

### Model Management (`models`)

```bash
python generate.py models list
python generate.py models search "female singer"
python generate.py models search "anime" --limit 50
python generate.py models install User/ModelRepo
python generate.py models install User/MultiModelRepo --file "specific_voice"
python generate.py models install User/Repo --name "my-custom-name"
python generate.py models install "https://example.com/model.zip"
python generate.py models remove my-model
python generate.py models calibrate my-model        # guess target F0 from model name (heuristic)
python generate.py models set-pitch my-model 120   # male ~120 Hz
python generate.py models set-pitch my-model 220   # female ~220 Hz
python generate.py models set-pitch my-model 280   # child ~280 Hz
```

<details>
<summary>Details</summary>

- Accepts HuggingFace repo IDs or direct download URLs
- Supports `.pth`, `.zip`, `.rar`, `.7z` archives
- Multi-model repos: auto-installs all models found
- Auto-calibrates target F0 after install

**Calibration heuristics:**
- Child/young voices: 280 Hz
- Female voices: 220 Hz
- Male voices: 120 Hz

**Model config:** `rvc_models/<model>/revoicer.json` with `target_f0` and optional `hf_repo_id`.

</details>

---

### Server Management (`server`)

The RVC worker runs as a background API server on port 5100.

```bash
python generate.py server start              # default port 5100
python generate.py server start -p 5200      # custom port
python generate.py server status
python generate.py server stop
```

---

### System Status (`ps`)

```bash
python generate.py ps
python generate.py ps --screen-log-format json
```

Shows installed models, target pitch settings, server status, and supported formats.

---

---

<details>
<summary><strong>Environments</strong></summary>

`setup.sh` creates 9 isolated conda environments + 1 uv project:

| Env | Python | Manager | Purpose |
|-----|--------|---------|---------|
| `tts-mist` | 3.11 | conda | Main CLI |
| `rvc` | 3.10 | conda | RVC voice conversion worker |
| `enhance` | 3.12 | conda | Audio enhancement (resemble-enhance) |
| `whisper` | 3.12 | conda | Audio transcription (mlx-whisper) |
| `heartmula` | 3.10 | conda | HeartMuLa music + lyrics transcription |
| `diarize` | 3.10 | conda | Speaker diarization (pyannote.audio) |
| `separate` | 3.10 | conda | Audio source separation (demucs) |
| `ai-tts` | 3.11 | conda | Qwen3-TTS neural speech (mlx-audio) |
| `lang-detect` | 3.11 | conda | Language detection (langdetect) |
| `ace_worker` | 3.11+ | uv | ACE-Step 1.5 music generation |

</details>

<details>
<summary><strong>Offline Safety (Wheels Cache)</strong></summary>

Each worker has a local `wheels/` directory with cached `.whl` files and `requirements.lock`. Install scripts check:
1. `wheels/` present → Offline install from local wheels
2. `requirements.lock` → Online install with pinned versions
3. Neither → Fallback to PyPI, then generate lockfile

</details>

<details>
<summary><strong>progress.py — Worker subprocess streaming</strong></summary>

Internal library for real-time streaming of worker output. All workers run through `run_worker()` which streams stderr, parses progress events (tqdm, ffmpeg, counters), and collects stdout for JSON results.

</details>

<details>
<summary><strong>Environment Variables</strong></summary>

| Variable | Default | Purpose |
|----------|---------|---------|
| `RVC_API_URL` | `http://127.0.0.1:5100` | RVC worker API endpoint |
| `CONDA_BIN` | `/opt/miniconda3/bin/conda` | Path to conda binary |
| `UV_BIN` | `~/.local/bin/uv` or `uv` in PATH | Path to uv binary |
| `HF_TOKEN` | — | HuggingFace token (diarize, model downloads) |
| `HF_HOME` | `~/.cache/huggingface` | HuggingFace cache directory |

</details>

<details>
<summary><strong>Project Structure</strong></summary>

```
tts-mist/
├── generate.py             # Unified CLI entry point
├── progress.py             # Worker subprocess streaming library
├── setup.sh                # Master installer (all envs + models)
├── backup-models.sh        # Model checkpoint backup
├── requirements.txt        # Main env dependencies (pinned)
├── rvc_worker/             # RVC voice conversion worker
├── enhance_worker/         # resemble-enhance worker
├── music_worker/           # HeartMuLa worker
├── ace_worker/             # ACE-Step 1.5 worker (uv project)
├── whisper_worker/         # mlx-whisper worker
├── diarize_worker/         # pyannote diarization worker
├── separate_worker/        # demucs separation worker
├── tts_worker/             # Qwen3-TTS worker (mlx-audio)
├── langdetect_worker/      # Language detection worker
├── rvc_models/             # Installed RVC voice models
├── models/                 # All model checkpoints (gitignored)
├── tests/                  # Test scripts
├── demos/                  # Demo output files
└── prompt-guides/          # ACE-Step + HeartMuLa prompt guides
```

</details>
