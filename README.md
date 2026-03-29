# generate — Unified Media Generation Toolkit

CLI for voice synthesis, voice conversion, audio enhancement, AI music generation, sound effects, lyrics transcription, speaker diarization, and audio source separation. macOS Apple Silicon only.

**Entry point:** `generate.py`

## Quick Setup

```bash
# 1. Install all environments
bash setup.sh

# 2. Start RVC worker (needed for voice conversion)
python generate.py server start

# 3. Install a voice model
python generate.py models rvc search "neutral male"
python generate.py models rvc install User/ModelName

# 4. Ready!
python generate.py voice rvc --model my-voice input.wav
```

### Requirements

- macOS (Apple Silicon)
- Miniconda (`brew install --cask miniconda`)
- uv (`brew install uv`) — for ACE-Step music generation
- ffmpeg (`brew install ffmpeg`)
- git-lfs (`brew install git-lfs`)
- Xcode CLI tools with accepted license (`sudo xcodebuild -license accept`)
- ~305 GB disk for model checkpoints

---

## ABI Overview

```
generate.py <medium> <engine> [--model <variant>] [input] [options]
```

| Medium | Engine | Purpose |
|--------|--------|---------|
| `voice` | `ai-tts` | Neural TTS via Qwen3-TTS (text → audio) |
| `voice` | `say` | macOS native TTS (text → audio) |
| `voice` | `rvc` | RVC voice conversion (audio → audio) |
| `voice` | `clone-tts` | Zero-shot voice cloning via Qwen3-TTS Base (reference + text → cloned voice) |
| `audio` | `enhance` | Denoise + super-resolution |
| `audio` | `demucs` | Source separation (stems) |
| `audio` | `ace-step` | AI music generation |
| `audio` | `heartmula` | AI music generation (alt engine) |
| `audio` | `sfx` | Sound effects generation (EzAudio) |
| `audio` | `voice-removal` | Remove vocals (demucs → remix) |
| `audio` | `ltx2.3` | Audio generation via LTX-2.3 video engine (virtual) |
| `audio` | `diarize` | Speaker diarization |
| `text` | `whisper` | Audio transcription |
| `text` | `heartmula-transcribe` | Lyrics extraction |
| `text` | `ollama` | LLM inference via Ollama |
| `output` | `audio-concatenate` | Concatenate audio files (with per-clip trim, fades, volume, crossfade) |
| `output` | `audio-mucs` | Mix/overlay audio files in parallel (with per-clip trim, fades, volume, pan) |
| `server` | — | RVC worker management |
| `models` | `rvc`, `ollama`, `huggingface` | Model management (per engine) |
| `ps` | — | Active models across all engines |
| `image` | `flux.2` | Image generation & editing (FLUX.2 Klein, PyTorch MPS) |
| `image` | `sd1.5` | Image generation (Stable Diffusion 1.5, CivitAI models + LoRAs) |
| `image` | `openpose` | Pose estimation via DWPose (body, hands, face) |
| `image` | `depth` | Depth estimation via Depth Anything V2 (zero-shot) |
| `image` | `lineart` | Line art extraction via TEED/AnyLine |
| `image` | `normalmap` | Normal map estimation via Marigold-Normals |
| `image` | `sketch` | Sketch/edge extraction via HED |
| `image` | `upscale` | Image upscaling via Real-ESRGAN |
| `image` | `segment` | Background removal / object segmentation |
| `video` | `ltx2.3` | Video generation (LTX-2.3, 22B, T2V/I2V/A2V/Extend/Retake/Clone, PyTorch MPS) |

Future mediums (stubs): `translation`, `comparison`

**Planned:** `video-vision` — video input for Ollama vision models (blocked by upstream Ollama bug)

---

## Global Options

### `--screen-log-format json`

Switches all status/progress output from human-readable TUI to machine-readable JSON events. Works with every command.

```bash
python generate.py voice rvc --model my-voice input.wav --screen-log-format json
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

### Voice — AI TTS (`voice ai-tts`)

Neural text-to-speech via Qwen3-TTS (mlx-audio). Runs locally on Apple Silicon.

```bash
# Basic
python generate.py voice ai-tts --text "Hello world" -o demos/

# With preset voice
python generate.py voice ai-tts -v Serena --text "Hello" -o demos/

# With style instructions (refine mode)
python generate.py voice ai-tts -v Aiden -t "dramatic, slow" --text "Silence." -o demos/

# Smaller model (faster, less quality)
python generate.py voice ai-tts --tts-model small --text "Quick test" -o demos/

# Auto-detect language
python generate.py voice ai-tts --text "Der Fuchs springt über den Bach" -o demos/

# Explicit language
python generate.py voice ai-tts -v Dylan --language de --text "Hallo Welt" -o demos/

# Text from file
python generate.py voice ai-tts --text-file story.txt -o demos/

# Dialog (multiple voices)
python generate.py voice ai-tts --text "[Aiden] Hi! [Serena] Hello there!" -o demos/

# Dialog with per-segment style instructions
python generate.py voice ai-tts --text "[Aiden: excited, fast] Hi! [Serena: calm, slow] Hello there." -o demos/

# Reproduce from prompt sidecar file
python generate.py voice ai-tts --prompt-file demos/speech.txt -o demos/

# Reproduce with different voice (CLI flags override sidecar values)
python generate.py voice ai-tts --prompt-file demos/speech.txt -v Dylan -o demos/

# List available voices
python generate.py voice ai-tts --list-voices
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
- `--prompt-file`, `-pf` — Load text + params from prompt sidecar (.txt)
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

### Voice — macOS TTS (`voice say`)

Generate speech from text using macOS `say` command. Optionally pipe through RVC for voice conversion.

```bash
# System default voice
python generate.py voice say --text "Hallo Welt" -o demos/

# Specific macOS voice
python generate.py voice say -v Anna --text "Hallo Welt" -o demos/

# With speaking rate
python generate.py voice say -v Samantha --rate 180 --text "Hello world" -o demos/

# Text from file
python generate.py voice say --text-file story.txt -o demos/

# Say + RVC voice conversion pipeline
python generate.py voice say --model my-voice --text "Hallo Welt" -o demos/
python generate.py voice say -v Anna --model my-voice --text "Hallo" -o demos/
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

### Voice — RVC Conversion (`voice rvc`)

Convert audio to a different voice using RVC with automatic pitch detection. Requires running server (`generate.py server start`).

```bash
# Single file (auto-pitch from model config)
python generate.py voice rvc --model my-voice input.wav

# Batch conversion
python generate.py voice rvc --model my-voice *.wav -o ./output/

# Manual pitch shift (semitones, disables auto-pitch)
python generate.py voice rvc --model my-voice input.wav --pitch 12

# Set target pitch directly (Hz)
python generate.py voice rvc --model my-voice input.wav --target-hz 280

# Choose pitch detection algorithm
python generate.py voice rvc --model my-voice input.wav --decoder crepe
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
2. Reads model's `target_f0` from `worker/rvc/models/<model>/revoicer.json`
3. Computes pitch shift: `semitones = 12 * log2(target_f0 / input_f0)`
4. Applies the shift via RVC's `f0up_key` parameter

If the model has no `target_f0`, run `models rvc set-pitch` first.

**Implicit behaviors:**
- Non-WAV files are auto-converted to WAV (44.1 kHz) via ffmpeg
- Output format is always WAV
- All output paths are printed as JSON array to stdout

</details>

---

### Voice Cloning (`voice clone-tts`)

Zero-shot voice cloning via Qwen3-TTS Base. Provide a 3–10 second reference audio sample and text to synthesize in that voice. Falls back to `voice/default-reference.m4a` if no `--reference` given.

```bash
# Basic voice cloning (uses default reference)
python generate.py voice clone-tts --text "Hello world" -o output.wav

# With explicit reference audio
python generate.py voice clone-tts --reference ref.wav --text "Hello world" -o output.wav

# German with explicit reference text (auto-transcribed if omitted)
python generate.py voice clone-tts --reference ref.wav --language de --text "Guten Tag" -o output.wav
```

<details>
<summary>Options</summary>

- `--reference` — Reference audio (3–10s of the voice to clone; falls back to `voice/default-reference.m4a`)
- `--text` / `-l` — Text to synthesize in the cloned voice
- `--ref-text` — Text spoken in the reference audio (auto-transcribed via Whisper if omitted)
- `--language` — Language code: de, en, fr, ja, zh, it, es, pt, hi, ko, ru (autodetect if omitted)
- `-o, --output` — Output WAV path or directory

**Reference audio tips:**
- 3–10 seconds, clean speech, minimal background noise
- Auto-trimmed at sentence boundaries if too long (via Whisper timestamps)

</details>

---

### Audio Enhancement (`audio enhance`)

Denoise and super-resolve audio using resemble-enhance.

```bash
python generate.py audio enhance input.wav -o ./enhanced/
python generate.py audio enhance input.wav --denoise-only
python generate.py audio enhance input.wav --enhance-only
python generate.py audio enhance *.wav -o ./enhanced/
```

<details>
<summary>Options</summary>

- `-o, --output` — Output directory
- `--denoise-only` — Skip super-resolution
- `--enhance-only` — Only super-resolution, skip denoising

</details>

---

### Audio Separation (`audio demucs`)

Split audio into stems (vocals, drums, bass, other) using demucs.

```bash
python generate.py audio demucs song.mp3 -o ./stems/
python generate.py audio demucs song.mp3 -o ./stems/ --model htdemucs_ft
```

Output: `song_vocals.wav`, `song_drums.wav`, `song_bass.wav`, `song_other.wav`

<details>
<summary>Options</summary>

- `-o, --output` — Output directory
- `--model` — Demucs model (default: `htdemucs`, alt: `htdemucs_ft`)

</details>

---

### Voice Removal (`audio voice-removal`)

Remove vocals from audio — keeps drums, bass, and other instruments. Uses AI-based stem separation (demucs) instead of old-school frequency filtering, producing clean karaoke tracks with minimal artifacts.

```bash
python generate.py audio voice-removal song.mp3 -o ./karaoke/
python generate.py audio voice-removal song.mp3 -o ./karaoke/ --model htdemucs_ft
```

Output: `song_no_vocals.wav`

<details>
<summary>Options</summary>

- `-o, --output` — Output directory
- `--model` — Demucs model (default: `htdemucs`, alt: `htdemucs_ft`)
- `--tmp-dir` — Directory for temporary stems (default: `/tmp`)

</details>

---

### Music Generation (`audio ace-step` / `audio heartmula`)

Generate music from lyrics and style tags.

```bash
# ACE-Step (default turbo model)
python generate.py audio ace-step \
  -l "[Verse] La la la" -t "upbeat disco" -s 30 -o song.mp3

# ACE-Step SFT (50 steps)
python generate.py audio ace-step --model sft \
  -f lyrics.txt -t "cinematic orchestral" -s 60

# ACE-Step with language hint (improves vocal pronunciation)
python generate.py audio ace-step \
  -l "[Verse] Die Sonne geht auf" -t "german pop, female vocal" \
  --language de -s 60 -o deutsch.mp3

# HeartMuLa
python generate.py audio heartmula \
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
- `--top-k` — Top-k sampling
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
- `--language` — Vocal language code (e.g. `en`, `de`, `zh`, `ja`, `ko`)
- `--bpm` — Beats per minute
- `--keyscale` — Musical key (e.g. `"C Major"`, `"Am"`)
- `--timesignature` — Time signature: `2`=2/4, `3`=3/4, `4`=4/4, `6`=6/8

</details>

**Lyrics format:** Use section tags like `[Verse]`, `[Chorus]`, `[Bridge]`, `[Intro]`, `[Outro]`.

**Prompt guides:** See [prompt-guides/ACE-Step.md](prompt-guides/ACE-Step.md) and [prompt-guides/HeartMuLa.md](prompt-guides/HeartMuLa.md).

---

### Sound Effects (`audio sfx`)

Generate sound effects from text descriptions using EzAudio (diffusion-based text-to-audio).

```bash
# Basic sound effect
python generate.py audio sfx --text "a dog barking in the distance" -o sfx.wav

# Layered scene, 8 seconds
python generate.py audio sfx --text "rain falling on leaves as thunder rumbles" -s 8 -o rain.wav

# With custom steps and guidance
python generate.py audio sfx --text "a car horn honking" --steps 50 --cfg-scale 3.0 -o horn.wav

# Reproducible output
python generate.py audio sfx --text "waves crashing on a rocky shore" --seed 42 -o waves.wav
```

<details>
<summary>Options</summary>

- `--text`, `-l` — Text prompt describing the sound
- `--text-file`, `-f` — Read prompt from file
- `-o, --output` — Output WAV path
- `-s, --seconds` — Duration in seconds (1–10, default: 10)
- `--steps` — DDIM inference steps (default: 100, lower = faster)
- `--cfg-scale` — Guidance scale (default: 5.0)
- `--seed` — Random seed for reproducibility
- `--model` — Model variant: `s3_xl` (default), `s3_l`

**Output:** 24 kHz mono WAV.

</details>

**Prompt guide:** See [prompt-guides/sfx.md](prompt-guides/sfx.md).

---

### Video Generation (`video ltx2.3`)

Generate video using LTX-2.3 (Lightricks, 22B). Runs locally on Apple Silicon via PyTorch MPS.

```bash
# Text-to-video (distilled model, default, 8 steps, fast)
python generate.py video ltx2.3 -p "a cat running through a meadow" --ratio 16:9 --quality 480p -o cat.mp4

# Dev model (30+3 steps, two-stage, higher quality)
python generate.py video ltx2.3 --model dev -p "a cat running through a meadow" --ratio 16:9 --quality 480p -o cat.mp4

# Image-to-video (first frame conditioning)
python generate.py video ltx2.3 -p "animate this scene" --image-first start.jpg -o anim.mp4

# First + last frame (interpolation)
python generate.py video ltx2.3 -p "transition between scenes" --image-first a.jpg --image-last b.jpg -o interp.mp4

# Flexible keyframe conditioning (any frame, any strength)
python generate.py video ltx2.3 -p "..." --image photo.jpg 30 0.8 -o keyframe.mp4

# Custom dimensions and duration
python generate.py video ltx2.3 -p "a sunset" -W 768 -H 512 --num-frames 121 --frame-rate 24 -o sunset.mp4

# Audio-to-video
python generate.py video ltx2.3 -p "..." --audio music.wav -o music_video.mp4

# Video-only output (no audio)
python generate.py video ltx2.3 -p "a sunset" -o sunset.mp4v

# Audio-only output
python generate.py video ltx2.3 -p "ocean waves" -o waves.mp4a

# Split output (video + audio as separate files)
python generate.py video ltx2.3 -p "a sunset" -o sunset
# → sunset.mp4v + sunset.mp4a
```

| Parameter | Short | Default | Description |
|-----------|-------|---------|-------------|
| `--model` | `-m` | `distilled` | Model: `distilled` (8 steps), `dev` (40 steps) |
| `--prompt` | `-p` | required | Text prompt |
| `--output` | `-o` | `video.mp4` | Output path (.mp4, .mp4v, .mp4a, or no ext for split) |
| `--seed` | | random | Random seed |
| `--steps` | | model default | Inference steps |
| `--cfg-scale` | | model default | CFG guidance scale |
| `--ratio` | | | Aspect ratio (see resolution table below) |
| `--quality` | | | Quality tier (see resolution table below) |
| `--width` | `-W` | 768 | Video width (override, must be multiple of 64) |
| `--height` | `-H` | 512 | Video height (override, must be multiple of 64) |
| `--num-frames` | | 121 | Frame count (must be 8k+1, auto from audio) |
| `--frame-rate` | | 24 | FPS |
| `--image` | | | `PATH FRAME_IDX STRENGTH` (repeatable) |
| `--image-first` | | | First frame image (strength 1.0) |
| `--image-mid` | | | Middle frame image (strength 1.0) |
| `--image-last` | | | Last frame image (strength 1.0) |
| `--audio` | | | Audio file for A2V pipeline |
| `--lora` | | | LoRA: `PATH [STRENGTH]` (repeatable) |
| `--negative-prompt` | | | Negative prompt |
| `--enhance-prompt` | | | Auto-enhance prompt via Gemma |
| `--extend` | | | `VIDEO SECONDS` — extend video by N seconds |
| `--clone` | | | `VIDEO` — clone: generate new video from reference |
| `--seconds` | `-s` | 5 | [clone] Output duration in seconds |
| `--retake` | | | `VIDEO START END` — retake a time region (seconds) |
| `--ref-seconds` | | 2/5 | Context seconds from source (default: 2 for extend, 5 for clone) |
| `--no-rescale` | | | Keep reference images in original resolution |

#### Resolution Table (`--ratio` + `--quality`)

Dimensions are approximate — all values are aligned to /128 boundaries (VAE requirement: width//2 and height//2 must each be divisible by 32 and even). Higher tiers require more VRAM and processing time.

| Ratio | 240p | 360p | 480p | 720p | 1080p |
|-------|------|------|------|------|-------|
| 16:9 | 256x128 | 512x256 | 640x384 | 1152x640 | 1152x640 |
| 9:16 | 128x256 | 256x512 | 384x640 | 640x1152 | 640x1152 |
| 21:9 | 256x128 | 512x256 | 640x256 | 896x384 | 1792x768 |
| 9:21 | 128x256 | 256x512 | 256x640 | 384x896 | 768x1792 |
| 3:2 | 384x256 | 384x256 | 384x256 | 1152x768 | 1920x1280 |
| 2:3 | 256x384 | 256x384 | 256x384 | 768x1152 | 1280x1920 |
| 4:3 | 384x256 | 512x384 | 512x384 | 1024x768 | 1536x1152 |
| 3:4 | 256x384 | 384x512 | 384x512 | 768x1024 | 1152x1536 |
| 4:5 | 256x384 | 384x512 | 512x640 | 1024x1280 | 1536x1920 |
| 5:4 | 384x384 | 512x384 | 640x512 | 1280x1024 | 1920x1536 |
| 1:1 | 384x384 | 512x512 | 640x640 | 1280x1280 | 1920x1920 |
| 1:2 | 128x256 | 256x512 | 256x512 | 640x1280 | 896x1792 |
| 2:1 | 256x128 | 512x256 | 512x256 | 1280x640 | 1792x896 |

Additional tiers available: 1440p, 2160p (4K). Use `--width`/`--height` for custom dimensions (must be multiples of 64).

#### Extend, Clone & Retake

Extend appends new content, clone generates new content from a reference (keeping only the new part), retake replaces a time region.

```bash
# Extend a video by 5 seconds
python generate.py video ltx2.3 --model dev -p "The man says: hello" --extend video.mp4 5 --ratio 16:9 --quality 240p -o extended.mp4

# Clone: generate 5s new video using reference as visual context (default 5s)
python generate.py video ltx2.3 --model dev -p "The person says: hello" --clone video.mp4 --ratio 16:9 --quality 720p -o clone.mp4

# Clone with custom duration (8s)
python generate.py video ltx2.3 --model dev -p "She laughs" --clone video.mp4 --seconds 8 --ratio 16:9 --quality 720p -o clone.mp4

# Retake a passage (replace 3.5s–5.0s with new content)
python generate.py video ltx2.3 --model dev -p "Full scene description with changed dialog" --retake video.mp4 3.5 5.0 --ratio 16:9 --quality 240p -o retake.mp4

# Extend with more context (default: 2s)
python generate.py video ltx2.3 --model dev -p "..." --extend video.mp4 5 --ref-seconds 4 -o extended.mp4
```

**Extend** trims the source to the last `--ref-seconds` as context, generates context + new frames, then concatenates the original (minus context) with the full pipeline output. This keeps VRAM usage constant regardless of source video length.

**Clone** generates a new video using the reference as visual initialization. Uses RetakePipeline over the full output duration (start=0), so audio and video are fully regenerated from the prompt while the source latents provide appearance and motion context. Source videos longer than 5s are trimmed to the last 5s.

**Retake** loads the entire source video, applies a temporal mask to the specified time region, and regenerates only that region. The prompt should describe the **entire scene** with the change embedded — not just the changed part.

### Image Generation & Editing (`image flux.2`)

Generate and edit images using FLUX.2 Klein (Black Forest Labs). Runs locally on Apple Silicon via PyTorch MPS.

```bash
# Text-to-image (4B distilled, fast)
python generate.py image flux.2 --model 4b-distilled -p "a cat on a cliff overlooking the ocean" -o cat.png

# Higher quality (4B base, more steps)
python generate.py image flux.2 --model 4b -p "a cat on a cliff" --steps 20 -o cat.png

# 9B model (best quality)
python generate.py image flux.2 --model 9b-distilled -p "a portrait of a woman" -o portrait.png

# Image editing with reference image
python generate.py image flux.2 --model 4b-distilled -p "a man standing in front of a mountain lake" --images ref.png -o edited.png

# Multi-reference editing (combine elements from multiple images)
python generate.py image flux.2 --model 4b-distilled -p "the person from image 1 petting the cat from image 2" --images person.png cat.png -o combined.png

# Custom dimensions (16:9)
python generate.py image flux.2 -p "a wide cinematic landscape" -W 1360 -H 768 -o landscape.png

# Reproducible output
python generate.py image flux.2 -p "a sunset" --seed 42 -o sunset.png

# Image editing (no mask needed — FLUX.2 Klein is a native edit model)
python generate.py image flux.2 --images photo.png -p "remove the glasses" -o no_glasses.png
python generate.py image flux.2 --images photo.png -p "add a painting on the wall" -o with_painting.png

# Iterative editing (output of step 1 → input of step 2)
python generate.py image flux.2 --images scene.png -p "remove the lamp" -o step1.png
python generate.py image flux.2 --images step1.png -p "add a bookshelf where the lamp was" -o step2.png
```

**Image editing:** FLUX.2 Klein is a native edit model — no inpainting masks needed. Just describe what to change. For complex edits, use iterative steps: remove first, then add.

<details>
<summary>Models</summary>

| `--model` | Parameters | Steps | License | Best for |
|-----------|------------|-------|---------|----------|
| `4b-distilled` | 4B | 4 (fixed) | Apache 2.0 | Fast generation, real-time |
| `4b` (default) | 4B | Configurable | Apache 2.0 | Fine-tuning, quality control |
| `9b-distilled` | 9B | 4 (fixed) | Non-Commercial | Best quality, fast |
| `9b` | 9B | Configurable | Non-Commercial | Research, maximum flexibility |

</details>

<details>
<summary>Options</summary>

- `flux.2` — Engine (positional, required)
- `--model`, `-m` — Model variant (default: `4b`)
- `--prompt`, `-p` — Text prompt (required)
- `-o, --output` — Output file path (default: `image.png`)
- `--seed` — Random seed for reproducibility
- `--steps` — Inference steps (only for base models; distilled = fixed 4 steps)
- `--cfg-scale` — Guidance scale
- `--ratio` — Aspect ratio (e.g. `16:9`, `1:1`). Requires `--quality`
- `--quality` — Quality tier (e.g. `480p`, `720p`). Requires `--ratio`
- `-W, --width` — Image width (default: 1360). Overridden by `--ratio`/`--quality`
- `-H, --height` — Image height (default: 768). Overridden by `--ratio`/`--quality`
- `--images` — Reference image path(s) for editing (up to 10). Auto-rescaled to target dimensions via Pan & Scan.
- `--no-rescale` — Pass reference images in original resolution (skip Pan & Scan)
- `--controlnet` — Conditioning: `mode:filepath` (e.g. `depth:depth.png`, `pose:pose.png`, `lineart:lines.png`, `normalmap:normals.png`, `sketch:sketch.png`). Always rescaled.

</details>

<details>
<summary>Setup</summary>

```bash
bash worker/image/install.sh
```

Creates `flux2` conda env with PyTorch 2.8 + FLUX.2 dependencies. Models are downloaded from HuggingFace on first use.

</details>

**Prompt guide:** See [prompt-guides/FLUX2.md](prompt-guides/FLUX2.md).

---

### Pose Estimation (`image openpose`)

Extract body, hand, and face poses from images using DWPose. Outputs skeleton visualization on black background.

```bash
# Wholebody (body + hands + face) — default
python generate.py image openpose --images person.png -o pose.png

# Body only
python generate.py image openpose --images person.png --pose-mode body -o pose_body.png

# Body + hands
python generate.py image openpose --images person.png --pose-mode bodyhand -o pose_hands.png

# Body + face
python generate.py image openpose --images person.png --pose-mode bodyface -o pose_face.png

# Multiple images
python generate.py image openpose --images img1.png img2.png -o poses/
```

<details>
<summary>Options</summary>

- `--images` — Input image(s) (required)
- `-o, --output` — Output file path (default: `pose.png`)
- `--pose-mode` — Detection mode: `wholebody` (default), `body`, `bodyhand`, `bodyface`

</details>

---

### Stable Diffusion 1.5 (`image sd1.5`)

Generate images using Stable Diffusion 1.5 checkpoints from CivitAI. Supports multiple LoRAs with per-LoRA intensity.

```bash
# MatureMaleMix (default model, includes add_detail LoRA at 1.2)
python generate.py image sd1.5 -p "a muscular man with a beard, studio lighting" -o man.png

# Without LoRA
python generate.py image sd1.5 --no-lora -p "a muscular man" -o man.png

# DreamShaper
python generate.py image sd1.5 --model dreamshaper -p "fantasy landscape" -o landscape.png

# Custom LoRA with intensity
python generate.py image sd1.5 --lora add_detail:1.5 -p "a warrior" -o warrior.png

# Multiple LoRAs
python generate.py image sd1.5 --lora add_detail:1.2 --lora style:0.8 -p "..." -o out.png

# Custom negative prompt
python generate.py image sd1.5 -p "a man" --negative-prompt "blurry, deformed" -o out.png

# ControlNet conditioning (depth, lineart, sketch, normalmap, pose)
python generate.py image sd1.5 --controlnet depth:depth.png -p "a man standing in a room" -o out.png
python generate.py image sd1.5 --controlnet lineart:lines.png -p "detailed illustration" -o out.png
python generate.py image sd1.5 --controlnet sketch:sketch.png -p "anime character" -o out.png
python generate.py image sd1.5 --controlnet normalmap:normals.png -p "relit scene" -o out.png
python generate.py image sd1.5 --controlnet pose:pose.png -p "woman in red dress" -o out.png
```

<details>
<summary>Options</summary>

- `sd1.5` — Engine (positional, required)
- `--model` — Checkpoint: `mm` (default), `dreamshaper`
- `-p, --prompt` — Text prompt (required)
- `-o, --output` — Output file path (default: `image.png`)
- `--negative-prompt` — Negative prompt (default: model-specific)
- `--ratio` — Aspect ratio (e.g. `16:9`, `1:1`). Requires `--quality`
- `--quality` — Quality tier (e.g. `480p`, `720p`). Requires `--ratio`
- `-W, --width` — Image width (default: 1280, must be multiple of 64). Overridden by `--ratio`/`--quality`
- `-H, --height` — Image height (default: 768, must be multiple of 64). Overridden by `--ratio`/`--quality`
- `--steps` — Inference steps (default: 20)
- `--cfg-scale` — Guidance scale (default: 3.5)
- `--seed` — Random seed (default: random)
- `--lora` — LoRA: `name:intensity` (repeatable, e.g. `add_detail:1.2`)
- `--no-lora` — Disable default LoRA(s)
- `--controlnet` — Conditioning: `mode:filepath` (modes: `depth`, `lineart`, `sketch`, `normalmap`, `pose`)

</details>

**Available models:**

| Model | `--model` | Description |
|---|---|---|
| MatureMaleMix v1.4 | `mm` | Realistic/2.5D mature male characters (default LoRA: add_detail at 1.2) |
| DreamShaper 8 | `dreamshaper` | Versatile artistic/photorealistic generation |

---

### Depth Estimation (`image depth`)

Estimate depth from any image using Depth Anything V2. Outputs a grayscale depth map (white = near, black = far).

```bash
# Single image
python generate.py image depth --images photo.png -o depth.png

# Multiple images
python generate.py image depth --images img1.png img2.png -o depths/
```

<details>
<summary>Options</summary>

- `--images` — Input image(s) (required)
- `-o, --output` — Output file path (default: `depth.png`)
- `--model` — Model size: `small` (default, fast), `large` (detailed)

</details>

**Depth-conditioned generation with FLUX.2:**

Use a depth map as structural reference for FLUX.2 image generation:

```bash
# Extract depth, then generate with same structure
python generate.py image depth --images photo.png -o depth.png
python generate.py image flux.2 --model 4b-distilled --controlnet depth:depth.png -p "turn image 1 into a cartoon character" -o cartoon.png
```

---

### Line Art Extraction (`image lineart`)

Extract line art from images using TEED (learned) or Canny (classical).

```bash
# Canny (default, classical edge detection)
python generate.py image lineart --images photo.png -o lines.png

# TEED (learned, thicker lines)
python generate.py image lineart --model teed --images photo.png -o lines_teed.png

# Use as ControlNet conditioning
python generate.py image flux.2 --controlnet lineart:lines.png -p "colorful illustration" -o out.png
```

<details>
<summary>Options</summary>

- `--images` — Input image(s) (required)
- `-o, --output` — Output file path (default: `lineart.png`)
- `--model` — Extraction method: `canny` (default, classical), `teed` (learned, thicker lines)

</details>

---

### Normal Map Estimation (`image normalmap`)

Estimate surface normals from images using Marigold-Normals v1.1. Outputs RGB normal map (standard color encoding).

```bash
python generate.py image normalmap --images photo.png -o normals.png

# Use as ControlNet conditioning
python generate.py image flux.2 --controlnet normalmap:normals.png -p "dramatic lighting" -o out.png
```

<details>
<summary>Options</summary>

- `--images` — Input image(s) (required)
- `-o, --output` — Output file path (default: `normalmap.png`)
- `--steps` — Denoising steps (default: 4)

</details>

---

### Sketch/Edge Extraction (`image sketch`)

Extract edges from images using HED (Holistically-Nested Edge Detection). Runs on CPU via OpenCV DNN — no PyTorch required.

```bash
python generate.py image sketch --images photo.png -o sketch.png

# Use as ControlNet conditioning
python generate.py image flux.2 --controlnet sketch:sketch.png -p "fantasy scene" -o out.png
```

<details>
<summary>Options</summary>

- `--images` — Input image(s) (required)
- `-o, --output` — Output file path (default: `sketch.png`)

</details>

---

### Image Upscaling (`image upscale`)

Upscale images using Real-ESRGAN. Standalone RRDBNet architecture with MPS acceleration.

```bash
# 4x upscale (default)
python generate.py image upscale --images photo.png -o upscaled.png

# 2x upscale
python generate.py image upscale --images photo.png --model 2x -o upscaled.png

# Anime-optimized 4x
python generate.py image upscale --images photo.png --model anime -o upscaled.png

# Batch upscale
python generate.py image upscale --images img1.png img2.png img3.png -o upscaled/
```

<details>
<summary>Options</summary>

- `--images` — Input image(s) (required)
- `-o, --output` — Output file path or directory
- `--model` — Model: `4x` (default), `2x`, `anime`

</details>

---

### Image Segmentation (`image segment`)

Separate foreground from background using BiRefNet (MIT license). Outputs transparent RGBA PNG.

```bash
# Foreground only (default)
python generate.py image segment --images photo.png -o foreground.png

# Background only
python generate.py image segment --output-layer background --images photo.png -o background.png

# Both (foreground + background in a directory)
python generate.py image segment --output-layer both --images photo.png -o output/
```

<details>
<summary>Options</summary>

- `--images` — Input image(s) (required)
- `-o, --output` — Output file path or directory
- `--output-layer` — What to output: `foreground` (default), `background`, `both`

</details>

---

### Speaker Diarization (`audio diarize`)

Split dialogue audio into separate tracks per speaker using pyannote.audio (MPS).

```bash
python generate.py audio diarize interview.wav -o ./speakers/
python generate.py audio diarize podcast.wav -o ./speakers/ --speakers 3
python generate.py audio diarize dialog.wav -o ./speakers/ --verify
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

### Transcription (`text whisper`)

Transcribe audio to text using mlx-whisper (Apple Silicon optimized).

```bash
python generate.py text whisper audio.wav
python generate.py text whisper audio.wav --format srt -o ./transcripts/
python generate.py text whisper audio.wav --language de --word-timestamps
python generate.py text whisper audio.wav --model large-v3
```

<details>
<summary>Options</summary>

- `--model` — Whisper model: `tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo` (default)
- `--language` — Language hint (e.g. `en`, `de`, `ja`)
- `--word-timestamps` — Word-level timing
- `--format` — Output: `json` (default), `txt`, `srt`, `vtt`, `tsv`, `all`
- `-o, --output` — Output directory

</details>

---

### Lyrics Extraction (`text heartmula-transcribe`)

Extract lyrics from audio using HeartTranscriptor.

```bash
python generate.py text heartmula-transcribe song.mp3
python generate.py text heartmula-transcribe song.mp3 -o lyrics.txt
```

---

### LLM Inference (`text ollama`)

Unified interface for local LLM inference via Ollama.

```bash
# Chat (messages as JSON string or file)
python generate.py text ollama --model qwen3.5:latest --endpoint chat \
  --messages '[{"role":"user","content":"Hello!"}]'
python generate.py text ollama --model qwen3.5:latest --endpoint chat \
  --messages chat.json --stream

# Generate (prompt + optional system)
python generate.py text ollama --model qwen3.5:latest --endpoint generate \
  --prompt "Explain quantum computing" --system "You are a physicist"

# Thinking (enable reasoning for supported models)
python generate.py text ollama --model qwen3.5:latest --endpoint chat \
  --messages '[{"role":"user","content":"What is 15*17?"}]' --thinking True --stream

# Vision (attach local images)
python generate.py text ollama --model qwen3.5:latest --endpoint chat \
  --messages '[{"role":"user","content":"What do you see?"}]' --images photo.jpg

# Vision (image URLs in prompt — auto-downloaded)
python generate.py text ollama --model qwen3.5:latest --endpoint chat \
  --messages '[{"role":"user","content":"Describe this. https://example.com/photo.jpg"}]'

# Config management (persistent default overrides)
python generate.py text ollama --model qwen3.5:latest --endpoint set \
  --context-length 256000 --temperature 0.7
python generate.py text ollama --model qwen3.5:latest --endpoint show
python generate.py text ollama --model qwen3.5:latest --endpoint reset

# Load/unload models
python generate.py text ollama --model qwen3.5:latest --endpoint load
python generate.py text ollama --model qwen3.5:latest --endpoint unload
```

Supported params: `--context-length`, `--max-tokens`, `--temperature`, `--top-p`, `--top-k`, `--repeat-penalty`, `--seed`, `--stop`, `--stream`, `--thinking`, `--images`, `--base-url`, `--api-key`.

---

### Model Management (`models`)

Per-engine model management.

```bash
# List all models across all engines
python generate.py models list

# List by medium (voice, audio, text, image)
python generate.py voice models list
python generate.py audio models list
python generate.py image models list

# List by specific engine
python generate.py voice rvc models list
python generate.py image flux.2 models list
python generate.py text whisper models list

# Ollama
python generate.py models ollama list
python generate.py models ollama pull qwen3.5:latest
python generate.py models ollama show qwen3.5:latest
python generate.py models ollama remove qwen3.5:latest
python generate.py models ollama unload qwen3.5:latest

# HuggingFace
python generate.py models huggingface list
python generate.py models huggingface search "qwen vision"
python generate.py models huggingface pull Qwen/Qwen2.5-VL-7B
```

---

### Process Status (`ps`)

Shows active/loaded models across all engines.

```bash
python generate.py ps
python generate.py ps --screen-log-format json
```

```
MODEL                          ENGINE        STATUS     VRAM        CTX        EXTRA
gerhard-v2                     rvc           loaded     -           -          target: 120 Hz
qwen3.5:latest                 ollama        running    13.2 GB     131072     qwen35 Q4_K_M
```

---

### Audio Concatenation (`output audio-concatenate`)

Concatenate multiple audio files into one. Supports per-clip trim, volume, fades, and crossfade via `--clip`.

```bash
# Simple concatenation
python generate.py output audio-concatenate a.wav b.wav c.mp3 -o out.wav

# With bitrate (for compressed formats)
python generate.py output audio-concatenate a.wav b.wav -o out.mp3 --output-bitrate 128k

# Per-clip options via --clip INDEX:key=val,key=val
python generate.py output audio-concatenate \
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

### Audio Mixing (`output audio-mucs`)

Mix multiple audio files in parallel (overlay). Supports per-clip trim, volume, pan, and fades via `--clip`.

```bash
# Simple mix (all tracks overlaid)
python generate.py output audio-mucs track1.wav track2.wav -o mix.wav

# Stereo remix with per-clip pan and volume
python generate.py output audio-mucs \
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
python generate.py models list                                          # all engines
python generate.py voice models list                                    # all voice engines
python generate.py voice rvc models list                                # RVC only
python generate.py models rvc search "female singer"
python generate.py models rvc search "anime" --limit 50
python generate.py models rvc install User/ModelRepo
python generate.py models rvc install User/MultiModelRepo --file "specific_voice"
python generate.py models rvc install User/Repo --name "my-custom-name"
python generate.py models rvc install "https://example.com/model.zip"
python generate.py models rvc remove my-model
python generate.py models rvc calibrate my-model                        # guess target F0 from model name (heuristic)
python generate.py models rvc set-pitch my-model 120                    # male ~120 Hz
python generate.py models rvc set-pitch my-model 220                    # female ~220 Hz
python generate.py models rvc set-pitch my-model 280                    # child ~280 Hz
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

**Model config:** `worker/rvc/models/<model>/revoicer.json` with `target_f0` and optional `hf_repo_id`.

</details>

---

### Server Management (`server`)

The RVC worker runs as a background API server on port 5100.

```bash
python generate.py server start              # default port 5100
python generate.py server start -pt 5200     # custom port
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

`setup.sh` creates 13 isolated conda environments + 1 uv project:

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
| `ezaudio` | 3.10 | conda | Sound effects generation (EzAudio) |
| `text` | 3.11 | conda | LLM inference (ollama, Pillow, requests) — auto-updates |
| `flux2` | 3.12 | conda | FLUX.2 image generation (PyTorch MPS) |
| `openpose` | 3.12 | conda | DWPose pose estimation (ONNX Runtime) |
| `sd15` | 3.12 | conda | Stable Diffusion 1.5 (PyTorch MPS, CivitAI models) |
| `depth` | 3.12 | conda | Depth Anything V2 depth estimation (PyTorch MPS) |
| `lineart` | 3.12 | conda | TEED/AnyLine line art extraction (controlnet-aux) |
| `normalmap` | 3.12 | conda | Marigold-Normals v1.1 surface normal estimation |
| `sketch` | 3.12 | conda | HED edge extraction (OpenCV DNN, no PyTorch) |
| `upscale` | 3.12 | conda | Real-ESRGAN image upscaling (PyTorch MPS) |
| `segment` | 3.12 | conda | BiRefNet foreground/background separation (rembg, CoreML) |
| `ace` (uv) | 3.11+ | uv | ACE-Step 1.5 music generation |

</details>

<details>
<summary><strong>Offline Safety (Wheels Cache)</strong></summary>

Each worker has a local `wheels/` directory with cached `.whl` files and `requirements.lock`. Install scripts check:
1. `wheels/` present → Offline install from local wheels
2. `requirements.lock` → Online install with pinned versions
3. Neither → Fallback to PyPI, then generate lockfile

**Exception:** The **text** worker has no pinned wheels. It installs `ollama` directly from PyPI and auto-updates the package once per day on first use.

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
├── worker/
│   ├── ace/                # ACE-Step 1.5 worker (uv project)
│   ├── diarize/            # pyannote diarization worker
│   ├── enhance/            # resemble-enhance worker
│   ├── langdetect/         # Language detection worker
│   ├── music/              # HeartMuLa worker
│   ├── rvc/                # RVC voice conversion worker
│   ├── separate/           # demucs separation worker
│   ├── tts/                # Qwen3-TTS worker (mlx-audio)
│   ├── sfx/                # EzAudio sound effects worker
│   ├── image/              # FLUX.2 image generation (PyTorch MPS)
│   ├── pose/               # DWPose pose estimation (ONNX Runtime)
│   └── whisper/            # mlx-whisper worker
├── models/                 # All model checkpoints (gitignored)
├── tests/                  # Test scripts
├── demos/                  # Demo output files
└── prompt-guides/          # ACE-Step, HeartMuLa & SFX prompt guides
```

</details>
