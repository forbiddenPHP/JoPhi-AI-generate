# LTX-2.3 Video Worker

Text/Image/Audio-to-Video generation using LTX-2.3 (22B parameters). macOS Apple Silicon (MPS) only.

## Models

| File | Size | Pipeline |
|------|------|----------|
| `ltx-2.3-22b-distilled.safetensors` | 43 GB | Distilled (8 steps) |
| `ltx-2.3-22b-dev.safetensors` | 43 GB | Dev (30+3 steps, two-stage) |
| `ltx-2.3-22b-distilled-lora-384.safetensors` | 7.1 GB | Stage 2 refinement LoRA |
| `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | 950 MB | 2x spatial upscaler |
| `ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors` | 624 MB | IC-LoRA Union Control |
| `ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors` | 312 MB | IC-LoRA Motion Track |
| `gemma-3-12b-it/` | ~24 GB | Text encoder (Gemma 3 12B) |

## Pipelines

- **Distilled** — Single checkpoint, 8 denoising steps. Fast, good quality.
- **Dev (Two-Stage)** — Stage 1: 30 steps at half resolution → 2x upscale → Stage 2: 3 steps refinement with distilled LoRA. Slower, higher quality.
- **A2V** — Audio-to-video. Same two-stage process with audio conditioning. Video length matches audio duration.
- **Extend** — Append new frames to an existing video. Uses last N seconds as context (default: 2s), generates continuation, then concatenates. Source video is auto-transcoded to target resolution/FPS if needed.
- **Clone** — Generates a new video using a reference video as visual initialization. Uses RetakePipeline over the full output duration (start=0), so audio and video are fully regenerated from the prompt while the source latents provide appearance and motion context. Source videos longer than 5s are trimmed to the last 5s. Auto-transcoded to target resolution/FPS if needed.
- **Retake** — Regenerate a time region of an existing video. Encodes full video to latents, applies temporal mask to the specified time window, regenerates only that section. Prompt should describe the entire scene with the desired change.

## Usage (via root generate.py)

```bash
# Text-to-video (distilled, default)
python generate.py video ltx2.3 -p "A cat running through a meadow" --ratio 16:9 --quality 480p -o output.mp4

# Text-to-video (dev, higher quality)
python generate.py video ltx2.3 --model dev -p "A cat running through a meadow" --ratio 16:9 --quality 480p -o output.mp4

# Image-to-video
python generate.py video ltx2.3 -p "The man smiles and waves" --image-first photo.png --ratio 1:1 --quality 480p -o output.mp4

# Audio-to-video
python generate.py video ltx2.3 -p "Two people talking in a café" --audio dialog.wav --ratio 16:9 --quality 240p -o output.mp4

# Extend (+5 seconds)
python generate.py video ltx2.3 --model dev -p "The man says: hello" --extend video.mp4 5 --ratio 16:9 --quality 240p -o extended.mp4

# Extend with more context (3s instead of default 2s)
python generate.py video ltx2.3 --model dev -p "She walks away" --extend video.mp4 5 --ref-seconds 3 --ratio 16:9 --quality 240p -o extended.mp4

# Clone (generate 5s new video from reference, default)
python generate.py video ltx2.3 --model dev -p "The person says: hello" --clone video.mp4 --ratio 16:9 --quality 720p -o clone.mp4

# Clone with custom duration (8s)
python generate.py video ltx2.3 --model dev -p "She laughs" --clone video.mp4 --seconds 8 --ratio 16:9 --quality 720p -o clone.mp4

# Retake (replace 3.5s–5.0s)
python generate.py video ltx2.3 --model dev -p "Full scene description with the changed dialog" --retake video.mp4 3.5 5.0 --ratio 16:9 --quality 240p -o retake.mp4
```

## Dimensions

Width and height must be multiples of 64. Use `--ratio` + `--quality` for automatic resolution.

Supported ratios: 16:9, 9:16, 21:9, 9:21, 4:3, 3:4, 4:5, 5:4, 1:1, 1:2, 2:1
Quality tiers: 240p, 360p, 480p, 720p, 1080p, 1440p, 2160p, 4k

## Frames

- Must satisfy `8k+1` constraint (e.g., 9, 17, 25, 33, 41, 49, ..., 121, ..., 481)
- 24 FPS default
- For A2V: frames derived automatically from audio duration

## Setup

```bash
bash worker/ltx2/install.sh
```

## License

Apache License 2.0 (Lightricks). See `licenses/ltx2.md`.
