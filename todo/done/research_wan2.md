# Wan 2.2 — Research

> Repo: https://github.com/Wan-Video/Wan2.2
> Geclont nach: /tmp/wan22-research/

## Projektstruktur (aus dem Code)

```
wan/                    # Core-Library
├── __init__.py         # Exportiert: WanT2V, WanI2V, WanTI2V, WanS2V, WanAnimate
├── configs/            # Modell-Konfigurationen (5 Configs)
├── modules/
│   ├── attention.py    # Flash Attention (CUDA-only!)
│   ├── model.py        # DiT-Modell + RoPE
│   ├── vae2_1.py       # VAE für T2V/I2V/S2V/Animate
│   ├── vae2_2.py       # VAE für TI2V-5B (höhere Kompression)
│   ├── t5.py           # UMT5 Text-Encoder
│   ├── s2v/            # Speech-to-Video Module
│   └── animate/        # Character Animation Module
├── distributed/        # FSDP, Ulysses (CUDA-only)
├── utils/              # Prompt Extension, Solver etc.
├── text2video.py       # WanT2V Pipeline
├── image2video.py      # WanI2V Pipeline
├── textimage2video.py  # WanTI2V Pipeline
├── speech2video.py     # WanS2V Pipeline
└── animate.py          # WanAnimate Pipeline
generate.py             # CLI Entry Point
```

Build-System: **pip** + `requirements.txt`

## Pipelines im Code (tatsächlich im Repo)

5 Pipeline-Klassen:

| Klasse | Task-Flag | Beschreibung |
|--------|-----------|-------------|
| **WanT2V** | `t2v-A14B` | Text-to-Video, MoE 27B (14B aktiv) |
| **WanI2V** | `i2v-A14B` | Image-to-Video, MoE 27B (14B aktiv) |
| **WanTI2V** | `ti2v-5B` | Unified Text+Image-to-Video, Dense 5B |
| **WanS2V** | `s2v-14B` | Speech/Audio-to-Video, audio-driven |
| **WanAnimate** | `animate-14B` | Character Animation + Replacement |

### NICHT im Repo-Code enthalten (extern)
- **VACE** — separates Projekt (Fun-VACE), nicht in diesem Repo
- **Camera Control** — separates Projekt (Fun-Camera)
- **FLF2V** — nur via diffusers/ComfyUI
- **Chrono Edit** — nur via ComfyUI

## CLI (generate.py)

```bash
python generate.py \
    --task t2v-A14B \
    --ckpt_dir /path/to/model \
    --prompt "..." \
    --size "1280*720" \
    --frame_num 81 \
    --base_seed 42 \
    --sample_steps 40 \
    --sample_guide_scale 5.0 \
    --sample_solver unipc \
    --offload_model True \
    --save_file output.mp4
```

**Args (alle Tasks):**
- `--task` — `{t2v-A14B, i2v-A14B, ti2v-5B, s2v-14B, animate-14B}`
- `--ckpt_dir` — Checkpoint-Verzeichnis
- `--prompt` — Text-Prompt
- `--size` — z.B. "1280*720"
- `--frame_num` — Frames (muss `4k + 1` sein)
- `--base_seed` — Seed (-1 = random)
- `--sample_steps` — Diffusion Steps
- `--sample_shift` — Noise Schedule Shift
- `--sample_guide_scale` — CFG Scale (kann Tuple für MoE)
- `--sample_solver` — `{unipc, dpm++}`
- `--convert_model_dtype` — Konvertiert zu config dtype
- `--offload_model` — CPU-Offloading (default True)

**I2V:**
- `--image` — Input-Bild

**S2V:**
- `--audio` — Audio-Datei
- `--enable_tts` — CosyVoice TTS
- `--tts_prompt_audio` + `--tts_prompt_text` — Voice Reference
- `--tts_text` — Text für TTS
- `--pose_video` — Pose-Guidance (DW-Pose Format)
- `--num_clip`, `--infer_frames` — Clip-Kontrolle

**Animate:**
- `--src_root_path` — Preprocessed Data
- `--refert_num` — Temporal Guidance Frames (1 oder 5)
- `--replace_flag` — Replacement Mode
- `--use_relighting_lora` — Relighting LoRA

**Prompt Extension:**
- `--use_prompt_extend` + `--prompt_extend_method {dashscope, local_qwen}`

## Modelle

### Architektur & Größen

| Modell | Params | Architektur | Layers | Hidden | Heads | VAE | FPS |
|--------|--------|-------------|--------|--------|-------|-----|-----|
| T2V-A14B | 27B (14B aktiv) | MoE (2 Experten) | 40 | 5120 | 40 | 2.1 (4×8×8) | 16 |
| I2V-A14B | 27B (14B aktiv) | MoE | 40 | 5120 | 40 | 2.1 (4×8×8) | 16 |
| TI2V-5B | 5B | Dense | 30 | 3072 | 24 | 2.2 (4×16×16) | 24 |
| S2V-14B | 14B | MoE-basiert | 40 | 5120 | 40 | 2.1 (4×8×8) | 16 |
| Animate-14B | 14B | MoE-basiert | 40 | 5120 | 40 | 2.1 (4×8×8) | 30 |

**MoE-Detail:** High-Noise Expert (frühe Denoising-Stages) + Low-Noise Expert (Refinement). Umschaltung bei 87.5% der Timesteps (SNR-basiert).

### Checkpoint-Format im Original-Repo

Das Repo lädt `.pth`-Dateien aus `--ckpt_dir`. S2V nutzt teilweise `safetensors.safe_open()`.

**Safetensors-Varianten existieren** (ComfyUI-repackaged, HuggingFace):
- FP16: T2V 14B (~28.6 GB), I2V 14B (~28.6 GB), TI2V 5B (~10 GB)
- BF16: S2V, Animate, Fun-Varianten (~29-35 GB)
- Diffusers-Format: `Wan-AI/Wan2.2-*-Diffusers` (safetensors in Subfolders)

**Distilled:** `lightx2v/Wan2.2-Distill-Models` — 4-Step, BF16

### RAM-Anforderung auf MPS

| Modell | FP16/BF16 Größe | RAM-Minimum |
|--------|----------------|-------------|
| **TI2V-5B** | ~10 GB | 16+ GB |
| **T2V-A14B** | ~57 GB (beide Experten) | 64+ GB |
| **I2V-A14B** | ~57 GB | 64+ GB |
| **S2V-14B** | ~57 GB | 64+ GB |
| **Distilled (BF16)** | ~28 GB | 36+ GB |

**NICHT verwenden**: FP8 — MPS unterstützt kein Float8_e4m3fn.

### Text-Encoder
**UMT5-XXL** (Google) — läuft lokal. Kann via `--t5_cpu` auf CPU gehalten werden.

### Image-Encoder (I2V)
**CLIP Vision Model**

## LoRAs

- **Animate**: Built-in Relighting LoRA (`relighting_lora.ckpt`), PEFT-basiert (rank=128, alpha=128)
- **Andere Pipelines**: LoRA-Support nur via externe Tools (DiffSynth-Studio, Diffusers)
- **Community**: 47+ Adapter auf HuggingFace für T2V-A14B
- **Im Repo-Code**: Nur Animate hat direkten LoRA-Code

## Audio/Speech Features (S2V-Pipeline)

### Implementiert im Code (`wan/speech2video.py`, `wan/modules/s2v/`):

**Audio-Encoding:**
- Encoder: **WAV2Vec2** (`wav2vec2-large-xlsr-53-english`)
- Input: Audio-Dateien (wav, mp3)
- Output: 1024-dim Audio-Features

**Audio-Injection ins DiT:**
- 12 Injection-Layers: `[0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36, 39]`
- Methode: AdaIN (Adaptive Instance Normalization) in Attention Blocks

**Features:**
- **Video-Länge auto-sync** zu Audio-Dauer
- **Pose-Guidance**: Optionales `--pose_video` (DW-Pose Format)
- **TTS-Integration**: CosyVoice (zero-shot Voice Cloning)
- **Motion Frames**: 73-Frame Motion Buffer für temporale Konsistenz
- **Lip-Sync**: Implizit via Audio-Visual Alignment (kein separates Modul)

**KEIN Audio-Output** — Wan konsumiert Audio als Input, generiert keins (Gegensatz zu LTX-2!)

### S2V zusätzliche Dependencies (`requirements_s2v.txt`)
```
openai-whisper, HyperPyYAML, onnxruntime, inflect, wetext, omegaconf,
conformer, hydra-core, lightning, rich, gdown, matplotlib, wget, pyarrow,
pyworld, librosa, decord, modelscope, GitPython
```

### Animate zusätzliche Dependencies (`requirements_animate.txt`)
```
decord, peft, onnxruntime, pandas, matplotlib,
sam2 (git+https://github.com/facebookresearch/sam2.git),
loguru, sentencepiece
```

## Generation Capabilities

| Parameter | Werte |
|-----------|-------|
| **Auflösungen T2V/I2V** | 720×1280, 1280×720, 480×832, 832×480 |
| **Auflösungen TI2V-5B** | 1280×704, 704×1280 (720P@24fps) |
| **Auflösungen S2V** | + 1024×704, 704×1024 (Cinematic) |
| **Auflösungen Animate** | 720×1280, 1280×720 |
| **Frames** | `4k + 1` (z.B. 81, 121). Default T2V/I2V: 81, TI2V: 121, Animate: 77 |
| **FPS** | T2V/I2V/S2V: 16, TI2V-5B: 24, Animate: 30 |
| **Steps** | 40-50 typisch. Solver: UniPC (default) oder DPM++ |
| **Guidance** | CFG Scale, dual guidance für MoE |
| **Dauer** | ~5 Sek (81@16fps), S2V auto nach Audio-Länge |

## Prompt Guide

**Struktur (80-120 Wörter empfohlen):**
```
Subjekt + Aktion + Kamerabewegung + Beleuchtung + Stil/Medium + Objektiv/Ära + Color Grade + Stimmung
```

- Beschreiben wie sich Dinge **bewegen**, nicht nur was erscheint
- Konkrete Verben, Kinematische Begriffe, Parallax-Hinweise
- Beleuchtung explizit angeben
- **Prompt Extension**: `--use_prompt_extend` mit `dashscope` (remote) oder `local_qwen` (lokal, Qwen2.5-14B/7B/3B)
- Default Negative Prompt eingebaut (Chinese+English)

## MPS-Kompatibilität — Analyse des Codes

### Status: CUDA-ONLY, kein MPS-Support

**Device-Handling** (alle Pipeline-Dateien):
```python
self.device = torch.device(f"cuda:{device_id}")  # ← hardcoded
```

**CUDA-Blocker im Code:**

1. **flash_attn** (`wan/modules/attention.py`):
   - `assert q.device.type == 'cuda' and q.size(-1) <= 256`
   - Importiert `flash_attn_interface` (FA3) oder `flash_attn` (FA2), beide CUDA-only
   - **HAT CPU-Fallback** (Standard-Attention ab Zeile ~150) — aber kein MPS-Pfad

2. **`torch.amp.autocast('cuda')`** — in allen Pipelines, muss auf `'mps'` oder weg

3. **RoPE** (`modules/model.py`):
   - `@torch.amp.autocast('cuda', enabled=False)` Decorator

4. **`torch.cuda.empty_cache()`** — überall

5. **Distributed** (NCCL Backend) — für uns irrelevant (Single-Device)

### Was wir patchen müssen

1. **Device-Detection**: `cuda:{id}` → `mps` (oder parametrisch)
2. **flash_attn** → `torch.nn.functional.scaled_dot_product_attention` (CPU-Fallback existiert, muss für MPS erweitert werden)
3. **`torch.amp.autocast('cuda')`** → `torch.amp.autocast('mps')` oder entfernen
4. **`@torch.amp.autocast('cuda', enabled=False)`** → Device-agnostisch
5. **`torch.cuda.empty_cache()`** → `torch.mps.empty_cache()`
6. **Checkpoint-Loading**: .pth → ggf. safetensors-Loader, oder .pth mit `map_location='mps'`
7. **numpy<2** pinnen
8. **Eigenes Chunking (K*)** wahrscheinlich nötig wie bei Flux

## Core Dependencies (requirements.txt)

```
torch>=2.4.0
torchvision>=0.19.0
torchaudio
diffusers>=0.31.0
transformers>=4.49.0,<=4.51.3
tokenizers>=0.20.3
accelerate>=1.1.1
opencv-python>=4.9.0.80
imageio[ffmpeg]
imageio-ffmpeg
flash_attn              # ← CUDA-only, NICHT installierbar auf macOS
numpy>=1.23.5,<2
easydict
ftfy
dashscope
tqdm
```

## Vergleich LTX-2.3 vs Wan 2.2

| Feature | LTX-2.3 | Wan 2.2 |
|---------|---------|---------|
| **Audio-Generation** | JA (joint mit Video) | NEIN (nur Audio-Input) |
| **Audio-to-Video** | JA (A2Vid) | JA (S2V-14B) |
| **Lipsync** | Nur API/ComfyUI | Implizit via S2V |
| **Video Extension** | JA (RetakePipeline) | Nicht im Repo |
| **Control/Guided** | IC-LoRA | Nicht im Repo (extern: VACE) |
| **Inpainting** | Nein | Nicht im Repo (extern: VACE) |
| **Character Animation** | Nein | JA (Animate-14B) |
| **Kleinstes Modell** | 22B | 5B (TI2V) |
| **Text-Encoder** | Gemma 3 12B | UMT5-XXL |
| **Build-System** | uv | pip |
| **Diffusers-basiert** | NEIN | NEIN (eigener Code) |
| **MPS-Patches nötig** | Mittel (Device, Attention, Vocoder) | Hoch (flash_attn, Autocast, Device) |
| **Checkpoint-Format** | .safetensors | .pth (safetensors via HF/ComfyUI) |
