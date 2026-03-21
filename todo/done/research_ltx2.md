# LTX-2 / LTX Video 2.3 — Research

> Repo: https://github.com/Lightricks/LTX-2
> Geclont nach: /tmp/ltx2-research/

## Projektstruktur (aus dem Code)

Monorepo mit 3 Packages:
- **ltx-core** — Modell-Implementierung, Inference-Komponenten, Utilities
- **ltx-pipelines** — Pipeline-Klassen für alle Generierungsmodi
- **ltx-trainer** — Training/Fine-Tuning (LoRA, Full)

Build-System: **uv** (nicht pip). `pyproject.toml` + `uv.lock`.
**NICHT diffusers-basiert** — eigene Pipeline-Implementierung.

## Pipelines im Code

8 Pipeline-Klassen, alle als Module ausführbar (`python -m ltx_pipelines.<name>`):

| Pipeline-Klasse | Beschreibung |
|----------------|-------------|
| **TI2VidTwoStagesPipeline** | Text/Image-to-Video, 2-Stage mit Upsampling (Produktion) |
| **TI2VidTwoStagesHQPipeline** | Wie oben, res_2s Second-Order-Sampler, weniger Steps |
| **TI2VidOneStagePipeline** | Single-Stage, kein Upsampling (Prototyping) |
| **DistilledPipeline** | Schnellste Inferenz: 8 vordefinierte Sigmas |
| **ICLoraPipeline** | Video-to-Video / Image-to-Video via IC-LoRA Conditioning (nur distilled) |
| **KeyframeInterpolationPipeline** | Interpolation zwischen Keyframe-Bildern |
| **A2VidPipelineTwoStage** | Audio-to-Video: Input-Audio → passendes Video (2-Stage) |
| **RetakePipeline** | Regeneriert Zeitbereiche in existierenden Videos (1-Stage) |

### Modi pro Pipeline
- **T2V** (Text-to-Video): alle Pipelines
- **I2V** (Image-to-Video): alle Pipelines via `ImageConditioningInput`
- **V2V** (Video-to-Video): nur ICLoraPipeline
- **Audio-to-Video**: A2VidPipelineTwoStage
- **Video Extension**: RetakePipeline (Frames als Condition)
- **FLF2V** (First+Last Frame): Conditioning auf beide Frames

### Weitere Fähigkeiten
- **Spatial Upscaling**: x1.5 und x2 Upscaler-Modelle
- **Temporal Upscaling**: x2 für mehr FPS (Frame-Interpolation)
- **Multi-Language Prompts**: EN, DE, ES, FR, JA, KO, ZH, IT, PT (Gemma 3)
- **Lipsync**: Nur via API/ComfyUI erwähnt, kein Standalone-Pipeline-Code im Repo

## CLI-Argumente (aus dem Code)

```bash
python -m ltx_pipelines.ti2vid_two_stages \
    --checkpoint-path model.safetensors \
    --distilled-lora lora.safetensors 0.8 \
    --spatial-upsampler-path upsampler.safetensors \
    --gemma-root /path/to/gemma \
    --prompt "..." --negative-prompt "..." \
    --height 512 --width 768 --num-frames 121 --frame-rate 24 \
    --num-inference-steps 40 \
    --seed 42 \
    --output-path output.mp4
```

**Gemeinsame Args:**
- `--checkpoint-path` / `--distilled-checkpoint-path`
- `--gemma-root` — lokales Gemma-3-Verzeichnis
- `--spatial-upsampler-path` — für 2-Stage
- `--distilled-lora` — Pfad + Stärke
- `--lora` — mehrere LoRAs: `pfad stärke` Paare
- `--prompt`, `--negative-prompt`, `--enhance-prompt`
- `--seed`, `--height`, `--width`, `--num-frames`, `--frame-rate`
- `--num-inference-steps`
- `--quantization` (fp8-cast / fp8-scaled-mm) — NVIDIA-only, für uns irrelevant

**Pipeline-spezifisch:**
- A2Vid: `--audio-path`, `--audio-start-time`, `--audio-max-duration`
- Retake: `--video-path`, `--start-time`, `--end-time`
- ICLora: `--video-conditioning`, `--conditioning-attention-mask`, `--skip-stage-2`

## Modelle (.safetensors, bf16)

### LTX-2.3 (22B) — AKTUELL

| Datei | Größe | Zweck |
|-------|-------|-------|
| `ltx-2.3-22b-dev.safetensors` | 46.1 GB | Vollmodell, bf16, mit CFG |
| `ltx-2.3-22b-distilled.safetensors` | 46.1 GB | Distilled, 8 Steps, CFG=1 |
| `ltx-2.3-22b-distilled-lora-384.safetensors` | 7.61 GB | LoRA für Stage-2 Refinement |
| `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | 996 MB | x2 Spatial Upscaler |
| `ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors` | 1.09 GB | x1.5 Spatial Upscaler |
| `ltx-2.3-temporal-upscaler-x2-1.0.safetensors` | 262 MB | x2 Temporal Upscaler |

**NICHT verwenden**: fp8, nvfp4 — NVIDIA-only.

### LTX-2 (19B) — ALT, NICHT VERWENDEN

19B ist Vorgänger. LoRAs inkompatibel mit 22B.

**Arbeitspräzision im Code**: `torch.bfloat16` hardcoded in Pipelines.

### Text-Encoder

**Gemma 3 12B Instruct** — läuft lokal, kein API-Call. QAT q4_0 Quantisierung.

## LoRAs (22B)

Format: Standard `.safetensors` mit `.lora_A.weight` / `.lora_B.weight` Keys.
Fusion: `apply_loras()` fusioniert LoRA-Weights in Modell-Weights beim Laden.
Mehrere LoRAs kombinierbar, jeweils mit Stärke-Koeffizient (0.0-1.0).

| LoRA | Funktion |
|------|----------|
| `LTX-2.3-22b-IC-LoRA-Union-Control` | Multi-Control (Canny, Depth, Pose, etc.) |
| `LTX-2.3-22b-IC-LoRA-Motion-Track-Control` | Objekt-Bewegung via Sparse Point Trajectories |
| `ltx-2.3-22b-distilled-lora-384.safetensors` | Distilled LoRA für Stage-2 |
| Camera LoRAs (7 Varianten) | Dolly In/Out/Left/Right, Jib Up/Down, Static |

## Audio-Features (im Code)

- **Audio VAE** (Encoder/Decoder) — Spectral Encoding mit Causal Convolutions
- **Vocoder** + **Vocoder with BWE** (Bandwidth Extension)
- **Audio Processor** — Waveform → Mel-Spectrogram
- **TMRoPE** (Time-aligned Multimodal RoPE) — Audio-Video Synchronisation
- **Modality CFG** — Separate Guidance für Video/Audio (default 3.0/3.0)
- **Joint Generation**: Audio und Video in einem DiT-Pass synchron
- **A2Vid**: Audio-Input wird via Audio-VAE encoded, in Stage 1 eingefroren, in Stage 2 verfeinert
- Audio-Latent Shape: `[B, 8, T, 16]` (8 Channels, 16 Mel Bins)

## Generation Capabilities

| Parameter | Werte |
|-----------|-------|
| **Auflösung** | Teilbar durch 64. Typisch: Stage-1 512×768 → Upscale 1024×1536 |
| **Frames** | `frames = 8k + 1` (1, 9, 17, 25, ..., 121) |
| **FPS** | Default 24, konfigurierbar |
| **Default** | 121 Frames @ 24fps = ~5 Sekunden |
| **Steps (Dev)** | 40 (default), 20-30 mit Gradient Estimation |
| **Steps (Distilled)** | 8 (Stage 1) + 4 (Stage 2) |

**Guidance:**
- **CFG Scale** — Prompt-Stärke (2.0-5.0, default 3.0 Video / 7.0 Audio)
- **STG Scale** — Temporal Coherence (0.5-1.5, default 1.0)
- **Rescale Scale** — Variance Matching (default 0.7)
- **Enhance Prompt** — Auto-Verbesserung via Gemma

## Prompt Guide

- Wie ein Kameramann beschreiben: detaillierte, chronologische Absätze, max 200 Wörter
- Enthalten: Szene, Subjekt+Aktion, Kamera+Objektiv, visueller Stil, Bewegung, Audio
- Kinematische Begriffe: "macro lens", "tracking shot", "shallow depth of field"
- Beleuchtung, Farben, Umgebungsdetails konkret angeben
- Präsens verwenden
- `--enhance-prompt` für Auto-Verbesserung
- Negative Prompts unterstützt
- Separate System-Prompts für T2V/I2V im Gemma-Encoder

## MPS-Kompatibilität — Analyse des Codes

### Status: NULL MPS-Support im Code

**Device-Detection** (`ltx_pipelines/utils/helpers.py`):
```python
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")  # ← kein MPS!
```

**CUDA-spezifische Stellen im Code:**
1. `torch.cuda.synchronize()` — mehrfach in Pipelines
2. `torch.cuda.empty_cache()` — Memory Cleanup
3. `str(device).startswith("cuda")` — Device-Checks in LoRA-Fusion
4. Attention: xFormers oder Flash Attention 3 (beide nicht MPS-kompatibel)
5. FP8-Quantisierung (TensorRT-LLM) — CUDA-only
6. `torch.device("meta")` bei Modell-Konstruktion

### Was wir patchen müssen

1. **`get_device()`** → MPS-Detection hinzufügen
2. **`torch.cuda.synchronize()`** → entfernen oder `torch.mps.synchronize()`
3. **`torch.cuda.empty_cache()`** → `torch.mps.empty_cache()`
4. **Attention** → Fallback auf `torch.nn.functional.scaled_dot_product_attention`
5. **RoPE** — `double_precision_rope` steht auf False (fp32), kein float64-Problem
6. **bfloat16** — MPS unterstützt bf16 ab PyTorch 2.3+, sollte funktionieren
7. **Vocoder conv1d** — MPS max 65536 Output-Channels, Workaround nötig bei großen Frame-Counts
8. **Eigenes Chunking (K*)** wahrscheinlich nötig wie bei Flux

### Bekannte MPS-Probleme (Community)
- PyTorch >2.4.1: MPS produziert nur Noise (strided API Bug)
- Distilled-Modell: Artefakte auf MPS → primär Dev-Modell verwenden
- Vocoder: Nur bei 21/61 Frames funktional auf MPS (conv1d Limit)

**Referenz für Patches:** `Pocket-science/ltx2-mps` (Diffusers-basiert, aber zeigt welche Stellen problematisch sind)

## Dependencies (aus pyproject.toml)

```
torch ~=2.7
transformers >= 4.52
safetensors
einops
torchaudio
scipy >= 1.14
accelerate
av (PyAV)
pillow
tqdm
```
Optional: `xformers`, `tensorrt-llm` (NVIDIA-only)
