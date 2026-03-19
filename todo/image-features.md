# Weitere Bild-Features

## ControlNet ABI (geplant, vereinheitlicht)

Zwei getrennte Schritte: **Konvertieren** (Foto → Conditioning-Bild) und **Generieren** (Conditioning-Bild → neues Bild).

### Schritt 1: Konvertieren (eigene Engines)
```bash
python generate.py image --engine depth     --images photo.png -o depth.png
python generate.py image --engine openpose  --images photo.png -o pose.png
python generate.py image --engine lineart   --images photo.png -o lines.png
python generate.py image --engine normalmap --images photo.png -o normals.png
python generate.py image --engine sketch    --images photo.png -o sketch.png
```

### Schritt 2: Generieren mit Conditioning (`--controlnet mode:file`)
```bash
python generate.py image --engine flux.2 --controlnet depth:depth.png -p "cartoon version" -o out.png
python generate.py image --engine flux.2 --controlnet pose:pose.png --images person.png -p "change pose to match controlnet" -o out.png
python generate.py image --engine flux.2 --controlnet lineart:lines.png -p "colorful illustration" -o out.png
python generate.py image --engine flux.2 --controlnet normalmap:normals.png -p "relight with dramatic side lighting" -o out.png
python generate.py image --engine flux.2 --controlnet sketch:sketch.png -p "detailed fantasy scene" -o out.png
python generate.py image --engine sd1.5  --controlnet lineart:lines.png -p "detailed fantasy illustration" -o out.png
python generate.py image --engine sd1.5  --controlnet sketch:sketch.png -p "anime character" -o out.png
```

**Behandlung nach Mode:**
- `depth`, `normalmap`, `lineart`, `sketch` → pixel-aligned, Pan & Scan auf Output-Dimensionen
- `pose` → normales Referenzbild (keine Pixel-Alignment)

**Migration:** `--depth` wird zu `--controlnet depth:file`. Tests müssen aktualisiert werden.

---

## Inpainting & Outpainting
- [ ] Inpainting: Teile eines Bildes maskieren und neu generieren
- [ ] Outpainting: Bild über die Ränder hinaus erweitern
- [ ] Masken-Unterstützung (Schwarz/Weiß-Maske, binary)

**Lösungsweg FLUX.2 Klein:** Infrastruktur zu ~70% vorhanden (VAE, Encoder, Denoise, Reference-Conditioning). Fehlende Teile:
1. `--mask` Parameter in argparse + generate.py
2. Mask laden + auf Latent-Dimensionen resizen (H//16, W//16)
3. Original-Bild VAE-encoden → Referenz-Latent
4. Im Denoise-Loop: pro Step unmaskierte Latents mit Original-Latent (+ passendem Noise-Level) ersetzen (**SetLatentNoiseMask**-Prinzip)
5. Outpainting: Canvas padden → automatische Mask über gepaddte Bereiche → gleiche Pipeline
- ⚠️ NICHT "VAE Encode for Inpainting" (graut Pixel aus) — normales VAE Encode + Latent Noise Mask
- Optional: LanPaint-Sampling-Logik für bessere Qualität (~5x langsamer)
- Bestätigt durch mehrere ComfyUI-Workflows (Civitai, RunComfy, MyAIForce)

**Lösungsweg SD 1.5:** `runwayml/stable-diffusion-inpainting` + `StableDiffusionInpaintPipeline` aus diffusers. Natives Inpainting mit dediziertem Modell, funktioniert auf MPS, ~4GB RAM.

**ABI:**
```bash
python generate.py image --engine flux.2 --images orig.png --mask mask.png -p "garden" -o out.png
python generate.py image --engine sd1.5  --images orig.png --mask mask.png -p "garden" -o out.png
python generate.py image --engine flux.2 --images orig.png --outpaint top=200,right=300 -p "extend sky"
```

→ Details: `todo/research/inpainting-outpainting.md`

## Upscaling & Schärfung
- [ ] Bild-Upscaling (z.B. 2x, 4x)
- [ ] Schärfung / Detail-Enhancement

**Lösungsweg:** Real-ESRGAN Fork mit MPS-Patches (PR #836). Modelle ~64MB, RAM ~1-2GB. Kein MLX-Upscaler verfügbar.
```bash
python generate.py image --engine upscale --images input.png -o output.png
```

→ Details: `todo/research/upscaling.md`

## Bildebenen trennen (Image Demux)
- [ ] Vordergrund/Hintergrund trennen (Segmentierung)
- [ ] Einzelne Objekte/Personen isolieren
- [ ] Transparenter Hintergrund (PNG mit Alpha-Kanal)
- [ ] Mehrere Ebenen als separate Dateien ausgeben

**Lösungsweg Background Removal:** RMBG 2.0 (`briaai/RMBG-2.0`) via transformers auf MPS, ~1-2GB. Output: Alpha-Matte → RGBA PNG.
**Lösungsweg Multi-Object:** SAM 2 Core ML (Apple-offizielle Conversion `apple/coreml-sam2-large`). ⚠️ SAM 3 geht NICHT (Triton).
```bash
python generate.py image --engine segment --images input.png -o layers/
```

→ Details: `todo/research/segmentation.md`

## Stil-Transfer
- [ ] Stil eines Bildes auf ein anderes übertragen (Lighting, Farben, Ästhetik)

**Lösungsweg:** IP-Adapter + SD1.5 (`h94/IP-Adapter`, ~100MB, ~6-8GB). Strength steuerbar via `scale` 0.0-1.0. Für FLUX braucht man 32GB+ (oder GGUF).
```bash
python generate.py image --engine sd1.5  --style-ref style.png -p "a warrior" -o out.png
python generate.py image --engine flux.2 --style-ref style.png -p "a warrior" -o out.png
```

→ Details: `todo/research/style-transfer.md`

## Color Palette Conditioning
- [ ] Farbpalette aus Bild extrahieren oder manuell vorgeben
- [ ] Generierung an bestimmte Farbstimmung binden

**Lösungsweg:** FLUX.2 unterstützt Hex-Codes direkt im Prompt — zero cost. Für SD1.5: T2I-Adapter Color (`TencentARC/t2iadapter_color_sd14v1`, ~17MB). Palette-Extraktion: `color-thief-py` (CPU, pure Python).

→ Details: `todo/research/color-palette-and-lineart.md`

## Neue Worker (Konvertierungs-Engines)

| Engine | Modell | Installation | Modellgröße |
|--------|--------|-------------|-------------|
| lineart | AnyLine/TEED via `controlnet-aux` | `pip install controlnet-aux` | 58K Params, winzig |
| normalmap | Marigold-Normals v1.1 via `diffusers` | `pip install diffusers transformers accelerate torch` | ~2.5GB (fp16) |
| sketch | HED via OpenCV DNN | `pip install opencv-python-headless` | ~56MB (kein PyTorch!) |

Jeder Worker bekommt ein eigenes conda env.

→ Details: `todo/research/lineart-worker.md`, `normalmap-worker.md`, `sketch-worker.md`

## Erledigt
- ~~Posen & ControlNet~~ (OpenPose/DWPose + Depth Anything V2)
