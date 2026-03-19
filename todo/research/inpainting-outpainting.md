# Research: Inpainting & Outpainting

## Grundprinzip
- Input (Bild + optional Mask + Prompt) rein → neues Bild raus. Original wird nie modifiziert.

## FLUX.2 Klein 4B — Architektur-Verständnis

Klein ist ein **natives Edit-Modell**, kein dediziertes Inpainting-Modell (wie Flux.1 Fill).
- Reference Image wird VAE-encoded → `reference_latents` [1, 128, H, W]
- Referenz-Latents werden patchifiziert und mit dem Noisy Latent **konkateniert** (nicht in den Text-Stream gemerged)
- Dual-Stream: Text + Bild parallel durch `double_blocks`, dann merged in `single_blocks`
- **Kein nativer Mask-Channel** im Modell selbst — Masks werden extern im Sampling/Conditioning gehandhabt

## FLUX.2 Klein: Drei bestätigte Inpainting-Methoden

### 1. Prompt-Only (masklos) — EXISTIERT BEREITS
- Referenzbild + Textanweisung → Modell entscheidet selbst wo es eingreift
- Das ist unser bestehender `--images` Pfad!
- Bestätigt: Civitai 4B GGUF Workflow, 376 Downloads, 5-Sterne Reviews
- Outpainting: Einfach größere Output-Dimensionen als Input → Modell erweitert natürlich

### 2. SetLatentNoiseMask (explizite Mask)
- Bild VAE-encoden → Latents
- Mask an Latents anhängen via SetLatentNoiseMask
- Sampler denoised NUR maskierte Bereiche, behält Rest
- ⚠️ NICHT "VAE Encode for Inpainting" nutzen (graut Pixel aus) → normales VAE Encode + SetLatentNoiseMask
- Optional: LanPaint KSampler statt Standard-KSampler für bessere Qualität (~5x langsamer)
- Bestätigt: Mehrere ComfyUI Workflows (lilys.ai, Civitai, aistudynow.com)

### 3. Crop & Composite (High-Res)
- Mask definiert Region → Inpaint Crop schneidet Quadrat um Mask aus
- Klein regeneriert nur den Crop bei nativer Auflösung
- Inpaint Stitch blendet Ergebnis zurück ins Original
- `context_expand_factor` >= 1.2 für saubere Übergänge
- Bestätigt: MyAIForce 4K Inpainting Guide

## SD 1.5 Inpainting (sofort einsatzbereit)
- Model: `runwayml/stable-diffusion-inpainting`
- `StableDiffusionInpaintPipeline` aus diffusers — **funktioniert auf MPS**
- Bild + Mask separat an Pipeline (natives Inpainting mit dediziertem Modell)
- RAM: ~4GB, sehr leichtgewichtig
- Outpainting: Bild padden, gepaddte Bereiche maskieren → funktioniert

## Geplante ABI
```bash
# Prompt-only Editing (Klein-native, existiert quasi schon)
python generate.py image --engine flux.2 --images orig.png -p "remove the tree"

# Mit Mask (Inpainting)
python generate.py image --engine flux.2 --images orig.png --mask mask.png -p "garden with flowers"
python generate.py image --engine sd1.5  --images orig.png --mask mask.png -p "garden with flowers"

# Outpainting (Canvas erweitern)
python generate.py image --engine flux.2 --images orig.png --outpaint top=200,right=300 -p "extend the sky"
python generate.py image --engine sd1.5  --images orig.png --outpaint top=200,right=300 -p "extend the sky"
```

## Bekannte Limitierungen
- Artefakte an Mask-Kanten bei Denoise-Strength 1.0 → 0.85-0.90 besser
- Outpainting >256px pro Seite wird ohne Crop & Stitch schlechter
- Klein: Desaturation-Bug — inpaintete Bereiche können übersättigt werden (Fix: Desaturation-Preprocessing 0.05-0.20)
- Ein Workflow empfiehlt SDXL für initiales Outpainting, FLUX nur für Refinement (große leere Flächen)

## Quellen
- ComfyUI Official: docs.comfy.org/tutorials/flux/flux-2-klein
- 4-in-1 Workflow: civitai.com/articles/25307
- 4K Inpainting: myaiforce.com/flux-2-klein-inpaiting/
- LanPaint: github.com/scraed/LanPaint
- 4B GGUF Workflow: civitai.com/models/2322343
- Klein Enhancer: github.com/capitan01R/ComfyUI-Flux2Klein-Enhancer
- Unified Editing: runcomfy.com (FLUX Klein Unified Image Editing)
- Outpainting: comfyui.org/en/beyond-the-frame-flux-model-image-outpainting
