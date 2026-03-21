# Image Features Research Report
**Date:** 2026-03-19
**Target:** macOS Apple Silicon (MPS/MLX), no CUDA, no cloud APIs
**Existing stack:** FLUX.2 Klein (PyTorch MPS), Stable Diffusion 1.5 (PyTorch MPS)

---

## 1. Style Transfer

### 1.1 IP-Adapter (RECOMMENDED - Best Option)

IP-Adapter is the current state-of-the-art approach for style transfer within diffusion pipelines. It works by injecting image embeddings into the cross-attention layers.

**For SD1.5:**
- Model: `h94/IP-Adapter` on HuggingFace
  - `ip-adapter_sd15.safetensors` (~100MB, general purpose)
  - `ip-adapter_sd15_light_v11.bin` (lighter, less intense transfer)
  - `ip-adapter-plus_sd15.safetensors` (stronger style transfer)
  - `ip-adapter-plus-face_sd15.safetensors` (face-focused)
- Only 22M parameters added on top of SD1.5
- Works natively with `diffusers` library via `load_ip_adapter()`
- MPS: Works on Apple Silicon. SD1.5 + IP-Adapter fits in ~6-8GB RAM
- HuggingFace: https://huggingface.co/h94/IP-Adapter
- Diffusers docs: https://huggingface.co/docs/diffusers/using-diffusers/ip_adapter

**For FLUX.1/FLUX.2:**
- XLabs IP-Adapter V1 and V2 for FLUX
  - V2 trained for 350k steps at 1024x1024 resolution
  - HuggingFace: `XLabs-AI/flux-ip-adapter` and `XLabs-AI/flux-ip-adapter-v2`
  - GitHub: https://github.com/XLabs-AI/x-flux-comfyui
- MPS: Works on Apple Silicon, but FLUX base model needs 32GB+ unified memory
- With GGUF quantized FLUX models, memory drops to ~12GB
- FP8 is NOT supported on MPS -- use GGUF quantization instead
- Generation time: ~85s on M4 Max, ~105s on M3 Max, ~145s on M2 Max (1024x1024)

**Workflow:** Load style reference image -> IP-Adapter encodes it -> combine with text prompt -> generate. Strength controllable via `scale` parameter (0.0-1.0).

### 1.2 Dedicated Style Transfer Models

**AesPA-Net (ICCV 2023):**
- GitHub: https://github.com/Kibeom-Hong/AesPA-Net
- Pure PyTorch, should work on MPS with minor adjustments
- Introduces "pattern repeatability" metric for style quantification
- Standalone model, not diffusion-based
- Smaller memory footprint than diffusion approaches (~2-4GB)
- Quality: Good for artistic/pattern-heavy styles

**StyTr2 (CVPR 2022):**
- Transformer-based style transfer
- PyTorch implementation available
- Should work on MPS (standard transformer ops are supported)
- ~2-3GB memory

**CAST (AAAI 2023):**
- Contrastive Arbitrary Style Transfer
- PyTorch-based
- Standard ops, likely MPS-compatible

**Classic Neural Style Transfer (Gatys et al.):**
- VGG-based, pure PyTorch
- Works perfectly on MPS, minimal memory (~1-2GB)
- Slow iterative optimization but reliable

### 1.3 Recommendation

**Primary:** IP-Adapter with SD1.5 for diffusion-based style transfer. Low memory overhead, well-integrated in diffusers, proven on MPS.

**Secondary:** IP-Adapter with FLUX for higher quality, but only viable with 32GB+ RAM (or GGUF quantized FLUX).

**Standalone fallback:** AesPA-Net for non-diffusion style transfer. Lighter, faster, but less flexible.

### 1.4 Known Issues on macOS

- FP8 not supported on MPS (affects FLUX IP-Adapter, use GGUF instead)
- Some IP-Adapter CLIP image encoder ops may need explicit `.to('mps')` calls
- IP-Adapter face models can be unstable with certain MPS PyTorch versions
- SD1.5 IP-Adapter is the most battle-tested path on Apple Silicon

---

## 2. Scribble/Sketch-to-Image

### 2.1 ControlNet Scribble for SD1.5 (RECOMMENDED)

**Models (all by lllyasviel):**
- `lllyasviel/sd-controlnet-scribble` (v1.0, original)
- `lllyasviel/control_v11p_sd15_scribble` (v1.1, improved)
- HuggingFace: https://huggingface.co/lllyasviel/control_v11p_sd15_scribble
- Model size: ~1.4GB each
- Combined with SD1.5: ~6-8GB total RAM

**MPS Status:** Works with diffusers on Apple Silicon. ControlNet SD1.5 pipelines are well-tested on MPS via `StableDiffusionControlNetPipeline`.

**Preprocessors:**
- **Scribble HED:** Holistically-Nested Edge Detection. Captures more detail, good for converting photos to scribble-like inputs. Works on MPS.
- **Scribble Pidinet:** Pixel Difference Network. Cleaner lines, fewer details than HED. **KNOWN BUG on Apple Silicon** -- the pidinet processor has documented preview/generation errors on MPS (GitHub issue #966 on sd-webui-controlnet). May need workaround or CPU fallback for preprocessing.
- **Fake Scribble:** Simplifies HED output to thick lines. Good middle ground.

**Quality comparison:**
- Scribble mode: Best for rough hand-drawn sketches, very forgiving
- HED mode: Better for photo-to-artwork, captures structure
- Pidinet mode: Cleanest edges, but MPS compatibility issues

### 2.2 T2I-Adapter Sketch for SD1.5

- Model: `TencentARC/t2iadapter_sketch_sd15v2`
- HuggingFace: https://huggingface.co/TencentARC/t2iadapter_sketch_sd15v2
- Lighter than ControlNet (~300MB adapter vs ~1.4GB ControlNet)
- Uses `StableDiffusionAdapterPipeline` from diffusers
- MPS: Should work (standard PyTorch ops), but less tested than ControlNet
- Quality: Slightly less precise control than ControlNet, but faster inference

### 2.3 FLUX ControlNet Sketch/Scribble

**Shakker-Labs ControlNet Union Pro 2.0:**
- HuggingFace: `Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0`
- Supports: canny, soft edge, depth, pose, gray (removed tile in v2.0)
- Soft edge mode is closest to sketch/scribble
- Architecture: 6 double blocks, 0 single blocks
- Trained for 300k steps at 512x512 on 20M images
- No dedicated "scribble" mode -- soft edge is the substitute

**XLabs FLUX ControlNet:**
- HuggingFace: `XLabs-AI/flux-controlnet-collections`
- Includes HED/scribble conditioning
- Works via ComfyUI on Apple Silicon (tested M1-M4)

**InstantX FLUX ControlNet Union:**
- HuggingFace: `InstantX/FLUX.1-dev-Controlnet-Union`
- Multi-condition support including scribble

**MPS Status for FLUX ControlNet:**
- Works on Apple Silicon with 32GB+ RAM
- GGUF quantized FLUX base reduces memory to ~12GB
- Generation slower than SD1.5 (~85-145s vs ~5-15s)
- Scribble conditioning scale optimal range: 0.65-0.80

### 2.4 Recommendation

**Primary:** ControlNet v1.1 Scribble (`control_v11p_sd15_scribble`) with SD1.5. Best tested on MPS, fast generation, good quality.

**Preprocessor:** Use HED preprocessor (more reliable on MPS than Pidinet). Pidinet needs CPU fallback.

**For higher quality:** FLUX + Shakker-Labs ControlNet Union Pro 2.0 soft edge mode, but only with 32GB+ RAM.

**Lightweight alternative:** T2I-Adapter sketch for SD1.5 if ControlNet is too heavy.

### 2.5 Known Issues on macOS

- Pidinet preprocessor has documented bugs on Apple Silicon MPS
- HED preprocessor works but can be slow on first run (model download)
- FLUX ControlNet scribble support is "basic" compared to SD1.5 ecosystem
- controlnet_aux package (preprocessors) may need specific versions for MPS compatibility

---

## 3. NormalMap Estimation

### 3.1 Marigold Normals (RECOMMENDED)

**The best option for Apple Silicon in 2025-2026.**

- **Marigold-Normals v1.1** released 2025-05-15
- Diffusion-based surface normal estimation, fine-tuned from Stable Diffusion
- Integrated into diffusers core since v0.28.0 -- first-class citizen
- Pipeline: `MarigoldNormalsPipeline` in diffusers
- HuggingFace: `prs-eth/marigold-normals-v1-1`
- LCM variant (faster): `prs-eth/marigold-normals-lcm-v0-1`
- Effective resolution: ~768px (inherited from SD base)
- Output: 3D unit vectors in screen-space camera coordinates
- CVPR 2024 Oral, Best Paper Award Candidate

**MPS Status:** Since it's built on diffusers and uses the same SD pipeline architecture, it runs on MPS via `.to('mps')`. Same memory profile as SD inference (~4-6GB).

**LCM variant** reduces inference from ~10-20 denoising steps to 1-4 steps, significantly faster.

### 3.2 DSINE (CVPR 2024 Oral)

- GitHub: https://github.com/baegwangbin/DSINE
- Discriminative model (not diffusion-based), faster inference
- Uses per-pixel ray direction and relative rotation encoding
- DPT-based architecture
- ComfyUI node available: https://github.com/kijai/ComfyUI-DSINE

**MPS Status:** Officially targets CUDA only (install instructions specify pytorch-cuda). However, the architecture uses standard PyTorch ops (ViT backbone, DPT decoder). Should work on MPS with device changes, but untested officially. Torch Hub loader available: https://github.com/hugoycj/DSINE-hub

**Memory:** ~2-4GB (discriminative model, single forward pass)

### 3.3 Omnidata v2

- GitHub: https://github.com/EPFL-VILAB/omnidata
- DPT-Hybrid architecture, trained on large-scale 3D scan data
- V2 released March 2022 with improved training
- Torch Hub: `torch.hub.load('alexsax/omnidata_models', 'surface_normal_dpt_hybrid_384')`
- Input resolution: 384x384 (can process higher with tiling)

**MPS Status:** Pure PyTorch DPT model. Standard ops should work on MPS. No official MPS testing documented. Memory: ~2-3GB for inference.

### 3.4 Using Normals as Conditioning for Image Generation

**SD1.5 ControlNet Normal:**
- Model: `lllyasviel/sd-controlnet-normal` (v1.0)
- Uses Bae's normal estimation, follows ScanNet standard
- Color encoding: Blue=front, Red=left, Green=top
- HuggingFace: https://huggingface.co/lllyasviel/sd-controlnet-normal
- Works with diffusers `StableDiffusionControlNetPipeline` on MPS
- ~1.4GB model + SD1.5 base = ~6-8GB total

**FLUX ControlNet with normals:**
- No dedicated normal map mode in current FLUX ControlNets
- Depth mode is the closest substitute
- Shakker-Labs Union Pro 2.0 does NOT include normal conditioning
- Workaround: Convert normals to depth-like representation, use depth ControlNet
- This is a gap in the FLUX ecosystem as of early 2026

### 3.5 Recommendation

**For normal estimation:** Marigold-Normals v1.1 via diffusers. Best integration, proven architecture, LCM variant for speed. First-class diffusers support means `.to('mps')` just works.

**For using normals as conditioning:** SD1.5 ControlNet Normal. FLUX currently lacks native normal conditioning.

**Alternative estimator:** DSINE if you need faster inference (single pass vs diffusion denoising). May need minor MPS porting work.

### 3.6 Known Issues on macOS

- Marigold inherits SD's ~768px effective resolution limit
- DSINE and Omnidata need manual `.to('mps')` device changes (no official MPS support)
- No FLUX ControlNet normal mode exists yet -- depth is the closest proxy
- Normal map color conventions differ between models (ScanNet vs OpenGL vs DirectX) -- ensure consistency between estimator output and ControlNet input

---

## Summary Table

| Feature | Best SD1.5 Option | Best FLUX Option | Min RAM | MPS Status |
|---------|-------------------|------------------|---------|------------|
| Style Transfer | IP-Adapter SD1.5 | XLabs IP-Adapter V2 | 6-8GB / 32GB+ | Works / Works (GGUF) |
| Scribble-to-Image | ControlNet v1.1 Scribble | Shakker-Labs Union Pro 2.0 (soft edge) | 6-8GB / 32GB+ | Works / Works |
| Normal Estimation | Marigold-Normals v1.1 | -- | 4-6GB | Works (diffusers) |
| Normal Conditioning | ControlNet Normal | None (use depth proxy) | 6-8GB / -- | Works / N/A |

## Key Takeaways

1. **SD1.5 is the safer bet** for all three features on Apple Silicon. Smaller models, faster inference, better MPS testing coverage.

2. **FLUX offers higher quality** but requires 32GB+ RAM (or GGUF quantization) and has gaps in the ecosystem (no normal conditioning, basic scribble support).

3. **IP-Adapter is the modern style transfer** -- forget standalone neural style transfer models unless you specifically need non-diffusion approaches.

4. **Marigold-Normals is the clear winner** for normal estimation -- diffusers-native, CVPR award-winning, works on the same SD infrastructure you already have.

5. **Pidinet preprocessor is broken on MPS** -- use HED for scribble preprocessing.

6. **FP8 does not work on MPS** -- always use FP16 or GGUF quantization for FLUX models.
