# Research: Color Palette Conditioning & Line Art Extraction
**Date:** 2026-03-19
**Focus:** macOS Apple Silicon (MPS / MLX), local inference only, no CUDA, no cloud APIs

---

## 1. Color Palette Conditioning

### 1.1 FLUX.2 Native: Hex Color Prompting (BEST OPTION)

FLUX.2 has **native hex color support** baked into the model. No additional adapters needed.

**How it works:**
- Include hex codes directly in prompts: `"a sunset landscape, sky color #FF6B35, mountains #2C3E50"`
- Hex codes work best when associated with specific objects
- Structured JSON prompts supported with a `color_palette` field:

```json
{
  "scene": "forest clearing at dawn",
  "color_palette": ["#2C3E50", "#E74C3C", "#F39C12"],
  "subjects": [{ "description": "deer", "position": "center" }],
  "style": "watercolor illustration"
}
```

**Apple Silicon status:** Works wherever FLUX.2 runs. This is prompt-level, not a separate model.

**Limitation:** This is a soft conditioning. The model interprets hex codes as guidance, not pixel-perfect enforcement. Brand-exact color matching is approximate, not guaranteed. Klein (4B) follows color prompts less precisely than the larger Dev/Pro models.

**Verdict:** Already available in your FLUX.2 Klein pipeline. Just change the prompt format. Zero additional downloads, zero additional memory.

### 1.2 T2I-Adapter Color (SD1.5)

**Model:** `TencentARC/t2iadapter_color_sd14v1` (works with SD1.5 too)
**Size:** ~17M parameters (tiny adapter)
**HuggingFace:** https://huggingface.co/TencentARC/t2iadapter_color_sd14v1

**How it works:**
1. Take a reference image (or create a solid-color grid)
2. Resize to 8x8 pixels (extracts dominant colors)
3. Resize back to 512x512 with nearest-neighbor resampling
4. Feed this pixelated color map as conditioning alongside the text prompt

**Apple Silicon status:**
- Uses `diffusers` `StableDiffusionAdapterPipeline`
- Diffusers supports MPS device for SD1.5
- The adapter itself is tiny (17M params), so memory is not a concern
- SD1.5 + T2I-Adapter Color: ~4-5 GB unified memory total

**Limitation:** Trained for SD1.4, works with SD1.5 but not extensively tested on all checkpoints. No SDXL or FLUX version exists.

**Verdict:** Viable for SD1.5 pipeline. Very lightweight. Good for enforcing a color mood/atmosphere.

### 1.3 IP-Adapter with Color Reference Images

**Concept:** Feed a reference image that has the desired color mood. IP-Adapter extracts style/color features and injects them into generation.

**For SD1.5:**
- `h94/IP-Adapter` on HuggingFace
- IP-Adapter-Plus extracts more detailed style info including colors
- Works via diffusers on MPS

**For FLUX:**
- `InstantX/FLUX.1-dev-IP-Adapter` on HuggingFace
- Injects image embeddings as key-value pairs in attention layers
- Style transfer includes color schemes

**Apple Silicon status:**
- SD1.5 + IP-Adapter: ~6-7 GB unified memory (SD1.5 + CLIP image encoder + adapter)
- FLUX + IP-Adapter: ~20-24 GB unified memory (FLUX model + image encoder + adapter)
- Both work on MPS through diffusers

**Limitation:** IP-Adapter transfers the full style (textures, composition, mood), not just colors. You cannot isolate "only the color palette." Best for "make it look like this reference image" rather than "use exactly these 5 colors."

**Verdict:** Overkill if you only want color control. Good if you want full style-matching from a reference image.

### 1.4 Palette Extraction Tools (Image -> Dominant Colors)

For extracting a palette from an existing image to feed back into generation:

**color-thief-py** (recommended):
- GitHub: https://github.com/fengsp/color-thief-py
- Pure Python + Pillow, no GPU needed
- `get_palette(color_count=5)` returns list of (R,G,B) tuples
- Convert to hex, feed into FLUX.2 prompt

**colorgram.py:**
- GitHub: https://github.com/obskyr/colorgram.py
- Similar functionality, also returns proportions (how much of the image each color occupies)

**DIY with Pillow:**
- Resize image to 8x8, read pixel values -> instant 64-color palette
- K-means clustering on pixel values for N dominant colors (scikit-learn)

**Pipeline idea:**
```
Input image -> color-thief-py -> 5 hex codes -> FLUX.2 prompt with color_palette
```

All of these are CPU-only, no GPU, no compatibility issues on macOS.

### 1.5 Recommendation for Implementation

| Approach | Works Now? | Additional Downloads | Memory Overhead | Color Control Quality |
|---|---|---|---|---|
| FLUX.2 hex prompts | YES | None | None | Medium (soft guidance) |
| T2I-Adapter Color + SD1.5 | Yes (needs adapter) | ~70 MB | ~17M params | Good (spatial color map) |
| IP-Adapter color ref | Yes (needs adapter) | ~100-500 MB | ~1-2 GB | Medium (style, not just color) |
| Palette extraction | YES | pip install | None (CPU) | N/A (extraction only) |

**Recommended approach:**
1. **Quick win:** Add hex color palette support to FLUX.2 prompts. Zero cost.
2. **Better control:** T2I-Adapter Color with SD1.5 for cases where hex prompts are too imprecise.
3. **Extraction:** color-thief-py for pulling palettes from reference images.

---

## 2. ControlNet: Line Art Extraction

### 2.1 Canny Edge Detection (OpenCV, No Model)

**What:** Classical edge detection algorithm. Not learned, purely mathematical.

**Implementation:** `cv2.Canny(image, low_threshold, high_threshold)`

**Pros:**
- Zero model downloads, zero GPU usage
- Instant, deterministic
- Works perfectly on macOS
- Adjustable thresholds for fine/coarse edges

**Cons:**
- Detects ALL edges including noise, textures, shadows
- No semantic understanding (cannot distinguish important contours from texture)
- Produces noisy lineart for photographic images
- Not great for anime/illustration style

**Memory:** Negligible (CPU-only, OpenCV)

**Best for:** Technical/architectural images, simple clean images, quick preprocessing.

### 2.2 Informative Drawings (Learned Lineart)

**GitHub:** https://github.com/carolineec/informative-drawings
**Paper:** "Learning to Generate Line Drawings that Convey Geometry and Semantics"

**What:** Trained model that extracts semantically meaningful line drawings. Understands which edges are important for conveying the geometry and meaning of a scene.

**Architecture:** PyTorch model, unpaired training on photographs + line drawings.

**Apple Silicon status:**
- PyTorch-based, should work on MPS with `PYTORCH_ENABLE_MPS_FALLBACK=1`
- Original repo uses PyTorch 1.7.1 -- needs updating for modern PyTorch
- Small model, CPU inference is fast enough for single images

**Quality:** Produces clean, artist-like line drawings. Much better semantic understanding than Canny. Good for realistic/photographic content.

**Limitation:** Older project (2022), not actively maintained. May need dependency updates.

### 2.3 Anime2Sketch

**GitHub:** https://github.com/Mukosame/Anime2Sketch

**What:** Sketch extractor specifically for anime and illustration styles.

**Apple Silicon status:**
- PyTorch-based, MPS compatible with fallback flag
- Small model, CPU inference viable

**Best for:** Anime/manga content specifically. Not ideal for photographic content.

### 2.4 AnyLine Preprocessor (State of the Art)

**GitHub:** https://github.com/TheMistoAI/ComfyUI-Anyline
**Based on:** TEED (Tiny and Efficient Model for Edge Detection Generalization)

**What:** Fast, accurate line detection that handles any type of input image. Extracts object edges, details, and even text content. Combines the best of classical edge detection and learned approaches.

**Apple Silicon status:**
- Available as ComfyUI node
- The underlying TEED model is small and PyTorch-based
- Should work on MPS

**Quality:** Best general-purpose lineart extractor as of 2025. Better edge fidelity than Canny, better detail than pure learned approaches. Handles both photographic and illustration content.

**Limitation:** Primarily distributed as ComfyUI extension. Using it standalone requires extracting the model code from the ComfyUI wrapper.

### 2.5 ControlNet Lineart Models

#### For SD1.5 (Proven, Stable)

**Model:** `lllyasviel/control_v11p_sd15_lineart`
**HuggingFace:** https://huggingface.co/lllyasviel/control_v11p_sd15_lineart
**Preprocessor:** `LineartDetector` from `lllyasviel/Annotators`

**How it works:**
1. Extract lineart from input image (using any of the methods above)
2. Feed lineart + text prompt to ControlNet-conditioned SD1.5
3. Generate image that follows the line structure

**Apple Silicon status:**
- Works via diffusers on MPS
- SD1.5 + ControlNet: ~6-8 GB unified memory
- Lineart preprocessor runs separately (small model, CPU or MPS)
- Known to work. Some users report OOM with complex ControlNets on 16GB machines, but lineart is one of the lighter ones.

**Also available:**
- `lllyasviel/control_v11p_sd15_lineart` -- realistic lineart
- `ControlNet-1-1-preview/control_v11p_sd15_lineart` -- preview version
- `refiners/sd15.controlnet.lineart` -- refiners library format

#### For FLUX.1 (Newer, Less Tested)

**Models:**
- `promeai/FLUX.1-controlnet-lineart-promeai` -- dedicated lineart ControlNet for FLUX.1-dev
- `TheMistoAI/MistoLine_Flux.dev` -- versatile line ControlNet (~1.4B params)
- `InstantX/FLUX.1-dev-Controlnet-Union` -- multi-condition ControlNet (includes lineart mode)

**Apple Silicon status:**
- FLUX.1-dev + ControlNet: ~24-28 GB unified memory (needs 32GB+ Mac)
- Works via diffusers `FluxControlNetPipeline`
- MPS acceleration supported but slower than CUDA
- FP8 quantization NOT supported on MPS (still a known limitation)

**For FLUX.2:**
- `alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union` -- ControlNet Union for FLUX.2-dev
- Very new, less community testing
- No FLUX.2-Klein-specific ControlNet exists yet (as of March 2026)

### 2.6 Using Lineart as FLUX.2 Reference Image

FLUX.2 supports multi-reference image conditioning (up to 4 images). You can:
1. Extract lineart from source image
2. Pass it as a reference image to FLUX.2
3. The model will use it as structural guidance

**This is NOT the same as ControlNet.** Reference images influence style/composition loosely, while ControlNet enforces spatial structure precisely. For tight line-following, ControlNet is necessary. For "inspired by these lines," reference images work.

### 2.7 Quality Comparison

| Method | Semantic Understanding | Clean Lines | Speed | Best For |
|---|---|---|---|---|
| Canny (OpenCV) | None | Noisy | Instant | Technical images, quick tests |
| Informative Drawings | High | Very clean | ~1-2s/img | Photographic content |
| Anime2Sketch | Medium (anime-specific) | Clean | ~1s/img | Anime/illustration |
| AnyLine (TEED) | High | Very clean | Fast | Everything (best general) |
| LineartDetector (ControlNet preprocessor) | High | Clean | ~1-2s/img | Preprocessing for ControlNet |

### 2.8 Recommendation for Implementation

**Lineart Extraction Engine (standalone):**
1. **Primary:** AnyLine/TEED -- best quality, fast, handles all content types
2. **Fallback:** Canny (OpenCV) -- zero dependencies, always works
3. **Anime content:** Anime2Sketch if needed

**Lineart-conditioned Generation:**
1. **SD1.5 + ControlNet lineart** -- proven, stable, ~6-8 GB memory, works on 16GB Macs
2. **FLUX.1 + ControlNet lineart** -- better quality, needs 32GB+ Mac, less tested on MPS
3. **FLUX.2 + reference image** -- loose structural guidance only, not precise line-following

**Suggested pipeline:**
```
Input image --> AnyLine extraction --> lineart.png (standalone output)
                                   --> SD1.5 + ControlNet lineart + prompt --> new image
                                   --> (optional) FLUX.2 reference conditioning --> new image
```

---

## Known Issues on macOS Apple Silicon

1. **FP8 not supported on MPS** -- quantized FLUX models must use FP16 or BF16, increasing memory usage
2. **Some PyTorch ops fall back to CPU on MPS** -- set `PYTORCH_ENABLE_MPS_FALLBACK=1`
3. **Diffusers MPS memory management** -- add `torch.mps.empty_cache()` after generation to free memory
4. **ControlNet + SDXL on MPS** -- known OOM issues on 16GB machines. SD1.5 ControlNet is safer.
5. **mflux (MLX native FLUX)** -- only supports Canny ControlNet currently, no lineart ControlNet
6. **FLUX.2 Klein ControlNet** -- no Klein-specific ControlNet models exist yet. ControlNets are trained for FLUX.1-dev or FLUX.2-dev (12B), not Klein (4B). Using dev-trained ControlNets with Klein may not work correctly.

---

## Memory Budget Summary (Apple Silicon Unified Memory)

| Configuration | Memory Required | 16GB Mac | 32GB Mac | 64GB Mac |
|---|---|---|---|---|
| SD1.5 + T2I-Adapter Color | ~5 GB | OK | OK | OK |
| SD1.5 + ControlNet Lineart | ~6-8 GB | OK | OK | OK |
| SD1.5 + ControlNet + IP-Adapter | ~8-10 GB | Tight | OK | OK |
| FLUX.2 Klein (4B, FP16) | ~10-13 GB | Tight | OK | OK |
| FLUX.2 Klein + hex color prompts | ~10-13 GB | Tight | OK | OK |
| FLUX.1-dev + ControlNet | ~24-28 GB | NO | Tight | OK |
| FLUX.1-dev + IP-Adapter | ~20-24 GB | NO | Tight | OK |

---

## Key Links

- T2I-Adapter Color: https://huggingface.co/TencentARC/t2iadapter_color_sd14v1
- T2I-Adapter repo: https://github.com/TencentARC/T2I-Adapter
- ControlNet SD1.5 Lineart: https://huggingface.co/lllyasviel/control_v11p_sd15_lineart
- FLUX.1 ControlNet Lineart: https://huggingface.co/promeai/FLUX.1-controlnet-lineart-promeai
- MistoLine (FLUX.1): https://huggingface.co/TheMistoAI/MistoLine_Flux.dev
- AnyLine preprocessor: https://github.com/TheMistoAI/ComfyUI-Anyline
- FLUX.2 ControlNet Union: https://huggingface.co/alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union
- Informative Drawings: https://github.com/carolineec/informative-drawings
- Anime2Sketch: https://github.com/Mukosame/Anime2Sketch
- color-thief-py: https://github.com/fengsp/color-thief-py
- mflux (MLX FLUX): https://github.com/filipstrand/mflux
- IP-Adapter (diffusers): https://huggingface.co/docs/diffusers/en/using-diffusers/ip_adapter
- FLUX.2 Prompting Guide: https://docs.bfl.ai/guides/prompting_guide_flux2
- Diffusers MPS Guide: https://huggingface.co/docs/diffusers/en/optimization/mps
