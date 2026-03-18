"""FLUX.2 image generation wrapper for generate.py.

Usage:
  python generate.py image --engine flux.2 --model 4b -p "a cat" -o cat.png
  python generate.py image --engine flux.2 --model 4b -p "transform background" --images ref.png -o out.png
"""

import argparse
import gc
import os
import random
import sys
from pathlib import Path

# Store models alongside this script, not in ~/.cache
_MODELS_DIR = Path(__file__).resolve().parent / "models"
_MODELS_DIR.mkdir(exist_ok=True)
os.environ["HF_HOME"] = str(_MODELS_DIR)

# Ensure HF token is available (for gated repos like 9B)
_DEFAULT_TOKEN = Path.home() / ".cache" / "huggingface" / "token"
if "HF_TOKEN" not in os.environ and _DEFAULT_TOKEN.exists():
    os.environ["HF_TOKEN"] = _DEFAULT_TOKEN.read_text().strip()

import torch
from einops import rearrange
from PIL import Image

from flux2.sampling import (
    batched_prc_img,
    batched_prc_txt,
    denoise,
    denoise_cached,
    denoise_cfg,
    encode_image_refs,
    get_schedule,
    scatter_ids,
)
from flux2.util import FLUX2_MODEL_INFO, load_ae, load_flow_model, load_text_encoder


def _pan_and_scan(img, target_w, target_h):
    """Upscale if needed, center-crop to target aspect ratio, then resize (Pan & Scan)."""
    src_w, src_h = img.size
    # Upscale so both sides are >= target
    scale = max(target_w / src_w, target_h / src_h)
    if scale > 1:
        img = img.resize((int(src_w * scale), int(src_h * scale)), Image.LANCZOS)
        src_w, src_h = img.size
    # Center-crop to target ratio
    target_ratio = target_w / target_h
    src_ratio = src_w / src_h
    if src_ratio > target_ratio:
        new_w = int(src_h * target_ratio)
        left = (src_w - new_w) // 2
        img = img.crop((left, 0, left + new_w, src_h))
    elif src_ratio < target_ratio:
        new_h = int(src_w / target_ratio)
        top = (src_h - new_h) // 2
        img = img.crop((0, top, src_w, top + new_h))
    return img.resize((target_w, target_h), Image.LANCZOS)


def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _dtype(device: torch.device):
    # MPS doesn't support bfloat16; use float32 for numerical stability
    if device.type == "mps":
        return torch.float32
    return torch.bfloat16


def _free_memory() -> int:
    """Return free RAM in bytes."""
    import psutil
    return psutil.virtual_memory().available


def _memory_budget() -> int:
    """Return usable memory: free RAM minus 1/9 safety margin."""
    free = _free_memory()
    return free - free // 9


def _check_memory(label: str, needed_estimate: int):
    """Check if enough memory is available before a step. Exit if not."""
    budget = _memory_budget()
    if needed_estimate > budget:
        free_gb = _free_memory() / (1024**3)
        need_gb = needed_estimate / (1024**3)
        print(f"ERROR: Not enough memory for '{label}'. "
              f"Need ~{need_gb:.1f} GB, have {free_gb:.1f} GB free.",
              file=sys.stderr)
        sys.exit(1)


def _auto_chunk_dimensions(width: int, height: int, dtype: torch.dtype):
    """If requested image exceeds memory budget, reduce dimensions proportionally."""
    budget = _memory_budget()
    bytes_per_pixel = 200 if dtype == torch.float32 else 100
    max_pixels = budget // bytes_per_pixel
    requested = width * height
    if requested <= max_pixels:
        return width, height
    scale = (max_pixels / requested) ** 0.5
    new_w = int(width * scale) // 16 * 16
    new_h = int(height * scale) // 16 * 16
    print(f"WARNING: {width}x{height} exceeds memory budget. "
          f"Reducing to {new_w}x{new_h}", file=sys.stderr)
    return new_w, new_h


def main():
    parser = argparse.ArgumentParser(description="FLUX.2 image generation")
    parser.add_argument("--model", required=True, help="Model name (e.g. flux.2-klein-4b)")
    parser.add_argument("--prompt", "-p", required=True, help="Text prompt")
    parser.add_argument("-o", "--output", default="image.png", help="Output file path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--width", "-W", type=int, default=1360, help="Image width")
    parser.add_argument("--height", "-H", type=int, default=768, help="Image height")
    parser.add_argument("--steps", type=int, default=None, help="Inference steps")
    parser.add_argument("--guidance", type=float, default=None, help="Guidance scale")
    parser.add_argument("--images", nargs="+", default=None, help="Reference image path(s)")
    parser.add_argument("--depth", default=None, help="Depth map image for structural conditioning")
    parser.add_argument("--depth-strength", type=float, default=0.5,
                        help="Depth conditioning strength 0.0-1.0 (default: 0.5)")
    args = parser.parse_args()

    model_name = args.model.lower()
    if model_name not in FLUX2_MODEL_INFO:
        print(f"ERROR: Unknown model '{model_name}'", file=sys.stderr)
        print(f"Available: {', '.join(FLUX2_MODEL_INFO.keys())}", file=sys.stderr)
        sys.exit(1)

    model_info = FLUX2_MODEL_INFO[model_name]
    device = _device()
    dtype = _dtype(device)

    # Steps and guidance from model defaults or user override
    defaults = model_info.get("defaults", {})
    num_steps = args.steps if args.steps is not None else defaults.get("num_steps", 50)
    guidance = args.guidance if args.guidance is not None else defaults.get("guidance", 4.0)

    seed = args.seed if args.seed is not None else random.randrange(2**31)
    width, height = _auto_chunk_dimensions(args.width, args.height, dtype)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Model: {model_name}", file=sys.stderr)
    print(f"Device: {device}", file=sys.stderr)
    print(f"Seed: {seed}", file=sys.stderr)
    print(f"Steps: {num_steps}, Guidance: {guidance}", file=sys.stderr)
    print(f"Dimensions: {width}x{height}", file=sys.stderr)

    import gc

    # Step 1: Load autoencoder (small, ~500MB, needed for ref images + decode)
    print("Loading autoencoder …", file=sys.stderr)
    ae = load_ae(model_name, device=device)
    ae.eval()

    # Step 2: Encode reference images (if any) — depth map is added as first reference
    ref_tokens = None
    ref_ids = None
    all_images = list(args.images or [])
    if args.depth:
        all_images.insert(0, args.depth)
    if all_images:
        img_ctx = [Image.open(p) for p in all_images]
        # Pan & Scan depth map to target aspect ratio + dimensions
        if args.depth:
            img_ctx[0] = _pan_and_scan(img_ctx[0], width, height)
        print(f"Encoding {len(img_ctx)} reference image(s) …", file=sys.stderr)
        with torch.no_grad():
            ref_tokens, ref_ids = encode_image_refs(ae, img_ctx)

    # Step 3: Load text encoder, encode prompt, then free it
    print("Loading text encoder …", file=sys.stderr)
    text_encoder = load_text_encoder(model_name, device=device)
    text_encoder.eval()

    with torch.no_grad():
        print("Encoding prompt …", file=sys.stderr)
        if model_info["guidance_distilled"]:
            ctx = text_encoder([args.prompt]).to(dtype)
        else:
            ctx_empty = text_encoder([""]).to(dtype)
            ctx_prompt = text_encoder([args.prompt]).to(dtype)
            ctx = torch.cat([ctx_empty, ctx_prompt], dim=0)
        ctx, ctx_ids = batched_prc_txt(ctx)

    # Free text encoder to make room for flow model
    print("Freeing text encoder …", file=sys.stderr)
    del text_encoder
    gc.collect()
    if device.type == 'mps':
        torch.mps.empty_cache()

    # Step 4: Load flow model on MPS
    print("Loading flow model …", file=sys.stderr)
    model = load_flow_model(model_name, device=device)
    model.eval()

    with torch.no_grad():
        # Create noise latent
        shape = (1, 128, height // 16, width // 16)
        generator = torch.Generator(device=device).manual_seed(seed)
        randn = torch.randn(shape, generator=generator, dtype=dtype, device=device)

        x, x_ids = batched_prc_img(randn)

        # Denoise (attention auto-chunks based on available memory)
        timesteps = get_schedule(num_steps, x.shape[1])
        print("Denoising …", file=sys.stderr)

        if model_info["guidance_distilled"]:
            denoise_fn = (
                denoise_cached
                if (model_info.get("use_kv_cache") and ref_tokens is not None)
                else denoise
            )
            x = denoise_fn(
                model, x, x_ids, ctx, ctx_ids,
                timesteps=timesteps, guidance=guidance,
                img_cond_seq=ref_tokens, img_cond_seq_ids=ref_ids,
            )
        else:
            x = denoise_cfg(
                model, x, x_ids, ctx, ctx_ids,
                timesteps=timesteps, guidance=guidance,
                img_cond_seq=ref_tokens, img_cond_seq_ids=ref_ids,
            )

        # Free flow model before decoding
        del model
        gc.collect()
        if device.type == 'mps':
            torch.mps.empty_cache()

        # Decode
        print("Decoding …", file=sys.stderr)
        x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
        x = ae.decode(x).float()

    x = x.clamp(-1, 1)
    x = rearrange(x[0], "c h w -> h w c")
    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
    img.save(str(out_path), quality=95, subsampling=0)

    print(f"Saved: {out_path}", file=sys.stderr)
    # Output path to stdout (standard worker pattern)
    import json
    print(json.dumps([str(out_path)]))


if __name__ == "__main__":
    main()
