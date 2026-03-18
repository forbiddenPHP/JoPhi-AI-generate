"""FLUX.2 image generation wrapper for generate.py.

Usage:
  python generate.py image --engine flux.2 --model 4b -p "a cat" -o cat.png
  python generate.py image --engine flux.2 --model 4b -p "transform background" --images ref.png -o out.png
"""

import argparse
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
    width = args.width
    height = args.height

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Model: {model_name}", file=sys.stderr)
    print(f"Device: {device}", file=sys.stderr)
    print(f"Seed: {seed}", file=sys.stderr)
    print(f"Steps: {num_steps}, Guidance: {guidance}", file=sys.stderr)
    print(f"Dimensions: {width}x{height}", file=sys.stderr)

    # Load models
    print("Loading text encoder ...", file=sys.stderr)
    text_encoder = load_text_encoder(model_name, device=device)
    text_encoder.eval()

    print("Loading flow model ...", file=sys.stderr)
    model = load_flow_model(model_name, device=device)
    model.eval()

    print("Loading autoencoder ...", file=sys.stderr)
    ae = load_ae(model_name, device=device)
    ae.eval()

    # Reference images
    ref_tokens = None
    ref_ids = None
    if args.images:
        img_ctx = [Image.open(p) for p in args.images]
        print(f"Encoding {len(img_ctx)} reference image(s) ...", file=sys.stderr)
        with torch.no_grad():
            ref_tokens, ref_ids = encode_image_refs(ae, img_ctx)

    with torch.no_grad():
        # Encode text
        if model_info["guidance_distilled"]:
            ctx = text_encoder([args.prompt]).to(dtype)
        else:
            ctx_empty = text_encoder([""]).to(dtype)
            ctx_prompt = text_encoder([args.prompt]).to(dtype)
            ctx = torch.cat([ctx_empty, ctx_prompt], dim=0)
        ctx, ctx_ids = batched_prc_txt(ctx)

        # Create noise
        shape = (1, 128, height // 16, width // 16)
        generator = torch.Generator(device=device).manual_seed(seed)
        randn = torch.randn(shape, generator=generator, dtype=dtype, device=device)
        x, x_ids = batched_prc_img(randn)

        # Denoise
        timesteps = get_schedule(num_steps, x.shape[1])
        print("Generating ...", file=sys.stderr)

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

        # Decode
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
