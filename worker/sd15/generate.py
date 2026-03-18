"""Stable Diffusion 1.5 image generation wrapper.

Usage:
  python generate.py image --engine sd1.5 -p "a muscular man with beard" -o out.png
  python generate.py image --engine sd1.5 --lora add_detail:1.2 -p "..." -o out.png
  python generate.py image --engine sd1.5 --lora add_detail:1.2 --lora other:0.8 -p "..." -o out.png
  python generate.py image --engine sd1.5 --no-lora -p "..." -o out.png
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

_SCRIPT_DIR = Path(__file__).resolve().parent
_MODELS_DIR = _SCRIPT_DIR / "models"
_LORAS_DIR = _SCRIPT_DIR / "loras"

# Model registry: --model name → checkpoint filename
MODEL_REGISTRY = {
    "mm": {
        "filename": "maturemalemix_v14.safetensors",
        "description": "MatureMaleMix v1.4 — realistic/2.5D mature male characters",
        "default_loras": [("add_detail", 1.2)],
        "recommended_negative": "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark",
    },
    "dreamshaper": {
        "filename": "dreamshaper_8.safetensors",
        "description": "DreamShaper 8 — versatile artistic/photorealistic generation",
        "recommended_negative": "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark",
    },
}


def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _emit(msg, level="info"):
    prefix = {"info": "●", "warning": "⚠", "error": "✗"}.get(level, "·")
    print(f"\n  {prefix} {msg}", file=sys.stderr)


def _parse_lora(spec):
    """Parse 'name:intensity' or 'name' → (name, intensity)."""
    if ":" in spec:
        name, intensity = spec.rsplit(":", 1)
        return name, float(intensity)
    return spec, 1.0


def _resolve_lora_path(name):
    """Find LoRA file by name, auto-appending .safetensors if needed."""
    path = _LORAS_DIR / name
    if path.exists():
        return path
    path = _LORAS_DIR / f"{name}.safetensors"
    if path.exists():
        return path
    return None


def main():
    parser = argparse.ArgumentParser(description="SD 1.5 image generation")
    parser.add_argument("--model", default="mm", choices=list(MODEL_REGISTRY.keys()),
                        help="Checkpoint to use (default: mm)")
    parser.add_argument("-p", "--prompt", required=True, help="Text prompt")
    parser.add_argument("--negative-prompt", default=None, help="Negative prompt")
    parser.add_argument("-o", "--output", default="image.png", help="Output file path")
    parser.add_argument("-W", "--width", type=int, default=1280, help="Image width (default: 1280)")
    parser.add_argument("-H", "--height", type=int, default=768, help="Image height (default: 768)")
    parser.add_argument("--steps", type=int, default=20, help="Inference steps (default: 20)")
    parser.add_argument("--cfg", type=float, default=3.5, help="Guidance scale (default: 3.5)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--lora", action="append", default=None,
                        help="LoRA to apply: name:intensity (e.g. add_detail:1.2). Can be repeated.")
    parser.add_argument("--no-lora", action="store_true", help="Disable default LoRA(s)")
    args = parser.parse_args()

    info = MODEL_REGISTRY[args.model]
    device = _device()
    dtype = torch.float32  # MPS: float32 required

    # Seed
    seed = args.seed if args.seed is not None else torch.randint(0, 2**32, (1,)).item()
    generator = torch.Generator(device="cpu").manual_seed(seed)

    # Negative prompt
    negative_prompt = args.negative_prompt
    if negative_prompt is None:
        negative_prompt = info.get("recommended_negative", "")

    # Resolve LoRAs: explicit --lora overrides defaults
    lora_specs = []
    if args.no_lora:
        pass  # no LoRAs
    elif args.lora:
        lora_specs = [_parse_lora(s) for s in args.lora]
    else:
        lora_specs = list(info.get("default_loras", []))

    # Header
    _emit(f"Generating image (sd1.5/{args.model}) …")
    print(f"  · Model: {info['filename']}", file=sys.stderr)
    print(f"  · Device: {device}", file=sys.stderr)
    print(f"  · Seed: {seed}", file=sys.stderr)
    print(f"    Steps: {args.steps}, CFG: {args.cfg}", file=sys.stderr)
    print(f"    Dimensions: {args.width}x{args.height}", file=sys.stderr)

    # Load checkpoint
    checkpoint_path = _MODELS_DIR / info["filename"]
    if not checkpoint_path.exists():
        print(f"  ✗ ERROR: Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    _emit("Loading checkpoint …")
    pipe = StableDiffusionPipeline.from_single_file(
        str(checkpoint_path),
        torch_dtype=dtype,
        local_files_only=True,
    ).to(device)

    pipe.enable_attention_slicing()

    # Load LoRAs
    loaded_loras = []
    for name, intensity in lora_specs:
        lora_path = _resolve_lora_path(name)
        if lora_path:
            adapter_name = name.replace(".", "_")
            _emit(f"Loading LoRA: {name} (intensity: {intensity}) …")
            pipe.load_lora_weights(str(lora_path), adapter_name=adapter_name)
            loaded_loras.append((adapter_name, intensity))
        else:
            print(f"  ⚠ WARNING: LoRA not found: {name}", file=sys.stderr)

    # Set adapter weights if multiple LoRAs
    if len(loaded_loras) > 1:
        names = [n for n, _ in loaded_loras]
        weights = [w for _, w in loaded_loras]
        pipe.set_adapters(names, adapter_weights=weights)
        cross_attention_kwargs = {"scale": 1.0}
    elif len(loaded_loras) == 1:
        cross_attention_kwargs = {"scale": loaded_loras[0][1]}
    else:
        cross_attention_kwargs = None

    # Generate
    _emit("Generating …")

    result = pipe(
        prompt=args.prompt,
        negative_prompt=negative_prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        generator=generator,
        cross_attention_kwargs=cross_attention_kwargs,
    )

    image = result.images[0]

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(out_path))
    print(f"  · Saved: {out_path}", file=sys.stderr)

    print(json.dumps([str(out_path)]))


if __name__ == "__main__":
    main()
