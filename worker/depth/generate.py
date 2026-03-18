"""Depth estimation wrapper using Depth Anything V2.

Usage:
  python generate.py image --engine depth --images photo.png -o depth.png
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import pipeline

_SCRIPT_DIR = Path(__file__).resolve().parent
_MODELS_DIR = _SCRIPT_DIR / "models"
os.environ["HF_HOME"] = str(_MODELS_DIR)

# Available models (small is fast, large is detailed)
_MODEL_MAP = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}


def _device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _emit(msg, level="info"):
    prefix = {"info": "●", "warning": "⚠", "error": "✗"}.get(level, "·")
    print(f"\n  {prefix} {msg}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Depth Anything V2 depth estimation")
    parser.add_argument("--images", nargs="+", required=True, help="Input image(s)")
    parser.add_argument("-o", "--output", default="depth.png", help="Output file path")
    parser.add_argument("--model", default="small", choices=list(_MODEL_MAP.keys()),
                        help="Model size: small (fast), large (detailed). Default: small")
    args = parser.parse_args()

    device = _device()
    model_id = _MODEL_MAP[args.model]

    _emit(f"Loading Depth Anything V2 ({args.model}) …")
    pipe = pipeline("depth-estimation", model=model_id, device=device)

    outputs = []
    for img_path in args.images:
        print(f"    Processing: {img_path}", file=sys.stderr)

        img = Image.open(img_path)
        result = pipe(img)
        depth_image = result["depth"]  # already a PIL Image (grayscale)

        if len(args.images) == 1:
            out_path = Path(args.output)
        else:
            stem = Path(img_path).stem
            out_path = Path(args.output).parent / f"{stem}_depth.png"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        depth_image.save(str(out_path))
        print(f"  · Saved: {out_path}", file=sys.stderr)
        outputs.append(str(out_path))

    print(json.dumps(outputs))


if __name__ == "__main__":
    main()
