"""Normal map estimation using Marigold-Normals v1.1.

Usage:
  python generate.py image --engine normalmap --images photo.png -o normals.png
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_MODELS_DIR = _SCRIPT_DIR / "models"
os.environ["HF_HOME"] = str(_MODELS_DIR)

_MODEL_ID = "prs-eth/marigold-normals-v1-1"


def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    if "--list-models" in sys.argv:
        print(json.dumps([{"model": "", "notice": "single model"}]))
        return

    parser = argparse.ArgumentParser(description="Marigold normal map estimation")
    parser.add_argument("--images", nargs="+", required=True, help="Input image(s)")
    parser.add_argument("-o", "--output", default="normalmap.png", help="Output file path")
    parser.add_argument("--steps", type=int, default=4, help="Denoising steps (default: 4)")
    args = parser.parse_args()

    import numpy as np
    from diffusers import MarigoldNormalsPipeline
    from PIL import Image

    device = _device()
    # MPS produces NaN with fp16 — use float32 for stability
    dtype = torch.float32

    print("Loading Marigold-Normals v1.1 …", file=sys.stderr)
    pipe = MarigoldNormalsPipeline.from_pretrained(
        _MODEL_ID,
        torch_dtype=dtype,
        use_safetensors=True,
    ).to(device)
    pipe.enable_attention_slicing()

    outputs = []
    for img_path in args.images:
        print(f"  Processing: {img_path}", file=sys.stderr)

        img = Image.open(img_path)
        result = pipe(img, num_inference_steps=args.steps)
        normals = result.prediction[0]  # shape: (H, W, 3), range [-1, 1]
        rgb = ((normals + 1) * 0.5 * 255).clip(0, 255).astype(np.uint8)
        normal_pil = Image.fromarray(rgb)

        if len(args.images) == 1:
            out_path = Path(args.output)
        else:
            stem = Path(img_path).stem
            out_path = Path(args.output).parent / f"{stem}_normalmap.png"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        normal_pil.save(str(out_path))
        print(f"  Saved: {out_path}", file=sys.stderr)
        outputs.append(str(out_path))

    print(json.dumps(outputs))


if __name__ == "__main__":
    main()
