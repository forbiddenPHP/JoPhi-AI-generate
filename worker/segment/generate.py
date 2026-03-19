"""Image segmentation — foreground/background separation.

Uses BiRefNet via rembg (MIT license). CoreML acceleration on Apple Silicon.

Usage:
  python generate.py image --engine segment --images photo.png -o foreground.png
  python generate.py image --engine segment --images photo.png --output-layer background -o bg.png
  python generate.py image --engine segment --images photo.png --output-layer both -o output/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_SCRIPT_DIR = Path(__file__).resolve().parent
_MODELS_DIR = _SCRIPT_DIR / "models"
_MODELS_DIR.mkdir(exist_ok=True)


def _segment(img):
    """Remove background using BiRefNet (rembg). Returns foreground RGBA + background RGBA."""
    from rembg import new_session, remove

    print("Segmenting (BiRefNet) …", file=sys.stderr)
    session = new_session("birefnet-general")
    foreground = remove(img, session=session)  # RGBA

    # Create background: invert the alpha mask
    fg_np = np.array(foreground)
    alpha = fg_np[:, :, 3]
    inv_alpha = 255 - alpha

    img_np = np.array(img.convert("RGB"))
    bg_rgba = np.concatenate([img_np, inv_alpha[:, :, np.newaxis]], axis=2)
    background = Image.fromarray(bg_rgba, "RGBA")

    return foreground, background


def main():
    if "--list-models" in sys.argv:
        print(json.dumps([{"model": "", "notice": "single model"}]))
        return

    parser = argparse.ArgumentParser(description="Image segmentation (foreground/background)")
    parser.add_argument("--images", nargs="+", required=True, help="Input image(s)")
    parser.add_argument("-o", "--output", default="segment.png", help="Output file path or directory")
    parser.add_argument("--output-layer", default="foreground", dest="output_layer",
                        choices=["foreground", "background", "both"],
                        help="What to output: foreground (default), background, both")
    args = parser.parse_args()

    outputs = []
    for img_path in args.images:
        print(f"  Processing: {img_path}", file=sys.stderr)
        img = Image.open(img_path).convert("RGB")

        foreground, background = _segment(img)
        stem = Path(img_path).stem

        if args.output_layer == "both":
            out_dir = Path(args.output)
            out_dir.mkdir(parents=True, exist_ok=True)
            fg_path = out_dir / f"{stem}_foreground.png"
            bg_path = out_dir / f"{stem}_background.png"
            foreground.save(str(fg_path))
            background.save(str(bg_path))
            print(f"  Saved: {fg_path}", file=sys.stderr)
            print(f"  Saved: {bg_path}", file=sys.stderr)
            outputs.extend([str(fg_path), str(bg_path)])
        else:
            result = foreground if args.output_layer == "foreground" else background

            if len(args.images) == 1:
                out_path = Path(args.output)
            else:
                suffix = "_foreground" if args.output_layer == "foreground" else "_background"
                out_path = Path(args.output).parent / f"{stem}{suffix}.png"

            out_path.parent.mkdir(parents=True, exist_ok=True)
            result.save(str(out_path))
            print(f"  Saved: {out_path}", file=sys.stderr)
            outputs.append(str(out_path))

    print(json.dumps(outputs))


if __name__ == "__main__":
    main()
