"""Line art extraction using TEED (AnyLine) or Canny edge detection.

Usage:
  python generate.py image --engine lineart --images photo.png -o lines.png
  python generate.py image --engine lineart --images photo.png --model canny -o edges.png
"""

import argparse
import json
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_MODELS_DIR = _SCRIPT_DIR / "models"
os.environ["HF_HOME"] = str(_MODELS_DIR)


def _extract_teed(img):
    """Extract line art using TEED (Tiny Efficient Edge Detector)."""
    from controlnet_aux import TEEDdetector

    teed = TEEDdetector.from_pretrained("fal-ai/teed", filename="5_model.pth")
    return teed(img)


def _extract_canny(img, low=100, high=200):
    """Extract edges using Canny (OpenCV, no ML model)."""
    import cv2
    import numpy as np

    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low, high)
    from PIL import Image as PILImage
    return PILImage.fromarray(edges)


def main():
    parser = argparse.ArgumentParser(description="Line art extraction")
    parser.add_argument("--images", nargs="+", required=True, help="Input image(s)")
    parser.add_argument("-o", "--output", default="lineart.png", help="Output file path")
    parser.add_argument("--model", default="teed", choices=["teed", "canny"],
                        help="Extraction method: teed (default, learned), canny (classical)")
    args = parser.parse_args()

    from PIL import Image

    extract_fn = _extract_teed if args.model == "teed" else _extract_canny

    print(f"Extracting line art ({args.model}) …", file=sys.stderr)

    outputs = []
    for img_path in args.images:
        print(f"  Processing: {img_path}", file=sys.stderr)

        img = Image.open(img_path)
        result = extract_fn(img)

        if len(args.images) == 1:
            out_path = Path(args.output)
        else:
            stem = Path(img_path).stem
            out_path = Path(args.output).parent / f"{stem}_lineart.png"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(str(out_path))
        print(f"  Saved: {out_path}", file=sys.stderr)
        outputs.append(str(out_path))

    print(json.dumps(outputs))


if __name__ == "__main__":
    main()
