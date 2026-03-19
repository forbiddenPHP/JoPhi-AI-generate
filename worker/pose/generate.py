"""DWPose pose estimation wrapper for generate.py.

Usage:
  python generate.py image --engine openpose --images person.png -o pose.png
  python generate.py image --engine openpose --images person.png --pose-mode body -o pose.png
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="USE_SYMLINKS")

# Store models alongside this script
_MODELS_DIR = Path(__file__).resolve().parent / "models"
_MODELS_DIR.mkdir(exist_ok=True)
os.environ["HF_HOME"] = str(_MODELS_DIR)
os.environ["USE_SYMLINKS"] = "False"

from dwpose.wholebody import Wholebody
from dwpose import DwposeDetector
from PIL import Image

# Local model paths (downloaded by setup.sh)
_DET_MODEL = "yolox_l.onnx"
_POSE_MODEL = "dw-ll_ucoco_384.onnx"


def _find_model(name):
    """Find model file in local models dir, fallback to HF download."""
    # Check HF cache structure
    hf_dir = _MODELS_DIR / "hub" / "models--yzd-v--DWPose"
    if hf_dir.exists():
        for onnx in hf_dir.rglob(name):
            return str(onnx)
    # Flat layout
    flat = _MODELS_DIR / name
    if flat.exists():
        return str(flat)
    # Fallback: download
    from huggingface_hub import hf_hub_download
    return hf_hub_download("yzd-v/DWPose", name)


def _load_detector():
    """Load ONNX models and create detector."""
    det_path = _find_model(_DET_MODEL)
    pose_path = _find_model(_POSE_MODEL)
    wb = Wholebody(det_model_path=det_path, pose_model_path=pose_path, torchscript_device="cpu")
    return DwposeDetector(wb)


def main():
    if "--list-models" in sys.argv:
        import json
        print(json.dumps([{"model": "", "notice": "single model"}]))
        return

    parser = argparse.ArgumentParser(description="DWPose pose estimation")
    parser.add_argument("--images", nargs="+", required=True, help="Input image(s)")
    parser.add_argument("-o", "--output", default="pose.png", help="Output file path")
    parser.add_argument("--mode", default="wholebody",
                        choices=["wholebody", "body", "bodyhand", "bodyface"],
                        help="Detection mode (default: wholebody)")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Detection resolution (default: 512)")
    args = parser.parse_args()

    include_body = True
    include_hand = args.mode in ("wholebody", "bodyhand")
    include_face = args.mode in ("wholebody", "bodyface")

    print("Loading DWPose models ...", file=sys.stderr)
    detector = _load_detector()

    outputs = []
    for img_path in args.images:
        img = Image.open(img_path)
        print(f"Processing: {img_path}", file=sys.stderr)

        result = detector(
            img,
            detect_resolution=args.resolution,
            include_body=include_body,
            include_hand=include_hand,
            include_face=include_face,
            output_type="pil",
        )

        if len(args.images) == 1:
            out_path = Path(args.output)
        else:
            stem = Path(img_path).stem
            out_path = Path(args.output).parent / f"{stem}_pose.png"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(str(out_path))
        print(f"Saved: {out_path}", file=sys.stderr)
        outputs.append(str(out_path))

    print(json.dumps(outputs))


if __name__ == "__main__":
    main()
