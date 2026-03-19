"""Sketch/edge extraction using HED (Holistically-Nested Edge Detection).

Uses OpenCV DNN with the original HED Caffe model. No PyTorch required.

Usage:
  python generate.py image --engine sketch --images photo.png -o sketch.png
"""

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_MODELS_DIR = _SCRIPT_DIR / "models"
_MODELS_DIR.mkdir(exist_ok=True)

_PROTO_URL = "https://raw.githubusercontent.com/s9xie/hed/master/examples/hed/deploy.prototxt"
_MODEL_URL = "https://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel"
_PROTO_PATH = _MODELS_DIR / "deploy.prototxt"
_MODEL_PATH = _MODELS_DIR / "hed_pretrained_bsds.caffemodel"


class _CropLayer:
    """OpenCV DNN custom layer for HED center-cropping."""

    def __init__(self, params, blobs):
        self.x_start = 0
        self.x_end = 0
        self.y_start = 0
        self.y_end = 0

    def getMemoryShapes(self, inputs):
        input_shape, target_shape = inputs[0], inputs[1]
        batch, channels = input_shape[0], input_shape[1]
        h, w = target_shape[2], target_shape[3]

        self.y_start = (input_shape[2] - target_shape[2]) // 2
        self.x_start = (input_shape[3] - target_shape[3]) // 2
        self.y_end = self.y_start + h
        self.x_end = self.x_start + w

        return [[batch, channels, h, w]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.y_start:self.y_end, self.x_start:self.x_end]]


def _ensure_model():
    """Download HED model files if not present."""
    if not _PROTO_PATH.exists():
        print("  Downloading HED prototxt …", file=sys.stderr)
        urllib.request.urlretrieve(_PROTO_URL, str(_PROTO_PATH))
    if not _MODEL_PATH.exists():
        print("  Downloading HED model (~56 MB) …", file=sys.stderr)
        urllib.request.urlretrieve(_MODEL_URL, str(_MODEL_PATH))


def main():
    parser = argparse.ArgumentParser(description="HED sketch/edge extraction")
    parser.add_argument("--images", nargs="+", required=True, help="Input image(s)")
    parser.add_argument("-o", "--output", default="sketch.png", help="Output file path")
    args = parser.parse_args()

    _ensure_model()

    cv2.dnn_registerLayer("Crop", _CropLayer)
    net = cv2.dnn.readNetFromCaffe(str(_PROTO_PATH), str(_MODEL_PATH))

    print("Extracting sketch (HED) …", file=sys.stderr)

    outputs = []
    for img_path in args.images:
        print(f"  Processing: {img_path}", file=sys.stderr)

        img = cv2.imread(img_path)
        if img is None:
            print(f"  ERROR: Cannot read: {img_path}", file=sys.stderr)
            sys.exit(1)

        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(
            img, scalefactor=1.0, size=(w, h),
            mean=(104.00698793, 116.66876762, 122.67891434),
            swapRB=False, crop=False,
        )
        net.setInput(blob)
        result = net.forward()

        # Convert to uint8 edge image
        edges = result[0, 0]
        edges = (255 * edges).clip(0, 255).astype(np.uint8)

        if len(args.images) == 1:
            out_path = Path(args.output)
        else:
            stem = Path(img_path).stem
            out_path = Path(args.output).parent / f"{stem}_sketch.png"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), edges)
        print(f"  Saved: {out_path}", file=sys.stderr)
        outputs.append(str(out_path))

    print(json.dumps(outputs))


if __name__ == "__main__":
    main()
