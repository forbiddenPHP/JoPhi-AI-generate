"""Test: OpenPose — DWPose pose estimation."""

import sys
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
REF_IMAGE = ASSETS_DIR / "johannes.png"


def register(suite):
    out = suite.out_dir

    suite.add(
        name="OpenPose: wholebody (johannes)",
        cmd=[
            sys.executable, "generate.py", "image",
            "--engine", "openpose",
            "--images", str(REF_IMAGE),
            "-o", str(out / "pose_wholebody.png"),
        ],
        output=out / "pose_wholebody.png",
    )

    suite.add(
        name="OpenPose: body only (johannes)",
        cmd=[
            sys.executable, "generate.py", "image",
            "--engine", "openpose",
            "--images", str(REF_IMAGE),
            "--pose-mode", "body",
            "-o", str(out / "pose_body.png"),
        ],
        output=out / "pose_body.png",
    )
