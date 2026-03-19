"""Test: Upscale — Real-ESRGAN image upscaling (4x, 2x, anime)."""

import sys
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
REF_IMAGE = ASSETS_DIR / "johannes.png"
LIVINGROOM = ASSETS_DIR / "livingroom.png"


def register(suite):
    out = suite.out_dir

    suite.add(
        name="Upscale: johannes 4x",
        cmd=[
            sys.executable, "generate.py", "image",
            "--engine", "upscale",
            "--images", str(REF_IMAGE),
            "--model", "4x",
            "-o", str(out / "johannes_4x.png"),
        ],
        output=out / "johannes_4x.png",
    )

    suite.add(
        name="Upscale: johannes 2x",
        cmd=[
            sys.executable, "generate.py", "image",
            "--engine", "upscale",
            "--images", str(REF_IMAGE),
            "--model", "2x",
            "-o", str(out / "johannes_2x.png"),
        ],
        output=out / "johannes_2x.png",
    )

    suite.add(
        name="Upscale: livingroom 4x",
        cmd=[
            sys.executable, "generate.py", "image",
            "--engine", "upscale",
            "--images", str(LIVINGROOM),
            "--model", "4x",
            "-o", str(out / "livingroom_4x.png"),
        ],
        output=out / "livingroom_4x.png",
    )

    suite.add(
        name="Upscale: johannes anime",
        cmd=[
            sys.executable, "generate.py", "image",
            "--engine", "upscale",
            "--images", str(REF_IMAGE),
            "--model", "anime",
            "-o", str(out / "johannes_anime.png"),
        ],
        output=out / "johannes_anime.png",
    )
