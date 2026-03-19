"""Test: Style Transfer — SD1.5 generates style reference, FLUX.2 transfers it onto content."""

import sys
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
JOHANNES = ASSETS_DIR / "johannes.png"


def register(suite):
    out = suite.out_dir

    # ── Step 1: Generate style references with SD1.5 ───────────────────────

    suite.add(
        name="Style: SD1.5 oil painting reference",
        cmd=[
            sys.executable, "generate.py", "image",
            "--engine", "sd1.5",
            "--model", "dreamshaper",
            "--no-lora",
            "-p", "oil painting with thick impasto brushstrokes, portrait of a nobleman in a dark room, Rembrandt lighting, golden frame",
            "-W", "512", "-H", "512",
            "--seed", "42",
            "-o", str(out / "style_oil.png"),
        ],
        output=out / "style_oil.png",
    )

    suite.add(
        name="Style: SD1.5 watercolor reference",
        cmd=[
            sys.executable, "generate.py", "image",
            "--engine", "sd1.5",
            "--model", "dreamshaper",
            "--no-lora",
            "-p", "delicate watercolor painting, soft washes of color bleeding into wet paper, a garden scene with flowers, pastel tones",
            "-W", "512", "-H", "512",
            "--seed", "73",
            "-o", str(out / "style_watercolor.png"),
        ],
        output=out / "style_watercolor.png",
    )

    suite.add(
        name="Style: SD1.5 anime reference",
        cmd=[
            sys.executable, "generate.py", "image",
            "--engine", "sd1.5",
            "--model", "dreamshaper",
            "--no-lora",
            "-p", "anime style illustration, vibrant cel shading, a hero character with dramatic pose, bold outlines, colorful background",
            "-W", "512", "-H", "512",
            "--seed", "99",
            "-o", str(out / "style_anime.png"),
        ],
        output=out / "style_anime.png",
    )

    # ── Step 2: Transfer styles onto johannes with FLUX.2 ──────────────────

    suite.add(
        name="Style: johannes as oil painting",
        cmd=[
            sys.executable, "generate.py", "image",
            "--engine", "flux.2",
            "--model", "4b-distilled",
            "--images", str(JOHANNES), str(out / "style_oil.png"),
            "-p", "turn image 1 into a painting like image 2",
            "-W", "504", "-H", "504",
            "-o", str(out / "transfer_oil.png"),
        ],
        output=out / "transfer_oil.png",
    )

    suite.add(
        name="Style: johannes as watercolor",
        cmd=[
            sys.executable, "generate.py", "image",
            "--engine", "flux.2",
            "--model", "4b-distilled",
            "--images", str(JOHANNES), str(out / "style_watercolor.png"),
            "-p", "turn image 1 into a watercolor artwork in the style of image 2",
            "-W", "504", "-H", "504",
            "-o", str(out / "transfer_watercolor.png"),
        ],
        output=out / "transfer_watercolor.png",
    )

    suite.add(
        name="Style: johannes as anime",
        cmd=[
            sys.executable, "generate.py", "image",
            "--engine", "flux.2",
            "--model", "4b-distilled",
            "--images", str(JOHANNES), str(out / "style_anime.png"),
            "-p", "transform image 1 into the art style of image 2",
            "-W", "504", "-H", "504",
            "-o", str(out / "transfer_anime.png"),
        ],
        output=out / "transfer_anime.png",
    )
