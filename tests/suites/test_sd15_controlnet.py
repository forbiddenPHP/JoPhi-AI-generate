"""Test: SD1.5 ControlNet — conditioned generation with depth, lineart, sketch, normalmap, pose."""

import sys
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
REF_IMAGE = ASSETS_DIR / "johannes.png"

PROMPT = "a 30-year-old man with messy brown hair, bright blue eyes and a confident smile, wearing glasses, warm studio lighting"
NEGATIVE = "lowres, bad anatomy, bad hands, text, error, worst quality, low quality"


def register(suite):
    out = suite.out_dir

    # ── Step 1: Extract conditioning maps (reuse from other tests if available) ──

    suite.add(
        name="SD1.5 CN: extract depth",
        cmd=[
            sys.executable, "generate.py", "image",
            "depth",
            "--images", str(REF_IMAGE),
            "-o", str(out / "cn_depth.png"),
        ],
        output=out / "cn_depth.png",
    )

    suite.add(
        name="SD1.5 CN: extract lineart (TEED)",
        cmd=[
            sys.executable, "generate.py", "image",
            "lineart",
            "--images", str(REF_IMAGE),
            "-o", str(out / "cn_lineart_teed.png"),
        ],
        output=out / "cn_lineart_teed.png",
    )

    suite.add(
        name="SD1.5 CN: extract lineart (Canny)",
        cmd=[
            sys.executable, "generate.py", "image",
            "lineart",
            "--model", "canny",
            "--images", str(REF_IMAGE),
            "-o", str(out / "cn_lineart_canny.png"),
        ],
        output=out / "cn_lineart_canny.png",
    )

    suite.add(
        name="SD1.5 CN: extract sketch",
        cmd=[
            sys.executable, "generate.py", "image",
            "sketch",
            "--images", str(REF_IMAGE),
            "-o", str(out / "cn_sketch.png"),
        ],
        output=out / "cn_sketch.png",
    )

    suite.add(
        name="SD1.5 CN: extract normalmap",
        cmd=[
            sys.executable, "generate.py", "image",
            "normalmap",
            "--images", str(REF_IMAGE),
            "-o", str(out / "cn_normalmap.png"),
        ],
        output=out / "cn_normalmap.png",
    )

    # ── Step 2: SD1.5 ControlNet generation ──────────────────────────────────

    suite.add(
        name="SD1.5 CN: depth-conditioned",
        cmd=[
            sys.executable, "generate.py", "image",
            "sd1.5",
            "--controlnet", "depth:" + str(out / "cn_depth.png"),
            "-p", PROMPT,
            "--negative-prompt", NEGATIVE,
            "-W", "512", "-H", "512",
            "--seed", "42",
            "-o", str(out / "sd15_cn_depth.png"),
        ],
        output=out / "sd15_cn_depth.png",
    )

    suite.add(
        name="SD1.5 CN: lineart-conditioned (TEED)",
        cmd=[
            sys.executable, "generate.py", "image",
            "sd1.5",
            "--controlnet", "lineart:" + str(out / "cn_lineart_teed.png"),
            "-p", PROMPT,
            "--negative-prompt", NEGATIVE,
            "-W", "512", "-H", "512",
            "--seed", "42",
            "-o", str(out / "sd15_cn_lineart_teed.png"),
        ],
        output=out / "sd15_cn_lineart_teed.png",
    )

    suite.add(
        name="SD1.5 CN: lineart-conditioned (Canny)",
        cmd=[
            sys.executable, "generate.py", "image",
            "sd1.5",
            "--controlnet", "lineart:" + str(out / "cn_lineart_canny.png"),
            "-p", PROMPT,
            "--negative-prompt", NEGATIVE,
            "-W", "512", "-H", "512",
            "--seed", "42",
            "-o", str(out / "sd15_cn_lineart_canny.png"),
        ],
        output=out / "sd15_cn_lineart_canny.png",
    )

    suite.add(
        name="SD1.5 CN: sketch-conditioned",
        cmd=[
            sys.executable, "generate.py", "image",
            "sd1.5",
            "--controlnet", "sketch:" + str(out / "cn_sketch.png"),
            "-p", PROMPT,
            "--negative-prompt", NEGATIVE,
            "-W", "512", "-H", "512",
            "--seed", "42",
            "-o", str(out / "sd15_cn_sketch.png"),
        ],
        output=out / "sd15_cn_sketch.png",
    )

    suite.add(
        name="SD1.5 CN: normalmap-conditioned",
        cmd=[
            sys.executable, "generate.py", "image",
            "sd1.5",
            "--controlnet", "normalmap:" + str(out / "cn_normalmap.png"),
            "-p", PROMPT,
            "--negative-prompt", NEGATIVE,
            "-W", "512", "-H", "512",
            "--seed", "42",
            "-o", str(out / "sd15_cn_normalmap.png"),
        ],
        output=out / "sd15_cn_normalmap.png",
    )
