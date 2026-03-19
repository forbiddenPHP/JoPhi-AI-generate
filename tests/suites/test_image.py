"""Test: Image — FLUX.2 image generation (4 models + reference images)."""

import sys
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
REF_IMAGE = ASSETS_DIR / "johannes.png"

PROMPT = "a small orange cat sitting on a mossy cliff overlooking the ocean"
PROMPT_REF = "a man standing in front of a mountain lake, cinematic photography, golden hour"


def register(suite):
    out = suite.out_dir

    # ── Text-to-Image ────────────────────────────────────────────────────

    suite.add(
        name="Image: FLUX.2 4b-distilled",
        cmd=[
            sys.executable, "generate.py", "image",
            "flux.2", "--model", "4b-distilled",
            "-p", PROMPT, "--seed", "42",
            "-W", "896", "-H", "504",
            "-o", str(out / "flux2_4b_distilled.png"),
        ],
        output=out / "flux2_4b_distilled.png",
    )

    suite.add(
        name="Image: FLUX.2 4b (base)",
        cmd=[
            sys.executable, "generate.py", "image",
            "flux.2", "--model", "4b",
            "-p", PROMPT, "--seed", "42",
            "--steps", "20",
            "-W", "896", "-H", "504",
            "-o", str(out / "flux2_4b_base.png"),
        ],
        output=out / "flux2_4b_base.png",
    )

    suite.add(
        name="Image: FLUX.2 9b-distilled",
        cmd=[
            sys.executable, "generate.py", "image",
            "flux.2", "--model", "9b-distilled",
            "-p", PROMPT, "--seed", "42",
            "-W", "896", "-H", "504",
            "-o", str(out / "flux2_9b_distilled.png"),
        ],
        output=out / "flux2_9b_distilled.png",
    )

    suite.add(
        name="Image: FLUX.2 9b (base)",
        cmd=[
            sys.executable, "generate.py", "image",
            "flux.2", "--model", "9b",
            "-p", PROMPT, "--seed", "42",
            "--steps", "20",
            "-W", "896", "-H", "504",
            "-o", str(out / "flux2_9b_base.png"),
        ],
        output=out / "flux2_9b_base.png",
    )

    # ── Reference image tests (johannes.png) ─────────────────────────────

    suite.add(
        name="Image: FLUX.2 4b-distilled + ref (johannes)",
        cmd=[
            sys.executable, "generate.py", "image",
            "flux.2", "--model", "4b-distilled",
            "-p", PROMPT_REF,
            "--images", str(REF_IMAGE),
            "--seed", "42",
            "-W", "896", "-H", "504",
            "-o", str(out / "flux2_4b_distilled_ref.png"),
        ],
        output=out / "flux2_4b_distilled_ref.png",
    )

    suite.add(
        name="Image: FLUX.2 9b-distilled + ref (johannes)",
        cmd=[
            sys.executable, "generate.py", "image",
            "flux.2", "--model", "9b-distilled",
            "-p", PROMPT_REF,
            "--images", str(REF_IMAGE),
            "--seed", "42",
            "-W", "896", "-H", "504",
            "-o", str(out / "flux2_9b_distilled_ref.png"),
        ],
        output=out / "flux2_9b_distilled_ref.png",
    )
