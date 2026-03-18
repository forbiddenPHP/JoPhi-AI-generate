"""Test: SD 1.5 — Stable Diffusion 1.5 image generation (MatureMaleMix, DreamShaper)."""

import sys
from pathlib import Path


def register(suite):
    out = suite.out_dir
    NEGATIVE ="(worst quality, low quality:1.4), deformed, disfigured, distorted, blurry, deformed iris, deformed pupils, (bad anatomy), (bad hands)"

    # ── MatureMaleMix ────────────────────────────────────────────────────────

    suite.add(
        name="SD1.5: mm default (with LoRA)",
        cmd=[
            sys.executable, "generate.py", "image",
            "--engine", "sd1.5",
            "--model", "mm",
            "-p", "a muscular man with a beard, looking at the camera, studio lighting, sharp focus",
            "-W", "512", "-H", "512",
            # "--seed", "42",
            "-o", str(out / "mm_default.png"),
        ],
        output=out / "mm_default.png",
    )

    suite.add(
        name="SD1.5: mm without LoRA",
        cmd=[
            sys.executable, "generate.py", "image",
            "--engine", "sd1.5",
            "--model", "mm",
            "--no-lora",
            "-p", "a muscular man with a beard, looking at the camera, studio lighting, sharp focus",
            "-W", "512", "-H", "512",
            # "--seed", "42",
            "-o", str(out / "mm_no_lora.png"),
        ],
        output=out / "mm_no_lora.png",
    )

    suite.add(
        name="SD1.5: mm portrait (16:9)",
        cmd=[
            sys.executable, "generate.py", "image",
            "--engine", "sd1.5",
            "--model", "mm",
            "--no-lora",
            "-p", "a handsome man in a leather jacket, cinematic lighting, bokeh background",
            "--negative-prompt", NEGATIVE,
            "-W", "512", "-H", "320",
            # "--seed", "73",
            "-o", str(out / "mm_landscape.png"),
        ],
        output=out / "mm_landscape.png",
    )

    # ── DreamShaper ──────────────────────────────────────────────────────────

    suite.add(
        name="SD1.5: dreamshaper landscape (16:9)",
        cmd=[
            sys.executable, "generate.py", "image",
            "--engine", "sd1.5",
            "--model", "dreamshaper",
            "--no-lora",
            "-p", "a fantasy landscape with floating islands, dramatic clouds, golden sunlight, highly detailed digital painting",
            "-W", "512", "-H", "320",
            # "--seed", "42",
            "-o", str(out / "dreamshaper_landscape.png"),
        ],
        output=out / "dreamshaper_landscape.png",
    )

    suite.add(
        name="SD1.5: dreamshaper portrait (9:16)",
        cmd=[
            sys.executable, "generate.py", "image",
            "--engine", "sd1.5",
            "--model", "dreamshaper",
            "--no-lora",
            "-p", "portrait of an elven warrior princess with silver hair, intricate armor, forest background, fantasy art style",
            "-W", "320", "-H", "512",
            # "--seed", "99",
            "-o", str(out / "dreamshaper_portrait.png"),
        ],
        output=out / "dreamshaper_portrait.png",
    )
