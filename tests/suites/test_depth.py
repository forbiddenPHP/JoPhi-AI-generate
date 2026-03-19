"""Test: Depth — Depth Anything V2 estimation + depth-conditioned FLUX.2 generation."""

import sys
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
REF_IMAGE = ASSETS_DIR / "johannes.png"


def register(suite):
    out = suite.out_dir

    # ── Depth extraction ─────────────────────────────────────────────────────

    suite.add(
        name="Depth: extract from johannes",
        cmd=[
            sys.executable, "generate.py", "image",
            "depth",
            "--images", str(REF_IMAGE),
            "-o", str(out / "depth_johannes.png"),
        ],
        output=out / "depth_johannes.png",
    )

    # ── Depth-conditioned generation (empty latent + depth as reference) ─────

    suite.add(
        name="Depth: generate man from depth",
        cmd=[
            sys.executable, "generate.py", "image",
            "flux.2",
            "--model", "4b-distilled",
            "-p", "close-up, turn image 1 into a 30-year-old male cartoon character shaped like image 1, wearing glasses, with messy brown hair, bright blue eyes and a confident sexy smile, chair backrest visible in the lower right background, warm studio lighting",
            "--controlnet", "depth:" + str(out / "depth_johannes.png"),
            "-W", "504", "-H", "504",
            "-o", str(out / "depth_gen_man.png"),
        ],
        output=out / "depth_gen_man.png",
    )

    suite.add(
        name="Depth: generate woman from depth",
        cmd=[
            sys.executable, "generate.py", "image",
            "flux.2",
            "--model", "4b-distilled",
            "-p", "close-up, turn image 1 into a 25-year-old East African woman shaped like image 1, wearing glasses, with short curly hair, golden earrings, deep brown eyes and a wide smile, chair backrest visible in the lower right background, sunset lighting from the left",
            "--controlnet", "depth:" + str(out / "depth_johannes.png"),
            "-W", "504", "-H", "504",
            "-o", str(out / "depth_gen_woman.png"),
        ],
        output=out / "depth_gen_woman.png",
    )

    suite.add(
        name="Depth: generate creature from depth",
        cmd=[
            sys.executable, "generate.py", "image",
            "flux.2",
            "--model", "4b-distilled",
            "-p", "close-up, turn image 1 into a friendly swamp creature shaped like image 1, wearing glasses, with mossy green skin, glowing yellow eyes, small antlers and a wide toothy grin, chair backrest visible in the lower right background, misty forest lighting",
            "--controlnet", "depth:" + str(out / "depth_johannes.png"),
            "-W", "504", "-H", "504",
            "-o", str(out / "depth_gen_creature.png"),
        ],
        output=out / "depth_gen_creature.png",
    )

    # ── Pan & Scan test: 16:9 from square depth map ──────────────────────

    suite.add(
        name="Depth: man 16:9 (pan&scan)",
        cmd=[
            sys.executable, "generate.py", "image",
            "flux.2",
            "--model", "4b-distilled",
            "-p", "close-up, turn image 1 into a 30-year-old male cartoon character shaped like image 1, wearing glasses, with messy brown hair, bright blue eyes and a confident sexy smile, warm studio lighting",
            "--controlnet", "depth:" + str(out / "depth_johannes.png"),
            "-W", "896", "-H", "504",
            "-o", str(out / "depth_gen_man_16x9.png"),
        ],
        output=out / "depth_gen_man_16x9.png",
    )

    suite.add(
        name="Depth: woman 9:16 (pan&scan)",
        cmd=[
            sys.executable, "generate.py", "image",
            "flux.2",
            "--model", "4b-distilled",
            "-p", "close-up, turn image 1 into a 25-year-old East African woman shaped like image 1, wearing glasses, with short curly hair, golden earrings, deep brown eyes and a wide smile, sunset lighting from the left",
            "--controlnet", "depth:" + str(out / "depth_johannes.png"),
            "-W", "504", "-H", "896",
            "-o", str(out / "depth_gen_woman_9x16.png"),
        ],
        output=out / "depth_gen_woman_9x16.png",
    )

    suite.add(
        name="Depth: creature 16:9 (pan&scan)",
        cmd=[
            sys.executable, "generate.py", "image",
            "flux.2",
            "--model", "4b-distilled",
            "-p", "close-up, turn image 1 into a friendly swamp creature shaped like image 1, wearing glasses, with mossy green skin, glowing yellow eyes, small antlers and a wide toothy grin, misty forest lighting",
            "--controlnet", "depth:" + str(out / "depth_johannes.png"),
            "-W", "896", "-H", "504",
            "-o", str(out / "depth_gen_creature_16x9.png"),
        ],
        output=out / "depth_gen_creature_16x9.png",
    )
