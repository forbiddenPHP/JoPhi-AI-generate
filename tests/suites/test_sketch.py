"""Test: Sketch — HED edge extraction + ControlNet generation (3 personas)."""

import sys
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
REF_IMAGE = ASSETS_DIR / "johannes.png"

PROMPT_MAN = "close-up, turn image 1 into a 30-year-old male cartoon character shaped like image 1, wearing glasses, with messy brown hair, bright blue eyes and a confident sexy smile, warm studio lighting"
PROMPT_WOMAN = "close-up, turn image 1 into a 25-year-old East African woman shaped like image 1, wearing glasses, with short curly hair, golden earrings, deep brown eyes and a wide smile, sunset lighting from the left"
PROMPT_MONSTER = "close-up, turn image 1 into a friendly swamp creature shaped like image 1, wearing glasses, with mossy green skin, glowing yellow eyes, small antlers and a wide toothy grin, misty forest lighting"


def register(suite):
    out = suite.out_dir

    # ── Extraction ────────────────────────────────────────────────────────

    suite.add(
        name="Sketch: HED extraction",
        cmd=[
            sys.executable, "generate.py", "image",
            "sketch",
            "--images", str(REF_IMAGE),
            "-o", str(out / "sketch_hed.png"),
        ],
        output=out / "sketch_hed.png",
    )

    # ── ControlNet generation from sketch ─────────────────────────────────

    suite.add(
        name="Sketch: generate man from HED",
        cmd=[
            sys.executable, "generate.py", "image",
            "flux.2",
            "--model", "4b-distilled",
            "--controlnet", "sketch:" + str(out / "sketch_hed.png"),
            "-p", PROMPT_MAN,
            "-W", "504", "-H", "504",
            "--seed", "42",
            "-o", str(out / "gen_sketch_man.png"),
        ],
        output=out / "gen_sketch_man.png",
    )

    suite.add(
        name="Sketch: generate woman from HED",
        cmd=[
            sys.executable, "generate.py", "image",
            "flux.2",
            "--model", "4b-distilled",
            "--controlnet", "sketch:" + str(out / "sketch_hed.png"),
            "-p", PROMPT_WOMAN,
            "-W", "504", "-H", "504",
            "--seed", "73",
            "-o", str(out / "gen_sketch_woman.png"),
        ],
        output=out / "gen_sketch_woman.png",
    )

    suite.add(
        name="Sketch: generate monster from HED",
        cmd=[
            sys.executable, "generate.py", "image",
            "flux.2",
            "--model", "4b-distilled",
            "--controlnet", "sketch:" + str(out / "sketch_hed.png"),
            "-p", PROMPT_MONSTER,
            "-W", "504", "-H", "504",
            "--seed", "99",
            "-o", str(out / "gen_sketch_monster.png"),
        ],
        output=out / "gen_sketch_monster.png",
    )
