"""Test: NormalMap — Marigold-Normals estimation + ControlNet generation (3 personas)."""

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
        name="NormalMap: estimate normals",
        cmd=[
            sys.executable, "generate.py", "image",
            "normalmap",
            "--images", str(REF_IMAGE),
            "-o", str(out / "normalmap_johannes.png"),
        ],
        output=out / "normalmap_johannes.png",
    )

    # ── ControlNet generation from normalmap ──────────────────────────────

    suite.add(
        name="NormalMap: generate man from normals",
        cmd=[
            sys.executable, "generate.py", "image",
            "flux.2",
            "--model", "4b-distilled",
            "--controlnet", "normalmap:" + str(out / "normalmap_johannes.png"),
            "-p", PROMPT_MAN,
            "-W", "504", "-H", "504",
            "--seed", "42",
            "-o", str(out / "gen_normalmap_man.png"),
        ],
        output=out / "gen_normalmap_man.png",
    )

    suite.add(
        name="NormalMap: generate woman from normals",
        cmd=[
            sys.executable, "generate.py", "image",
            "flux.2",
            "--model", "4b-distilled",
            "--controlnet", "normalmap:" + str(out / "normalmap_johannes.png"),
            "-p", PROMPT_WOMAN,
            "-W", "504", "-H", "504",
            "--seed", "73",
            "-o", str(out / "gen_normalmap_woman.png"),
        ],
        output=out / "gen_normalmap_woman.png",
    )

    suite.add(
        name="NormalMap: generate monster from normals",
        cmd=[
            sys.executable, "generate.py", "image",
            "flux.2",
            "--model", "4b-distilled",
            "--controlnet", "normalmap:" + str(out / "normalmap_johannes.png"),
            "-p", PROMPT_MONSTER,
            "-W", "504", "-H", "504",
            "--seed", "99",
            "-o", str(out / "gen_normalmap_monster.png"),
        ],
        output=out / "gen_normalmap_monster.png",
    )
