"""Test: Video — LTX-2.3 video generation."""

import sys
from pathlib import Path

VIDEO_QUALITY = "480p"

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
REF_IMAGE = ASSETS_DIR / "johannes.png"

PROMPT = "A small orange cat running through a sunlit meadow, camera slowly tracking"
PROMPT_I2V = "A man smiles, nods slowly and says in German: \"Hallo, ich bin Johannes.\""


def register(suite):
    out = suite.out_dir

    # T2V with distilled model — 16:9 240p (fast test)
    suite.add(
        name="Video T2V distilled 2s",
        cmd=[
            sys.executable, "generate.py", "video", "ltx2.3",
            "-p", PROMPT,
            "--ratio", "16:9", "--quality", VIDEO_QUALITY,
            "--num-frames", "48", "--frame-rate", "24",
            "--seed", "42",
            "-o", str(out / "ltx2_distilled_t2v.mp4"),
        ],
        output=out / "ltx2_distilled_t2v.mp4",
    )

    # I2V with distilled model — 1:1 240p, image-first conditioning (johannes.png)
    suite.add(
        name="Video I2V distilled 5s (johannes)",
        cmd=[
            sys.executable, "generate.py", "video", "ltx2.3",
            "-p", PROMPT_I2V,
            "--image-first", str(REF_IMAGE),
            "--ratio", "1:1", "--quality", VIDEO_QUALITY,
            "--num-frames", "120", "--frame-rate", "24",
            "--seed", "42",
            "-o", str(out / "ltx2_distilled_i2v.mp4"),
        ],
        output=out / "ltx2_distilled_i2v.mp4",
    )

    # video models list → save to txt (TUI output goes to stderr)
    suite.add(
        name="Video models list",
        cmd=[
            "bash", "-c",
            f"{sys.executable} generate.py video ltx2.3 models list 2>{out / 'ltx2_models.txt'}",
        ],
        output=out / "ltx2_models.txt",
    )
