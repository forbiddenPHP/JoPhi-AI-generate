"""Test: Video — LTX-2.3 dev (full model, 40 steps)."""

import sys
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
REF_IMAGE = ASSETS_DIR / "johannes.png"

PROMPT = "A small orange cat running through a sunlit meadow, camera slowly tracking"
PROMPT_I2V = "A man smiles, nods slowly and says in German: \"Hallo, ich bin Johannes.\""


def register(suite):
    out = suite.out_dir

    # T2V with dev model — 16:9 240p
    suite.add(
        name="Video T2V dev 2s",
        cmd=[
            sys.executable, "generate.py", "video", "ltx2.3",
            "--model", "dev",
            "-p", PROMPT,
            "--ratio", "16:9", "--quality", "240p",
            "--num-frames", "48", "--frame-rate", "24",
            "--seed", "42",
            "-o", str(out / "ltx2_dev_t2v.mp4"),
        ],
        output=out / "ltx2_dev_t2v.mp4",
    )

    # I2V with dev model — 1:1 240p, image-first conditioning (johannes.png)
    suite.add(
        name="Video I2V dev 5s (johannes)",
        cmd=[
            sys.executable, "generate.py", "video", "ltx2.3",
            "--model", "dev",
            "-p", PROMPT_I2V,
            "--image-first", str(REF_IMAGE),
            "--ratio", "1:1", "--quality", "240p",
            "--num-frames", "120", "--frame-rate", "24",
            "--seed", "42",
            "-o", str(out / "ltx2_dev_i2v.mp4"),
        ],
        output=out / "ltx2_dev_i2v.mp4",
    )

    # ltx2.3 models list → save to txt
    suite.add(
        name="LTX2.3 models list",
        cmd=[
            "bash", "-c",
            f"{sys.executable} generate.py video ltx2.3 models list 2>{out / 'ltx2_models.txt'}",
        ],
        output=out / "ltx2_models.txt",
    )
