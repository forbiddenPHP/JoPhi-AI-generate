"""Test: Video — LTX-2.3 video generation."""

import sys
from pathlib import Path

import os
VIDEO_QUALITY = "720p" if os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") > 64 * 1024**3 else "480p"

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
REF_IMAGE = ASSETS_DIR / "johannes.png"

PROMPT = (
    "Wide shot of a small orange tabby cat bounding through a sunlit wildflower meadow. "
    "Warm golden hour light, shallow depth of field, soft bokeh in the background. "
    "The cat leaps over tall grass, ears perked forward, tail streaming behind. "
    "The camera tracks alongside at low angle, keeping pace. "
    "Pollen and tiny seeds drift through the warm air. "
    "Birdsong fills the background, soft rustling of grass underfoot."
)
PROMPT_I2V = "A man smiles, nods slowly and says in German: \"Hallo, ich bin Johannes.\""


def register(suite):
    out = suite.out_dir

    # T2V with distilled model — 16:9 240p (fast test)
    suite.add(
        name="Video T2V distilled 5s",
        cmd=[
            sys.executable, "generate.py", "video", "ltx2.3",
            "--model", "distilled",
            "-p", PROMPT,
            "--ratio", "16:9", "--quality", VIDEO_QUALITY,
            "--num-frames", "120", "--frame-rate", "24",
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
            "--model", "distilled",
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
