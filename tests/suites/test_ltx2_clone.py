"""Test: Video — LTX-2.3 clone.

Uses VideoCloneTest.mov as reference video. RetakePipeline extends the video
in latent space (denoise_mask=0 for context, =1 for new frames), then crops
the context out — preserving person identity via hard latent conditioning.
"""

import subprocess
import sys
from pathlib import Path

_mem = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
VIDEO_QUALITY = "720p" if _mem > 64 * 1024**3 else "480p"

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
CLONE_VIDEO = ASSETS_DIR / "VideoCloneTest.mov"

CLONE_PROMPT = (
    'A bearded man with glasses wearing a black sleeveless shirt, '
    'speaking directly to camera in a cozy living room with a decorative wall shelf, '
    'ceramic plates, wooden objects and a framed painting in the background, '
    'says in German: "Sag mal, wir testen einfach, ob das funktioniert und ich bin schon gespannt darauf." '
    'The man continues in German: "Das Hörgerät meiner Großmutter wurde vom Blitz getroffen."'
)


def register(suite):
    out = suite.out_dir

    suite.add(
        name="Clone 5s (default, distilled)",
        cmd=[
            sys.executable, "generate.py", "video", "ltx2.3",
            "--model", "distilled",
            "-p", CLONE_PROMPT,
            "--clone", str(CLONE_VIDEO),
            "--seconds", "5",
            "--ratio", "16:9", "--quality", VIDEO_QUALITY,
            "--seed", "42",
            "-o", str(out / "ltx2_clone.mp4"),
        ],
        output=out / "ltx2_clone.mp4",
    )

    suite.add(
        name="Clone 5s (dev)",
        cmd=[
            sys.executable, "generate.py", "video", "ltx2.3",
            "--model", "dev",
            "-p", CLONE_PROMPT,
            "--clone", str(CLONE_VIDEO),
            "--seconds", "5",
            "--ratio", "16:9", "--quality", VIDEO_QUALITY,
            "--seed", "42",
            "-o", str(out / "ltx2_clone_dev.mp4"),
        ],
        output=out / "ltx2_clone_dev.mp4",
    )
