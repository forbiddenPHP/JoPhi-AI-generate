"""Test: Diarize — speaker diarization (2 modes)."""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
PODCAST = SCRIPT_DIR / "tests" / "assets" / "podcast-5min.mp3"


def register(suite):
    out = suite.out_dir

    suite.add(
        name="Diarize 3 speakers + verify",
        cmd=[
            sys.executable, "generate.py", "audio", "--engine", "diarize",
            str(PODCAST),
            "--speakers", "3", "--verify",
            "-o", str(out / "3-speakers"),
        ],
        output=out / "3-speakers",
    )

    suite.add(
        name="Diarize auto-detect speakers",
        cmd=[
            sys.executable, "generate.py", "audio", "--engine", "diarize",
            str(PODCAST),
            "-o", str(out / "auto-speakers"),
        ],
        output=out / "auto-speakers",
    )
