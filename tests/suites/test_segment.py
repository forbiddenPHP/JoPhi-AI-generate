"""Test: Segment — foreground/background separation via BiRefNet."""

import sys
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
JOHANNES = ASSETS_DIR / "johannes.png"
LIVINGROOM = ASSETS_DIR / "livingroom.png"


def register(suite):
    out = suite.out_dir

    # ── Johannes ──────────────────────────────────────────────────────────

    suite.add(
        name="Segment: johannes foreground",
        cmd=[
            sys.executable, "generate.py", "image",
            "--engine", "segment",
            "--images", str(JOHANNES),
            "-o", str(out / "johannes" / "foreground.png"),
        ],
        output=out / "johannes" / "foreground.png",
    )

    suite.add(
        name="Segment: johannes background",
        cmd=[
            sys.executable, "generate.py", "image",
            "--engine", "segment",
            "--output-layer", "background",
            "--images", str(JOHANNES),
            "-o", str(out / "johannes" / "background.png"),
        ],
        output=out / "johannes" / "background.png",
    )

    suite.add(
        name="Segment: johannes both",
        cmd=[
            sys.executable, "generate.py", "image",
            "--engine", "segment",
            "--output-layer", "both",
            "--images", str(JOHANNES),
            "-o", str(out / "johannes_both/"),
        ],
        output=out / "johannes_both",
    )

    # ── Livingroom ────────────────────────────────────────────────────────

    suite.add(
        name="Segment: livingroom foreground",
        cmd=[
            sys.executable, "generate.py", "image",
            "--engine", "segment",
            "--images", str(LIVINGROOM),
            "-o", str(out / "livingroom" / "foreground.png"),
        ],
        output=out / "livingroom" / "foreground.png",
    )

    suite.add(
        name="Segment: livingroom background",
        cmd=[
            sys.executable, "generate.py", "image",
            "--engine", "segment",
            "--output-layer", "background",
            "--images", str(LIVINGROOM),
            "-o", str(out / "livingroom" / "background.png"),
        ],
        output=out / "livingroom" / "background.png",
    )

    suite.add(
        name="Segment: livingroom both",
        cmd=[
            sys.executable, "generate.py", "image",
            "--engine", "segment",
            "--output-layer", "both",
            "--images", str(LIVINGROOM),
            "-o", str(out / "livingroom_both/"),
        ],
        output=out / "livingroom_both",
    )
