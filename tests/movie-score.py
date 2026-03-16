#!/usr/bin/env python3
"""Test: ACE-Step instrumental — cinematic movie score.

Generates an epic orchestral film score using [inst] mode.
Output: demos/ace-step/movie-score.mp3
"""

import subprocess
import sys
from pathlib import Path


TAGS = (
    "cinematic orchestral film score, 85 bpm, D minor. "
    "Starts with a slow lonely piano solo, single notes. "
    "Low strings enter with cellos and violas tremolo, building tension. "
    "French horns rise with timpani rumble, growing intensity. "
    "Full orchestra erupts with brass fanfare, epic percussion, crashing cymbals. "
    "Quiet break with solo violin over sustained strings, fragile and haunting. "
    "Orchestra returns even bigger with choir-like pads and triumphant brass melody. "
    "Music fades out, piano returns alone, final lingering note."
)

OUTPUT = Path(__file__).parent.parent / "demos" / "ace-step" / "movie-score.mp3"
SECONDS = 120


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(Path(__file__).parent.parent / "generate.py"),
        "audio",
        "--engine", "ace-step",
        "-l", "[instrumental]",
        "-t", TAGS,
        "-s", str(SECONDS),
        "-o", str(OUTPUT),
    ]

    print(f"Generating: {TAGS}")
    print(f"Output: {OUTPUT}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"FAILED (exit {result.returncode})")
        sys.exit(result.returncode)

    if not OUTPUT.exists():
        print("FAILED: output file not created")
        sys.exit(1)

    size = OUTPUT.stat().st_size
    if size == 0:
        print("FAILED: output file is empty")
        sys.exit(1)

    print(f"OK: {OUTPUT.name} ({size:,} bytes)")


if __name__ == "__main__":
    main()
