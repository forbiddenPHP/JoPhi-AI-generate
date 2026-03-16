#!/usr/bin/env python3
"""Test: SFX generation — spaceship flyby.

Generates a sci-fi spaceship engine sound (X-wing style).
Output: demos/sfx/spaceship-flyby.wav
"""

import subprocess
import sys
from pathlib import Path


PROMPT = "a spaceship engine roaring and whooshing as it flies by at high speed"
OUTPUT = Path(__file__).parent.parent / "demos" / "sfx" / "spaceship-flyby.wav"
DURATION = 10


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(Path(__file__).parent.parent / "generate.py"),
        "audio",
        "--engine", "sfx",
        "--text", PROMPT,
        "-s", str(DURATION),
        "-o", str(OUTPUT),
    ]

    print(f"Generating SFX: {PROMPT}")
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
