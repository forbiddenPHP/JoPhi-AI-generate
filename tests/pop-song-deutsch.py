#!/usr/bin/env python3
"""Test: ACE-Step — German pop song with --language de.

Generates a German-language pop song to verify vocal_language parameter.
Output: demos/ace-step/pop-song-deutsch.mp3
"""

import subprocess
import sys
from pathlib import Path


LYRICS = """
Er konnte einfach nicht widerstehen,
und musste den Klo-Besen wieder sehen.
Er sah ihn an, er war braun gefleckt.
Darum hat er genüsslich an ihm geleckt!
Er hat am Klo-Besen geleckt,
weil es ihm so schmeckt!
Er hat ma Klo-Besen geleckt,
uns hat es total gereckt!
"""

TAGS = "eurodance, dancefloor, disco"
OUTPUT = Path(__file__).parent.parent / "demos" / "ace-step" / "pop-song-deutsch.mp3"
SECONDS = 200


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(Path(__file__).parent.parent / "generate.py"),
        "audio",
        "--engine", "ace-step",
        "-l", LYRICS,
        "-t", TAGS,
        "--language", "de",
        "-s", str(SECONDS),
        "-o", str(OUTPUT),
    ]

    print(f"Generating: {TAGS}")
    print(f"Language: de")
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
