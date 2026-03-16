#!/usr/bin/env python3
"""Test: Clone-TTS — Clone a voice from a reference audio sample.

Uses a real reference audio file and generates new speech in that voice
via Qwen3-TTS Base (ICL voice cloning).

Output: demos/clone-tts/voice-cloning-test.wav
"""

import subprocess
import sys
from pathlib import Path


REFERENCE = Path(__file__).parent.parent / "voice" / "default-reference.m4a"
OUTPUT = Path(__file__).parent.parent / "demos" / "clone-tts" / "voice-cloning-test.wav"

TEXT = (
    'Hello friends! Welcome to a new episode. '
    'I hope you had a wonderful week!'
)


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    if not REFERENCE.exists():
        print(f"FAILED: Reference audio not found: {REFERENCE}")
        sys.exit(1)

    print(f"=== Voice Cloning Test (Qwen3-TTS Base) ===")
    print(f"Reference: {REFERENCE}")
    print(f"Text: {TEXT}")
    print(f"Output: {OUTPUT}")

    cmd = [
        sys.executable,
        str(Path(__file__).parent.parent / "generate.py"),
        "voice",
        "--engine", "clone-tts",
        "--reference", str(REFERENCE),
        "--text", TEXT,
        "-o", str(OUTPUT),
    ]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"FAILED: voice cloning (exit {result.returncode})")
        sys.exit(result.returncode)

    if not OUTPUT.exists():
        print(f"FAILED: output not created: {OUTPUT}")
        sys.exit(1)

    size = OUTPUT.stat().st_size
    if size == 0:
        print("FAILED: output is empty")
        sys.exit(1)

    print(f"\nOK: {OUTPUT.name} ({size:,} bytes)")


if __name__ == "__main__":
    main()
