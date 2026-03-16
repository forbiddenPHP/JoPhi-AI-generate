#!/usr/bin/env python3
"""Test: Clone-TTS — Voice cloning with default reference (your voice).

Generates two samples using the default reference audio (voice/default-reference.m4a):
  1. German:  demos/clone-tts/clone_deutsch.wav
  2. English: demos/clone-tts/clone_english.wav
"""

import subprocess
import sys
from pathlib import Path


GENERATE = str(Path(__file__).parent.parent / "generate.py")
OUTPUT_DIR = Path(__file__).parent.parent / "demos" / "clone-tts"

SAMPLES = [
    {
        "name": "Deutsch",
        "text": "Herzlich willkommen! Oh mein Gott, Leute, ich kann es kaum erwarten! Heute wird es absolut grandios — ihr werdet nicht glauben, was ich vorbereitet habe!",
        "language": "de",
        "output": OUTPUT_DIR / "clone_deutsch.wav",
    },
    {
        "name": "English",
        "text": "Welcome everyone! Oh my God, you guys, I can barely contain myself! Today is going to be absolutely amazing — you won't believe what I've got in store for you!",
        "language": "en",
        "output": OUTPUT_DIR / "clone_english.wav",
    },
]


def run_clone(sample):
    print(f"\n=== {sample['name']} ===")
    print(f"  Text: {sample['text'][:70]}…")
    print(f"  Output: {sample['output']}")

    cmd = [
        sys.executable, GENERATE,
        "voice", "--engine", "clone-tts",
        "--language", sample["language"],
        "--text", sample["text"],
        "-o", str(sample["output"]),
    ]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})")
        return False

    if not sample["output"].exists() or sample["output"].stat().st_size == 0:
        print(f"  FAILED: output missing or empty")
        return False

    size = sample["output"].stat().st_size
    print(f"  OK: {sample['output'].name} ({size:,} bytes)")
    return True


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ok = 0
    for sample in SAMPLES:
        if run_clone(sample):
            ok += 1

    print(f"\n{'='*40}")
    print(f"  {ok}/{len(SAMPLES)} samples generated")
    if ok < len(SAMPLES):
        sys.exit(1)


if __name__ == "__main__":
    main()
