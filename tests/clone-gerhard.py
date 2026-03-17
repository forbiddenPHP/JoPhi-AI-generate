#!/usr/bin/env python3
"""Standalone test: Clone-TTS with Gerhard voice reference."""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
REFERENCE = SCRIPT_DIR / "demos" / "custom-voice" / "gerhard" / "sample.m4a"
OUTPUT = SCRIPT_DIR / "demos" / "custom-voice" / "gerhard" / "clone_output.wav"

TEXT = "Ich bin der kleine dicke, ich stell mich in die Mitte, ich mach' einen Knicks, denn sonst kann ich nix'!"

cmd = [
    sys.executable, "generate.py", "voice", "--engine", "clone-tts",
    "--reference", str(REFERENCE),
    "--language", "de",
    "--text", TEXT,
    "-o", str(OUTPUT),
]

print(f"Reference: {REFERENCE}")
print(f"Output:    {OUTPUT}")
print()

result = subprocess.run(cmd, cwd=SCRIPT_DIR)
sys.exit(result.returncode)
