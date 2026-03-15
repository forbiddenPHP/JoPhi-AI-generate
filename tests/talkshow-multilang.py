#!/usr/bin/env python3
"""Test: Multilingual talkshow dialog with per-segment language switching.

Generates a German dialog where speakers switch to English for direct quotes,
using the [Voice:instruct:language] tag syntax.
"""

import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT / "demos" / "ai-tts-multilanguage-syntax"
OUTPUT = OUTPUT_DIR / "talkshow-multilang.wav"

TEXT = (
    '[Dylan - calm- German] Und Peter sagte: '
    '[English - energetic] "Hey there, how are you?" '
    '[Uncle_Fu - excited - German] und wir freuten uns so sehr, dass wir '
    '[English] "Oh my god, what the quack!" '
    '[German] riefen und in großes Gelächter ausbrachen.'
)

# ── Generate ─────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cmd = [
    sys.executable, str(PROJECT / "generate.py"), "voice",
    "--engine", "ai-tts",
    "--language", "de",
    "--text", TEXT,
    "-o", str(OUTPUT),
]

print(f"=== Talkshow Multilang (5 segments, DE/EN) ===")
print(f"Output: {OUTPUT}")
print()

result = subprocess.run(cmd)

if result.returncode != 0:
    print("FAILED: generation returned non-zero", file=sys.stderr)
    sys.exit(1)

if not OUTPUT.exists():
    print(f"FAILED: {OUTPUT} not created", file=sys.stderr)
    sys.exit(1)

size = OUTPUT.stat().st_size
if size == 0:
    print(f"FAILED: {OUTPUT} is empty", file=sys.stderr)
    sys.exit(1)

print(f"OK — {OUTPUT} ({size:,} bytes)")
