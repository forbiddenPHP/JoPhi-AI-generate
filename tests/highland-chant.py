#!/usr/bin/env python3
"""Test 23: Ethereal Nordic Chant
Wordless female vocals — aahs, oohs, sustained notes.
Celtic/Nordic atmosphere, epic and cinematic.
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_duration(s):
    """Parse 'M:SS' or 'MM:SS' or plain seconds into int seconds."""
    m = re.match(r'^(\d+):(\d{2})$', s.strip())
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    return int(s)


# ── Config ────────────────────────────────────────────────────────
DURATION = "2:00"
SEED = 42
ENGINE = "ace"
OUTPUT = Path(__file__).parent.parent / "demos" / "highland-chant-2-minutes.wav"

CAPTION = (
    "ethereal female voice, wordless vocalizations, "
    "Celtic orchestral, Nordic atmosphere, epic cinematic"
)

# ── Song structure ───────────────────────────────────────────────

SONG = [
    {"tag": "Verse", "lines": [
        "Aaaaaah, aaaaaah,",
        "ooooooh, ooooooh.",
    ]},

    {"tag": "Verse", "lines": [
        "Aaaaaah, aaaaaah,",
        "ooooooh, aaaaaah.",
    ]},

    {"tag": "Chorus", "lines": [
        "Ooooooh, ooooooh,",
        "aaaaaah, aaaaaah,",
        "ooooooh.",
    ]},

    {"tag": "Verse", "lines": [
        "Aaaaaah, ooooooh,",
        "aaaaaah, ooooooh.",
    ]},

    {"tag": "Chorus", "lines": [
        "Ooooooh, ooooooh,",
        "aaaaaah, aaaaaah,",
        "ooooooh.",
    ]},
]


# ── Build lyrics ─────────────────────────────────────────────────

def build_lyrics(song):
    """Build lyrics with section tags, separated by blank lines."""
    parts = []
    for section in song:
        tag = section["tag"].upper()
        lines = section.get("lines", [])
        vocal = [l for l in lines if l.strip()]
        if vocal:
            parts.append(f"[{tag}]\n" + "\n".join(vocal))
        else:
            parts.append(f"[{tag}]")
    return "\n\n".join(parts)


duration_s = parse_duration(DURATION)
lyrics = build_lyrics(SONG)

n_lines = sum(len([l for l in s.get("lines", []) if l.strip()]) for s in SONG)
print(f"=== {n_lines} lines, {duration_s}s ({duration_s // 60}:{duration_s % 60:02d}) ===")

# Write lyrics to temp file
lyrics_file = Path(tempfile.mktemp(suffix=".txt", prefix="lyrics-23-"))
lyrics_file.write_text(lyrics, encoding="utf-8")

# ── Generate ─────────────────────────────────────────────────────
cmd = [
    sys.executable, str(Path(__file__).parent.parent / "revoicer.py"), "music",
    "--engine", ENGINE,
    "-f", str(lyrics_file),
    "-t", CAPTION,
    "--seed", str(SEED),
    "-s", str(duration_s),
    "-o", str(OUTPUT),
]

print(f"Generating: {OUTPUT}")
result = subprocess.run(cmd)
lyrics_file.unlink(missing_ok=True)
sys.exit(result.returncode)
