#!/usr/bin/env python3
"""Test: transcribe the first audio file in demos/ using whisper worker.

Picks the first .wav/.mp3/.flac in demos/, transcribes it with
--input-language en, outputs all formats into demos/<filename-without-ext>/.
"""

import json
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DEMOS = PROJECT / "demos"

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def find_first_audio():
    """Return the first audio file in demos/, sorted by name."""
    for f in sorted(DEMOS.iterdir()):
        if f.suffix.lower() in AUDIO_EXTENSIONS and f.is_file():
            return f
    return None


def run():
    demo_file = find_first_audio()
    if not demo_file:
        print(f"ERROR: No audio files found in {DEMOS}", file=sys.stderr)
        sys.exit(1)

    output_dir = DEMOS / demo_file.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run transcription via revoicer.py
    cmd = [
        sys.executable, str(PROJECT / "revoicer.py"),
        "transcribe",
        str(demo_file),
        "--input-language", "en",
        "--word-timestamps",
        "--format", "all",
        "-o", str(output_dir),
    ]

    print(f"Running: {' '.join(cmd)}")
    print()

    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    # Show stderr (progress)
    if r.stderr:
        for line in r.stderr.strip().split("\n"):
            print(f"  {line}", file=sys.stderr)

    if r.returncode != 0:
        print(f"FAILED (exit {r.returncode})", file=sys.stderr)
        sys.exit(1)

    # Parse JSON output
    if r.stdout.strip():
        results = json.loads(r.stdout.strip())
        print(f"Transcribed {len(results)} file(s)")
        print()

        for entry in results:
            lang = entry.get("language", "?")
            text = entry.get("text", "").strip()
            n_segments = len(entry.get("segments", []))
            print(f"  Language: {lang}")
            print(f"  Segments: {n_segments}")
            print(f"  Text ({len(text)} chars):")
            print()
            # Print first 500 chars
            print(text[:500])
            if len(text) > 500:
                print("...")
            print()

    # Check output files
    print(f"Output files ({output_dir}):")
    for f in sorted(output_dir.iterdir()):
        if f.name.startswith("."):
            continue
        size = f.stat().st_size
        print(f"  {f.name} ({size:,} bytes)")

    print()
    print("OK")


if __name__ == "__main__":
    run()
