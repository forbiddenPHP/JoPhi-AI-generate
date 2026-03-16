#!/usr/bin/env python3
"""Test: Voice removal — generate a song, then strip vocals.

Step 1: Generate a 200s upbeat party song via ACE-Step.
Step 2: Remove vocals using demucs separation + non-vocal stem remix.

Output: demos/voice-removal/party-anthem.mp3 (original)
        demos/voice-removal/party-anthem_no_vocals.wav (instrumental)
"""

import subprocess
import sys
from pathlib import Path


LYRICS = """[Intro]
Oh yeah, here we go!

[Verse 1]
Friday night the city lights are calling
Dancing shoes are on and we ain't stalling
Bass is pumping through the floor
Open up that door we want more

[Chorus]
We're living for the weekend
Hands up touch the ceiling
Nothing gonna stop us now
We're screaming loud and proud

[Verse 2]
DJ spinning records til the sunrise
Every single moment feels like a prize
Strangers turning into friends
Hope this night never ends

[Chorus]
We're living for the weekend
Hands up touch the ceiling
Nothing gonna stop us now
We're screaming loud and proud

[Bridge]
Turn it up turn it up
Can you feel the beat
Turn it up turn it up
Move your feet

[Chorus]
We're living for the weekend
Hands up touch the ceiling
Nothing gonna stop us now
We're screaming loud and proud

[Outro]
Yeah we're living for the weekend
Living for the weekend
"""

TAGS = (
    "upbeat party pop, energetic, 128 bpm, C major, "
    "female vocal, catchy synth hooks, four on the floor drums, "
    "disco strings, funky bass, euphoric drop"
)

OUTPUT_DIR = Path(__file__).parent.parent / "demos" / "voice-removal"
SONG = OUTPUT_DIR / "party-anthem.mp3"
INSTRUMENTAL = OUTPUT_DIR / "party-anthem_no_vocals.wav"
SECONDS = 200


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Generate song ─────────────────────────────────────────────
    print(f"=== Step 1: Generating {SECONDS}s party anthem ===")
    print(f"Tags: {TAGS}")
    print(f"Output: {SONG}")

    gen_cmd = [
        sys.executable,
        str(Path(__file__).parent.parent / "generate.py"),
        "audio",
        "--engine", "ace-step",
        "-l", LYRICS,
        "-t", TAGS,
        "--language", "en",
        "-s", str(SECONDS),
        "-o", str(SONG),
    ]

    result = subprocess.run(gen_cmd)
    if result.returncode != 0:
        print(f"FAILED: song generation (exit {result.returncode})")
        sys.exit(result.returncode)

    if not SONG.exists() or SONG.stat().st_size == 0:
        print("FAILED: song not created or empty")
        sys.exit(1)

    print(f"Song OK: {SONG.name} ({SONG.stat().st_size:,} bytes)")

    # ── Step 2: Remove vocals ─────────────────────────────────────────────
    print(f"\n=== Step 2: Removing vocals ===")
    print(f"Input: {SONG.name}")
    print(f"Output: {INSTRUMENTAL.name}")

    rm_cmd = [
        sys.executable,
        str(Path(__file__).parent.parent / "generate.py"),
        "audio",
        "--engine", "voice-removal",
        str(SONG),
        "-o", str(OUTPUT_DIR),
    ]

    result = subprocess.run(rm_cmd)
    if result.returncode != 0:
        print(f"FAILED: voice removal (exit {result.returncode})")
        sys.exit(result.returncode)

    if not INSTRUMENTAL.exists():
        print(f"FAILED: output not created: {INSTRUMENTAL}")
        sys.exit(1)

    size = INSTRUMENTAL.stat().st_size
    if size == 0:
        print("FAILED: instrumental output is empty")
        sys.exit(1)

    print(f"\nOK: {INSTRUMENTAL.name} ({size:,} bytes)")
    print(f"    {SONG.name} ({SONG.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
