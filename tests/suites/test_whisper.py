"""Test: Whisper — audio transcription (3 variants)."""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
PODCAST = SCRIPT_DIR / "tests" / "assets" / "podcast-5min.mp3"

PREP_LYRICS = """[verse]
Stars are falling from the midnight sky
Catching wishes as they tumble by
Every corner of the universe
Sings a melody rehearsed

[chorus]
Hold on tight the world is spinning round
Lost and found we hit the ground
Dancing through the fire and the rain
Nothing ever feels the same

[verse]
Neon rivers flowing through the streets
Heartbeat syncing to the city beats
Shadows whispering our names out loud
Rising far above the crowd

[chorus]
Hold on tight the world is spinning round
Lost and found we hit the ground
Dancing through the fire and the rain
Nothing ever feels the same"""


def register(suite):
    out = suite.out_dir
    prep = suite.prep_dir
    prep_song = prep / "prep_song_60s.wav"

    # Prep: generate a 60s song for transcription input
    suite.add(
        name="Prep: generate 60s song (ACE-Step)",
        cmd=[
            sys.executable, "generate.py", "audio", "ace-step",
            "--model", "turbo",
            "-l", PREP_LYRICS,
            "-t", "pop,vocal,english,clear vocals",
            "-s", "60", "-o", str(prep_song),
        ],
        output=prep_song,
        prep=True,
    )

    suite.add(
        name="Whisper EN transcribe (all formats)",
        cmd=[
            sys.executable, "generate.py", "text", "whisper",
            str(prep_song),
            "--language", "en", "--format", "all",
            "-o", str(out / "ace-song-en"),
        ],
        output=out / "ace-song-en",
    )

    suite.add(
        name="Whisper DE transcribe (podcast)",
        cmd=[
            sys.executable, "generate.py", "text", "whisper",
            str(PODCAST),
            "--language", "de", "--format", "all",
            "-o", str(out / "podcast-de"),
        ],
        output=out / "podcast-de",
    )

    suite.add(
        name="Whisper SRT output",
        cmd=[
            sys.executable, "generate.py", "text", "whisper",
            str(prep_song),
            "--language", "en", "--format", "srt",
            "-o", str(out / "ace-song-srt"),
        ],
        output=out / "ace-song-srt",
    )
