"""Test: Enhance — audio denoising + super-resolution (2 modes)."""

import sys
from pathlib import Path

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
    prep_vocals = prep / "prep_song_60s_vocals.wav"

    # Prep 1: generate a 60s song
    suite.add(
        name="Prep: generate 60s song (ACE-Step)",
        cmd=[
            sys.executable, "generate.py", "audio", "--engine", "ace-step",
            "--model", "turbo",
            "-l", PREP_LYRICS,
            "-t", "rock,drums,bass,guitar,vocal",
            "-s", "60", "-o", str(prep_song),
        ],
        output=prep_song,
        prep=True,
    )

    # Prep 2: separate vocals
    suite.add(
        name="Prep: demucs separation (vocals)",
        cmd=[
            sys.executable, "generate.py", "audio", "--engine", "demucs",
            str(prep_song),
            "-o", str(prep),
        ],
        output=prep_vocals,
        prep=True,
    )

    suite.add(
        name="Enhance full (denoise + super-res)",
        cmd=[
            sys.executable, "generate.py", "audio", "--engine", "enhance",
            str(prep_vocals),
            "-o", str(out / "full"),
        ],
        output=out / "full",
    )

    suite.add(
        name="Enhance denoise only",
        cmd=[
            sys.executable, "generate.py", "audio", "--engine", "enhance",
            str(prep_vocals),
            "--denoise-only",
            "-o", str(out / "denoise-only"),
        ],
        output=out / "denoise-only",
    )
