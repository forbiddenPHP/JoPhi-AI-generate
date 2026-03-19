"""Test: Demucs — audio source separation (2 models)."""

import sys

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

    # Prep: generate a 60s song
    suite.add(
        name="Prep: generate 60s song (ACE-Step)",
        cmd=[
            sys.executable, "generate.py", "audio", "ace-step",
            "--model", "turbo",
            "-l", PREP_LYRICS,
            "-t", "rock,drums,bass,guitar,vocal",
            "-s", "60", "-o", str(prep_song),
        ],
        output=prep_song,
        prep=True,
    )

    suite.add(
        name="Demucs default model (4 stems)",
        cmd=[
            sys.executable, "generate.py", "audio", "demucs",
            str(prep_song),
            "-o", str(out / "default"),
        ],
        output=out / "default",
    )

    suite.add(
        name="Demucs fine-tuned model (htdemucs_ft)",
        cmd=[
            sys.executable, "generate.py", "audio", "demucs",
            str(prep_song),
            "--model", "htdemucs_ft",
            "-o", str(out / "fine-tuned"),
        ],
        output=out / "fine-tuned",
    )
