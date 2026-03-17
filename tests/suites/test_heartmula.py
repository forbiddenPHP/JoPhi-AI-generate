"""Test: HeartMuLa — AI music generation (max 60s, slow engine)."""

import sys

LYRICS = """[verse]
Neon lights are burning through the haze
Walking down the boulevard in a purple daze
Every heartbeat echoes off the walls
Dancing shadows answer when the city calls

[chorus]
We are the midnight riders, chasing dreams
Nothing is as broken as it seems
Light it up and let the music flow
We are the midnight riders, steal the show"""


def register(suite):
    out = suite.out_dir

    suite.add(
        name="HeartMuLa 60s synthwave",
        cmd=[
            sys.executable, "generate.py", "audio", "--engine", "heartmula",
            "-l", LYRICS,
            "-t", "synthwave,electronic,upbeat,80s,energetic",
            "-s", "60", "-o", str(out / "synthwave_60s.wav"),
        ],
        output=out / "synthwave_60s.wav",
    )
