"""Test: Audio — LTX-2.3 audio generation (virtual audio worker)."""

import sys
from pathlib import Path

PROMPT = (
    "Wide shot of a crowded Oktoberfest beer tent. "
    "People in lederhosen and dirndls raise massive beer steins and cheer. "
    "A brass band plays on a small stage. "
    "Waitresses carry six steins at once through the narrow rows. "
    "Pan across laughing faces, clinking glasses, and swaying crowds."
)


def register(suite):
    out = suite.out_dir

    suite.add(
        name="Audio LTX distilled (Oktoberfest 10s)",
        cmd=[
            sys.executable, "generate.py", "audio", "ltx2.3",
            "--model", "distilled",
            "--text", PROMPT,
            "--seconds", "10",
            "--seed", "42",
            "-o", str(out / "ltx_oktoberfest_distilled.wav"),
        ],
        output=out / "ltx_oktoberfest_distilled.wav",
    )
