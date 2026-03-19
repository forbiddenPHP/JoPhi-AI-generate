"""Test: SFX — sound effects generation via EzAudio (3 variants)."""

import sys


def register(suite):
    out = suite.out_dir

    suite.add(
        name="SFX dog barking 5s",
        cmd=[
            sys.executable, "generate.py", "audio", "sfx",
            "--text", "a dog barking in the distance",
            "-s", "5", "-o", str(out / "dog_barking_5s.wav"),
        ],
        output=out / "dog_barking_5s.wav",
    )

    suite.add(
        name="SFX rain + thunder 8s",
        cmd=[
            sys.executable, "generate.py", "audio", "sfx",
            "--text", "rain falling on leaves as thunder rumbles in the distance",
            "-s", "8", "-o", str(out / "rain_thunder_8s.wav"),
        ],
        output=out / "rain_thunder_8s.wav",
    )

    suite.add(
        name="SFX car horn 3s (seed 42)",
        cmd=[
            sys.executable, "generate.py", "audio", "sfx",
            "--text", "a car horn honking twice on a busy street",
            "-s", "3", "--seed", "42",
            "-o", str(out / "car_horn_3s.wav"),
        ],
        output=out / "car_horn_3s.wav",
    )
