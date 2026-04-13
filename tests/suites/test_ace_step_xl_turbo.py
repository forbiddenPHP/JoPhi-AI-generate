"""Test: ACE-Step XL-Turbo — 4B distilled model, movie score instrumental (3 durations)."""

import sys


def register(suite):
    out = suite.out_dir

    for duration in ("30", "60", "120"):
        suite.add(
            name=f"ACE-Step XL-Turbo {duration}s movie score",
            cmd=[
                sys.executable, "generate.py", "audio", "ace-step",
                "--model", "xl-turbo",
                "-l", "[instrumental]",
                "-t", "A slow piano intro that builds into soaring strings, followed by a brass crescendo leading to a full orchestra climax, ending with a soft emotional fade out. Cinematic film soundtrack.",
                "-s", duration, "-o", str(out / f"movie_score_xl_turbo_{duration}s.wav"),
            ],
            output=out / f"movie_score_xl_turbo_{duration}s.wav",
        )
