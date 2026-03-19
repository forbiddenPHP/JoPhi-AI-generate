"""Test: Voice Removal — generate song then strip vocals."""

import sys

LYRICS = """[verse]
Friday night the city lights are calling
Dancing shoes are on and we ain't stalling
Bass is pumping through the floor
Open up that door we want more

[chorus]
We're living for the weekend
Hands up touch the ceiling
Nothing gonna stop us now
We're screaming loud and proud"""

TAGS = "upbeat party pop, energetic, 128 bpm, female vocal, catchy synth hooks, four on the floor drums"


def register(suite):
    out = suite.out_dir
    prep = suite.prep_dir
    prep_song = prep / "party_anthem_60s.wav"

    # Prep: generate 60s song
    suite.add(
        name="Prep: generate 60s party song (ACE-Step)",
        cmd=[
            sys.executable, "generate.py", "audio", "ace-step",
            "--model", "turbo",
            "-l", LYRICS, "-t", TAGS,
            "-s", "60", "-o", str(prep_song),
        ],
        output=prep_song,
        prep=True,
    )

    suite.add(
        name="Voice removal (strip vocals)",
        cmd=[
            sys.executable, "generate.py", "audio", "voice-removal",
            str(prep_song),
            "-o", str(out),
        ],
        output=out / "party_anthem_60s_no_vocals.wav",
    )
