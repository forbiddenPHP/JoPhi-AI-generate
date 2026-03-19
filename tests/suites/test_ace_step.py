"""Test: ACE-Step — AI music generation (3 variants).

Works best if duration is >60s. The 30s cinematic test proves this —
shorter durations tend to produce less coherent results.
"""

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
We are the midnight riders, steal the show

[verse]
Starlight dripping from the rooftop edge
Balancing our futures on a razor ledge
Every whisper carries through the night
Hold my hand and we will be alright

[chorus]
We are the midnight riders, chasing dreams
Nothing is as broken as it seems
Light it up and let the music flow
We are the midnight riders, steal the show"""

LYRICS_DE = """[verse]
Die Sonne geht auf über den Dächern der Stadt
Ein neuer Tag beginnt und die Welt wird nicht satt
Die Straßen erwachen im goldenen Licht
Ein Lächeln im Gesicht das die Stille durchbricht

[chorus]
Wir tanzen durch den Morgen Hand in Hand
Zusammen sind wir stark in diesem Land
Die Musik trägt uns weit über das Meer
Wir kommen immer wieder zu dir her"""


def register(suite):
    out = suite.out_dir

    suite.add(
        name="ACE-Step turbo 120s synthwave",
        cmd=[
            sys.executable, "generate.py", "audio", "ace-step",
            "--model", "turbo",
            "-l", LYRICS,
            "-t", "synthwave,electronic,upbeat,80s,energetic",
            "-s", "120", "-o", str(out / "synthwave_turbo_120s.wav"),
        ],
        output=out / "synthwave_turbo_120s.wav",
    )

    suite.add(
        name="ACE-Step SFT 30s cinematic",
        cmd=[
            sys.executable, "generate.py", "audio", "ace-step",
            "--model", "sft",
            "-l", LYRICS,
            "-t", "cinematic,orchestral,epic,dramatic",
            "-s", "30", "-o", str(out / "cinematic_sft_30s.wav"),
        ],
        output=out / "cinematic_sft_30s.wav",
    )

    suite.add(
        name="ACE-Step Deutsch 60s pop",
        cmd=[
            sys.executable, "generate.py", "audio", "ace-step",
            "-l", LYRICS_DE,
            "-t", "german pop,female vocal,upbeat,catchy",
            "--language", "de",
            "-s", "120", "-o", str(out / "deutsch_pop_120s.wav"),
        ],
        output=out / "deutsch_pop_120s.wav",
    )
