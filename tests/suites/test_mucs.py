"""Test: Audio Mucs — stereo remix (generate → separate → enhance → mix)."""

import sys

PREP_LYRICS = """[verse]
Walking through the city in the morning light
Every sound a melody every face a sight
Strangers passing by with stories yet untold
Brave and beautiful and bold

[chorus]
Feel the rhythm of the streets below
Let the music carry you and let it flow
Every step a beat every breath a song
This is where we all belong

[verse]
Traffic lights are painting red and green
Buskers playing songs from some forgotten dream
Coffee steam is rising to the sky
Watch the world go rushing by

[chorus]
Feel the rhythm of the streets below
Let the music carry you and let it flow
Every step a beat every breath a song
This is where we all belong"""


def register(suite):
    out = suite.out_dir
    prep = suite.prep_dir
    prep_song = prep / "song_60s.wav"
    stems_dir = prep / "stems"
    enhanced_dir = prep / "enhanced"

    # Prep 1: generate 60s song
    suite.add(
        name="Prep: generate 60s song (ACE-Step)",
        cmd=[
            sys.executable, "generate.py", "audio", "--engine", "ace-step",
            "--model", "turbo",
            "-l", PREP_LYRICS,
            "-t", "indie rock,drums,bass,guitar,vocal,warm",
            "-s", "60", "-o", str(prep_song),
        ],
        output=prep_song,
        prep=True,
    )

    # Prep 2: separate into stems
    suite.add(
        name="Prep: demucs separation",
        cmd=[
            sys.executable, "generate.py", "audio", "--engine", "demucs",
            str(prep_song),
            "-o", str(stems_dir),
        ],
        output=stems_dir,
        prep=True,
    )

    # Prep 3: enhance vocals
    vocals_stem = stems_dir / "song_60s_vocals.wav"
    suite.add(
        name="Prep: enhance vocals",
        cmd=[
            sys.executable, "generate.py", "audio", "--engine", "enhance",
            str(vocals_stem),
            "-o", str(enhanced_dir),
        ],
        output=enhanced_dir,
        prep=True,
    )

    # Test: stereo remix with pan and volume
    # Use original stems + enhanced vocals
    drums = stems_dir / "song_60s_drums.wav"
    bass = stems_dir / "song_60s_bass.wav"
    other = stems_dir / "song_60s_other.wav"
    enhanced_vocals = enhanced_dir / "song_60s_vocals.wav"

    suite.add(
        name="Mucs stereo remix (4 stems, pan + volume)",
        cmd=[
            sys.executable, "generate.py", "output", "--engine", "audio-mucs",
            str(enhanced_vocals), str(drums), str(bass), str(other),
            "--clip", "0:pan=-0.2,volume=0.9",   # vocals slightly left
            "--clip", "1:pan=0.3,volume=0.6",     # drums right
            "--clip", "2:pan=0.0,volume=0.7",     # bass center
            "--clip", "3:pan=-0.5,volume=0.5",    # other far left
            "-o", str(out / "stereo_remix.wav"),
        ],
        output=out / "stereo_remix.wav",
    )
