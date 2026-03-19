"""Test: Say — macOS native TTS (3 variants)."""

import sys


def register(suite):
    out = suite.out_dir

    # say generates: say_{voice}_{first5words}.wav in the -o directory
    suite.add(
        name="Say default voice (DE)",
        cmd=[
            sys.executable, "generate.py", "voice", "say",
            "--text", "Der Fuchs springt über den Bach und klaut dem Ofen die Tür.",
            "-o", str(out),
        ],
        output=out / "say_default_der_fuchs_springt_über_den.wav",
    )

    suite.add(
        name="Say Anna voice (DE)",
        cmd=[
            sys.executable, "generate.py", "voice", "say",
            "-v", "Anna",
            "--text", "Petra hat es gesehen und gelacht. Das war wirklich unglaublich.",
            "-o", str(out),
        ],
        output=out / "say_Anna_petra_hat_es_gesehen_und.wav",
    )

    suite.add(
        name="Say custom rate 180 (DE)",
        cmd=[
            sys.executable, "generate.py", "voice", "say",
            "-v", "Anna", "--rate", "180",
            "--text", "Manchmal muss man einfach schneller reden, damit die Leute zuhören.",
            "-o", str(out),
        ],
        output=out / "say_Anna_manchmal_muss_man_einfach_schneller.wav",
    )
