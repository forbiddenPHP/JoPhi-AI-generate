"""Test: RVC — voice conversion (all installed models + say+RVC pipeline).

Dynamically discovers all installed RVC models and tests each one.
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
RVC_MODELS_DIR = SCRIPT_DIR / "worker" / "rvc" / "models"


def _get_voices():
    """Get list of installed RVC voice models."""
    if not RVC_MODELS_DIR.exists():
        return []
    return sorted(d.name for d in RVC_MODELS_DIR.iterdir() if d.is_dir())


def register(suite):
    out = suite.out_dir
    prep = suite.prep_dir
    # say generates: say_{voice}_{first5words}.wav in the -o directory
    prep_say = prep / "say_default_das_ist_ein_test_der.wav"

    voices = _get_voices()
    if not voices:
        suite.add(
            name="RVC convert (no models installed — SKIP)",
            cmd=[sys.executable, "-c", "import sys; print('No RVC models'); sys.exit(1)"],
        )
        return

    # Prep: generate Say output for conversion
    suite.add(
        name="Prep: generate Say input",
        cmd=[
            sys.executable, "generate.py", "voice", "--engine", "say",
            "--text", "Das ist ein Test der RVC Stimmkonvertierung.",
            "-o", str(prep),
        ],
        output=prep_say,
        prep=True,
    )

    # Start RVC server
    suite.add(
        name="RVC server start",
        cmd=[sys.executable, "generate.py", "server", "start"],
        prep=True,
    )

    # Test: Convert Say output through every installed model
    prep_say_name = prep_say.name
    for voice in voices:
        suite.add(
            name=f"RVC convert → {voice}",
            cmd=[
                sys.executable, "generate.py", "voice", "--engine", "rvc",
                "--model", voice,
                str(prep_say),
                "-o", str(out / "convert" / voice),
            ],
            output=out / "convert" / voice / prep_say_name,
        )

    # Test: Say+RVC pipeline (first model)
    suite.add(
        name=f"Say+RVC pipeline ({voices[0]})",
        cmd=[
            sys.executable, "generate.py", "voice", "--engine", "say",
            "--model", voices[0],
            "--text", "Das ist ein Test der Say plus RVC Pipeline.",
            "-o", str(out),
        ],
    )

    # Stop RVC server
    suite.add(
        name="RVC server stop",
        cmd=[sys.executable, "generate.py", "server", "stop"],
    )
