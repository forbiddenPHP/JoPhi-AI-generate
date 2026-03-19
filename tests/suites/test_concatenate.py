"""Test: Audio Concatenation — jingle + dialog, simple concat."""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
PODCAST = SCRIPT_DIR / "tests" / "assets" / "podcast-5min.mp3"

DIALOG = [
    ("Dylan", "Hey, hast du schon das neue Zelda gespielt? Ich komme da nicht mehr weg!"),
    ("Uncle_Fu", "Zelda? Nee, ich hänge gerade bei Baldur's Gate 3. Hundert Stunden und ich bin erst im zweiten Akt."),
    ("Dylan", "Oh Mann, das kenne ich. Ich hab allein drei Stunden damit verbracht, meinen Charakter zu erstellen."),
    ("Uncle_Fu", "Genau! Und dann speichert man vor jeder Entscheidung, weil man bloß nichts verpassen will."),
    ("Dylan", "Weißt du was mich nervt? Wenn ein Spiel einen Fotomodus hat. Dann mach ich nur noch Screenshots."),
    ("Uncle_Fu", "Ha! Bei mir sind es die Nebenquests. Ich kann einfach nicht an einem Fragezeichen vorbeigehen."),
]


def register(suite):
    out = suite.out_dir
    prep = suite.prep_dir

    # Prep: generate dialog snippets
    snippet_paths = []
    for i, (voice, text) in enumerate(DIALOG):
        snippet = prep / f"dialog_{i:02d}_{voice.lower()}.wav"
        snippet_paths.append(snippet)
        suite.add(
            name=f"Prep: dialog snippet {i+1} ({voice})",
            cmd=[
                sys.executable, "generate.py", "voice", "ai-tts",
                "-v", voice, "--language", "de",
                "--text", text,
                "-o", str(snippet),
            ],
            output=snippet,
            prep=True,
        )

    # Test 1: Jingle + Dialog with per-clip options
    all_files = []
    clip_args = []

    if PODCAST.exists():
        all_files.append(str(PODCAST))
        clip_args += ["--clip", "0:start=0,end=6.9,volume=0.5,fade-in=0.3,fade-out=0.3"]

    for i, snippet in enumerate(snippet_paths):
        file_idx = len(all_files)
        all_files.append(str(snippet))
        voice = DIALOG[i][0]
        pan = -0.5 if voice == "Dylan" else 0.5
        clip_args += ["--clip", f"{file_idx}:pan={pan}"]

    suite.add(
        name="Concatenate jingle + dialog (pan, fades)",
        cmd=[
            sys.executable, "generate.py", "output", "audio-concatenate",
            *all_files,
            *clip_args,
            "-o", str(out / "jingle_dialog.wav"),
        ],
        output=out / "jingle_dialog.wav",
    )

    # Test 2: Simple concat with fade-in/out
    simple_files = [str(p) for p in snippet_paths[:3]]
    suite.add(
        name="Concatenate simple (3 clips, fade-in/out)",
        cmd=[
            sys.executable, "generate.py", "output", "audio-concatenate",
            *simple_files,
            "--clip", "0:fade-in=0.3",
            "--clip", "2:fade-out=0.5",
            "-o", str(out / "simple_concat.wav"),
        ],
        output=out / "simple_concat.wav",
    )
