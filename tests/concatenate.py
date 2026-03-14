#!/usr/bin/env python3
"""
Concatenate Test — Jingle + Dialog über Computerspiele.

1. Jingle (podcast-5min.mp3, erste 6.9s, halbe Lautstärke, fade-in/out)
2. Dylan und Uncle_Fu unterhalten sich über Computerspiele
3. Alles zusammen via generate.py output --engine audio-concatenate

Usage:  python tests/concatenate.py
"""

import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
SNIPPETS_DIR = SCRIPT_DIR / "demos" / "schnippsel"
OUT = SCRIPT_DIR / "demos" / "concatinate-test"
PODCAST_FILE = SCRIPT_DIR / "demos" / "test-input-audio-file" / "podcast-5min.mp3"

# ── Dialog: Computerspiele ───────────────────────────────────────────────────

SNIPPETS = [
    {
        "name": "dialog_01_dylan",
        "voice": "Dylan",
        "language": "de",
        "text": "Hey, hast du schon das neue Zelda gespielt? Ich komme da nicht mehr weg!",
    },
    {
        "name": "dialog_02_uncle_fu",
        "voice": "Uncle_Fu",
        "language": "de",
        "text": "Zelda? Nee, ich hänge gerade bei Baldur's Gate 3. Hundert Stunden und ich bin erst im zweiten Akt.",
    },
    {
        "name": "dialog_03_dylan",
        "voice": "Dylan",
        "language": "de",
        "text": "Oh Mann, das kenne ich. Ich hab allein drei Stunden damit verbracht, meinen Charakter zu erstellen.",
    },
    {
        "name": "dialog_04_uncle_fu",
        "voice": "Uncle_Fu",
        "language": "de",
        "text": "Genau! Und dann speichert man vor jeder Entscheidung, weil man bloß nichts verpassen will.",
    },
    {
        "name": "dialog_05_dylan",
        "voice": "Dylan",
        "language": "de",
        "text": "Weißt du was mich nervt? Wenn ein Spiel einen Fotomodus hat. Dann mach ich nur noch Screenshots statt zu spielen.",
    },
    {
        "name": "dialog_06_uncle_fu",
        "voice": "Uncle_Fu",
        "language": "de",
        "text": "Ha! Bei mir sind es die Nebenquests. Ich kann einfach nicht an einem Fragezeichen auf der Karte vorbeigehen.",
    },
]

passed = 0
failed = 0
results = []


def run(name: str, cmd: list[str]) -> bool:
    print()
    print("━" * 64)
    print(f"  TEST: {name}")
    print("━" * 64)
    print(f"  CMD: {' '.join(cmd)}")
    print()
    result = subprocess.run(cmd)
    ok = result.returncode == 0
    tag = "PASS" if ok else "FAIL"
    print(f"\n  >>> {tag}")
    results.append(f"{tag}  {name}")
    return ok


def main():
    global passed, failed

    os.chdir(SCRIPT_DIR)
    SNIPPETS_DIR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Generate dialog snippets (only if missing) ───────────────────

    wav_files: list[str] = []

    for s in SNIPPETS:
        out_path = SNIPPETS_DIR / f"{s['name']}.wav"
        if out_path.exists():
            print(f"  CACHED: {out_path}")
            wav_files.append(str(out_path))
            continue

        result = subprocess.run([
            sys.executable, "generate.py", "voice", "--engine", "ai-tts",
            "-v", s["voice"],
            "--language", s["language"],
            "--text", s["text"],
            "-o", str(out_path),
        ])
        if result.returncode == 0 and out_path.exists():
            wav_files.append(str(out_path))
        else:
            print(f"  ERROR: Failed to generate {s['name']}")
            sys.exit(1)

    # ── Step 2: Concatenate — Jingle + Dialog → WAV ─────────────────────────
    #   Reihenfolge: Jingle (6.9s, leise) → Dialog (6 Schnippsel)

    all_files = []
    clip_args = []

    # Jingle (podcast-5min.mp3, 0–6.9s, volume 0.5, fade-in + fade-out)
    if PODCAST_FILE.exists():
        jingle_idx = len(all_files)
        all_files.append(str(PODCAST_FILE))
        clip_args += ["--clip", f"{jingle_idx}:start=0,end=6.9,volume=0.5,fade-in=0.3,fade-out=0.3"]
    else:
        print(f"  INFO: {PODCAST_FILE} not found, skipping jingle.")

    # Dialog-Schnippsel (Dylan leicht links, Uncle_Fu leicht rechts)
    for i, s in enumerate(SNIPPETS):
        file_idx = len(all_files)
        all_files.append(wav_files[i])
        pan = -0.5 if s["voice"] == "Dylan" else 0.5
        clip_args += ["--clip", f"{file_idx}:pan={pan}"]

    # Fade-out auf dem letzten Schnippsel
    last_idx = len(all_files) - 1
    clip_args += ["--clip", f"{last_idx}:fade-out=0.5"]

    concat_out = str(OUT / "computerspiele_dialog.wav")
    ok = run(
        "Jingle + Dialog Computerspiele → WAV",
        [
            sys.executable, "generate.py", "output",
            "--engine", "audio-concatenate",
            *all_files,
            *clip_args,
            "-o", concat_out,
        ],
    )
    if ok and Path(concat_out).exists():
        passed += 1
        size = Path(concat_out).stat().st_size
        print(f"  Output: {concat_out} ({size:,} bytes)")
    else:
        failed += 1

    # ── Summary ──────────────────────────────────────────────────────────────

    total = passed + failed
    print()
    print("━" * 64)
    print("  RESULTS")
    print("━" * 64)
    for r in results:
        print(f"  {r}")
    print()
    print(f"  Total: {passed} passed, {failed} failed (of {total})")
    print(f"  Output: {OUT}/")
    print("━" * 64)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
