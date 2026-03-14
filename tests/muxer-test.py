#!/usr/bin/env python3
"""
Muxer-Test — Full Roundtrip: Generate → Separate → Enhance → Stereo-Remix.

1. ACE-Step:   Generate a 2-minute song
2. demucs:     Separate into stems (vocals, drums, bass, other)
3. enhance:    Enhance ALL stems (1x + 2x for stereo widening)
4. audio-mucs: Remix 8 tracks (each stem duplicated L/R)

Ergebnis:
  demos/muxer-test/original.wav     ← Original zum Vergleich
  demos/muxer-test/remixed.wav      ← Stereo-Remix

Zwischenschritte:
  demos/zwischenschritte/muxer-test/stems/        ← demucs Stems
  demos/zwischenschritte/muxer-test/enhanced/      ← 1x + 2x enhanced

Usage:  python tests/muxer-test.py
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent

# Endergebnis: Original + Remix nebeneinander
OUT = SCRIPT_DIR / "demos" / "muxer-test"
ORIGINAL = OUT / "song.wav"
REMIX = OUT / "remixed.wav"

# Zwischenschritte
WORK = SCRIPT_DIR / "demos" / "zwischenschritte" / "muxer-test"
STEMS_DIR = WORK / "stems"
ENHANCED_DIR = WORK / "enhanced"

LYRICS = """\
[verse]
Bits und Bytes im Morgenlicht,
der Server summt, die Leitung spricht.
Ein Cursor blinkt im leeren Raum,
die Nacht war kurz, fast wie ein Traum.

[chorus]
Wir coden durch die Dunkelheit,
in Nullen, Einsen, Ewigkeit.
Der Compiler schweigt, der Build ist grün,
ein kleines Wunder, digital und kühn.

[verse]
Stack Overflow um halb drei,
ein Semikolon war nicht dabei.
Der Bug versteckt sich tief im Code,
doch wir debuggen bis zum Morgenrot.

[chorus]
Wir coden durch die Dunkelheit,
in Nullen, Einsen, Ewigkeit.
Der Compiler schweigt, der Build ist grün,
ein kleines Wunder, digital und kühn.

[bridge]
Und wenn die Sonne langsam steigt,
der letzte Test sich grün anzeigt,
dann lehnen wir uns still zurück —
ein Push, ein Merge, ein kleines Glück.

[chorus]
Wir coden durch die Dunkelheit,
in Nullen, Einsen, Ewigkeit.
Der Compiler schweigt, der Build ist grün,
ein kleines Wunder, digital und kühn.
"""

TAGS = "indie rock, german, male vocals, electric guitar, drums, 120 bpm, energetic"

# Stem-Konfiguration: name, pan_left, pan_right, volume
STEM_CONFIG = [
    #  name      pan_L   pan_R   vol
    ("vocals",  -0.2,   +0.2,   0.7),
    ("drums",   -0.4,   +0.4,   0.6),
    ("bass",    -0.15,  +0.15,  0.6),
    ("other",   -0.6,   +0.6,   0.5),
]

passed = 0
failed = 0
results = []


def run(name: str, cmd: list[str]) -> bool:
    print()
    print("━" * 64)
    print(f"  STEP: {name}")
    print("━" * 64)
    print(f"  CMD: {' '.join(cmd)}")
    print()
    result = subprocess.run(cmd)
    ok = result.returncode == 0
    tag = "PASS" if ok else "FAIL"
    print(f"\n  >>> {tag}")
    results.append(f"{tag}  {name}")
    return ok


def enhance_stem(input_path: Path, output_path: Path, label: str,
                  mode: str = "denoise") -> bool:
    """Enhance a single stem, with caching.

    mode: "denoise" (vocals), "enhance" (full), "enhance-only" (no denoise)
    enhance -o expects a DIRECTORY, output file = dir/input_name.
    We use a temp dir, then rename the result to output_path.
    """
    global passed, failed
    if output_path.exists() and output_path.is_file():
        print(f"  CACHED: {output_path.name}")
        passed += 1
        results.append(f"SKIP  {label} (cached)")
        return True

    # enhance writes into a directory, keeping the input filename
    tmp_dir = output_path.parent / f".tmp_{output_path.stem}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    mode_flag = {"denoise": "--denoise-only", "enhance-only": "--enhance-only"}.get(mode)
    cmd = [
        sys.executable, "generate.py", "audio",
        "--engine", "enhance",
        str(input_path),
        "-o", str(tmp_dir),
    ]
    if mode_flag:
        cmd.insert(4, mode_flag)

    ok = run(label, cmd)

    # Result is tmp_dir / input_path.name
    result_file = tmp_dir / input_path.name
    if ok and result_file.exists():
        result_file.rename(output_path)
        tmp_dir.rmdir()
        passed += 1
        return True
    else:
        failed += 1
        return False


def main():
    global passed, failed

    os.chdir(SCRIPT_DIR)
    OUT.mkdir(parents=True, exist_ok=True)
    STEMS_DIR.mkdir(parents=True, exist_ok=True)
    ENHANCED_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Generate song with ACE-Step (2 min) ──────────────────────────

    if ORIGINAL.exists():
        print(f"  CACHED: {ORIGINAL}")
        passed += 1
        results.append("SKIP  Song generieren (cached)")
    else:
        ok = run(
            "ACE-Step: Song generieren (2 Min)",
            [
                sys.executable, "generate.py", "audio",
                "--engine", "ace-step",
                "-l", LYRICS,
                "-t", TAGS,
                "--seconds", "120",
                "-o", str(ORIGINAL),
            ],
        )
        if ok and ORIGINAL.exists():
            passed += 1
            size = ORIGINAL.stat().st_size
            print(f"  Output: {ORIGINAL} ({size:,} bytes)")
        else:
            failed += 1
            print("  ABORT: Song konnte nicht erzeugt werden.")
            sys.exit(1)

    # ── Step 2: Separate with demucs ─────────────────────────────────────────

    stem_names = [s[0] for s in STEM_CONFIG]
    # demucs prefixes stems with input filename: original_vocals.wav etc.
    input_prefix = ORIGINAL.stem  # "original"
    expected_stems = {s: f"{input_prefix}_{s}.wav" for s in stem_names}
    stems_exist = all((STEMS_DIR / f).exists() for f in expected_stems.values())

    if stems_exist:
        print(f"  CACHED: Stems in {STEMS_DIR}")
        passed += 1
        results.append("SKIP  demucs Separation (cached)")
    else:
        ok = run(
            "demucs: Stems trennen",
            [
                sys.executable, "generate.py", "audio",
                "--engine", "demucs",
                str(ORIGINAL),
                "-o", str(STEMS_DIR),
            ],
        )
        stems_exist_after = all((STEMS_DIR / f).exists() for f in expected_stems.values())
        if ok and stems_exist_after:
            passed += 1
            for stem_name, filename in expected_stems.items():
                size = (STEMS_DIR / filename).stat().st_size
                print(f"  Stem: {filename} ({size:,} bytes)")
        else:
            failed += 1
            print("  ABORT: Stems konnten nicht erzeugt werden.")
            print(f"  Vorhandene Dateien in {STEMS_DIR}:")
            for f in sorted(STEMS_DIR.rglob("*")):
                if f.is_file():
                    print(f"    {f.relative_to(STEMS_DIR)}")
            sys.exit(1)

    # ── Step 3: Enhance ALL stems (1x + 2x) ──────────────────────────────────
    #   1x enhanced → links
    #   2x enhanced (1x nochmal enhanced) → rechts

    print()
    print("━" * 64)
    print("  ENHANCE: Alle Stems 1x + 2x für Stereo-Widening")
    print("━" * 64)

    enhanced_files = {}  # stem_name -> (1x_path, 2x_path)

    for stem_name in stem_names:
        raw = STEMS_DIR / expected_stems[stem_name]
        enh_1x = ENHANCED_DIR / f"{stem_name}_1x.wav"
        enh_2x = ENHANCED_DIR / f"{stem_name}_2x.wav"

        # Vocals: denoise (removes background noise, keeps speech)
        # Instrumentals: enhance-only (super-resolution, NO denoising —
        #   denoiser treats instruments as noise and silences them)
        mode = "denoise" if stem_name == "vocals" else "enhance-only"

        # 1x enhance
        ok1 = enhance_stem(raw, enh_1x, f"enhance 1x: {stem_name}", mode=mode)
        if not ok1:
            print(f"  WARNING: 1x enhance von {stem_name} fehlgeschlagen, nutze Original.")
            shutil.copy2(raw, enh_1x)

        # 2x enhance (enhance the 1x version again)
        ok2 = enhance_stem(enh_1x, enh_2x, f"enhance 2x: {stem_name}", mode=mode)
        if not ok2:
            print(f"  WARNING: 2x enhance von {stem_name} fehlgeschlagen, nutze 1x.")
            shutil.copy2(enh_1x, enh_2x)

        enhanced_files[stem_name] = (enh_1x, enh_2x)

    # ── Step 4: Stereo-Remix with audio-mucs ─────────────────────────────────
    #   8 Tracks: je Stem 1x (links) + 2x (rechts)

    all_files = []
    clip_args = []

    for stem_name, pan_l, pan_r, vol in STEM_CONFIG:
        enh_1x, enh_2x = enhanced_files[stem_name]

        # 1x enhanced → links
        idx_l = len(all_files)
        all_files.append(str(enh_1x))
        clip_args += ["--clip", f"{idx_l}:pan={pan_l},volume={vol}"]

        # 2x enhanced → rechts
        idx_r = len(all_files)
        all_files.append(str(enh_2x))
        clip_args += ["--clip", f"{idx_r}:pan={pan_r},volume={vol}"]

    # Fade-in/out auf dem gesamten Mix (erste und letzte Spur)
    clip_args += ["--clip", "0:fade-in=0.5"]
    clip_args += ["--clip", f"{len(all_files) - 1}:fade-out=1.0"]

    ok = run(
        "audio-mucs: 8-Track Stereo-Remix",
        [
            sys.executable, "generate.py", "output",
            "--engine", "audio-mucs",
            *all_files,
            *clip_args,
            "-o", str(REMIX),
        ],
    )
    if ok and REMIX.exists():
        passed += 1
        size = REMIX.stat().st_size
        print(f"  Output: {REMIX} ({size:,} bytes)")
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
    print()
    print(f"  ♫ Original:  {ORIGINAL}")
    print(f"  ♫ Remix:     {REMIX}")
    print(f"  Zwischenschritte: {WORK}/")
    print("━" * 64)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
