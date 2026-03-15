#!/usr/bin/env python3
"""Pipeline: Split music into stems (vocals, drums, bass, other) via demucs.

Usage:
    python split-music-tracks.py demos/cheerful-tendency.wav
    python split-music-tracks.py demos/cheerful-tendency.wav --model htdemucs_ft
    python split-music-tracks.py demos/cheerful-tendency.wav --verify

Output structure:
    demos/cheerful-tendency/
        cheerful-tendency_vocals.wav
        cheerful-tendency_drums.wav
        cheerful-tendency_bass.wav
        cheerful-tendency_other.wav
        (with --verify:)
        cheerful-tendency_stats.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

CONDA_BIN = "/opt/miniconda3/bin/conda"
PROJECT_DIR = Path(__file__).resolve().parent.parent

EXPECTED_STEMS = ["vocals", "drums", "bass", "other"]


def run(cmd: list[str], label: str, timeout: int = 3600):
    """Run a subprocess, stream stderr, check return code."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}\n")

    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    if r.stderr:
        for line in r.stderr.strip().split("\n"):
            if "FutureWarning" in line or "pynvml" in line:
                continue
            if "ds_accelerator" in line:
                continue
            print(f"  {line}")

    if r.stdout.strip():
        print(r.stdout.strip())

    if r.returncode != 0:
        print(f"\nERROR: {label} failed (exit code {r.returncode})")
        sys.exit(1)

    return r


def verify_stems(output_dir: Path, stem: str):
    """Verify stem separation output: check files exist, durations match."""
    print(f"\n  ── Stem Verification ──")

    stem_files = {}
    missing = []
    for s in EXPECTED_STEMS:
        p = output_dir / f"{stem}_{s}.wav"
        if p.exists():
            stem_files[s] = p
        else:
            missing.append(s)

    if missing:
        print(f"    ERROR: Missing stems: {', '.join(missing)}")
        sys.exit(1)

    print(f"    Stems found: {len(stem_files)}/{len(EXPECTED_STEMS)}")

    # Check durations via ffprobe
    durations = {}
    for s, p in stem_files.items():
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries",
                 "format=duration", "-of", "csv=p=0", str(p)],
                capture_output=True, text=True, timeout=10)
            dur = float(r.stdout.strip())
            durations[s] = dur
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"    {s:8s}: {dur:.1f}s  ({size_mb:.1f} MB)")
        except (ValueError, subprocess.TimeoutExpired):
            print(f"    {s:8s}: duration unknown")

    # Check all durations match (within 0.1s)
    if durations:
        ref_dur = list(durations.values())[0]
        mismatched = [s for s, d in durations.items()
                      if abs(d - ref_dur) > 0.1]
        if mismatched:
            print(f"\n    WARNING: Duration mismatch: {', '.join(mismatched)}")
        else:
            print(f"\n    All stems have matching duration ({ref_dur:.1f}s)")

    # Save stats JSON
    stats = {
        "stems": len(stem_files),
        "expected": EXPECTED_STEMS,
        "durations": {s: round(d, 3) for s, d in durations.items()},
        "sizes_mb": {s: round(p.stat().st_size / (1024 * 1024), 2)
                     for s, p in stem_files.items()},
    }
    stats_path = output_dir / f"{stem}_stats.json"
    stats_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"    Saved: {stats_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Split music into stems via demucs")
    parser.add_argument("input", type=Path, help="Input audio file")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output directory (default: same dir as input, named after file stem)")
    parser.add_argument("--model", default="htdemucs",
                        help="Demucs model (default: htdemucs, alt: htdemucs_ft)")
    parser.add_argument("--verify", action="store_true",
                        help="Verify output stems (check files, durations, sizes)")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: File not found: {args.input}")
        sys.exit(1)

    stem = args.input.stem
    output_dir = args.output or (args.input.parent / stem)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Check if output already exists ────────────────────────────────────
    existing_stems = [output_dir / f"{stem}_{s}.wav" for s in EXPECTED_STEMS]
    all_exist = all(p.exists() for p in existing_stems)

    if all_exist and args.verify:
        print(f"\n  Stems already exist, skipping separation.")
        print(f"  (Delete {output_dir} to force re-separation.)")
    else:
        # ── Run separation ────────────────────────────────────────────────
        separate_script = PROJECT_DIR / "worker/separate" / "separate.py"

        separate_cmd = [
            CONDA_BIN, "run", "-n", "separate",
            "python", str(separate_script),
            str(args.input),
            "-o", str(output_dir),
            "--model", args.model,
        ]

        run(separate_cmd, f"Separate → {output_dir}")

    # ── Verify ────────────────────────────────────────────────────────────
    if args.verify:
        verify_stems(output_dir, stem)

    # ── Summary ───────────────────────────────────────────────────────────
    stem_wavs = sorted(output_dir.glob(f"{stem}_*.wav"))

    print(f"\n{'=' * 60}")
    print(f"  Done!")
    print(f"{'=' * 60}")
    print(f"\n  Output: {output_dir}/")
    print(f"  Stems: {len(stem_wavs)}")
    for wav in stem_wavs:
        print(f"    {wav.name}")
    if not args.verify:
        print(f"\n  Tip: --verify to check output stems.")
    print()


if __name__ == "__main__":
    main()
