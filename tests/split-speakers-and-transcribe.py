#!/usr/bin/env python3
"""Pipeline: Split dialogue by speaker, optionally verify.

1. Diarize: Split audio into one WAV per speaker (full length, silence elsewhere)
2. Verify (optional, --verify): Show diarization stats (gaps, overlaps, coverage)
3. Transcribe (optional, --transcribe): Transcribe compact WAVs, remap timestamps

Usage:
    python split-speakers-and-transcribe.py demos/stay_forever_test/folge_5min.mp3
    python split-speakers-and-transcribe.py demos/stay_forever_test/folge_5min.mp3 --verify
    python split-speakers-and-transcribe.py demos/stay_forever_test/folge_5min.mp3 --transcribe --language de
    python split-speakers-and-transcribe.py demos/stay_forever_test/folge_5min.mp3 --verify --transcribe

Output structure:
    demos/stay_forever_test/folge_5min/
        folge_5min_SPEAKER_00.wav
        folge_5min_SPEAKER_01.wav
        folge_5min_diarize.json
        folge_5min_stats.json              (with --verify on fresh run)
        folge_5min_verify_stats.json       (with --verify on existing data)
        (with --transcribe:)
        SPEAKER_00/
            folge_5min_SPEAKER_00_verify.json   (+ _verify.srt/vtt/txt/tsv)
        SPEAKER_01/
            folge_5min_SPEAKER_01_verify.json   (+ _verify.srt/vtt/txt/tsv)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

CONDA_BIN = "/opt/miniconda3/bin/conda"
PROJECT_DIR = Path(__file__).resolve().parent.parent


def remap_timestamps(whisper_json: dict, diarize_segments: list[dict],
                     speaker: str) -> dict:
    """Remap whisper timestamps from compact WAV back to original timeline."""
    # Build mapping: list of (compact_start, compact_end, original_start)
    mapping = []
    compact_offset = 0.0
    for seg in diarize_segments:
        if seg["speaker"] != speaker:
            continue
        duration = seg["end"] - seg["start"]
        mapping.append((compact_offset, compact_offset + duration, seg["start"]))
        compact_offset += duration

    def remap(t: float) -> float:
        for compact_start, compact_end, orig_start in mapping:
            if compact_start <= t <= compact_end + 0.01:
                return round(orig_start + (t - compact_start), 3)
        if mapping:
            cs, ce, os_ = mapping[-1]
            return round(os_ + (t - cs), 3)
        return t

    remapped = dict(whisper_json)
    remapped["segments"] = []
    for seg in whisper_json.get("segments", []):
        new_seg = dict(seg)
        new_seg["start"] = remap(seg["start"])
        new_seg["end"] = remap(seg["end"])
        if "words" in seg:
            new_seg["words"] = [
                {**w, "start": remap(w["start"]), "end": remap(w["end"])}
                for w in seg["words"]
            ]
        remapped["segments"].append(new_seg)
    return remapped


def _fmt_ts_srt(s: float) -> str:
    h, m = int(s // 3600), int((s % 3600) // 60)
    sec, ms = int(s % 60), int((s % 1) * 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def _fmt_ts_vtt(s: float) -> str:
    h, m = int(s // 3600), int((s % 3600) // 60)
    sec, ms = int(s % 60), int((s % 1) * 1000)
    return f"{h:02d}:{m:02d}:{sec:02d}.{ms:03d}"


def _write_formats(entry: dict, out_dir: Path, stem: str, fmt: str):
    """Write transcript in requested formats."""
    formats = ["txt", "srt", "vtt", "tsv"] if fmt == "all" else [fmt]
    segs = entry.get("segments", [])

    for f in formats:
        path = out_dir / f"{stem}.{f}"
        if f == "txt":
            path.write_text(entry.get("text", "").strip() + "\n", encoding="utf-8")
        elif f == "srt":
            lines = []
            for i, seg in enumerate(segs, 1):
                lines.extend([str(i),
                    f"{_fmt_ts_srt(seg['start'])} --> {_fmt_ts_srt(seg['end'])}",
                    seg["text"].strip(), ""])
            path.write_text("\n".join(lines), encoding="utf-8")
        elif f == "vtt":
            lines = ["WEBVTT", ""]
            for seg in segs:
                lines.extend([
                    f"{_fmt_ts_vtt(seg['start'])} --> {_fmt_ts_vtt(seg['end'])}",
                    seg["text"].strip(), ""])
            path.write_text("\n".join(lines), encoding="utf-8")
        elif f == "tsv":
            lines = ["start\tend\ttext"]
            for seg in segs:
                lines.append(f"{int(seg['start']*1000)}\t{int(seg['end']*1000)}\t{seg['text'].strip()}")
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"    Saved: {path}")


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


def verify_transcribe(output_dir: Path, stem: str,
                      diarize_segments: list[dict], args):
    """Transcribe compact WAVs and compare with diarization segments."""
    compact_wavs = sorted(output_dir.glob(f"{stem}_SPEAKER_*_compact.wav"))

    if not compact_wavs:
        print("ERROR: No compact WAV files found for verification.")
        sys.exit(1)

    print(f"\n  Verifying {len(compact_wavs)} speakers via transcription ...\n")

    transcribe_script = PROJECT_DIR / "worker/whisper" / "transcribe.py"

    for i, compact_path in enumerate(compact_wavs, 1):
        speaker_name = compact_path.stem.replace(f"{stem}_", "").replace("_compact", "")
        transcript_dir = output_dir / speaker_name
        transcript_dir.mkdir(parents=True, exist_ok=True)

        # Transcribe compact WAV (no silence = no hallucinations)
        transcribe_cmd = [
            CONDA_BIN, "run", "-n", "whisper",
            "python", str(transcribe_script),
            str(compact_path),
            "--model", args.transcribe_model,
            "--format", "json",
            "-o", str(transcript_dir),
        ]
        if args.language:
            transcribe_cmd.extend(["--language", args.language])

        r = run(transcribe_cmd,
                f"Verify: Transcribe {speaker_name} ({i}/{len(compact_wavs)})")

        # Remap timestamps and write output
        try:
            whisper_results = json.loads(r.stdout)
            if whisper_results:
                remapped = remap_timestamps(
                    whisper_results[0], diarize_segments, speaker_name)

                out_stem = f"{stem}_{speaker_name}_verify"
                remapped_json_path = transcript_dir / f"{out_stem}.json"
                remapped_json_path.write_text(
                    json.dumps(remapped, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )

                if args.format == "all" or args.format != "json":
                    _write_formats(remapped, transcript_dir, out_stem, args.format)

                # ── Coverage report ───────────────────────────────────
                speaker_segs = [s for s in diarize_segments
                                if s["speaker"] == speaker_name]
                diarize_duration = sum(s["end"] - s["start"]
                                       for s in speaker_segs)
                whisper_segs = remapped.get("segments", [])
                transcribed_duration = sum(s["end"] - s["start"]
                                           for s in whisper_segs)
                non_empty = [s for s in whisper_segs if s["text"].strip()]

                print(f"\n  ── {speaker_name} Verification ──")
                print(f"    Diarize segments: {len(speaker_segs)}"
                      f" ({diarize_duration:.1f}s)")
                print(f"    Transcribed segments: {len(non_empty)}"
                      f" ({transcribed_duration:.1f}s)")

                if diarize_duration > 0:
                    coverage = min(100, transcribed_duration / diarize_duration * 100)
                    print(f"    Coverage: {coverage:.0f}%")

                # Flag segments with no speech detected
                empty = [s for s in whisper_segs if not s["text"].strip()]
                if empty:
                    print(f"    Empty segments: {len(empty)}"
                          " (may need manual review)")

        except (json.JSONDecodeError, IndexError, KeyError) as e:
            print(f"  WARNING: Could not verify {speaker_name}: {e}")

    # Clean up compact WAVs
    for p in compact_wavs:
        p.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Split dialogue by speaker, optionally verify via transcription")
    parser.add_argument("input", type=Path, help="Input audio file")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output directory (default: same dir as input, named after file stem)")
    parser.add_argument("--speakers", type=int, default=None,
                        help="Number of speakers (auto-detect if not set)")
    parser.add_argument("--hf-token", default=None,
                        help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--verify", action="store_true",
                        help="Show diarization statistics (segments, coverage, gaps, overlaps)")
    parser.add_argument("--transcribe", action="store_true",
                        help="Transcribe compact WAVs to verify speaker content (implies --verify)")
    parser.add_argument("--transcribe-model", default="large-v3-turbo",
                        help="mlx-whisper model for verification (default: large-v3-turbo)")
    parser.add_argument("--format", default="all",
                        help="Transcript format: json, txt, srt, vtt, tsv, all (default: all)")
    parser.add_argument("--language", default=None,
                        help="Language hint for transcription (e.g. 'de', 'en')")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: File not found: {args.input}")
        sys.exit(1)

    stem = args.input.stem
    output_dir = args.output or (args.input.parent / stem)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Diarize (skip if output already exists) ─────────────────
    diarize_json_path = output_dir / f"{stem}_diarize.json"
    existing = diarize_json_path.exists()

    if existing and (args.verify or args.transcribe):
        print(f"\n  Diarize output exists, skipping diarization.")
        print(f"  (Delete {output_dir} to force re-diarization.)")
    else:
        diarize_script = PROJECT_DIR / "worker/diarize" / "diarize.py"

        diarize_cmd = [
            CONDA_BIN, "run", "-n", "diarize",
            "python", str(diarize_script),
            str(args.input),
            "-o", str(output_dir),
        ]
        if args.speakers:
            diarize_cmd.extend(["--speakers", str(args.speakers)])
        if args.hf_token:
            diarize_cmd.extend(["--hf-token", args.hf_token])
        if args.verify:
            diarize_cmd.append("--verify")

        run(diarize_cmd, f"Diarize → {output_dir}")

    # ── Verify stats (on existing data → verify_ prefix) ────────────────
    if args.verify and existing:
        diarize_script = PROJECT_DIR / "worker/diarize" / "diarize.py"
        verify_cmd = [
            CONDA_BIN, "run", "-n", "diarize",
            "python", str(diarize_script),
            str(args.input),
            "-o", str(output_dir),
            "--stats-only",
            "--prefix", "verify_",
        ]
        run(verify_cmd, f"Verify stats → {output_dir}")

    # ── Collect results ──────────────────────────────────────────────────
    speaker_wavs = sorted(output_dir.glob(f"{stem}_SPEAKER_[0-9][0-9].wav"))

    # ── Transcribe: whisper verification ─────────────────────────────────
    if args.transcribe:
        diarize_json_path = output_dir / f"{stem}_diarize.json"
        if not diarize_json_path.exists():
            print(f"ERROR: {diarize_json_path} not found.")
            sys.exit(1)
        diarize_segments = json.loads(
            diarize_json_path.read_text(encoding="utf-8"))
        verify_transcribe(output_dir, stem, diarize_segments, args)
    else:
        # Clean up compact WAVs (not needed without --transcribe)
        for p in output_dir.glob(f"{stem}_SPEAKER_*_compact.wav"):
            p.unlink(missing_ok=True)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Done!")
    print(f"{'=' * 60}")
    print(f"\n  Output: {output_dir}/")
    print(f"  Speaker tracks: {len(speaker_wavs)}")
    for wav in speaker_wavs:
        speaker = wav.stem.replace(f"{stem}_", "")
        print(f"    {wav.name} → {speaker}/")
    if not args.verify and not args.transcribe:
        print(f"\n  Tip: --verify for stats, --transcribe for full verification.")
    print()


if __name__ == "__main__":
    main()
