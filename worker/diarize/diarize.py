#!/usr/bin/env python3
"""Diarize Worker — Split dialogue audio into separate tracks per speaker.

Uses pyannote.audio directly for speaker diarization (runs on MPS).
No transcription here — that's done separately by mlx-whisper.

Each output track has the full length of the original, with silence where
the speaker is not active.

Usage:
    python diarize.py input.wav -o out/
    python diarize.py input.wav -o out/ --speakers 3
    python diarize.py input.wav -o out/ --verify
    python diarize.py input.wav -o out/ --stats-only --prefix verify_
"""

import argparse
import json
import os
import sys
from pathlib import Path

import subprocess
import tempfile

import numpy as np
import soundfile as sf
import torch
from pyannote.audio import Pipeline


def get_device():
    """Best available device — pyannote supports mps."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def verify_stats(segments_json: list[dict], total_duration: float,
                 output_dir: Path, stem: str, prefix: str = ""):
    """Analyze diarization quality, print report, save as JSON.

    Args:
        prefix: File name prefix, e.g. "verify_" for verify runs.
    """
    speakers = sorted(set(s["speaker"] for s in segments_json))

    # ── Build stats ───────────────────────────────────────────────────────
    stats = {
        "audio_duration": round(total_duration, 3),
        "num_speakers": len(speakers),
        "speakers": {},
        "gaps": [],
        "overlaps": [],
    }

    total_speech = 0.0
    for speaker in speakers:
        segs = [s for s in segments_json if s["speaker"] == speaker]
        duration = sum(s["end"] - s["start"] for s in segs)
        total_speech += duration
        stats["speakers"][speaker] = {
            "segments": len(segs),
            "total_duration": round(duration, 3),
            "percent": round(duration / total_duration * 100, 1),
            "avg_segment": round(duration / len(segs), 3) if segs else 0,
            "shortest": round(min(s["end"] - s["start"] for s in segs), 3),
            "longest": round(max(s["end"] - s["start"] for s in segs), 3),
        }

    # Gaps (time not assigned to any speaker)
    all_sorted = sorted(segments_json, key=lambda s: s["start"])
    for i in range(1, len(all_sorted)):
        gap = all_sorted[i]["start"] - all_sorted[i - 1]["end"]
        if gap > 0.5:
            stats["gaps"].append({
                "start": round(all_sorted[i - 1]["end"], 3),
                "end": round(all_sorted[i]["start"], 3),
                "duration": round(gap, 3),
            })

    # Overlaps
    for i in range(1, len(all_sorted)):
        overlap = all_sorted[i - 1]["end"] - all_sorted[i]["start"]
        if overlap > 0.1:
            stats["overlaps"].append({
                "start": round(all_sorted[i]["start"], 3),
                "end": round(all_sorted[i - 1]["end"], 3),
                "duration": round(overlap, 3),
                "speakers": [all_sorted[i - 1]["speaker"],
                             all_sorted[i]["speaker"]],
            })

    stats["total_speech"] = round(total_speech, 3)
    stats["total_silence"] = round(max(0, total_duration - total_speech), 3)

    # ── Save stats JSON ──────────────────────────────────────────────────
    stats_path = output_dir / f"{stem}_{prefix}stats.json"
    stats_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"  ✓ {stats_path.name}")

    # ── Print report ─────────────────────────────────────────────────────
    print(f"\n  ── Diarization Stats ──")
    print(f"    Audio duration: {total_duration:.1f}s")
    print(f"    Speakers: {len(speakers)}")

    for speaker, info in stats["speakers"].items():
        print(f"\n    {speaker}:")
        print(f"      Segments: {info['segments']}")
        print(f"      Total: {info['total_duration']:.1f}s"
              f" ({info['percent']:.0f}% of audio)")
        print(f"      Avg segment: {info['avg_segment']:.1f}s")
        print(f"      Shortest: {info['shortest']:.1f}s")
        print(f"      Longest: {info['longest']:.1f}s")

    print(f"\n    Coverage: {stats['total_speech']:.1f}s speech"
          f" / {stats['total_silence']:.1f}s silence")

    if stats["gaps"]:
        print(f"    Gaps (>0.5s): {len(stats['gaps'])}")
        for g in stats["gaps"][:5]:
            print(f"      {g['start']:.1f}s – {g['end']:.1f}s ({g['duration']:.1f}s)")
        if len(stats["gaps"]) > 5:
            print(f"      ... and {len(stats['gaps']) - 5} more")

    if stats["overlaps"]:
        print(f"    Overlaps (>0.1s): {len(stats['overlaps'])}")
        for o in stats["overlaps"][:5]:
            print(f"      {o['start']:.1f}s – {o['end']:.1f}s"
                  f" ({o['duration']:.1f}s, {' vs '.join(o['speakers'])})")
        if len(stats["overlaps"]) > 5:
            print(f"      ... and {len(stats['overlaps']) - 5} more")


CONDA_BIN = "/opt/miniconda3/bin/conda"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent


def verify_transcription(speaker_segments: dict[str, list[tuple[float, float]]],
                         speakers: list[str], data: np.ndarray, sr: int,
                         output_dir: Path, stem: str):
    """Verify diarization by transcribing each segment clip individually.

    For each speaker segment:
    1. Extract the clip from the speaker track via soundfile to /tmp
    2. Transcribe the clip with whisper
    3. Report the text per segment for manual comparison
    """
    print("\n  ── Verify: Transcribing segments ──")

    transcribe_script = PROJECT_DIR / "worker" / "whisper" / "transcribe.py"
    total_samples = len(data)
    is_stereo = data.ndim == 2
    verify_results = []

    for speaker in speakers:
        segs = speaker_segments[speaker]
        print(f"\n    {speaker}: {len(segs)} segments")

        for i, (start, end) in enumerate(segs):
            duration = end - start
            if duration < 0.5:
                # Too short to transcribe meaningfully
                verify_results.append({
                    "speaker": speaker, "segment": i,
                    "start": round(start, 3), "end": round(end, 3),
                    "text": "(too short)",
                })
                continue

            # Extract clip from original audio at segment timestamps
            start_sample = max(0, int(start * sr))
            end_sample = min(total_samples, int(end * sr))
            clip = data[start_sample:end_sample]

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            sf.write(tmp_path, clip, sr)

            try:
                result = subprocess.run(
                    [CONDA_BIN, "run", "-n", "whisper",
                     "python", str(transcribe_script), tmp_path],
                    capture_output=True, text=True, timeout=120,
                )
                if result.returncode == 0:
                    # Parse JSON output from whisper
                    out = result.stdout.strip()
                    if out:
                        transcripts = json.loads(out)
                        text = transcripts[0].get("text", "").strip() if transcripts else ""
                    else:
                        text = "(empty)"
                else:
                    text = f"(error: {result.stderr.strip()[:100]})"
            except subprocess.TimeoutExpired:
                text = "(timeout)"
            finally:
                Path(tmp_path).unlink(missing_ok=True)

            verify_results.append({
                "speaker": speaker, "segment": i,
                "start": round(start, 3), "end": round(end, 3),
                "duration": round(duration, 3),
                "text": text,
            })

            # Print inline
            text_preview = text[:60] + "..." if len(text) > 60 else text
            print(f"      [{start:.1f}s–{end:.1f}s] {text_preview}")

    # Save verify results
    verify_path = output_dir / f"{stem}_verify.json"
    verify_path.write_text(
        json.dumps(verify_results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\n  ✓ {verify_path.name} ({len(verify_results)} segments transcribed)")


def diarize(input_path: Path, output_dir: Path,
            num_speakers: int | None = None, hf_token: str | None = None,
            verify: bool = False) -> list[str]:
    """Run pyannote diarization, then split audio into speaker tracks.
    Returns list of all output file paths."""

    device = get_device()
    print(f"Diarizing audio …", file=sys.stderr)
    print(f"Device: {device}", file=sys.stderr)
    print(f"Input: {input_path}", file=sys.stderr)

    # ── Convert to WAV if needed (pyannote has issues with MP3 sample counts)
    tmp_wav = None
    if input_path.suffix.lower() != ".wav":
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_wav.close()
        print(f"Converting to WAV …", file=sys.stderr)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(input_path), "-ar", "16000", "-ac", "1", tmp_wav.name],
            capture_output=True, check=True,
        )
        wav_path = Path(tmp_wav.name)
    else:
        wav_path = input_path

    # ── HF token ──────────────────────────────────────────────────────────
    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        # Try cached token from huggingface-cli login
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists():
            token = token_path.read_text().strip()
    if not token:
        print("ERROR: No HuggingFace token found.", file=sys.stderr)
        print("Run: huggingface-cli login", file=sys.stderr)
        print("Or set HF_TOKEN env var", file=sys.stderr)
        sys.exit(1)

    # ── Load pipeline ─────────────────────────────────────────────────────
    print("Loading pyannote pipeline …", file=sys.stderr)
    os.environ["HF_TOKEN"] = token
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
    )
    pipeline.to(device)

    # ── Run diarization ───────────────────────────────────────────────────
    print("Running diarization …", file=sys.stderr)
    diarize_kwargs = {}
    if num_speakers is not None:
        diarize_kwargs["num_speakers"] = num_speakers

    result = pipeline(str(wav_path), **diarize_kwargs)

    # pyannote 4.x returns DiarizeOutput dataclass
    if hasattr(result, "speaker_diarization"):
        diarization = result.speaker_diarization
    else:
        diarization = result

    # ── Clean up temp file ────────────────────────────────────────────────
    if tmp_wav is not None:
        Path(tmp_wav.name).unlink(missing_ok=True)

    # ── Read original audio ───────────────────────────────────────────────
    data, sr = sf.read(str(input_path), dtype="float32")
    total_samples = len(data)
    total_duration = total_samples / sr
    is_stereo = data.ndim == 2

    # ── Collect segments per speaker ──────────────────────────────────────
    speaker_segments: dict[str, list[tuple[float, float]]] = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in speaker_segments:
            speaker_segments[speaker] = []
        speaker_segments[speaker].append((turn.start, turn.end))

    if not speaker_segments:
        print("WARNING: No speakers detected.", file=sys.stderr)
        sys.exit(1)

    speakers = sorted(speaker_segments.keys())
    print(f"Speakers found: {len(speakers)} ({', '.join(speakers)})", file=sys.stderr)

    # ── Create speaker tracks ─────────────────────────────────────────────
    print("Writing speaker tracks …", file=sys.stderr)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    output_paths = []

    for speaker in speakers:
        if is_stereo:
            track = np.zeros_like(data)
        else:
            track = np.zeros(total_samples, dtype=np.float32)

        for start, end in speaker_segments[speaker]:
            start_sample = max(0, int(start * sr))
            end_sample = min(total_samples, int(end * sr))
            track[start_sample:end_sample] = data[start_sample:end_sample]

        out_path = output_dir / f"{stem}_{speaker}.wav"
        sf.write(str(out_path), track, sr)
        output_paths.append(str(out_path))
        print(f"✓ {out_path.name}", file=sys.stderr)

    # ── Save diarization JSON ─────────────────────────────────────────────
    segments_json = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments_json.append({
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "speaker": speaker,
        })

    json_path = output_dir / f"{stem}_diarize.json"
    json_path.write_text(
        json.dumps(segments_json, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_paths.append(str(json_path))
    print(f"✓ {json_path.name}", file=sys.stderr)

    # ── Verify stats ──────────────────────────────────────────────────────
    if verify:
        verify_stats(segments_json, total_duration, output_dir, stem)
        verify_transcription(speaker_segments, speakers, data, sr,
                             output_dir, stem)

    print(f"Done — {len(speakers)} speaker tracks written to {output_dir}", file=sys.stderr)
    return output_paths


def main():
    parser = argparse.ArgumentParser(description="Split dialogue into speaker tracks")
    parser.add_argument("input", type=Path, help="Input audio file")
    parser.add_argument("-o", "--output", type=Path, required=True,
                        help="Output directory")
    parser.add_argument("--speakers", type=int, default=None,
                        help="Number of speakers (auto-detect if not set)")
    parser.add_argument("--hf-token", default=None,
                        help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--verify", action="store_true",
                        help="Stats + transcribe each segment to verify diarization quality")
    parser.add_argument("--stats-only", action="store_true",
                        help="Only run stats on existing diarize JSON (no pyannote needed)")
    parser.add_argument("--prefix", default="",
                        help="Prefix for output files (e.g. 'verify_')")
    args = parser.parse_args()

    if args.stats_only:
        # Run stats on existing JSON without loading pyannote
        diarize_json = args.output / f"{args.input.stem}_diarize.json"
        if not diarize_json.exists():
            print(f"ERROR: {diarize_json} not found.", file=sys.stderr)
            sys.exit(1)
        segments = json.loads(diarize_json.read_text(encoding="utf-8"))
        total_duration = max(s["end"] for s in segments)
        verify_stats(segments, total_duration, args.output, args.input.stem,
                     prefix=args.prefix)
        return

    if not args.input.exists():
        print(f"ERROR: File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    output_paths = diarize(args.input, args.output,
                           num_speakers=args.speakers, hf_token=args.hf_token,
                           verify=args.verify)
    print(json.dumps(output_paths))


if __name__ == "__main__":
    main()
