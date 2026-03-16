#!/usr/bin/env python3
"""Audio transcription using mlx-whisper (Apple Silicon optimized).

Runs in the 'whisper' conda env. Called as subprocess from revoicer.py.
Supports word-level timestamps, multilingual transcription,
and multiple output formats (json, txt, srt, vtt, tsv).

Usage:
    python transcribe.py file.wav -o output/
    python transcribe.py file.wav --language de
    python transcribe.py file.wav --word-timestamps --format srt -o output/
    python transcribe.py file.wav --format all -o output/

Output: JSON on stdout with transcription results.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"

FORMATS = ["json", "txt", "srt", "vtt", "tsv"]

# Map short model names to HuggingFace repos
MODEL_MAP = {
    "tiny": "mlx-community/whisper-tiny",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx-fp32",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}


def resolve_model(name: str) -> str:
    """Resolve short model name to HuggingFace repo path."""
    return MODEL_MAP.get(name, name)


def transcribe_file(
    input_path: Path,
    model: str = DEFAULT_MODEL,
    language: str | None = None,
    word_timestamps: bool = False,
):
    """Transcribe a single audio file."""
    print("Loading whisper model …", file=sys.stderr)
    import mlx_whisper

    kwargs = {
        "word_timestamps": word_timestamps,
    }
    if language:
        kwargs["language"] = language

    result = mlx_whisper.transcribe(
        str(input_path),
        path_or_hf_repo=model,
        **kwargs,
    )

    return result


def format_timestamp_srt(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """Format seconds as VTT timestamp: HH:MM:SS.mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def write_txt(entry: dict, path: Path):
    """Write plain text transcript."""
    path.write_text(entry["text"].strip() + "\n", encoding="utf-8")


def write_srt(entry: dict, path: Path):
    """Write SRT subtitle file."""
    lines = []
    for i, seg in enumerate(entry["segments"], 1):
        lines.append(str(i))
        start = format_timestamp_srt(seg["start"])
        end = format_timestamp_srt(seg["end"])
        lines.append(f"{start} --> {end}")
        lines.append(seg["text"].strip())
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_vtt(entry: dict, path: Path):
    """Write WebVTT subtitle file."""
    lines = ["WEBVTT", ""]
    for seg in entry["segments"]:
        start = format_timestamp_vtt(seg["start"])
        end = format_timestamp_vtt(seg["end"])
        lines.append(f"{start} --> {end}")
        lines.append(seg["text"].strip())
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_tsv(entry: dict, path: Path):
    """Write TSV file (start_ms, end_ms, text)."""
    lines = ["start\tend\ttext"]
    for seg in entry["segments"]:
        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)
        lines.append(f"{start_ms}\t{end_ms}\t{seg['text'].strip()}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(entry: dict, path: Path):
    """Write JSON transcript."""
    path.write_text(
        json.dumps(entry, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


FORMAT_WRITERS = {
    "json": write_json,
    "txt": write_txt,
    "srt": write_srt,
    "vtt": write_vtt,
    "tsv": write_tsv,
}


def save_output(entry: dict, out_dir: Path, stem: str, formats: list[str]):
    """Save transcript in requested formats."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        writer = FORMAT_WRITERS[fmt]
        out_path = out_dir / f"{stem}.{fmt}"
        writer(entry, out_path)
        print(f"  Saved: {out_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using mlx-whisper")
    parser.add_argument("input", nargs="+", help="Input audio file(s)")
    parser.add_argument("--model", default="large-v3-turbo",
                        help=f"Model: {', '.join(MODEL_MAP.keys())} (default: large-v3-turbo)")
    parser.add_argument("--language", default=None,
                        help="Input language code (e.g. 'en', 'de', 'ja'). Auto-detect if omitted.")
    parser.add_argument("--word-timestamps", action="store_true",
                        help="Include word-level timestamps")
    parser.add_argument("--format", default="json",
                        help=f"Output format: {', '.join(FORMATS)}, all (default: json)")
    parser.add_argument("-o", "--output",
                        help="Output directory for transcript files")
    args = parser.parse_args()

    # Resolve model name
    model = resolve_model(args.model)

    # Resolve formats
    if args.format == "all":
        formats = FORMATS
    else:
        fmt = args.format.lower()
        if fmt not in FORMATS:
            print(f"ERROR: Unknown format '{fmt}'. Choose from: {', '.join(FORMATS)}, all",
                  file=sys.stderr)
            sys.exit(1)
        formats = [fmt]

    results = []

    for input_file in args.input:
        input_path = Path(input_file)
        if not input_path.exists():
            print(f"ERROR: File not found: {input_path}", file=sys.stderr)
            sys.exit(1)

        print(f"Transcribing …", file=sys.stderr)
        print(f"Input: {input_path.name}", file=sys.stderr)
        print(f"Model: {args.model}", file=sys.stderr)
        sys.stderr.flush()

        try:
            result = transcribe_file(
                input_path,
                model=model,
                language=args.language,
                word_timestamps=args.word_timestamps,
            )

            entry = {
                "file": str(input_path),
                "text": result.get("text", ""),
                "language": result.get("language", ""),
                "segments": [],
            }

            for seg in result.get("segments", []):
                segment = {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                }
                if args.word_timestamps and "words" in seg:
                    segment["words"] = [
                        {
                            "word": w["word"],
                            "start": w["start"],
                            "end": w["end"],
                        }
                        for w in seg["words"]
                    ]
                entry["segments"].append(segment)

            results.append(entry)

            # Save to files if output dir specified
            if args.output:
                save_output(entry, Path(args.output), input_path.stem, formats)

        except Exception as e:
            print(f"ERROR: {input_path.name}: {e}", file=sys.stderr)
            sys.exit(1)

    # Always output JSON on stdout (for revoicer.py to parse)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
