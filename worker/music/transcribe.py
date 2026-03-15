#!/usr/bin/env python3
"""Lyrics transcription from audio using HeartTranscriptor.

Runs in the 'heartmula' conda env. Called as subprocess from revoicer.py.
Loads HeartTranscriptor model and transcribes lyrics from an audio file.

Usage:
    python transcribe.py --audio input.mp3 --ckpt-dir /path/to/ckpt
    python transcribe.py --audio input.mp3 --ckpt-dir /path/to/ckpt -o lyrics.txt

Output: Transcribed lyrics text on stdout (or written to file with -o).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def get_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype():
    import torch

    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def transcribe(
    audio_path: Path,
    ckpt_dir: Path,
    max_new_tokens: int = 256,
    num_beams: int = 2,
):
    """Transcribe lyrics from an audio file."""
    import torch
    from heartlib import HeartTranscriptorPipeline

    device = get_device()
    dtype = get_dtype()

    print(f"Loading HeartTranscriptor from {ckpt_dir} ...", file=sys.stderr)
    print(f"  Device: {device}, Dtype: {dtype}", file=sys.stderr)

    pipe = HeartTranscriptorPipeline.from_pretrained(
        str(ckpt_dir),
        device=device,
        dtype=dtype,
    )

    print(f"Transcribing: {audio_path} ...", file=sys.stderr)
    sys.stderr.flush()
    start = time.time()

    with torch.no_grad():
        result = pipe(
            str(audio_path),
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            task="transcribe",
            condition_on_prev_tokens=False,
            compression_ratio_threshold=1.8,
            temperature=(0.0, 0.1, 0.2, 0.4),
            logprob_threshold=-1.0,
            no_speech_threshold=0.4,
        )

    elapsed = time.time() - start
    print(f"Transcribed in {elapsed:.1f}s", file=sys.stderr)
    sys.stderr.flush()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe lyrics from audio using HeartTranscriptor")

    parser.add_argument("--audio", required=True,
                        help="Path to input audio file (MP3, WAV, etc.)")
    parser.add_argument("--ckpt-dir", required=True,
                        help="Path to HeartMuLa checkpoint directory")
    parser.add_argument("-o", "--output",
                        help="Output file path for lyrics (default: stdout)")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="Max tokens to generate (default: 256)")
    parser.add_argument("--num-beams", type=int, default=2,
                        help="Beam search width (default: 2)")

    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"ERROR: Audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.exists():
        print(f"ERROR: Checkpoint directory not found: {ckpt_dir}", file=sys.stderr)
        print("  Run: bash worker/music/install.sh", file=sys.stderr)
        sys.exit(1)

    try:
        result = transcribe(
            audio_path=audio_path,
            ckpt_dir=ckpt_dir,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )
    except Exception as e:
        print(f"ERROR: Transcription failed: {e}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(str(result), encoding="utf-8")
        print(json.dumps([str(output_path)]))
    else:
        print(result)


if __name__ == "__main__":
    main()
