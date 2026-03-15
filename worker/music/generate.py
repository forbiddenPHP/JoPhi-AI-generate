#!/usr/bin/env python3
"""Music generation using HeartMuLa.

Runs in the 'heartmula' conda env. Called as subprocess from revoicer.py.
Loads model, generates music from lyrics + tags, saves to MP3.

Usage:
    python generate.py --lyrics "lyrics text" --tags "disco,happy" -o output.mp3
    python generate.py --lyrics-file lyrics.txt --tags "rock,guitar" -o output.mp3
    python generate.py --lyrics "text" --tags "ambient" --duration 60000 -o short.mp3

Output: JSON array of output file paths on stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path


def get_device() -> dict:
    """Pick the best available device for HeartMuLa.

    Returns device dict for HeartMuLaGenPipeline.from_pretrained().
    HeartMuLa uses separate devices for the language model ("mula")
    and the audio codec ("codec").
    Codec runs on CPU because MPS has issues with the audio codec ops.
    """
    import torch

    return {
        "mula": torch.device("mps"),
        "codec": torch.device("cpu"),
    }


def get_dtype() -> dict:
    """Pick dtype — float32 for MPS (bfloat16 not well supported on Apple Silicon)."""
    import torch

    return {
        "mula": torch.float32,
        "codec": torch.float32,
    }


def generate(
    lyrics: str,
    tags: str,
    output_path: Path,
    ckpt_dir: Path,
    duration_ms: int = 20000,
    topk: int = 50,
    temperature: float = 1.0,
    cfg_scale: float = 1.5,
    seed: int | None = None,
):
    """Generate music from lyrics and tags."""
    import torch
    from heartlib import HeartMuLaGenPipeline

    # Seed handling: only touch RNG if user explicitly provides a seed
    if seed is not None:
        torch.manual_seed(seed)
        print(f"Seed: {seed}", file=sys.stderr)

    device = get_device()
    dtype = get_dtype()

    print(f"Loading HeartMuLa model from {ckpt_dir} ...", file=sys.stderr)
    print(f"  Device: mula={device['mula']}, codec={device['codec']}", file=sys.stderr)
    print(f"  Dtype:  mula={dtype['mula']}, codec={dtype['codec']}", file=sys.stderr)

    # lazy_load only works reliably with CUDA + bitsandbytes
    use_lazy = torch.cuda.is_available()

    pipe = HeartMuLaGenPipeline.from_pretrained(
        str(ckpt_dir),
        device=device,
        dtype=dtype,
        version="3B",
        lazy_load=use_lazy,
    )

    # HeartMuLa API expects file paths for lyrics and tags
    lyrics_file = Path(tempfile.mktemp(suffix=".txt", prefix="heartmula_lyrics_"))
    tags_file = Path(tempfile.mktemp(suffix=".txt", prefix="heartmula_tags_"))

    lyrics_file.write_text(lyrics, encoding="utf-8")
    tags_file.write_text(tags, encoding="utf-8")

    print(f"Generating music (max {duration_ms / 1000:.0f}s) ...", file=sys.stderr)
    sys.stderr.flush()
    start = time.time()

    try:
        with torch.no_grad():
            pipe(
                {
                    "lyrics": str(lyrics_file),
                    "tags": str(tags_file),
                },
                max_audio_length_ms=duration_ms,
                save_path=str(output_path),
                topk=topk,
                temperature=temperature,
                cfg_scale=cfg_scale,
            )
    finally:
        lyrics_file.unlink(missing_ok=True)
        tags_file.unlink(missing_ok=True)

    elapsed = time.time() - start
    print(f"Generated in {elapsed:.1f}s: {output_path}", file=sys.stderr)
    sys.stderr.flush()


def main():
    parser = argparse.ArgumentParser(
        description="Generate music from lyrics and tags using HeartMuLa")

    # Lyrics input (mutually exclusive: inline or file)
    lyrics_group = parser.add_mutually_exclusive_group(required=True)
    lyrics_group.add_argument("--lyrics", help="Inline lyrics text")
    lyrics_group.add_argument("--lyrics-file", help="Path to lyrics text file")

    parser.add_argument("--tags", required=True,
                        help="Comma-separated tags (e.g. 'disco,happy,synthesizer')")
    parser.add_argument("-o", "--output", required=True,
                        help="Output file path (MP3)")
    parser.add_argument("--ckpt-dir", required=True,
                        help="Path to HeartMuLa checkpoint directory")
    parser.add_argument("--duration", type=int, default=20000,
                        help="Max audio length in ms (default: 20000 = 20s)")
    parser.add_argument("--topk", type=int, default=50,
                        help="Top-k sampling (default: 50)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (default: 1.0)")
    parser.add_argument("--cfg-scale", type=float, default=1.5,
                        help="Classifier-free guidance scale (default: 1.5)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (default: random, logged to stderr)")

    args = parser.parse_args()

    # Read lyrics
    if args.lyrics_file:
        lyrics_path = Path(args.lyrics_file)
        if not lyrics_path.exists():
            print(f"ERROR: Lyrics file not found: {lyrics_path}", file=sys.stderr)
            sys.exit(1)
        lyrics = lyrics_path.read_text(encoding="utf-8")
    else:
        lyrics = args.lyrics

    if not lyrics.strip():
        print("ERROR: Lyrics cannot be empty", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.exists():
        print(f"ERROR: Checkpoint directory not found: {ckpt_dir}", file=sys.stderr)
        print("  Run: bash worker/music/install.sh", file=sys.stderr)
        sys.exit(1)

    try:
        generate(
            lyrics=lyrics,
            tags=args.tags,
            output_path=output_path,
            ckpt_dir=ckpt_dir,
            duration_ms=args.duration,
            topk=args.topk,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
            seed=args.seed,
        )
    except Exception as e:
        print(f"ERROR: Generation failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Output: JSON array of paths on stdout (matching enhance.py protocol)
    print(json.dumps([str(output_path)]))


if __name__ == "__main__":
    main()
