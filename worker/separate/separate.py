#!/usr/bin/env python3
"""Separate Worker — Split audio into stems using demucs (Meta/Facebook).

Outputs: vocals, drums, bass, other — each track has the full length of the
original, with silence where the stem is not active (demucs does this natively).

Usage:
    python separate.py input.wav -o out/
    python separate.py input.wav -o out/ --model htdemucs_ft
"""

import argparse
import sys
from pathlib import Path

import soundfile as sf
import torch


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def separate(input_path: Path, output_dir: Path, model_name: str = "htdemucs"):
    """Run demucs stem separation."""

    device = get_device()
    print(f"  Device: {device}")
    print(f"  Model: {model_name}")
    print(f"  Input: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    # ── Load model ───────────────────────────────────────────────────────
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    from demucs.audio import AudioFile

    print("  Loading model ...")
    model = get_model(model_name)
    model.to(device)

    # ── Load audio ───────────────────────────────────────────────────────
    print("  Loading audio ...")
    wav = AudioFile(input_path).read(streams=0, samplerate=model.samplerate,
                                      channels=model.audio_channels)
    ref = wav.mean(0)
    wav -= ref.mean()
    wav /= ref.std() + 1e-8

    # ── Separate ─────────────────────────────────────────────────────────
    print("  Separating stems ...")
    sources = apply_model(model, wav[None], device=device, progress=True)[0]
    sources *= ref.std() + 1e-8
    sources += ref.mean()

    # ── Write output stems ───────────────────────────────────────────────
    for i, source_name in enumerate(model.sources):
        out_path = output_dir / f"{stem}_{source_name}.wav"
        sf.write(str(out_path), sources[i].cpu().numpy().T, model.samplerate)
        print(f"  ✓ {out_path.name}")

    print(f"\n  Done — {len(model.sources)} stems written to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Split audio into stems (demucs)")
    parser.add_argument("input", type=Path, help="Input audio file")
    parser.add_argument("-o", "--output", type=Path, required=True,
                        help="Output directory")
    parser.add_argument("--model", default="htdemucs",
                        help="Demucs model (default: htdemucs, alt: htdemucs_ft)")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"  ERROR: File not found: {args.input}")
        sys.exit(1)

    separate(args.input, args.output, model_name=args.model)


if __name__ == "__main__":
    main()
