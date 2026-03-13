#!/usr/bin/env python3
"""Batch audio enhancer using resemble-enhance.

Runs in the 'enhance' conda env. Called as subprocess from revoicer.py.
Loads models once and processes all input files sequentially.

Usage:
    python enhance.py file1.wav file2.wav -o output/
    python enhance.py file.wav --denoise-only
    python enhance.py file.wav --strength 0.5 --steps 32

Output: JSON array of output file paths on stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Enable MPS fallback for unsupported ops (e.g. aten::_weight_norm_interface)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torchaudio


def get_device() -> str:
    """Device for resemble-enhance — CPU only.

    MPS crashes with resemble-enhance (Abort trap on aten ops).
    """
    return "cpu"


def process_file(
    input_path: Path,
    output_path: Path,
    device: str,
    denoise_only: bool = False,
    enhance_only: bool = False,
    strength: float = 0.5,
    steps: int = 32,
):
    """Enhance a single audio file."""
    from resemble_enhance.enhancer.inference import denoise, enhance

    dwav, sr = torchaudio.load(str(input_path))

    # Convert to mono if stereo
    if dwav.shape[0] > 1:
        dwav = dwav.mean(dim=0)
    else:
        dwav = dwav.squeeze(0)

    if denoise_only:
        out_wav, new_sr = denoise(dwav, sr, device)
    elif enhance_only:
        out_wav, new_sr = enhance(
            dwav, sr, device,
            nfe=steps,
            solver="midpoint",
            lambd=0.0,
            tau=0.5,
        )
    else:
        out_wav, new_sr = enhance(
            dwav, sr, device,
            nfe=steps,
            solver="midpoint",
            lambd=strength,
            tau=0.5,
        )

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), out_wav.unsqueeze(0).cpu(), new_sr)


def main():
    parser = argparse.ArgumentParser(
        description="Enhance audio files (denoise + super-resolution)")
    parser.add_argument("input", nargs="+", help="Input audio file(s)")
    parser.add_argument("-o", "--output", required=True,
                        help="Output directory")
    parser.add_argument("--denoise-only", action="store_true",
                        help="Only denoise, skip super-resolution")
    parser.add_argument("--enhance-only", action="store_true",
                        help="Only super-resolution, skip denoising")
    parser.add_argument("--strength", type=float, default=0.5,
                        help="Denoising strength 0.0-1.0 (default: 0.5)")
    parser.add_argument("--steps", type=int, default=32,
                        help="Enhancement steps 1-128 (default: 32, higher=better+slower)")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}", file=sys.stderr)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = []
    total = len(args.input)

    for i, input_file in enumerate(args.input, 1):
        input_path = Path(input_file)
        if not input_path.exists():
            print(f"ERROR: File not found: {input_path}", file=sys.stderr)
            sys.exit(1)

        out_path = output_dir / input_path.name
        mode = "denoise" if args.denoise_only else "enhance-only" if args.enhance_only else "enhance"
        print(f"[{i}/{total}] {mode}: {input_path.name}", file=sys.stderr)

        try:
            process_file(
                input_path, out_path, device,
                denoise_only=args.denoise_only,
                enhance_only=args.enhance_only,
                strength=args.strength,
                steps=args.steps,
            )
            output_paths.append(str(out_path))
        except Exception as e:
            print(f"ERROR: {input_path.name}: {e}", file=sys.stderr)
            sys.exit(1)

    # Output paths as JSON on stdout (for revoicer.py to parse)
    print(json.dumps(output_paths))


if __name__ == "__main__":
    main()
