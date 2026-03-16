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

# MPS Conv1d is broken for >65536 output channels (PyTorch bug).
# The UnivNet vocoder's KernelPredictor has kernel_conv with 221184 channels.
# Fix: patch kernel_conv.forward to run in chunks of max 65536.
_MPS_CONV1D_MAX_OUT = 65536


def get_device() -> str:
    """Device for resemble-enhance."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _patch_mps_conv1d(enhancer):
    """Patch all Conv1d layers in vocoder that exceed MPS channel limit."""
    for block in enhancer.vocoder.blocks:
        kp = block.kernel_predictor
        conv = kp.kernel_conv
        out_channels = conv.weight.shape[0]
        if out_channels <= _MPS_CONV1D_MAX_OUT:
            continue

        _orig = conv.forward

        def chunked_forward(x, _conv=conv, _out=out_channels):
            # Compute the full weight once (triggers weight_norm parametrization)
            weight = _conv.weight
            bias = _conv.bias
            chunks_w = weight.split(_MPS_CONV1D_MAX_OUT, dim=0)
            chunks_b = bias.split(_MPS_CONV1D_MAX_OUT, dim=0) if bias is not None else [None] * len(chunks_w)
            parts = []
            for w, b in zip(chunks_w, chunks_b):
                parts.append(torch.nn.functional.conv1d(
                    x, w, b, stride=_conv.stride, padding=_conv.padding,
                    dilation=_conv.dilation, groups=_conv.groups,
                ))
            return torch.cat(parts, dim=1)

        conv.forward = chunked_forward


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
    from resemble_enhance.enhancer.inference import denoise, enhance, load_enhancer

    dwav, sr = torchaudio.load(str(input_path))

    # Convert to mono if stereo
    if dwav.shape[0] > 1:
        dwav = dwav.mean(dim=0)
    else:
        dwav = dwav.squeeze(0)

    if denoise_only:
        out_wav, new_sr = denoise(dwav, sr, device)
    else:
        # Patch MPS Conv1d bug (>65536 output channels broken)
        if device == "mps":
            enhancer = load_enhancer(None, device)
            if not hasattr(enhancer, '_mps_patched'):
                _patch_mps_conv1d(enhancer)
                enhancer._mps_patched = True

        if enhance_only:
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
    mode = "denoise" if args.denoise_only else "enhance-only" if args.enhance_only else "enhance"
    print(f"Loading enhance model …", file=sys.stderr)
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
        print(f"Enhancing audio …", file=sys.stderr) if total == 1 else None
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
