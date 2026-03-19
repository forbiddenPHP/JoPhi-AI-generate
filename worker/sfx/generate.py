#!/usr/bin/env python3
"""SFX Worker — Text-to-Audio generation via EzAudio.

Subprocess entry point. Reads args, loads model, generates audio,
writes WAV to output path. Emits unified stderr output for progress.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# EzAudio lives in the same directory
WORKER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKER_DIR))


def _ensure_t5_cached(model_name: str):
    """Pre-download T5 model with a tqdm bar that progress.py understands."""
    from huggingface_hub import try_to_load_from_cache, list_repo_files, hf_hub_download
    from tqdm import tqdm

    # Check if already cached (fast path)
    sentinel = try_to_load_from_cache(model_name, "config.json")
    if sentinel is not None and sentinel is not False:
        return  # already cached

    print(f"Downloading {model_name} …", file=sys.stderr)
    files = list_repo_files(model_name)
    for fname in tqdm(files, desc=f"Downloading {model_name}", file=sys.stderr):
        hf_hub_download(model_name, fname)


def main():
    # Suppress huggingface_hub's own tqdm bars (they break progress.py)
    import os
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    # EzAudio expects CWD = worker dir (ckpts/ paths are relative)
    os.chdir(WORKER_DIR)

    if "--list-models" in sys.argv:
        import json as _json
        print(_json.dumps([
            {"model": "s3_xl", "notice": "default"},
            {"model": "s3_l", "notice": "smaller"},
        ]))
        return

    parser = argparse.ArgumentParser(description="EzAudio SFX generation")
    parser.add_argument("--text", required=True, help="Text prompt for audio generation")
    parser.add_argument("-o", "--output", required=True, help="Output WAV path")
    parser.add_argument("--seconds", type=int, default=10, help="Duration in seconds (default: 10)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--steps", type=int, default=100, help="DDIM inference steps (default: 100)")
    parser.add_argument("--cfg-scale", type=float, default=5.0, help="Guidance scale (default: 5.0)")
    parser.add_argument("--model", default="s3_xl", choices=["s3_xl", "s3_l"],
                        help="Model variant (default: s3_xl)")
    args = parser.parse_args()

    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"Loading EzAudio model …", file=sys.stderr)
    print(f"Device: {device}", file=sys.stderr)
    print(f"Model: {args.model}", file=sys.stderr)

    t0 = time.time()

    # Pre-download T5 with our own tqdm (only on first run)
    _ensure_t5_cached("google/flan-t5-xl")

    from api.ezaudio import EzAudio

    ezaudio = EzAudio(model_name=args.model, device=device)
    t_load = time.time() - t0
    print(f"Model loaded in {t_load:.1f}s", file=sys.stderr)

    print(f"Generating audio …", file=sys.stderr)
    print(f"Prompt: {args.text}", file=sys.stderr)
    print(f"Duration: {args.seconds}s", file=sys.stderr)
    print(f"Steps: {args.steps}", file=sys.stderr)

    t1 = time.time()
    sr, audio = ezaudio.generate_audio(
        text=args.text,
        length=args.seconds,
        guidance_scale=args.cfg_scale,
        ddim_steps=args.steps,
        random_seed=args.seed,
        randomize_seed=(args.seed is None),
    )
    t_gen = time.time() - t1

    # Write output
    import soundfile as sf

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio, sr)

    print(f"Generated in {t_gen:.1f}s → {out_path.name} ({sr}Hz)", file=sys.stderr)

    # Stdout: JSON output path
    print(json.dumps([str(out_path)]))


if __name__ == "__main__":
    main()
