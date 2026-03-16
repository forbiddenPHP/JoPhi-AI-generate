#!/usr/bin/env python3
"""Music generation using ACE-Step 1.5.

Runs via uv in the ACE-Step-1.5 project. Called as subprocess from revoicer.py.
Loads model, generates music from lyrics + caption, saves to audio file.

Usage:
    python generate.py --lyrics "lyrics text" --tags "upbeat pop song" -o output.mp3
    python generate.py --lyrics-file lyrics.txt --tags "rock ballad" -o output.flac
    python generate.py --lyrics "text" --tags "ambient" --duration 30000 -o short.mp3

Output: JSON array of output file paths on stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path


def get_device() -> str:
    """Device for ACE-Step — always MPS on Apple Silicon."""
    return "mps"


def get_lm_backend(device: str) -> str:
    """LM backend — always MLX on Apple Silicon."""
    return "mlx"


def generate(
    lyrics: str,
    caption: str,
    output_path: Path,
    duration_s: float = 20.0,
    seed: int = -1,
    guidance_scale: float = 7.0,
    lm_temperature: float = 0.85,
    lm_top_k: int = 0,
    lm_top_p: float = 0.9,
    lm_cfg_scale: float = 2.0,
    inference_steps: int = 8,
    shift: float = 3.0,
    thinking: bool = True,
    infer_method: str = "ode",
    batch_size: int = 1,
    instrumental: bool = False,
    bpm: int | None = None,
    keyscale: str | None = None,
    timesignature: str | None = None,
    task_type: str = "text2music",
    src_audio: str | None = None,
    repainting_start: float = 0.0,
    repainting_end: float = -1,
    reference_audio: str | None = None,
    config_path: str = "acestep-v15-turbo",
):
    """Generate music from lyrics and caption using ACE-Step 1.5."""
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music
    from acestep.llm_inference import LLMHandler

    device = get_device()
    lm_backend = get_lm_backend(device)

    os.environ.setdefault("ACESTEP_LM_BACKEND", "mlx")

    project_root = str(Path(__file__).parent / "ACE-Step-1.5")

    print(f"Device: {device}, LM backend: {lm_backend}", file=sys.stderr)
    print(f"Loading ACE-Step model ({config_path}) …", file=sys.stderr)
    sys.stderr.flush()

    # ── Ensure model checkpoint is downloaded ──────────────────────────────
    from acestep.model_downloader import ensure_dit_model
    checkpoints_dir = Path(project_root) / "checkpoints"
    ok, msg = ensure_dit_model(config_path, checkpoints_dir)
    if not ok:
        print(f"ERROR: {msg}", file=sys.stderr)
        sys.exit(1)

    # ── Initialize DiT handler ────────────────────────────────────────────
    dit_handler = AceStepHandler()
    dit_handler.initialize_service(
        project_root=project_root,
        config_path=config_path,
        device=device,
    )

    # ── Initialize LLM handler ────────────────────────────────────────────
    llm_handler = LLMHandler()
    llm_handler.initialize(
        checkpoint_dir=project_root,
        lm_model_path="acestep-5Hz-lm-0.6B",
        backend=lm_backend,
        device=device,
    )

    # ── Build generation parameters ───────────────────────────────────────
    params = GenerationParams(
        task_type=task_type,
        caption=caption,
        lyrics=lyrics,
        instrumental=instrumental,
        duration=duration_s,
        seed=seed,
        guidance_scale=guidance_scale,
        inference_steps=inference_steps,
        shift=shift,
        infer_method=infer_method,
        thinking=thinking,
        lm_temperature=lm_temperature,
        lm_top_k=lm_top_k,
        lm_top_p=lm_top_p,
        lm_cfg_scale=lm_cfg_scale,
        bpm=bpm,
        keyscale=keyscale or "",
        timesignature=timesignature or "",
        src_audio=src_audio,
        repainting_start=repainting_start,
        repainting_end=repainting_end,
        reference_audio=reference_audio,
    )

    # Determine output format from extension
    ext = output_path.suffix.lstrip(".").lower()
    audio_format = ext if ext in ("mp3", "wav", "flac", "opus", "aac") else "mp3"

    config = GenerationConfig(
        batch_size=batch_size,
        use_random_seed=(seed < 0),
        seeds=[seed] if seed >= 0 else None,
        audio_format=audio_format,
    )

    # ── Generate ──────────────────────────────────────────────────────────
    print(f"Generating music (max {duration_s:.0f}s) …", file=sys.stderr)
    print(f"  Task: {task_type}", file=sys.stderr)
    if reference_audio:
        print(f"  Reference: {reference_audio}", file=sys.stderr)
    if task_type == "repaint":
        print(f"  Repaint: {repainting_start:.1f}s - {repainting_end:.1f}s", file=sys.stderr)
        print(f"  Source: {src_audio}", file=sys.stderr)
    if seed >= 0:
        print(f"  Seed: {seed}", file=sys.stderr)
    print(f"  Lyrics: {lyrics[:80]}{'...' if len(lyrics) > 80 else ''}", file=sys.stderr)
    sys.stderr.flush()

    start = time.time()

    # Use a temp dir for ACE-Step output, then move to final path
    save_dir = str(output_path.parent / ".ace_tmp")
    os.makedirs(save_dir, exist_ok=True)

    try:
        result = generate_music(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            params=params,
            config=config,
            save_dir=save_dir,
        )

        if not result.success:
            print(f"ERROR: Generation failed: {result.error}", file=sys.stderr)
            print(f"  Status: {result.status_message}", file=sys.stderr)
            sys.exit(1)

        # Move first audio to output path
        if result.audios:
            src = result.audios[0].get("path", "")
            if src and Path(src).exists():
                shutil.move(src, str(output_path))
            else:
                # Fallback: save tensor directly
                import torch
                import torchaudio

                audio_tensor = result.audios[0]["tensor"]
                sample_rate = result.audios[0]["sample_rate"]
                torchaudio.save(str(output_path), audio_tensor, sample_rate)
        else:
            print("ERROR: No audio generated", file=sys.stderr)
            sys.exit(1)

    finally:
        # Clean up temp dir
        shutil.rmtree(save_dir, ignore_errors=True)

    elapsed = time.time() - start
    print(f"Generated in {elapsed:.1f}s: {output_path}", file=sys.stderr)
    sys.stderr.flush()


def main():
    parser = argparse.ArgumentParser(
        description="Generate music from lyrics and caption using ACE-Step 1.5")

    # Lyrics input (mutually exclusive: inline or file)
    lyrics_group = parser.add_mutually_exclusive_group(required=True)
    lyrics_group.add_argument("--lyrics", help="Inline lyrics text")
    lyrics_group.add_argument("--lyrics-file", help="Path to lyrics text file")

    parser.add_argument("--tags", required=True,
                        help="Caption / style description (e.g. 'upbeat pop song with synths')")
    parser.add_argument("-o", "--output", required=True,
                        help="Output file path (MP3, FLAC, WAV)")
    parser.add_argument("--duration", type=int, default=20000,
                        help="Max audio length in ms (default: 20000 = 20s)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (default: random)")
    parser.add_argument("--guidance-scale", type=float, default=7.0,
                        help="DiT classifier-free guidance scale (default: 7.0)")
    parser.add_argument("--lm-temperature", type=float, default=0.85,
                        help="LM sampling temperature (default: 0.85)")
    parser.add_argument("--lm-top-k", type=int, default=0,
                        help="LM top-k sampling, 0=disabled (default: 0)")
    parser.add_argument("--lm-cfg", type=float, default=2.0,
                        help="LM guidance scale (default: 2.0)")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="LM nucleus sampling (default: 0.9)")
    parser.add_argument("--config-path", type=str, default="acestep-v15-turbo",
                        help="DiT config/checkpoint name (default: acestep-v15-turbo)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Inference steps (default: auto based on config)")
    parser.add_argument("--shift", type=float, default=3.0,
                        help="Timestep shift factor (default: 3.0)")
    parser.add_argument("--thinking", action="store_true", default=True,
                        help="Enable LM chain-of-thought (default: on)")
    parser.add_argument("--no-thinking", action="store_true",
                        help="Disable LM chain-of-thought")
    parser.add_argument("--infer-method", choices=["ode", "sde"], default="ode",
                        help="Inference method (default: ode)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Number of parallel samples (default: 1)")
    parser.add_argument("--instrumental", action="store_true",
                        help="Force instrumental output (no vocals)")
    parser.add_argument("--bpm", type=int, default=None,
                        help="Beats per minute (default: auto)")
    parser.add_argument("--keyscale", type=str, default=None,
                        help="Musical key (e.g., 'C Major', 'Am') (default: auto)")
    parser.add_argument("--timesignature", type=str, default=None,
                        help="Time signature: 2=2/4, 3=3/4, 4=4/4, 6=6/8 (default: auto)")

    # Repaint support
    parser.add_argument("--task", choices=["text2music", "repaint"],
                        default="text2music",
                        help="Task type (default: text2music)")
    parser.add_argument("--src-audio",
                        help="Source audio file for repaint task")
    parser.add_argument("--repaint-start", type=float, default=0.0,
                        help="Repaint region start in seconds (default: 0)")
    parser.add_argument("--repaint-end", type=float, default=-1,
                        help="Repaint region end in seconds (default: -1 = end)")
    parser.add_argument("--reference-audio",
                        help="Reference audio for style consistency (rolling window)")

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

    # Config path for DiT model
    config_path = args.config_path

    # Turbo is distilled for exactly 8 steps — always force 8
    # SFT/Base default to 50 but can be overridden via --steps
    if "turbo" in config_path:
        inference_steps = 8
    elif args.steps is not None:
        inference_steps = args.steps
    else:
        inference_steps = 50

    # Seed: None → -1 (ACE-Step convention for random)
    seed = args.seed if args.seed is not None else -1

    # Thinking: --no-thinking overrides
    thinking = not args.no_thinking

    try:
        generate(
            lyrics=lyrics,
            caption=args.tags,
            output_path=output_path,
            duration_s=args.duration / 1000.0,
            seed=seed,
            guidance_scale=args.guidance_scale,
            lm_temperature=args.lm_temperature,
            lm_top_k=args.lm_top_k,
            lm_top_p=args.top_p,
            lm_cfg_scale=args.lm_cfg,
            inference_steps=inference_steps,
            shift=args.shift,
            thinking=thinking,
            infer_method=args.infer_method,
            batch_size=args.batch_size,
            instrumental=args.instrumental,
            bpm=args.bpm,
            keyscale=args.keyscale,
            timesignature=args.timesignature,
            task_type=args.task,
            src_audio=args.src_audio,
            repainting_start=args.repaint_start,
            repainting_end=args.repaint_end,
            reference_audio=args.reference_audio,
            config_path=config_path,
        )
    except Exception as e:
        print(f"ERROR: Generation failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Output: JSON array of paths on stdout (matching HeartMuLa protocol)
    print(json.dumps([str(output_path)]))


if __name__ == "__main__":
    main()
