"""LTX-2.3 video generation worker.

Usage:
  generate.py video ltx2.3 -p "a cat running" -o cat.mp4
  generate.py video ltx2.3 --model dev -p "sunset over ocean" -o sunset.mp4
  generate.py video ltx2.3 --image-first start.jpg -p "animate this" -o anim.mp4
  generate.py video ltx2.3 --model dev -p "He says: hello" --extend video.mp4 5 -o extended.mp4
  generate.py video ltx2.3 --model dev -p "Full scene with change" --retake video.mp4 3.5 5.0 -o retake.mp4
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
os.environ["TQDM_DISABLE"] = "0"

# tqdm.auto disables itself when stderr is a pipe (no TTY).
# Force it on — progress.py on the other end parses and displays it.
import tqdm as _tqdm_mod, tqdm.auto as _tqdm_auto
_OrigTqdm = _tqdm_mod.tqdm
class _ForceTqdm(_OrigTqdm):
    def __init__(self, *a, **kw):
        kw["file"] = sys.stderr
        kw["disable"] = False
        super().__init__(*a, **kw)
        self.disable = False  # HF tqdm subclass passes disable=None → re-force
        self.refresh()  # flush initial 0% bar immediately (piped stderr buffers)
_tqdm_mod.tqdm = _ForceTqdm
_tqdm_auto.tqdm = _ForceTqdm

import torch


# Models alongside this script
_MODELS_DIR = Path(__file__).resolve().parent / "models"
_MODELS_DIR.mkdir(exist_ok=True)
os.environ["HF_HOME"] = str(_MODELS_DIR)

# HF token for gated models (gegebenenfalls)
_DEFAULT_TOKEN = Path.home() / ".cache" / "huggingface" / "token"
if "HF_TOKEN" not in os.environ and _DEFAULT_TOKEN.exists():
    os.environ["HF_TOKEN"] = _DEFAULT_TOKEN.read_text().strip()


# ── Model registry ────────────────────────────────────────────────────────
_MODEL_MAP = {
    "distilled": {
        "checkpoint": "ltx-2.3-22b-distilled.safetensors",
        "hf_repo": "Lightricks/LTX-2.3",
        "pipeline": "distilled",
    },
    "dev": {
        "checkpoint": "ltx-2.3-22b-dev.safetensors",
        "hf_repo": "Lightricks/LTX-2.3",
        "pipeline": "two_stage",
    },
}

_UPSCALER_FILE = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
_DISTILLED_LORA_FILE = "ltx-2.3-22b-distilled-lora-384.safetensors"
_IC_LORA_UNION_FILE = "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
_GEMMA_REPO = "google/gemma-3-12b-it"


# ── Memory management ────────────────────────────────────────────────────

def _free_memory() -> int:
    """Return free RAM in bytes."""
    import psutil
    return psutil.virtual_memory().available


def _auto_reduce_frames(num_frames: int, width: int, height: int) -> int:
    """Reduce frame count if video latents would exceed memory budget."""
    free = _free_memory()
    budget = free - free // 20  # 5% safety margin
    # Rough estimate: ~200 bytes per pixel per frame for full pipeline
    bytes_per_frame = width * height * 200
    max_frames = budget // bytes_per_frame
    if num_frames > max_frames:
        # Round down to nearest 8k+1
        reduced = (int(max_frames) // 8) * 8 + 1
        reduced = max(reduced, 9)  # minimum 9 frames
        print(f"WARNING: {num_frames} frames exceeds memory budget. "
              f"Reducing to {reduced} frames.", file=sys.stderr)
        return reduced
    return num_frames



# ── Model resolution ─────────────────────────────────────────────────────

def _resolve_model_path(filename: str) -> Path:
    """Return path to model file in models/, download from HF if missing."""
    path = _MODELS_DIR / filename
    if path.exists():
        return path
    # Try downloading from HF
    hf_repo = "Lightricks/LTX-2.3"
    print(f"Downloading {filename} from {hf_repo} …", file=sys.stderr)
    from huggingface_hub import hf_hub_download
    downloaded = hf_hub_download(repo_id=hf_repo, filename=filename, local_dir=str(_MODELS_DIR))
    return Path(downloaded)


def _resolve_gemma_path() -> Path:
    """Return path to Gemma text encoder, download if missing."""
    gemma_dir = _MODELS_DIR / "gemma-3-12b-it"
    if gemma_dir.exists() and any(gemma_dir.iterdir()):
        return gemma_dir
    print(f"Downloading Gemma 3 12B text encoder …", file=sys.stderr)
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=_GEMMA_REPO, local_dir=str(gemma_dir))
    return gemma_dir


def _output_paths(base: Path) -> dict:
    """Determine output paths and mode based on file extension."""
    suffix = base.suffix.lower()
    if suffix == ".mp4":
        return {"mode": "muxed", "video": base}
    elif suffix == ".mp4v":
        return {"mode": "video_only", "video": base}
    elif suffix == ".mp4a":
        return {"mode": "audio_only", "audio": base}
    else:
        # No recognized extension → split output
        return {
            "mode": "split",
            "video": base.with_suffix(".mp4v"),
            "audio": base.with_suffix(".mp4a"),
        }


def main():
    # ── Fast path: --list-models ──────────────────────────────────────
    if "--list-models" in sys.argv:
        models = [
            {"model": "distilled", "notice": "default, 22B, 8 steps"},
            {"model": "dev", "notice": "22B, 40 steps"},
        ]
        print(json.dumps(models))
        return

    parser = argparse.ArgumentParser(description="LTX-2.3 video generation")
    parser.add_argument("--model", "-m", default="distilled",
                        choices=list(_MODEL_MAP.keys()),
                        help="Model variant (default: distilled)")
    parser.add_argument("--prompt", "-p", required=True, help="Text prompt")
    parser.add_argument("-o", "--output", default="video.mp4", help="Output path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--width", "-W", type=int, default=768,
                        help="Video width (default: 768, stage-2 output)")
    parser.add_argument("--height", "-H", type=int, default=512,
                        help="Video height (default: 512, stage-2 output)")
    parser.add_argument("--num-frames", type=int, default=None, dest="num_frames",
                        help="Frames, must be 8k+1 (default: from audio length, or 121)")
    parser.add_argument("--frame-rate", type=int, default=24, dest="frame_rate",
                        help="FPS (default: 24)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Inference steps (default: model-dependent)")
    parser.add_argument("--cfg-scale", type=float, default=None, dest="cfg_scale",
                        help="Video CFG guidance scale")
    parser.add_argument("--negative-prompt", default=None, dest="negative_prompt",
                        help="Negative prompt")
    parser.add_argument("--enhance-prompt", action="store_true", dest="enhance_prompt",
                        help="Auto-enhance prompt via Gemma")

    # Image conditioning: flexible (--image PATH FRAME_IDX STRENGTH)
    parser.add_argument("--image", dest="images", action="append", nargs=3,
                        metavar=("PATH", "FRAME_IDX", "STRENGTH"),
                        help="Image conditioning: PATH FRAME_IDX STRENGTH (repeatable)")
    # Convenience shortcuts
    parser.add_argument("--image-first", dest="image_first", default=None,
                        help="Conditioning image for first frame (strength 1.0)")
    parser.add_argument("--image-mid", dest="image_mid", default=None,
                        help="Conditioning image for middle frame (strength 1.0)")
    parser.add_argument("--image-last", dest="image_last", default=None,
                        help="Conditioning image for last frame (strength 1.0)")

    # LoRA
    parser.add_argument("--lora", action="append", nargs="+",
                        metavar=("PATH", "STRENGTH"),
                        help="LoRA: PATH [STRENGTH] (repeatable)")

    # Audio input (for A2V pipeline)
    parser.add_argument("--audio", default=None, help="Audio file for audio-to-video")

    # Extend / Retake / Clone modes
    parser.add_argument("--extend", nargs=2, metavar=("VIDEO", "SECONDS"),
                        help="Extend an existing video by N seconds")
    parser.add_argument("--clone", nargs=2, metavar=("VIDEO", "SECONDS"),
                        help="Clone: extend video, keep only the new part")
    parser.add_argument("--retake", nargs=3, metavar=("VIDEO", "START", "END"),
                        help="Retake a time region of an existing video")
    parser.add_argument("--ref-seconds", type=float, default=None, dest="ref_seconds",
                        help="Context seconds from source video (default: 2 for extend, 5 for clone)")

    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.randrange(2**31)
    model_info = _MODEL_MAP[args.model]

    # ── Resolve image conditionings ───────────────────────────────────
    from ltx_pipelines.utils.args import ImageConditioningInput

    def _pan_scan_resize(img_path: str, target_w: int, target_h: int) -> str:
        """Pan&Scan: center-crop to target ratio, resize to target dims. Returns temp path."""
        from PIL import Image
        img = Image.open(img_path)
        w, h = img.size
        if w == target_w and h == target_h:
            return img_path
        target_ratio = target_w / target_h
        src_ratio = w / h
        if src_ratio > target_ratio:
            new_w = int(h * target_ratio)
            left = (w - new_w) // 2
            img = img.crop((left, 0, left + new_w, h))
        else:
            new_h = int(w / target_ratio)
            top = (h - new_h) // 2
            img = img.crop((0, top, w, top + new_h))
        img = img.resize((target_w, target_h), Image.LANCZOS)
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(tmp.name)
        print(f"  Image resized: {w}x{h} → {target_w}x{target_h} (Pan&Scan)", file=sys.stderr)
        return tmp.name

    image_conds = []
    if args.images:
        for path, frame_idx, strength in args.images:
            resized = _pan_scan_resize(str(Path(path).resolve()), args.width, args.height)
            image_conds.append(ImageConditioningInput(
                path=resized,
                frame_idx=int(frame_idx),
                strength=float(strength),
            ))
    if args.image_first:
        resized = _pan_scan_resize(str(Path(args.image_first).resolve()), args.width, args.height)
        image_conds.append(ImageConditioningInput(
            path=resized,
            frame_idx=0,
            strength=1.0,
        ))
    if args.image_mid:
        resized = _pan_scan_resize(str(Path(args.image_mid).resolve()), args.width, args.height)
        image_conds.append(ImageConditioningInput(
            path=resized,
            frame_idx=args.num_frames // 2,
            strength=1.0,
        ))
    if args.image_last:
        resized = _pan_scan_resize(str(Path(args.image_last).resolve()), args.width, args.height)
        image_conds.append(ImageConditioningInput(
            path=resized,
            frame_idx=args.num_frames - 1,
            strength=1.0,
        ))

    # ── Resolve LoRAs ─────────────────────────────────────────────────
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps

    loras = []
    if args.lora:
        for lora_args in args.lora:
            lora_path = str(Path(lora_args[0]).resolve())
            lora_strength = float(lora_args[1]) if len(lora_args) > 1 else 1.0
            loras.append(LoraPathStrengthAndSDOps(lora_path, lora_strength, LTXV_LORA_COMFY_RENAMING_MAP))

    # ── Derive num_frames from audio if not specified ──────────────────
    if args.num_frames is None:
        if args.audio:
            import subprocess, json as _json
            probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", args.audio],
                capture_output=True, text=True,
            )
            info = _json.loads(probe.stdout)
            for s in info.get("streams", []):
                if s.get("codec_type") == "audio":
                    duration = float(s["duration"])
                    raw = int(duration * args.frame_rate)
                    # nearest 8k+1 (round up)
                    args.num_frames = (((raw - 1 + 7) // 8) * 8) + 1
                    break
            if args.num_frames is None:
                args.num_frames = 121
        else:
            args.num_frames = 121

    # ── Pre-flight checks ────────────────────────────────────────────
    args.num_frames = _auto_reduce_frames(args.num_frames, args.width, args.height)

    # ── Resolve output paths ──────────────────────────────────────────
    out_base = Path(args.output)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    out = _output_paths(out_base)

    # ── Log ────────────────────────────────────────────────────────────
    free_gb = _free_memory() / (1024**3)
    print(f"Model: ltx-2.3-22b-{args.model}", file=sys.stderr)
    print(f"Seed: {seed}", file=sys.stderr)
    print(f"Dimensions: {args.width}x{args.height}, {args.num_frames} frames @ {args.frame_rate}fps", file=sys.stderr)
    print(f"Available memory: {free_gb:.1f} GB", file=sys.stderr)
    if image_conds:
        print(f"Image conditionings: {len(image_conds)}", file=sys.stderr)
    if args.audio:
        print(f"Audio input: {args.audio}", file=sys.stderr)

    # ── Resolve model files ───────────────────────────────────────────
    import torch
    checkpoint_path = str(_resolve_model_path(model_info["checkpoint"]))
    upscaler_path = str(_resolve_model_path(_UPSCALER_FILE))
    gemma_root = str(_resolve_gemma_path())

    # Distilled lora for stage-2 (only for dev/two_stage pipeline)
    distilled_lora_path = None
    if model_info["pipeline"] == "two_stage":
        distilled_lora_path = str(_resolve_model_path(_DISTILLED_LORA_FILE))

    # ── Run pipeline ──────────────────────────────────────────────────
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_pipelines.utils.media_io import encode_video

    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)

    if args.extend or args.retake or args.clone:
        # Extend / Retake / Clone pipeline
        from ltx_pipelines.retake import RetakePipeline
        from ltx_pipelines.utils.constants import detect_params
        from ltx_core.components.guiders import MultiModalGuiderParams
        from ltx_pipelines.utils.media_io import get_videostream_metadata

        import subprocess, tempfile

        extend_output_frames = None
        _extend_original_path = None  # for concat after pipeline
        _extend_ref_seconds = None
        _clone_mode = args.clone is not None

        if args.extend or args.clone:
            src = args.extend or args.clone
            video_path, seconds_str = src
            extend_seconds = float(seconds_str)
        else:
            video_path, start_str, end_str = args.retake
            start_time = float(start_str)
            end_time = float(end_str)

        fps, src_frames, src_w, src_h = get_videostream_metadata(video_path)
        target_w, target_h = args.width, args.height
        target_fps = args.frame_rate

        if args.extend or args.clone:
            ref_seconds = args.ref_seconds if args.ref_seconds is not None else (5.0 if _clone_mode else 2.0)
            video_duration = src_frames / fps

            if _clone_mode:
                # Trim ZUERST auf genau 1+8*k frames, dann Pan & Scan
                target_frames = max(9, ((int(ref_seconds * fps) - 1) // 8) * 8 + 1)
                skip_frames = max(0, src_frames - target_frames)
                if skip_frames > 0:
                    print(f"  Trimming to last {ref_seconds:.0f}s context ({target_frames} frames) …", file=sys.stderr)
                    tmp_trim = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                    subprocess.run([
                        "ffmpeg", "-y", "-i", video_path,
                        "-vf", f"select='gte(n\\,{skip_frames})',setpts=PTS-STARTPTS",
                        "-af", f"atrim=start={skip_frames / fps:.6f},asetpts=PTS-STARTPTS",
                        tmp_trim.name,
                    ], check=True, capture_output=True)
                    video_path = tmp_trim.name
                    fps, src_frames, src_w, src_h = get_videostream_metadata(video_path)
                    video_duration = src_frames / fps
            elif video_duration > ref_seconds:
                _extend_original_path = video_path  # for concat later (extend only)
                ref_frames = round(ref_seconds * fps)
                skip_frames = src_frames - ref_frames
                print(f"  Trimming to last {ref_seconds:.0f}s context ({ref_frames} frames) …", file=sys.stderr)
                tmp_trim = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                subprocess.run([
                    "ffmpeg", "-y", "-i", video_path,
                    "-vf", f"select='gte(n\\,{skip_frames})',setpts=PTS-STARTPTS",
                    "-af", f"atrim=start={skip_frames / fps:.6f},asetpts=PTS-STARTPTS",
                    tmp_trim.name,
                ], check=True, capture_output=True)
                video_path = tmp_trim.name
                fps, src_frames, src_w, src_h = get_videostream_metadata(video_path)
                video_duration = src_frames / fps

        # Pan & Scan nach dem Trim
        needs_transcode = (target_w != src_w or target_h != src_h
                           or target_fps != int(fps))
        if needs_transcode:
            vf_filters = []
            label_parts = []
            if target_w != src_w or target_h != src_h:
                src_ratio = src_w / src_h
                tgt_ratio = target_w / target_h
                if src_ratio > tgt_ratio:
                    crop_h = src_h
                    crop_w = int(src_h * tgt_ratio)
                else:
                    crop_w = src_w
                    crop_h = int(src_w / tgt_ratio)
                x_off = (src_w - crop_w) // 2
                y_off = (src_h - crop_h) // 2
                vf_filters.append(f"crop={crop_w}:{crop_h}:{x_off}:{y_off}")
                vf_filters.append(f"scale={target_w}:{target_h}")
                label_parts.append(f"{src_w}x{src_h} → {target_w}x{target_h}")
            if target_fps != int(fps):
                vf_filters.append(f"fps={target_fps}")
                label_parts.append(f"{int(fps)}fps → {target_fps}fps")
            print(f"  Transcoding ({', '.join(label_parts)}) …", file=sys.stderr)
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            subprocess.run([
                "ffmpeg", "-y", "-i", video_path,
                "-vf", ",".join(vf_filters),
                "-c:a", "copy", tmp.name,
            ], check=True, capture_output=True)
            video_path = tmp.name
            fps, src_frames, src_w, src_h = get_videostream_metadata(video_path)

        if args.extend or args.clone:
            video_duration = src_frames / fps
            start_time = video_duration
            end_time = video_duration + extend_seconds
            raw_frames = int(end_time * fps)
            extend_output_frames = ((raw_frames // 8) * 8) + 1
            if _clone_mode:
                mode_label = f"Cloning video ({extend_seconds:.1f}s, extend pipeline, context={video_duration:.1f}s)"
            else:
                mode_label = f"Extending video by {extend_seconds:.1f}s ({extend_output_frames} frames)"
        else:
            mode_label = f"Retaking {start_time:.1f}s–{end_time:.1f}s"

        print(f"  {mode_label} …", file=sys.stderr)
        print(f"  Source: {src_w}x{src_h}, {src_frames} frames @ {fps}fps", file=sys.stderr)

        is_distilled = model_info["pipeline"] == "distilled"

        print("  Loading Retake pipeline …", file=sys.stderr)
        pipeline = RetakePipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            loras=loras,
        )

        params = detect_params(checkpoint_path)
        vgp = params.video_guider_params
        agp = params.audio_guider_params
        if args.cfg_scale is not None and vgp is not None:
            vgp = MultiModalGuiderParams(
                cfg_scale=args.cfg_scale,
                stg_scale=vgp.stg_scale,
                rescale_scale=vgp.rescale_scale,
                modality_scale=vgp.modality_scale,
                skip_step=vgp.skip_step,
                stg_blocks=vgp.stg_blocks,
            )

        video, audio = pipeline(
            video_path=video_path,
            prompt=args.prompt,
            start_time=start_time,
            end_time=end_time,
            seed=seed,
            negative_prompt=args.negative_prompt or "",
            num_inference_steps=args.steps or params.num_inference_steps,
            video_guider_params=vgp if not is_distilled else None,
            audio_guider_params=agp if not is_distilled else None,
            distilled=is_distilled,
            tiling_config=tiling_config,
            enhance_prompt=args.enhance_prompt,
            output_frames=extend_output_frames,
        )
        # Recalculate chunks for the actual output frame count
        actual_frames = extend_output_frames if extend_output_frames else src_frames
        video_chunks_number = get_video_chunks_number(actual_frames, tiling_config)

        # If extend with trimmed context: trim original (remove last ref_seconds),
        # encode full pipeline output, concat original_trimmed + pipeline_output
        if _extend_original_path is not None:
            _tmp_pipeline_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            encode_video(video=video, fps=int(fps), audio=audio,
                         output_path=_tmp_pipeline_out.name,
                         video_chunks_number=video_chunks_number)
            # Trim original: remove last ref_seconds (those are now in pipeline output)
            _orig_fps, _orig_frames, _, _ = get_videostream_metadata(_extend_original_path)
            _keep_frames = _orig_frames - _extend_context_frames
            _tmp_orig_trimmed = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            subprocess.run([
                "ffmpeg", "-y", "-i", _extend_original_path,
                "-vf", f"select='lt(n\\,{_keep_frames})',setpts=PTS-STARTPTS",
                "-af", f"atrim=end={_keep_frames / _orig_fps:.6f},asetpts=PTS-STARTPTS",
                _tmp_orig_trimmed.name,
            ], check=True, capture_output=True)
            # Concat: original (minus context) + full pipeline output
            _concat_list = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
            _concat_list.write(f"file '{_tmp_orig_trimmed.name}'\n")
            _concat_list.write(f"file '{_tmp_pipeline_out.name}'\n")
            _concat_list.close()
            print("  Concatenating original + extension …", file=sys.stderr)
            out_path = str(Path(args.output))
            subprocess.run([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", _concat_list.name, "-c", "copy", out_path,
            ], check=True, capture_output=True)
            os.unlink(_tmp_pipeline_out.name)
            os.unlink(_tmp_orig_trimmed.name)
            os.unlink(_concat_list.name)
            # Skip normal encode — output already written
            sys.stderr.flush()
            print(json.dumps([out_path]))
            return

        if _clone_mode:
            # Encode full pipeline output (context + clone), then crop context off
            _tmp_full = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            encode_video(video=video, fps=int(fps), audio=audio,
                         output_path=_tmp_full.name,
                         video_chunks_number=video_chunks_number)
            out_path = str(Path(args.output))
            print(f"  Cropping source context ({video_duration:.2f}s / {src_frames} frames) from clone output …", file=sys.stderr)
            subprocess.run([
                "ffmpeg", "-y", "-i", _tmp_full.name,
                "-vf", f"select='gte(n\\,{src_frames})',setpts=PTS-STARTPTS",
                "-af", f"atrim=start={video_duration:.6f},asetpts=PTS-STARTPTS",
                "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                "-c:a", "aac", out_path,
            ], check=True, capture_output=True)
            os.unlink(_tmp_full.name)
            sys.stderr.flush()
            print(json.dumps([out_path]))
            return

    elif args.audio:
        # Audio-to-Video pipeline
        print("Loading A2V pipeline …", file=sys.stderr)
        from ltx_pipelines.a2vid_two_stage import A2VidPipelineTwoStage
        from ltx_pipelines.utils.constants import detect_params
        from ltx_core.components.guiders import MultiModalGuiderParams

        params = detect_params(checkpoint_path)
        pipeline = A2VidPipelineTwoStage(
            checkpoint_path=checkpoint_path,
            distilled_lora=_build_distilled_lora(distilled_lora_path),
            spatial_upsampler_path=upscaler_path,
            gemma_root=gemma_root,
            loras=tuple(loras),
        )
        video, audio = pipeline(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt or "",
            seed=seed,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            frame_rate=args.frame_rate,
            num_inference_steps=args.steps or params.num_inference_steps,
            video_guider_params=params.video_guider_params,
            images=image_conds,
            audio_path=args.audio,
            tiling_config=tiling_config,
            enhance_prompt=args.enhance_prompt,
        )

    elif model_info["pipeline"] == "distilled":
        print("  Loading Distilled pipeline …", file=sys.stderr)
        from ltx_pipelines.distilled import DistilledPipeline

        pipeline = DistilledPipeline(
            distilled_checkpoint_path=checkpoint_path,
            spatial_upsampler_path=upscaler_path,
            gemma_root=gemma_root,
            loras=tuple(loras),
        )

        video, audio = pipeline(
            prompt=args.prompt,
            seed=seed,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            frame_rate=args.frame_rate,
            images=image_conds,
            tiling_config=tiling_config,
            enhance_prompt=args.enhance_prompt,
        )

    else:
        # Dev / two-stage pipeline
        print("Loading Two-Stage pipeline …", file=sys.stderr)
        from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
        from ltx_pipelines.utils.constants import detect_params
        from ltx_core.components.guiders import MultiModalGuiderParams

        params = detect_params(checkpoint_path)
        distilled_lora = _build_distilled_lora(distilled_lora_path)

        pipeline = TI2VidTwoStagesPipeline(
            checkpoint_path=checkpoint_path,
            distilled_lora=distilled_lora,
            spatial_upsampler_path=upscaler_path,
            gemma_root=gemma_root,
            loras=tuple(loras),
        )

        vgp = params.video_guider_params
        agp = params.audio_guider_params
        if args.cfg_scale is not None:
            vgp = MultiModalGuiderParams(
                cfg_scale=args.cfg_scale,
                stg_scale=vgp.stg_scale,
                rescale_scale=vgp.rescale_scale,
                modality_scale=vgp.modality_scale,
                skip_step=vgp.skip_step,
                stg_blocks=vgp.stg_blocks,
            )

        video, audio = pipeline(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt or "",
            seed=seed,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            frame_rate=args.frame_rate,
            num_inference_steps=args.steps or params.num_inference_steps,
            video_guider_params=vgp,
            audio_guider_params=agp,
            images=image_conds,
            tiling_config=tiling_config,
            enhance_prompt=args.enhance_prompt,
        )

    # ── Encode output ─────────────────────────────────────────────────
    print("Encoding output …", file=sys.stderr)
    result_paths = []

    if out["mode"] == "muxed":
        encode_video(video=video, fps=args.frame_rate, audio=audio,
                     output_path=str(out["video"]),
                     video_chunks_number=video_chunks_number)
        result_paths.append(str(out["video"]))

    elif out["mode"] == "video_only":
        encode_video(video=video, fps=args.frame_rate, audio=None,
                     output_path=str(out["video"]),
                     video_chunks_number=video_chunks_number)
        result_paths.append(str(out["video"]))

    elif out["mode"] == "audio_only":
        # Encode with minimal video, extract audio
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            encode_video(video=video, fps=args.frame_rate, audio=audio,
                         output_path=tmp.name,
                         video_chunks_number=video_chunks_number)
        # Extract audio with ffmpeg
        import subprocess
        subprocess.run(["ffmpeg", "-y", "-i", tmp.name, "-vn", "-acodec", "copy",
                       str(out["audio"])], check=True, capture_output=True)
        os.unlink(tmp.name)
        result_paths.append(str(out["audio"]))

    elif out["mode"] == "split":
        # Video without audio
        encode_video(video=video, fps=args.frame_rate, audio=None,
                     output_path=str(out["video"]),
                     video_chunks_number=video_chunks_number)
        result_paths.append(str(out["video"]))
        # Audio separately
        if audio is not None:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                encode_video(video=video, fps=args.frame_rate, audio=audio,
                             output_path=tmp.name,
                             video_chunks_number=video_chunks_number)
            import subprocess
            subprocess.run(["ffmpeg", "-y", "-i", tmp.name, "-vn", "-acodec", "copy",
                           str(out["audio"])], check=True, capture_output=True)
            os.unlink(tmp.name)
            result_paths.append(str(out["audio"]))

    sys.stderr.flush()
    print(json.dumps(result_paths))


def _build_distilled_lora(path: str | None):
    """Build distilled LoRA list for two-stage pipeline."""
    if path is None:
        return []
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    return [LoraPathStrengthAndSDOps(path, 1.0, LTXV_LORA_COMFY_RENAMING_MAP)]


if __name__ == "__main__":
    with __import__("torch").inference_mode():
        main()
