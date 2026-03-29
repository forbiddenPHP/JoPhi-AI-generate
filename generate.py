#!/usr/bin/env python3
"""
generate — Unified media generation toolkit.

Usage:
  python generate.py ps                                                Show available models & status
  python generate.py voice ai-tts --text "Hello world" -o demos/         Neural TTS (Qwen3-TTS)
  python generate.py voice ai-tts -v Serena --text "Hello"               TTS with preset voice
  python generate.py voice ai-tts -v Aiden -t "dramatic" --text "Silence."  TTS with style
  python generate.py voice ai-tts --tts-model small --text "Hi"          Smaller model (0.6B)
  python generate.py voice ai-tts --text "[Aiden: excited] Hi! [Serena: calm] Hello."  Per-segment style
  python generate.py voice ai-tts --prompt-file demos/speech.txt -o demos/  From sidecar
  python generate.py voice ai-tts --list-voices                          Show available voices
  python generate.py voice say --text "Hello world" -o demos/            macOS TTS
  python generate.py voice say -v Anna --text "Hallo" -o demos/          TTS with specific voice
  python generate.py voice say --model my-voice --text "Hallo"           TTS + RVC voice conversion
  python generate.py voice rvc --model my-voice input.wav                Voice conversion
  python generate.py voice rvc --model my-voice input.wav --pitch 12
  python generate.py voice rvc --model my-voice input.wav --target-hz 280
  python generate.py voice rvc --model my-voice input.wav --decoder crepe
  python generate.py voice clone-tts --text "Hello" -o out.wav           Voice cloning (default ref)
  python generate.py voice clone-tts --reference ref.wav --text "Hello"  Voice cloning (custom ref)
  python generate.py audio enhance input.wav                             Denoise + enhance audio
  python generate.py audio enhance input.wav --denoise-only              Denoise only (faster)
  python generate.py audio enhance input.wav --enhance-only              Super-resolution only
  python generate.py audio demucs input.wav                              Separate into stems
  python generate.py audio demucs input.wav --model htdemucs_ft
  python generate.py audio ace-step -l "lyrics" -t "disco,happy" -o out.mp3
  python generate.py audio ace-step --model sft -f lyrics.txt -t "cinematic"
  python generate.py audio heartmula -l "lyrics" -t "disco,happy" -o out.mp3
  python generate.py audio diarize interview.wav                         Split dialogue by speaker
  python generate.py audio diarize interview.wav --speakers 3
  python generate.py audio diarize interview.wav --verify
  python generate.py audio sfx --text "a dog barking" -o sfx.wav         Sound effect generation
  python generate.py audio sfx --text "thunder and rain" -s 10 -o weather.wav
  python generate.py audio voice-removal song.mp3 -o karaoke/            Remove vocals
  python generate.py text whisper audio.wav                              Transcribe audio
  python generate.py text whisper audio.wav --model large-v3 --format srt
  python generate.py text whisper audio.wav --language de
  python generate.py text heartmula-transcribe song.mp3                  Extract lyrics
  python generate.py text ollama --model qwen3.5:latest --endpoint chat --messages '[{"role":"user","content":"Hi"}]'
  python generate.py text ollama --model qwen3.5:latest --endpoint generate --prompt "Explain X"
  python generate.py output audio-concatenate a.wav b.mp3 -o out.wav     Concatenate audio files
  python generate.py output audio-concatenate a.wav b.wav --output-bitrate 320k -o out.mp3
  python generate.py output audio-concatenate a.wav b.wav c.mp3 --clip 0:fade-in=0.3 --clip 1:crossfade=0.5,volume=1.2 --clip 2:fade-out=0.5 -o out.mp3
  python generate.py output audio-mucs vocals.wav drums.wav -o mix.wav   Mix/overlay audio
  python generate.py output audio-mucs v.wav d.wav --clip 0:pan=-0.2,volume=0.8 -o mix.wav
  python generate.py image flux.2 -p "a cat on a cliff" -o cat.png       Image generation (FLUX.2)
  python generate.py image flux.2 --model 4b-distilled -p "a cat" -o cat.png
  python generate.py image flux.2 --controlnet depth:depth.png -p "cartoon" -o out.png  ControlNet conditioning
  python generate.py image flux.2 --images ref.png -p "edit this" --no-rescale -o out.png  Keep ref original size
  python generate.py image sd1.5 -p "a warrior, studio lighting" -o out.png  Stable Diffusion 1.5
  python generate.py image openpose --images person.png -o pose.png      Pose estimation
  python generate.py image depth --images photo.png -o depth.png         Depth estimation
  python generate.py image lineart --images photo.png -o lines.png       Line art extraction
  python generate.py image normalmap --images photo.png -o normals.png   Normal map estimation
  python generate.py image sketch --images photo.png -o sketch.png       Sketch extraction
  python generate.py image upscale --images photo.png -o upscaled.png    Image upscaling
  python generate.py image segment --images photo.png -o transparent.png Background removal
  python generate.py video ltx2.3 -p "A cat running" --ratio 16:9 --quality 480p -o cat.mp4  Text-to-video (distilled)
  python generate.py video ltx2.3 --model dev -p "Sunset over ocean" --ratio 16:9 --quality 480p -o sunset.mp4  Text-to-video (dev)
  python generate.py video ltx2.3 -p "He smiles" --image-first photo.png --ratio 1:1 --quality 480p -o anim.mp4  Image-to-video
  python generate.py video ltx2.3 -p "Two people talking" --audio dialog.wav --ratio 16:9 --quality 240p -o a2v.mp4  Audio-to-video
  python generate.py video ltx2.3 --model dev -p "He says: hello" --extend video.mp4 5 --ratio 16:9 --quality 240p -o ext.mp4  Extend video (+5s)
  python generate.py video ltx2.3 --model dev -p "She says: hi" --clone ref.mp4 --ratio 16:9 --quality 720p -o clone.mp4  Clone (new video from reference)
  python generate.py video ltx2.3 --model dev -p "Full scene with change" --retake video.mp4 3.5 5.0 --ratio 16:9 --quality 240p -o ret.mp4  Retake passage
  python generate.py audio ltx2.3 -p "Oktoberfest crowd cheering" --ratio 1:1 --quality 240p -o crowd.wav  Audio-only (virtual)
  python generate.py models list                                         List all models
  python generate.py image models list                                   List image engine models
  python generate.py image flux.2 models list                            List FLUX.2 models only
  python generate.py models rvc list                                     List RVC models
  python generate.py models rvc search "neutral male"                    Search HuggingFace
  python generate.py models rvc install <id-or-url>                      Download from HuggingFace or URL
  python generate.py models rvc remove <name>                            Remove a model
  python generate.py models rvc set-pitch <name> <hz>                    Set target pitch manually
  python generate.py models rvc calibrate <name>                         Auto-detect target pitch
  python generate.py models ollama list                                  List Ollama models
  python generate.py models ollama pull qwen3.5:latest                   Pull Ollama model
  python generate.py models ollama remove <name>                         Remove Ollama model
  python generate.py models huggingface list                             List cached HF models
  python generate.py models huggingface search "qwen vision"             Search HuggingFace
  python generate.py models huggingface pull Org/Model                   Download HF model
  python generate.py server start                          Start RVC worker
  python generate.py server stop                           Stop RVC worker
  python generate.py server status                         Check worker status
  python generate.py <any command> --screen-log-format json JSON output (for automation)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

from progress import (
    run_worker, print_event_tui, print_event_json, finish_progress,
    ProgressEvent,
)

# ── Output mode ──────────────────────────────────────────────────────────────

_event_handler = print_event_tui


def _emit(message: str, type: str = "log"):
    """Route a status message through the event handler."""
    _event_handler(ProgressEvent(type=type, message=message))


# ── Config ───────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent

RVC_API_URL = os.environ.get("RVC_API_URL", "http://127.0.0.1:5100")
RVC_WORKER_DIR = SCRIPT_DIR / "worker" / "rvc"
RVC_MODELS_DIR = SCRIPT_DIR / "worker" / "rvc" / "models"
ENHANCE_WORKER_DIR = SCRIPT_DIR / "worker" / "enhance"
MUSIC_WORKER_DIR = SCRIPT_DIR / "worker" / "music"
MUSIC_MODELS_DIR = SCRIPT_DIR / "worker" / "music" / "models"
def _resolve_conda_bin() -> Path:
    """Resolve conda binary: $CONDA_BIN → ~/.ai-conda-path → dynamic search."""
    import shutil
    # 1. Env var
    env_val = os.environ.get("CONDA_BIN")
    if env_val and Path(env_val).is_file():
        return Path(env_val)
    # 2. Cached path from setup.sh
    cache = Path.home() / ".ai-conda-path"
    if cache.is_file():
        cached = cache.read_text().strip()
        if cached and Path(cached).is_file():
            return Path(cached)
    # 3. conda in PATH
    which = shutil.which("conda")
    if which:
        return Path(which)
    # 4. Common locations
    for p in [
        Path("/opt/miniconda3/bin/conda"),
        Path("/opt/homebrew/Caskroom/miniconda/base/bin/conda"),
        Path.home() / "miniconda3" / "bin" / "conda",
        Path.home() / "anaconda3" / "bin" / "conda",
    ]:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "conda not found. Run ./setup.sh first or set CONDA_BIN."
    )

CONDA_BIN = _resolve_conda_bin()
RVC_ENV = "rvc"
ENHANCE_ENV = "enhance"
HEARTMULA_ENV = "heartmula"
ACESTEP_DIR = SCRIPT_DIR / "worker" / "ace" / "ACE-Step-1.5"
ACESTEP_WORKER = SCRIPT_DIR / "worker" / "ace" / "generate.py"
WHISPER_WORKER_DIR = SCRIPT_DIR / "worker" / "whisper"
WHISPER_ENV = "whisper"
DIARIZE_WORKER_DIR = SCRIPT_DIR / "worker" / "diarize"
DIARIZE_ENV = "diarize"
SEPARATE_WORKER_DIR = SCRIPT_DIR / "worker" / "separate"
SEPARATE_ENV = "separate"
TTS_ENV = "ai-tts"
TTS_WORKER_DIR = SCRIPT_DIR / "worker" / "tts"
LANGDETECT_ENV = "lang-detect"
LANGDETECT_WORKER = SCRIPT_DIR / "worker" / "langdetect" / "detect.py"
SFX_WORKER_DIR = SCRIPT_DIR / "worker" / "sfx"
SFX_ENV = "ezaudio"
# Voice cloning uses Qwen3-TTS Base via TTS_WORKER_DIR / TTS_ENV
TEXT_WORKER_DIR = SCRIPT_DIR / "worker" / "text"
TEXT_MODELS_DIR = SCRIPT_DIR / "worker" / "text" / "models"
TEXT_ENGINES_FILE = SCRIPT_DIR / "worker" / "text" / "engines.json"


def _find_uv() -> Path:
    """Find uv binary (prefer ~/.local/bin, then PATH)."""
    if "UV_BIN" in os.environ:
        return Path(os.environ["UV_BIN"])
    local = Path.home() / ".local" / "bin" / "uv"
    if local.exists():
        return local
    return Path("uv")


UV_BIN = _find_uv()
PID_FILE = RVC_WORKER_DIR / ".server.pid"


# ── F0 Analysis (Auto-Pitch) ────────────────────────────────────────────────

def detect_input_f0(wav_path: str | Path) -> float | None:
    """Detect median fundamental frequency of a WAV file using pyworld."""
    script = (
        "import pyworld, soundfile, numpy, json; "
        "x, sr = soundfile.read(r'" + str(wav_path) + "'); "
        "x = x.mean(axis=1) if x.ndim > 1 else x; "
        "f0, _ = pyworld.harvest(x.astype('float64'), sr); "
        "v = f0[f0 > 0]; "
        "print(json.dumps({'median_f0': float(numpy.median(v))}) if len(v) > 0 else json.dumps({'median_f0': None}))"
    )
    try:
        result = run_worker(
            [str(CONDA_BIN), "run", "--no-capture-output", "-n", RVC_ENV, "python", "-c", script])
        if result.returncode != 0:
            _emit(f"  WARNING: F0 detection failed: {result.stderr_tail[:200]}", "warning")
            return None
        data = json.loads(result.stdout.strip())
        return data.get("median_f0")
    except Exception as e:
        _emit(f"  WARNING: F0 detection error: {e}", "warning")
        return None


def load_model_config(model_name: str) -> dict:
    """Load per-model config from worker/rvc/models/<model>/revoicer.json."""
    config_path = RVC_MODELS_DIR / model_name / "revoicer.json"
    if config_path.exists():
        return json.loads(config_path.read_text())
    return {}


def save_model_config(model_name: str, config: dict):
    """Save per-model config to worker/rvc/models/<model>/revoicer.json."""
    config_dir = RVC_MODELS_DIR / model_name
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "revoicer.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")


def compute_pitch_shift(input_f0: float, target_f0: float) -> int:
    """Compute pitch shift in semitones to go from input_f0 to target_f0."""
    return round(12 * math.log2(target_f0 / input_f0))


# ── API Client ───────────────────────────────────────────────────────────────

def api_get(endpoint: str) -> dict:
    try:
        r = requests.get(f"{RVC_API_URL}{endpoint}", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        _emit("ERROR: RVC worker not running.", "error")
        _emit("  Start with: python generate.py server start", "error")
        sys.exit(1)
    except requests.HTTPError as e:
        _emit(f"ERROR: API returned {e.response.status_code}", "error")
        sys.exit(1)


def api_post(endpoint: str, data: dict = None, files: dict = None,
             json: dict = None, timeout: int = None) -> requests.Response:
    try:
        r = requests.post(f"{RVC_API_URL}{endpoint}",
                          data=data, files=files, json=json, timeout=timeout)
        r.raise_for_status()
        return r
    except requests.ConnectionError:
        _emit("ERROR: RVC worker not running.", "error")
        _emit("  Start with: python generate.py server start", "error")
        sys.exit(1)
    except requests.HTTPError as e:
        _emit(f"ERROR: API returned {e.response.status_code}: {e.response.text}",
              "error")
        sys.exit(1)


def check_server() -> bool:
    try:
        requests.get(f"{RVC_API_URL}/models", timeout=3)
        return True
    except (requests.ConnectionError, requests.Timeout):
        return False


# ── Server Management ────────────────────────────────────────────────────────

def cmd_server_start(args):
    if check_server():
        _emit("RVC worker already running.")
        return

    if not CONDA_BIN.exists():
        _emit(f"ERROR: conda not found at {CONDA_BIN}", "error")
        sys.exit(1)

    _emit("Starting RVC worker …", "stage")
    port = args.port if hasattr(args, "port") else 5100
    start_sh = RVC_WORKER_DIR / "start.sh"

    cmd = f'bash {start_sh} {port}'
    proc = subprocess.Popen(
        cmd, shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setpgrp,
    )
    PID_FILE.write_text(str(proc.pid))

    for i in range(30):
        time.sleep(1)
        if check_server():
            _emit(f"RVC worker running on port {port} (PID {proc.pid})")
            return

    _emit("ERROR: RVC worker did not start within 30 seconds.", "error")
    proc.kill()
    sys.exit(1)


def cmd_server_stop(args):
    stopped = False

    # Try PID file first
    if PID_FILE.exists():
        pid = int(PID_FILE.read_text().strip())
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            _emit(f"RVC worker stopped (PID {pid})")
            stopped = True
        except ProcessLookupError:
            pass
        PID_FILE.unlink(missing_ok=True)

    # Fallback: kill whatever is on the port
    if not stopped and check_server():
        import subprocess
        port = RVC_API_URL.rsplit(":", 1)[-1].rstrip("/")
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True,
        )
        for pid_str in result.stdout.strip().split():
            try:
                os.kill(int(pid_str), signal.SIGTERM)
            except ProcessLookupError:
                pass
        _emit(f"RVC worker stopped (port {port})")
        stopped = True

    if not stopped:
        _emit("RVC worker is not running.")


def cmd_server_status(args):
    if check_server():
        models = api_get("/models")
        n = len(models.get("models", []))
        pid = PID_FILE.read_text().strip() if PID_FILE.exists() else "?"
        _emit(f"RVC worker: running (PID {pid})")
        _emit(f"  URL:    {RVC_API_URL}")
        _emit(f"  Models: {n} loaded")
    else:
        _emit("RVC worker: not running")
        _emit(f"  Start with: python generate.py server start")


# ── Models ───────────────────────────────────────────────────────────────────

def cmd_models(args):
    """Models dispatcher — enforce engine for non-list subcommands."""
    if not args.models_cmd:
        build_parser().parse_args(["models", "--help"])
        return
    if args.models_cmd != "list" and not args.engine:
        _emit(f"ERROR: engine required for 'models {args.models_cmd}'", "error")
        _emit(f"  Example: python generate.py models rvc {args.models_cmd} ...")
        sys.exit(1)

    # Validate subcommand is valid for engine
    engine = args.engine
    if engine and args.models_cmd:
        allowed = _MODELS_ENGINE_CMDS.get(engine, set())
        if args.models_cmd not in allowed:
            _emit(f"ERROR: '{args.models_cmd}' not supported for engine '{engine}'", "error")
            _emit(f"  Valid: {', '.join(sorted(allowed))}")
            sys.exit(1)

    args.models_func(args)


def _models_list_rvc():
    """List RVC voice models."""
    data = api_get("/models")
    models = data.get("models", [])
    if not models:
        _emit("No RVC models installed.")
        _emit("  Search: python generate.py models rvc search \"neutral male\"")
        _emit("  Install: python generate.py models rvc install <hf-model-id>")
        return
    _emit(f"RVC models ({len(models)}):\n")
    for m in models:
        if isinstance(m, str):
            _emit(f"  {m}")
        else:
            name = m.get("name", m.get("model_name", str(m)))
            _emit(f"  {name}")


def _query_worker_models(env: str, worker_script: str, runner: str = "conda") -> list[dict]:
    """Call a worker with --list-models and return parsed JSON, or [] on failure."""
    try:
        if runner == "uv":
            # ACE-Step uses uv run --project
            project_dir = str(Path(worker_script).resolve().parent / "ACE-Step-1.5")
            cmd = [str(UV_BIN), "run", "--project", project_dir,
                   "python", worker_script, "--list-models"]
        elif runner == "venv":
            # Worker with local .venv (e.g. MLX worker)
            venv_python = str(Path(worker_script).resolve().parent / ".venv" / "bin" / "python")
            cmd = [venv_python, worker_script, "--list-models"]
        elif runner == "native":
            # No conda env needed (e.g. say voices)
            cmd = [sys.executable, worker_script, "--list-models"]
        else:
            cmd = [str(CONDA_BIN), "run", "--no-capture-output", "-n", env,
                   "python", worker_script, "--list-models"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout.strip())
    except Exception:
        pass
    return []


# Worker registry: (modus, engine_name, conda_env, worker_script)
# Workers without a script (say, clone-tts, output engines) use static entries.
_WORKER_REGISTRY = []  # populated lazily


def _get_worker_registry():
    """Build the worker registry once, referencing the module-level constants."""
    if _WORKER_REGISTRY:
        return _WORKER_REGISTRY
    _WORKER_REGISTRY.extend([
        # voice
        ("voice", "ai-tts",             TTS_ENV,       str(TTS_WORKER_DIR / "generate_speech.py")),
        ("voice", "rvc",                RVC_ENV,       str(RVC_WORKER_DIR / "list_models.py")),
        ("voice", "say",                "",            str(SCRIPT_DIR / "worker" / "say" / "list_models.py"), "native"),
        # audio
        ("audio", "enhance",            ENHANCE_ENV,   str(SCRIPT_DIR / "worker" / "enhance" / "enhance.py")),
        ("audio", "demucs",             SEPARATE_ENV,  str(SEPARATE_WORKER_DIR / "separate.py")),
        ("audio", "ace-step",           "acestep",     str(ACESTEP_WORKER), "uv"),
        ("audio", "heartmula",          HEARTMULA_ENV, str(MUSIC_WORKER_DIR / "generate.py")),
        ("audio", "sfx",               SFX_ENV,       str(SFX_WORKER_DIR / "generate.py")),
        ("audio", "diarize",            DIARIZE_ENV,   str(DIARIZE_WORKER_DIR / "diarize.py")),
        # text
        ("text",  "whisper",            WHISPER_ENV,   str(WHISPER_WORKER_DIR / "transcribe.py")),
        ("text",  "ollama",             TEXT_ENV,      str(TEXT_WORKER_DIR / "inference.py")),
        ("text",  "heartmula-transcribe", HEARTMULA_ENV, str(MUSIC_WORKER_DIR / "transcribe.py")),
        # image
        ("image", "flux.2",             FLUX2_ENV,     str(IMAGE_WORKER)),
        ("image", "sd1.5",              SD15_ENV,      str(SD15_WORKER)),
        ("image", "depth",              DEPTH_ENV,     str(DEPTH_WORKER)),
        ("image", "lineart",            LINEART_ENV,   str(LINEART_WORKER)),
        ("image", "normalmap",          NORMALMAP_ENV, str(NORMALMAP_WORKER)),
        ("image", "sketch",             SKETCH_ENV,    str(SKETCH_WORKER)),
        ("image", "upscale",            UPSCALE_ENV,   str(UPSCALE_WORKER)),
        ("image", "segment",            SEGMENT_ENV,   str(SEGMENT_WORKER)),
        ("image", "openpose",           POSE_ENV,      str(POSE_WORKER)),
        # video
        ("video", "ltx2.3",             LTX2_ENV,      str(LTX2_WORKER)),
    ])
    return _WORKER_REGISTRY


# Static entries for engines that don't have a worker with --list-models
_STATIC_MODELS = [
    ("voice", "clone-tts",         "",  "clone any voice from audio"),
    ("audio", "voice-removal",     "",  "demucs-based"),
    ("audio", "ltx2.3",            "",  "audio from video engine (virtual)"),
    ("output", "audio-concatenate", "", ""),
    ("output", "audio-mucs",       "",  ""),
]


def _models_list_all(medium=None, engine=None):
    """Query all workers for their models and display the result."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    rows = []

    # Static entries first
    for modus, eng, model, notice in _STATIC_MODELS:
        if medium and modus != medium:
            continue
        if engine and eng != engine:
            continue
        rows.append((modus, eng, model, notice))

    # Dynamic entries from workers — query all in parallel
    filtered = []
    for entry in _get_worker_registry():
        modus, eng, env, script = entry[:4]
        runner = entry[4] if len(entry) > 4 else "conda"
        if medium and modus != medium:
            continue
        if engine and eng != engine:
            continue
        filtered.append((modus, eng, env, script, runner))

    with ThreadPoolExecutor(max_workers=len(filtered) or 1) as pool:
        futures = {
            pool.submit(_query_worker_models, env, script, runner=runner): (modus, eng)
            for modus, eng, env, script, runner in filtered
        }
        for future in as_completed(futures):
            modus, eng = futures[future]
            models = future.result()
            if models:
                for m in models:
                    rows.append((modus, eng, m.get("model", ""), m.get("notice", "")))
            else:
                rows.append((modus, eng, "", "unavailable"))

    if not rows:
        _emit(f"No models found for {medium or ''} {engine or ''}".strip(), "error")
        return

    # Sort: voice → audio → text → image → output, then by engine
    _ORDER = {"voice": 0, "audio": 1, "text": 2, "image": 3, "output": 4}
    rows.sort(key=lambda r: (_ORDER.get(r[0], 99), r[1], r[2]))

    # JSON output
    if _event_handler is print_event_json:
        out = [{"modus": m, "engine": e, "model": mdl, "notice": n}
               for m, e, mdl, n in rows]
        print(json.dumps(out, indent=2))
        return

    # Column widths
    w_mod = max(len(r[0]) for r in rows)
    w_eng = max(len(r[1]) for r in rows)
    w_mdl = max(len(r[2]) for r in rows) if any(r[2] for r in rows) else 5
    w_mod = max(w_mod, 5)
    w_eng = max(w_eng, 6)
    w_mdl = max(w_mdl, 5)

    # Header
    hdr = f"  {'MODUS':<{w_mod}}  {'ENGINE':<{w_eng}}  {'MODEL':<{w_mdl}}  NOTICE"
    sep = f"  {'-' * w_mod}  {'-' * w_eng}  {'-' * w_mdl}  ------"
    print(hdr, file=sys.stderr)
    print(sep, file=sys.stderr)

    for modus, eng, model, notice in rows:
        line = f"  {modus:<{w_mod}}  {eng:<{w_eng}}  {model:<{w_mdl}}"
        if notice:
            line += f"  {notice}"
        print(line, file=sys.stderr)

    print(file=sys.stderr)


def cmd_models_list(args):
    engine = getattr(args, "engine", None)
    if engine is None:
        # List all engines
        _models_list_rvc()
        print()
        _models_list_ollama(args)
        print()
        _models_list_huggingface(args)
        print()
    elif engine == "rvc":
        _models_list_rvc()
    elif engine == "ollama":
        _models_list_ollama(args)
    elif engine == "huggingface":
        _models_list_huggingface(args)


def _check_rvc_repo(repo_id: str, files: list[str]) -> tuple[bool, list[str], list[str]]:
    """Check if a repo has usable RVC .pth files."""
    pth = [f for f in files if f.endswith(".pth")]
    idx = [f for f in files if f.endswith(".index")]

    if not pth:
        return False, [], []

    voice_pth = [f for f in pth if not any(
        f.split("/")[-1].startswith(p) for p in ("D_", "G_", "D-", "G-", "f0")
    )]
    if not voice_pth and len(pth) != 1:
        return False, [], []

    return True, voice_pth or pth, idx


def _search_voice_models_com(query: str, limit: int = 25) -> list[dict]:
    """Search voice-models.com for RVC models."""
    import re
    try:
        r = requests.post(
            "https://voice-models.com/fetch_data.php",
            data={"page": 1, "search": query},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        html = data.get("table", "")
    except Exception as e:
        _emit(f"WARNING: Cannot fetch results from voice-models.com: {e}", "warning")
        return []

    results = []
    rows = re.findall(r"<tr>(.*?)</tr>", html, re.DOTALL)
    for row in rows[:limit]:
        title_m = re.search(r"class=['\"]fs-5['\"][^>]*>(.*?)</a>", row, re.DOTALL)
        url_m = re.search(r"data-clipboard-text=['\"]([^'\"]+)['\"]", row)
        size_m = re.search(r"badge[^>]*>([^<]+)</span>", row)

        if not (title_m and url_m):
            continue

        title_raw = title_m.group(1)
        title_clean = re.sub(r"<[^>]+>", "", title_raw).strip()

        url = url_m.group(1).strip()
        hf_repo_id = None
        hf_m = re.match(r"https://huggingface\.co/([^/]+/[^/]+)/resolve/", url)
        if hf_m:
            hf_repo_id = hf_m.group(1)

        results.append({
            "title": title_clean,
            "url": url,
            "hf_repo_id": hf_repo_id,
            "size": size_m.group(1).strip() if size_m else "?",
        })

    if not results and html.strip():
        _emit("WARNING: Cannot parse results from voice-models.com "
              "(page structure may have changed)", "warning")

    return results


def cmd_models_search(args):
    engine = getattr(args, "engine", None)
    if engine == "huggingface":
        _models_search_huggingface(args)
        return
    # Default: RVC search
    query = args.query
    _emit(f'Searching for: "{query}" ...', "stage")

    vm_results = _search_voice_models_com(query, limit=args.limit)
    if vm_results:
        _emit(f"voice-models.com ({len(vm_results)} results):\n")
        for i, r in enumerate(vm_results, 1):
            install_id = r["hf_repo_id"] or r["url"]
            _emit(f"  {i:2d}. {r['title']}")
            _emit(f"      Size: {r['size']}")
            _emit(f"      Install: python generate.py models install {install_id}")

    try:
        from huggingface_hub import HfApi
    except ImportError:
        if not vm_results:
            _emit("ERROR: pip install huggingface-hub", "error")
            sys.exit(1)
        _emit("(HuggingFace search skipped — pip install huggingface-hub)")
        return

    api = HfApi()
    search_queries = set()
    search_queries.add(f"rvc {query}")
    for w in query.strip().split():
        search_queries.add(f"rvc {w}")

    seen = {}
    for sq in search_queries:
        try:
            for m in api.list_models(search=sq, sort="downloads", limit=args.limit):
                if m.id not in seen:
                    seen[m.id] = m
        except Exception:
            pass

    if seen:
        results = sorted(seen.values(), key=lambda m: m.downloads or 0, reverse=True)
        _emit(f"HuggingFace ({len(results)} results):\n")
        _emit(f"{'#':>3}  {'Model ID':50s}  {'Downloads':>10s}")
        _emit("-" * 70)
        for i, m in enumerate(results, 1):
            dl = str(m.downloads) if m.downloads else "?"
            _emit(f"{i:3d}  {m.id:50s}  {dl:>10s}")

    if not vm_results and not seen:
        _emit("No models found. Try broader terms.")
        return

    _emit("Install: python generate.py models install <hf-repo-id or URL>")


def _extract_archive(archive_path: Path, extract_dir: Path) -> list[Path]:
    """Extract .zip, .rar, or .7z archive."""
    suffix = archive_path.suffix.lower()
    extract_dir.mkdir(parents=True, exist_ok=True)

    if suffix == ".zip":
        import zipfile
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(extract_dir)
    else:
        subprocess.run(["unar", "-o", str(extract_dir), "-f", str(archive_path)],
                       check=True, capture_output=True)

    return list(extract_dir.rglob("*"))


def _upload_model(name: str, pth_path: Path, idx_path: Path | None = None):
    """Pack .pth (+ optional .index) into a .zip and upload to the RVC worker."""
    import zipfile
    zip_path = Path(tempfile.mktemp(suffix=".zip", prefix="generate_"))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pth_path, f"{name}/{name}.pth")
        if idx_path:
            zf.write(idx_path, f"{name}/{name}.index")

    with open(zip_path, "rb") as f:
        api_post("/upload_model", files={"file": (f"{name}.zip", f)})

    zip_path.unlink(missing_ok=True)


def _download_url(url: str, dest_dir: Path) -> Path:
    """Download a file from a URL to dest_dir."""
    import re as _re
    r = requests.get(url, stream=True, timeout=120, allow_redirects=True)
    r.raise_for_status()

    cd = r.headers.get("Content-Disposition", "")
    fname_m = _re.search(r'filename="?([^";\n]+)"?', cd)
    if fname_m:
        fname = fname_m.group(1).strip()
    else:
        from urllib.parse import urlparse, unquote
        fname = unquote(urlparse(url).path.split("/")[-1])
        fname = fname.split("?")[0] or "download"

    dest = dest_dir / fname
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    return dest


def _install_from_archive(archive_path: Path, name: str) -> tuple[Path, Path | None]:
    """Extract archive, find .pth + .index."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="generate_"))
    _emit(f"Extracting {archive_path.name} …", "stage")
    try:
        extracted = _extract_archive(archive_path, tmp_dir)
    except FileNotFoundError:
        if subprocess.run(["which", "brew"], capture_output=True).returncode != 0:
            _emit("'unar' not found and brew is not installed.", "error")
            sys.exit(1)
        _emit("Installing unar via brew …", "stage")
        r = subprocess.run(["brew", "install", "unar"],
                           capture_output=True, text=True,
                           env={**os.environ, "HOMEBREW_NO_AUTO_UPDATE": "1"})
        if r.returncode != 0:
            _emit(f"brew install unar failed: {r.stderr[:200]}", "error")
            sys.exit(1)
        extracted = _extract_archive(archive_path, tmp_dir)
    except subprocess.CalledProcessError as e:
        _emit(f"Extraction failed: {e.stderr.decode()[:200]}", "error")
        sys.exit(1)

    ex_pth = [f for f in extracted if f.suffix == ".pth"]
    ex_idx = [f for f in extracted if f.suffix == ".index"]

    if not ex_pth:
        _emit("No .pth files found in archive", "error")
        sys.exit(1)

    _emit(f"Found: {ex_pth[0].name}")
    return ex_pth[0], ex_idx[0] if ex_idx else None


def _sanitize_model_name(raw: str) -> str:
    """Turn a filename stem into a clean model name."""
    import re as _re
    name = raw.lower().replace(" ", "_").replace("-", "_").strip("_")
    name = _re.sub(r"(_e\d+)?(_s\d+)?$", "", name)
    stripped = _re.sub(r"_?v\d+$", "", name)
    if stripped:
        name = stripped
    stripped = _re.sub(r"_rvc$", "", name)
    if stripped:
        name = stripped
    return name[:40]


def _is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def _hf_repo_from_url(url: str) -> str | None:
    """Extract HuggingFace repo ID from a HF download URL."""
    import re as _re
    m = _re.match(r"https://huggingface\.co/([^/]+/[^/]+)/resolve/", url)
    return m.group(1) if m else None


def _guess_f0_from_name(name: str) -> float | None:
    """Guess target F0 from model name keywords."""
    import re as _re
    name_lower = name.lower()

    def _has_word(keywords):
        for kw in keywords:
            if _re.search(r'(?:^|[\s_\-])' + _re.escape(kw) + r'(?:$|[\s_\-])', name_lower):
                return True
        return False

    if _has_word(["child", "kind", "kid", "girl", "mädchen", "boy", "junge",
                  "pippi", "anime", "loli", "young"]):
        return 280.0
    if _has_word(["female", "frau", "weiblich", "woman",
                  "soprano", "alto", "mezzo"]):
        return 220.0
    if _has_word(["male", "mann", "männlich", "man",
                  "tenor", "bass", "baritone", "bariton"]):
        return 120.0
    return None


def calibrate_model(model_name: str, hf_repo_id: str = None) -> float | None:
    """Calibrate target F0 for a model using name heuristics."""
    target_f0 = _guess_f0_from_name(model_name)
    if target_f0:
        _emit(f"Estimated F0 from model name: {target_f0:.0f} Hz")

    if target_f0 is None and hf_repo_id:
        target_f0 = _guess_f0_from_name(hf_repo_id)
        if target_f0:
            _emit(f"Estimated F0 from repo name: {target_f0:.0f} Hz")

    if target_f0 is None:
        _emit("WARNING: Could not determine target F0.", "warning")
        _emit("Set manually:", "log")
        _emit(f"  python generate.py models set-pitch {model_name} 120   # Male", "log")
        _emit(f"  python generate.py models set-pitch {model_name} 220   # Female", "log")
        _emit(f"  python generate.py models set-pitch {model_name} 280   # Child", "log")
        return None

    config = load_model_config(model_name)
    config["target_f0"] = round(target_f0, 1)
    if hf_repo_id:
        config["hf_repo_id"] = hf_repo_id
    save_model_config(model_name, config)

    _emit(f"Target F0: {target_f0:.1f} Hz (saved)")
    return target_f0


def cmd_models_install(args):
    model_id = args.model_id
    is_url = _is_url(model_id)

    hf_repo_id = None
    if is_url:
        hf_repo_id = _hf_repo_from_url(model_id)

    pth_path = None
    idx_path = None
    tmp_dir = None
    name = None

    if is_url:
        tmp_dir = Path(tempfile.mkdtemp(prefix="generate_"))
        _emit("Downloading from URL …", "stage")
        local_file = _download_url(model_id, tmp_dir)
        _emit(f"Downloaded: {local_file.name}")

        name = args.name or _sanitize_model_name(local_file.stem)

        if local_file.suffix.lower() in (".zip", ".rar", ".7z"):
            pth_path, idx_path = _install_from_archive(local_file, name)
        elif local_file.suffix.lower() == ".pth":
            pth_path = local_file
        else:
            _emit(f"Unexpected file type: {local_file.suffix}", "error")
            _emit("Expected .zip, .rar, .7z, or .pth", "error")
            sys.exit(1)

    else:
        hf_repo_id = model_id
        try:
            from huggingface_hub import HfApi, hf_hub_download
        except ImportError:
            _emit("pip install huggingface-hub", "error")
            sys.exit(1)

        api = HfApi()
        files = api.list_repo_files(model_id)

        pth_files = [f for f in files if f.endswith(".pth")
                     and not any(f.split("/")[-1].startswith(p)
                                 for p in ("D_", "G_", "D-", "G-", "f0"))]
        idx_files = [f for f in files if f.endswith(".index")]
        archive_files = [f for f in files
                         if f.lower().endswith((".zip", ".rar", ".7z"))]

        model_files = archive_files or pth_files

        target_file = getattr(args, "file", None)
        if target_file:
            if target_file not in files:
                matches = [f for f in files if target_file.lower() in f.lower()]
                if len(matches) == 1:
                    target_file = matches[0]
                elif len(matches) > 1:
                    _emit(f"'{target_file}' is ambiguous. Matches:", "error")
                    for m in matches:
                        _emit(f"  - {m}", "error")
                    sys.exit(1)
                else:
                    _emit(f"'{target_file}' not found in {model_id}", "error")
                    sys.exit(1)

            name = args.name or _sanitize_model_name(Path(target_file).stem)
            _emit(f"Installing '{name}' from {target_file} …", "stage")

            if target_file.lower().endswith((".zip", ".rar", ".7z")):
                _emit(f"Downloading {target_file} …", "stage")
                local_archive = Path(hf_hub_download(model_id, target_file))
                pth_path, idx_path = _install_from_archive(local_archive, name)
            elif target_file.endswith(".pth"):
                _emit(f"Downloading {target_file} …", "stage")
                pth_path = Path(hf_hub_download(model_id, target_file))
                base = Path(target_file).stem
                matching_idx = [f for f in idx_files if base in f]
                if matching_idx:
                    _emit(f"Downloading {matching_idx[0]} …", "stage")
                    idx_path = Path(hf_hub_download(model_id, matching_idx[0]))
            else:
                _emit("File must be .pth, .zip, .rar, or .7z", "error")
                sys.exit(1)

        elif len(model_files) > 1:
            total = len(model_files)
            _emit(f"Repo '{model_id}' contains {total} models — installing all …", "stage")
            installed = []
            for j, mf in enumerate(sorted(model_files), 1):
                mf_name = _sanitize_model_name(Path(mf).stem)
                _emit(f"[{j}/{total}] Installing '{mf_name}' from {mf} …", "stage")
                try:
                    if mf.lower().endswith((".zip", ".rar", ".7z")):
                        local_archive = Path(hf_hub_download(model_id, mf))
                        mf_pth, mf_idx = _install_from_archive(local_archive, mf_name)
                    elif mf.endswith(".pth"):
                        mf_pth = Path(hf_hub_download(model_id, mf))
                        base = Path(mf).stem
                        match_idx = [f for f in idx_files if base in f]
                        mf_idx = Path(hf_hub_download(model_id, match_idx[0])) if match_idx else None
                    else:
                        _emit("Skipping (unsupported format)", "warning")
                        continue
                    _upload_model(mf_name, mf_pth, mf_idx)
                    sys.stdout.flush()
                    calibrate_model(mf_name, hf_repo_id=hf_repo_id)
                    sys.stderr.flush()
                    installed.append(mf_name)
                except Exception as e:
                    _emit(f"ERROR: {e} — skipping", "error")
            _emit(f"Installed {len(installed)}/{total} models.")
            return

        elif pth_files:
            pth_file = pth_files[0]
            name = args.name or _sanitize_model_name(Path(pth_file).stem)
            _emit(f"Installing '{name}' …", "stage")
            _emit(f"Downloading {pth_file} …", "stage")
            pth_path = Path(hf_hub_download(model_id, pth_file))
            if idx_files:
                _emit(f"Downloading {idx_files[0]} …", "stage")
                idx_path = Path(hf_hub_download(model_id, idx_files[0]))

        elif archive_files:
            archive = archive_files[0]
            name = args.name or _sanitize_model_name(Path(archive).stem)
            _emit(f"Installing '{name}' …", "stage")
            _emit(f"Downloading {archive} …", "stage")
            local_archive = Path(hf_hub_download(model_id, archive))
            pth_path, idx_path = _install_from_archive(local_archive, name)

        else:
            _emit(f"No .pth or archive files found in {model_id}", "error")
            _emit(f"Files in repo: {', '.join(files[:10])}", "error")
            sys.exit(1)

    _emit("Uploading to RVC worker …", "stage")
    _upload_model(name, pth_path, idx_path)
    _emit(f"Uploaded {name}.zip")

    if tmp_dir:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    _emit(f"Model '{name}' installed.")
    calibrate_model(name, hf_repo_id=hf_repo_id)


def cmd_models_calibrate(args):
    config = load_model_config(args.name)
    hf_repo_id = config.get("hf_repo_id")
    result = calibrate_model(args.name, hf_repo_id=hf_repo_id)
    if result is None:
        sys.exit(1)


def cmd_models_set_f0(args):
    model_name = args.name
    target_f0 = args.hz

    config = load_model_config(model_name)
    config["target_f0"] = target_f0
    save_model_config(model_name, config)

    _emit(f"Target F0 for '{model_name}': {target_f0} Hz")
    _emit(f"Saved to: worker/rvc/models/{model_name}/revoicer.json")


def cmd_models_remove(args):
    engine = getattr(args, "engine", None)
    if engine == "ollama":
        _models_rm_ollama(args)
        return
    if engine == "huggingface":
        _models_rm_huggingface(args)
        return
    # Default: RVC
    _emit(f"Removing model '{args.name}' …", "stage")
    try:
        r = requests.delete(f"{RVC_API_URL}/models/{args.name}", timeout=10)
        if r.ok:
            _emit(f"Removed '{args.name}'")
        else:
            _emit(f"API returned {r.status_code}: {r.text}", "error")
    except Exception as e:
        _emit(f"ERROR: {e}", "error")


def cmd_models_pull(args):
    """Pull model — dispatches to engine-specific pull."""
    engine = args.engine
    if engine == "ollama":
        _models_pull_ollama(args)
    elif engine == "huggingface":
        _models_pull_huggingface(args)
    else:
        _emit(f"ERROR: 'pull' not supported for engine '{engine}'", "error")
        sys.exit(1)


def cmd_models_show(args):
    """Show model details — dispatches to engine-specific show."""
    engine = args.engine
    if engine == "ollama":
        _models_show_ollama(args)
    else:
        _emit(f"ERROR: 'show' not supported for engine '{engine}'", "error")
        sys.exit(1)


def cmd_models_load(args):
    """Load model — dispatches to engine-specific load."""
    _emit(f"ERROR: 'load' not supported for engine '{args.engine}'", "error")
    sys.exit(1)


def cmd_models_unload(args):
    """Unload model — dispatches to engine-specific unload."""
    engine = args.engine
    if engine == "ollama":
        _models_unload_ollama(args)
    else:
        _emit(f"ERROR: 'unload' not supported for engine '{engine}'", "error")
        sys.exit(1)


# ── Models — LLM Engines ────────────────────────────────────────────────────

def _models_engine_conn(engine: str, args):
    """Get base_url + api_key for an LLM engine from args/config."""
    base_url = _llm_engine_base_url(engine, args)
    api_key = _llm_api_key(engine, args)
    return base_url, api_key


def _models_list_ollama(args):
    """List models in Ollama."""
    base_url, api_key = _models_engine_conn("ollama", args)
    data = _llm_get("ollama", base_url, "/api/tags", api_key)
    if data is None:
        _emit("Ollama: (offline)")
        return
    models = data.get("models", [])
    if not models:
        _emit("Ollama: no models installed.")
        return
    _emit(f"Ollama models ({len(models)}):\n")
    for m in models:
        name = m.get("name", "?")
        size_gb = m.get("size", 0) / (1024**3)
        details = m.get("details", {})
        family = details.get("family", "")
        quant = details.get("quantization_level", "")
        extra = f"  {family} {quant}".strip() if (family or quant) else ""
        _emit(f"  {name:<30s} {size_gb:>5.1f} GB{extra}")


_OLLAMA_ENV_UPDATE_STAMP = TEXT_WORKER_DIR / ".last-update-check"
_OLLAMA_MODEL_UPDATE_STAMP = TEXT_WORKER_DIR / ".last-model-update"


def _ollama_maybe_update_env():
    """Once per day: pip upgrade ollama in the text env."""
    if _OLLAMA_ENV_UPDATE_STAMP.exists():
        age = time.time() - _OLLAMA_ENV_UPDATE_STAMP.stat().st_mtime
        if age < 86400:
            return
    _emit("Checking for ollama package updates …")
    result = subprocess.run(
        [str(CONDA_BIN), "run", "-n", TEXT_ENV,
         "pip", "install", "--upgrade", "ollama"],
        capture_output=True, text=True, timeout=120,
    )
    out = result.stdout.strip()
    if out and "Successfully installed" in out:
        msg = out.split("\n")[-1]
        _emit(msg)
    else:
        _emit("Environment up to date.")
    _OLLAMA_ENV_UPDATE_STAMP.touch()


def _ollama_maybe_update_models():
    """Once per day: pull all local Ollama models to check for updates + ensure num_ctx."""
    if _OLLAMA_MODEL_UPDATE_STAMP.exists():
        age = time.time() - _OLLAMA_MODEL_UPDATE_STAMP.stat().st_mtime
        if age < 86400:
            return

    base_url = "http://localhost:11434"
    if TEXT_ENGINES_FILE.exists():
        cfg = json.loads(TEXT_ENGINES_FILE.read_text())
        base_url = cfg.get("ollama", {}).get("base_url", base_url)

    # Get list of local models
    data = _llm_get("ollama", base_url, "/api/tags", None)
    if data is None:
        return  # Ollama not running

    models = data.get("models", [])
    if not models:
        _OLLAMA_MODEL_UPDATE_STAMP.touch()
        return

    _emit("Checking Ollama models for updates …", "stage")
    for m in models:
        name = m.get("name", m.get("model", ""))
        if not name:
            continue
        # Pull (no-op if already up to date)
        result = subprocess.run(["ollama", "pull", name],
                                capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            # Ensure num_ctx is set to max
            _ollama_set_max_context(base_url, None, name)

    _OLLAMA_MODEL_UPDATE_STAMP.touch()
    _emit("Model update check complete.")


def _ollama_set_max_context(base_url: str, api_key: str | None, model: str):
    """Set num_ctx to the model's maximum supported context length.

    Queries /api/show for the model's architecture context_length,
    then applies it via `ollama create` with a Modelfile.
    """
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Get model info
    try:
        resp = requests.post(f"{base_url}/api/show", json={"name": model},
                             headers=headers, timeout=30)
        resp.raise_for_status()
        info = resp.json()
    except Exception:
        return  # non-fatal — model works fine with default ctx

    # Find max context: model_info has keys like "<arch>.context_length"
    model_info = info.get("model_info", {})
    max_ctx = None
    for key, val in model_info.items():
        if key.endswith(".context_length") and isinstance(val, (int, float)):
            max_ctx = int(val)
            break

    if not max_ctx:
        return

    # Check current Modelfile — skip if num_ctx already set
    modelfile = info.get("modelfile", "")
    if "num_ctx" in modelfile.lower():
        _emit(f"  num_ctx already set ({max_ctx:,d}), skipping.")
        return

    # Apply via ollama CLI (API /api/create changed in 0.18+)
    import tempfile
    modelfile_content = f"FROM {model}\nPARAMETER num_ctx {max_ctx}\n"
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".modelfile",
                                         delete=False) as f:
            f.write(modelfile_content)
            tmp_path = f.name
        result = subprocess.run(["ollama", "create", model, "-f", tmp_path],
                                capture_output=True, text=True, timeout=60)
        os.unlink(tmp_path)
        if result.returncode == 0:
            _emit(f"  num_ctx set to {max_ctx:,d} (model maximum).")
        else:
            _emit(f"  WARNING: Could not set num_ctx: {result.stderr.strip()}", "log")
    except Exception:
        _emit(f"  WARNING: Could not set num_ctx to {max_ctx}.", "log")


def _models_pull_ollama(args):
    """Pull model from Ollama registry."""
    base_url, api_key = _models_engine_conn("ollama", args)
    model = args.model_id
    _emit(f"Pulling {model} via Ollama …", "stage")
    body = {"name": model, "stream": True}
    try:
        resp = requests.post(f"{base_url}/api/pull", json=body,
                             headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
                             stream=True, timeout=600)
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line.decode("utf-8"))
            status = chunk.get("status", "")
            total = chunk.get("total", 0)
            completed = chunk.get("completed", 0)
            if total:
                pct = completed / total * 100
                print(f"\r  {status}: {pct:.0f}%", end="", flush=True)
            else:
                print(f"\r  {status}", end="", flush=True)
        print()
    except requests.ConnectionError:
        _emit("ERROR: Could not connect to Ollama", "error")
        _emit("  Is Ollama running? Start with: ollama serve")
        sys.exit(1)
    except requests.HTTPError as e:
        _emit(f"ERROR: Ollama returned {resp.status_code}: {resp.text}", "error")
        sys.exit(1)

    # Create initial config backup
    cfg_dir = _llm_model_config_dir("ollama", model)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_file = cfg_dir / "config.json"
    if not cfg_file.exists():
        cfg_file.write_text("{}\n")
        ts = time.strftime("%Y-%m-%d_%H-%M-%S")
        backup = cfg_dir / f"config.{ts}.json"
        backup.write_text("{}\n")

    _emit(f"Model {model} pulled successfully.")

    # Set num_ctx to maximum supported context length
    _ollama_set_max_context(base_url, api_key, model)


def _models_rm_ollama(args):
    """Remove model from Ollama."""
    base_url, api_key = _models_engine_conn("ollama", args)
    model = args.name
    _emit(f"Removing {model} from Ollama …", "stage")
    _llm_delete("ollama", base_url, "/api/delete",
                {"name": model}, api_key)
    # Remove local config
    cfg_dir = _llm_model_config_dir("ollama", model)
    if cfg_dir.exists():
        import shutil
        shutil.rmtree(cfg_dir)
    _emit(f"Model {model} removed.")


def _models_unload_ollama(args):
    """Unload model from Ollama VRAM."""
    base_url, api_key = _models_engine_conn("ollama", args)
    model = args.name
    body = {"model": model, "prompt": "", "stream": False, "keep_alive": 0}
    _llm_request("ollama", base_url, "/api/generate", body, api_key)
    _emit(f"Model {model} unloaded from Ollama.")


def _models_show_ollama(args):
    """Show model details from Ollama (modelfile, parameters, context length)."""
    base_url, api_key = _models_engine_conn("ollama", args)
    model = args.name
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        resp = requests.post(f"{base_url}/api/show", json={"name": model},
                             headers=headers, timeout=30)
        resp.raise_for_status()
    except requests.ConnectionError:
        _emit("ERROR: Could not connect to Ollama", "error")
        sys.exit(1)
    except requests.HTTPError:
        _emit(f"ERROR: Model '{model}' not found", "error")
        sys.exit(1)

    info = resp.json()

    # JSON mode: output everything
    if _event_handler is print_event_json:
        print(json.dumps(info, ensure_ascii=False))
        return

    # TUI mode
    details = info.get("details", {})
    model_info = info.get("model_info", {})
    parameters = info.get("parameters", "")

    _emit(f"Model: {model}")
    if details.get("family"):
        _emit(f"  Family:         {details['family']}")
    if details.get("parameter_size"):
        _emit(f"  Parameters:     {details['parameter_size']}")
    if details.get("quantization_level"):
        _emit(f"  Quantization:   {details['quantization_level']}")
    if details.get("format"):
        _emit(f"  Format:         {details['format']}")

    # Context length from model_info
    for key, val in model_info.items():
        if key.endswith(".context_length"):
            _emit(f"  Max context:    {int(val):,d}")
            break

    # Active parameters from Modelfile
    if parameters:
        _emit(f"\n  Parameters:")
        for line in parameters.strip().split("\n"):
            _emit(f"    {line.strip()}")


def _models_list_huggingface(args):
    """List locally cached HuggingFace models."""
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if not hf_cache.exists():
        _emit("HuggingFace: no models cached.")
        return
    model_dirs = sorted(d for d in hf_cache.iterdir()
                        if d.is_dir() and d.name.startswith("models--"))
    if not model_dirs:
        _emit("HuggingFace: no models cached.")
        return
    _emit(f"HuggingFace models ({len(model_dirs)}):\n")
    for d in model_dirs:
        # models--org--name → org/name
        name = d.name.replace("models--", "").replace("--", "/", 1)
        # Get size of snapshots
        snapshots = d / "snapshots"
        size_bytes = sum(f.stat().st_size for f in snapshots.rglob("*")
                         if f.is_file()) if snapshots.exists() else 0
        size_gb = size_bytes / (1024**3)
        _emit(f"  {name:<40s} {size_gb:>5.1f} GB")


def _models_pull_huggingface(args):
    """Download model from HuggingFace Hub."""
    model = args.model_id
    _emit(f"Pulling {model} from HuggingFace …", "stage")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(model)
    except ImportError:
        _emit("ERROR: huggingface_hub not installed", "error")
        _emit("  pip install huggingface_hub")
        sys.exit(1)
    except Exception as e:
        _emit(f"ERROR: {e}", "error")
        sys.exit(1)

    # Create initial config backup
    cfg_dir = _llm_model_config_dir("huggingface", model)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_file = cfg_dir / "config.json"
    if not cfg_file.exists():
        cfg_file.write_text("{}\n")
        ts = time.strftime("%Y-%m-%d_%H-%M-%S")
        backup = cfg_dir / f"config.{ts}.json"
        backup.write_text("{}\n")

    _emit(f"Model {model} downloaded.")


def _models_rm_huggingface(args):
    """Remove locally cached HuggingFace model."""
    model = args.name
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    safe = "models--" + model.replace("/", "--")
    model_dir = hf_cache / safe
    if not model_dir.exists():
        _emit(f"Model {model} not found in HuggingFace cache.", "error")
        sys.exit(1)
    import shutil
    shutil.rmtree(model_dir)
    # Remove local config
    cfg_dir = _llm_model_config_dir("huggingface", model)
    if cfg_dir.exists():
        shutil.rmtree(cfg_dir)
    _emit(f"Model {model} removed from cache.")


def _models_search_huggingface(args):
    """Search HuggingFace Hub."""
    query = args.query
    _emit(f'Searching HuggingFace for: "{query}" …', "stage")
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        results = list(api.list_models(search=query, limit=args.limit,
                                       sort="downloads", direction=-1))
    except ImportError:
        _emit("ERROR: huggingface_hub not installed", "error")
        _emit("  pip install huggingface_hub")
        sys.exit(1)
    except Exception as e:
        _emit(f"ERROR: {e}", "error")
        sys.exit(1)

    if not results:
        _emit("No models found.")
        return

    _emit(f"HuggingFace ({len(results)} results):\n")
    for m in results:
        dl = m.downloads if hasattr(m, "downloads") else 0
        dl_str = f"{dl:>8,d} DL" if dl else ""
        _emit(f"  {m.modelId:<50s} {dl_str}")


# Engine → allowed subcommands
_MODELS_ENGINE_CMDS = {
    "rvc": {"list", "search", "install", "remove", "calibrate", "set-pitch"},
    "ollama": {"list", "pull", "remove", "unload", "show"},
    "huggingface": {"list", "pull", "remove", "search"},
}


# ── Voice Engines ────────────────────────────────────────────────────────────

def _tts_rvc(args):
    """Voice conversion via RVC."""
    input_paths = [Path(p) for p in args.input]

    for p in input_paths:
        if not p.exists():
            _emit(f"ERROR: File not found: {p}", "error")
            sys.exit(1)

    output_dir = Path(args.output) if args.output else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Set voice model
    model_name = args.voice
    if model_name:
        api_post("/models/" + model_name)

    # Determine target F0 for auto-pitch
    target_f0 = None
    manual_pitch = args.pitch is not None

    if not manual_pitch:
        if args.target_hz is not None:
            target_f0 = args.target_hz
        elif model_name:
            config = load_model_config(model_name)
            target_f0 = config.get("target_f0")
            if target_f0 is None:
                _emit("No target F0 for this model — calibrating automatically …", "stage")
                target_f0 = calibrate_model(model_name)
                if target_f0 is None:
                    _emit("  Calibration failed. Using pitch=0.", "warning")

    base_params = {"index_rate": 0.0}
    if args.decoder is not None:
        base_params["f0method"] = args.decoder
    if manual_pitch:
        base_params["f0up_key"] = args.pitch

    output_paths = []
    total = len(input_paths)

    for i, input_path in enumerate(input_paths, 1):
        if output_dir:
            out_path = output_dir / input_path.with_suffix(".wav").name
        else:
            out_path = input_path.with_stem(input_path.stem + "_converted").with_suffix(".wav")

        # Convert non-WAV to WAV via ffmpeg
        tmp_wav = None
        send_path = input_path
        if input_path.suffix.lower() != ".wav":
            tmp_wav = Path(tempfile.mktemp(suffix=".wav", prefix="generate_"))
            r_ff = run_worker(
                ["ffmpeg", "-y", "-i", str(input_path), "-ar", "44100", str(tmp_wav)])
            if r_ff.returncode != 0:
                _emit(f"ERROR: ffmpeg conversion failed: {r_ff.stderr_tail[:200]}", "error")
                sys.exit(1)
            send_path = tmp_wav

        # Auto-pitch: detect input F0 and compute shift per file
        params = dict(base_params)
        if not manual_pitch and target_f0 is not None:
            input_f0 = detect_input_f0(send_path)
            if input_f0 and input_f0 > 0:
                shift = compute_pitch_shift(input_f0, target_f0)
                params["f0up_key"] = shift
                _emit(f"[{i}/{total}] {input_path.name} → {out_path.name}  "
                      f"(F0: {input_f0:.0f} Hz → {target_f0:.0f} Hz, shift: {shift:+d})")
            else:
                params["f0up_key"] = 0
                _emit(f"[{i}/{total}] {input_path.name} → {out_path.name}  "
                      f"(F0 detection failed, pitch=0)")
        else:
            pitch_info = f"pitch={args.pitch}" if manual_pitch else "pitch=0"
            _emit(f"[{i}/{total}] {input_path.name} → {out_path.name}  ({pitch_info})")

        api_post("/params", json={"params": params})

        with open(send_path, "rb") as f:
            r = api_post("/convert_file", files={"file": (send_path.name, f, "audio/wav")})

        if tmp_wav:
            tmp_wav.unlink(missing_ok=True)

        out_path.write_bytes(r.content)
        output_paths.append(str(out_path))

    print(json.dumps(output_paths))


def _voice_say(args):
    """Text-to-speech via macOS say command, optionally piped through RVC."""
    text = _require_text(args, "text")

    output_dir = Path(args.output) if args.output else Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build filename from first few words
    slug = "_".join(text.split()[:5]).lower()
    slug = "".join(c for c in slug if c.isalnum() or c == "_")[:40]
    voice_tag = args.say_voice or "default"
    out_name = f"say_{voice_tag}_{slug}"
    wav_path = output_dir / f"{out_name}.wav"

    # Build say command — always 44100 Hz WAV
    cmd = ["say", "--file-format=WAVE", "--data-format=LEI16@44100",
           "-o", str(wav_path)]
    if args.say_voice:
        cmd += ["-v", args.say_voice]
    if args.rate:
        cmd += ["-r", str(args.rate)]
    cmd.append(text)

    _emit(f"say → {wav_path.name}" +
          (f"  (voice: {args.say_voice})" if args.say_voice else "  (system voice)"))

    r = run_worker(cmd)
    if r.returncode != 0:
        _emit(f"ERROR: say failed: {r.stderr_tail[:200]}", "error")
        sys.exit(1)

    # Optional RVC post-processing
    rvc_model = args.voice
    if rvc_model:
        _emit(f"RVC → {rvc_model}")
        # Reuse _tts_rvc by injecting the say output as input
        import argparse as _ap
        rvc_args = _ap.Namespace(**vars(args))
        rvc_args.input = [str(wav_path)]
        rvc_args.engine = "rvc"
        # Output to same dir with model name suffix
        rvc_out = output_dir / "rvc"
        rvc_out.mkdir(parents=True, exist_ok=True)
        rvc_args.output = str(rvc_out)
        _tts_rvc(rvc_args)
        # Clean up intermediate say wav
        wav_path.unlink(missing_ok=True)
    else:
        print(json.dumps([str(wav_path)]))


_AI_TTS_MODEL_MAP = {
    "large": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
    "small": "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit",
}

_AI_TTS_VOICES = [
    "Aiden", "Dylan", "Eric", "Ryan", "Uncle_Fu",
    "Vivian", "Serena", "Ono_Anna", "Sohee",
]


def _detect_language(text: str) -> str:
    """Detect language of text via worker/langdetect. Returns ISO code (de, en, ...)."""
    if not LANGDETECT_WORKER.exists():
        _emit("WARNING: worker/langdetect not found, defaulting to en", "warning")
        return "en"

    result = run_worker([
        str(CONDA_BIN), "run", "--no-capture-output", "-n", LANGDETECT_ENV,
        "python", str(LANGDETECT_WORKER),
        "--text", text[:500],  # first 500 chars is enough
    ])
    if result.returncode != 0:
        _emit("WARNING: Language detection failed, defaulting to en", "warning")
        return "en"
    return result.stdout.strip() or "en"


def _parse_prompt_sidecar(path: Path) -> dict:
    """Parse a prompt sidecar .txt file into {voice, language, model, tags, text}."""
    raw = path.read_text(encoding="utf-8")
    if "\n---\n" not in raw:
        # No header — treat entire file as text
        return {"voice": None, "language": None, "model": None, "tags": None, "text": raw.strip()}

    header, body = raw.split("\n---\n", 1)
    meta: dict[str, str | None] = {"voice": None, "language": None, "model": None, "tags": None}
    for line in header.splitlines():
        if ": " in line:
            key, val = line.split(": ", 1)
            key = key.strip().lower()
            val = val.strip()
            if key in meta and val and val != "default" and val != "auto":
                meta[key] = val

    # Reverse-map full model ID → "large"/"small"
    if meta["model"]:
        _reverse = {v: k for k, v in _AI_TTS_MODEL_MAP.items()}
        meta["model"] = _reverse.get(meta["model"], None)

    meta["text"] = body.strip()
    return meta


def _voice_ai_tts(args):
    """Text-to-speech via Qwen3-TTS (mlx-audio)."""
    # --list-voices
    if getattr(args, "list_voices", False):
        _emit("Available AI-TTS voices:\n")
        _emit("  Male:   Aiden, Dylan, Eric, Ryan, Uncle_Fu")
        _emit("  Female: Vivian, Serena, Ono_Anna, Sohee")
        return

    text = _require_text(args, "text")

    # Voice
    voice = args.say_voice  # -v flag

    # Language: pass through to worker (worker autodetects if empty)
    language = getattr(args, "language", None) or ""

    # Model
    tts_model = getattr(args, "tts_model", None) or "large"
    model_id = _AI_TTS_MODEL_MAP.get(tts_model, _AI_TTS_MODEL_MAP["large"])

    # Output — file path or directory
    out = Path(args.output) if args.output else Path(".")
    if out.suffix in (".wav", ".WAV"):
        wav_path = out
        wav_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out.mkdir(parents=True, exist_ok=True)
        slug = "_".join(text.split()[:5]).lower()
        slug = "".join(c for c in slug if c.isalnum() or c == "_")[:40]
        voice_tag = voice or "default"
        wav_path = out / f"ai_tts_{voice_tag}_{slug}.wav"

    # Worker script
    generate_script = TTS_WORKER_DIR / "generate_speech.py"
    if not generate_script.exists():
        _emit("ERROR: worker/tts/generate_speech.py not found", "error")
        _emit("  Run: bash worker/tts/install.sh", "error")
        sys.exit(1)

    cmd = [
        str(CONDA_BIN), "run", "--no-capture-output", "-n", TTS_ENV,
        "python", str(generate_script),
        "--text", text,
        "--model", model_id,
        "--output", str(wav_path),
    ]
    if language:
        cmd.extend(["--language", language])
    if voice:
        cmd.extend(["--voice", voice])
    tags = getattr(args, "tags", None)
    if tags:
        cmd.extend(["--instruct", tags])

    _emit(f"Generating speech (Qwen3-TTS {tts_model}) …", "stage")
    if voice:
        _emit(f"  Voice:    {voice}")
    if language:
        _emit(f"  Language: {language}")
    else:
        _emit(f"  Language: auto")
    _emit(f"  Output:   {wav_path}")

    result = run_worker(cmd, on_event=_event_handler)
    finish_progress()
    if result.returncode != 0:
        _emit("ERROR: AI-TTS generation failed.", "error")
        sys.exit(1)
    # Print JSON result (filter out non-JSON noise from stdout)
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if line.startswith("{") or line.startswith("["):
            print(line)

    # Save prompt sidecar (.txt with all generation parameters)
    prompt_path = wav_path.with_suffix(".txt")
    prompt_lines = [
        f"voice: {voice or 'default'}",
        f"language: {language or 'auto'}",
        f"model: {model_id}",
    ]
    if tags:
        prompt_lines.append(f"tags: {tags}")
    prompt_lines.append(f"---")
    prompt_lines.append(text)
    prompt_path.write_text("\n".join(prompt_lines), encoding="utf-8")
    _emit(f"  Prompt:   {prompt_path}", "info")


def _voice_clone_tts(args):
    """Zero-shot voice cloning via Qwen3-TTS Base."""
    text = _require_text(args, "text")

    # Strip [instructions] — Base model doesn't support inline tags
    import re as _re
    text = _re.sub(r"\[.*?\]", "", text, flags=_re.DOTALL).strip()

    # Reference audio — --reference flag(s), fallback to default
    DEFAULT_REF = SCRIPT_DIR / "worker" / "tts" / "default-reference.m4a"
    ref_flags = getattr(args, "reference", None) or []
    ref_paths = [Path(p) for p in ref_flags]
    if not ref_paths:
        if DEFAULT_REF.exists():
            ref_paths = [DEFAULT_REF]
            _emit(f"  Using default reference: {DEFAULT_REF.name}")
        else:
            _emit("ERROR: No --reference given and no default reference found", "error")
            _emit(f"  Place a reference audio at: {DEFAULT_REF}", "error")
            sys.exit(1)
    for rp in ref_paths:
        if not rp.exists():
            _emit(f"ERROR: Reference audio not found: {rp}", "error")
            sys.exit(1)
    # For now: use first reference (multi-reference is future work)
    ref_path = ref_paths[0]

    # Transcribe reference audio with Whisper → get ref_text + trim to sentence boundary
    ref_text = getattr(args, "ref_text", None) or ""
    if not ref_text:
        transcribe_script = WHISPER_WORKER_DIR / "transcribe.py"
        if transcribe_script.exists():
            _emit("Transcribing reference audio …", "stage")
            whisper_out = Path(tempfile.mkdtemp(prefix="whisper_clone_"))
            whisper_cmd = [
                str(CONDA_BIN), "run", "--no-capture-output", "-n", WHISPER_ENV,
                "python", str(transcribe_script),
                str(ref_path), "--model", "large-v3-turbo",
                "--format", "json",
                "--output", str(whisper_out),
            ]
            whisper_result = run_worker(whisper_cmd, on_event=_event_handler)
            finish_progress()
            whisper_json_path = whisper_out / f"{ref_path.stem}.json"
            if whisper_result.returncode == 0 and whisper_json_path.exists():
                try:
                    whisper_data = json.loads(whisper_json_path.read_text(encoding="utf-8"))
                    segments = whisper_data.get("segments", [])

                    # Find last segment that ends within MAX_REF_SECONDS
                    MAX_REF_SECONDS = 5.0
                    usable_segments = [s for s in segments if s["end"] <= MAX_REF_SECONDS]

                    if usable_segments:
                        # Use all segments up to the cutoff
                        cut_time = usable_segments[-1]["end"]
                        ref_text = " ".join(s["text"].strip() for s in usable_segments).strip()
                    elif segments:
                        # First segment exceeds 5s — use it anyway but trim at its end
                        cut_time = segments[0]["end"]
                        ref_text = segments[0]["text"].strip()
                    else:
                        cut_time = None
                        ref_text = whisper_data.get("text", "").strip()

                    # Trim reference audio at sentence boundary
                    if cut_time is not None and cut_time > 0:
                        trimmed_ref = Path(tempfile.mktemp(suffix="_ref_trimmed.wav"))
                        trim_cmd = [
                            "ffmpeg", "-y", "-i", str(ref_path),
                            "-ac", "1", "-ar", "24000", "-sample_fmt", "s16",
                            "-t", f"{cut_time:.2f}",
                            str(trimmed_ref),
                        ]
                        trim_r = subprocess.run(trim_cmd, capture_output=True, text=True)
                        if trim_r.returncode == 0:
                            ref_path = trimmed_ref

                    if ref_text:
                        _emit(f"  Reference text: {ref_text[:80]}{'…' if len(ref_text) > 80 else ''}")
                except (json.JSONDecodeError, KeyError, IndexError):
                    pass
        if not ref_text:
            _emit("  WARNING: Could not transcribe reference — quality may suffer", "warning")

    # Output — file path or directory
    out = Path(args.output) if args.output else Path(".")
    if out.suffix in (".wav", ".WAV"):
        wav_path = out
        wav_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out.mkdir(parents=True, exist_ok=True)
        slug = "_".join(text.split()[:5]).lower()
        slug = "".join(c for c in slug if c.isalnum() or c == "_")[:40]
        wav_path = out / f"clone_tts_{slug}.wav"

    # Voice cloning via Qwen3-TTS Base model (ref_audio + ref_text → ICL)
    tts_script = TTS_WORKER_DIR / "generate_speech.py"
    if not tts_script.exists():
        _emit("ERROR: ai-tts worker not found — needed for voice cloning", "error")
        _emit("  Run: bash worker/tts/install.sh", "error")
        sys.exit(1)

    language = getattr(args, "language", None)

    _emit(f"Cloning voice → {wav_path.name}", "stage")
    _emit(f"  Reference: {ref_path.name}")
    _emit(f"  Output:    {wav_path}")

    cmd = [
        str(CONDA_BIN), "run", "--no-capture-output", "-n", TTS_ENV,
        "python", str(tts_script),
        "--text", text,
        "--model", "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
        "--ref-audio", str(ref_path),
        "--output", str(wav_path),
    ]
    if ref_text:
        cmd.extend(["--ref-text", ref_text])
    if language:
        cmd.extend(["--language", language])

    result = run_worker(cmd, on_event=_event_handler)
    finish_progress()

    if result.returncode != 0:
        _emit("ERROR: Voice cloning failed.", "error")
        sys.exit(1)

    print(json.dumps([str(wav_path)]))


def cmd_voice(args):
    """Voice conversion — dispatch by engine."""
    engine = args.engine
    if engine == "rvc":
        _tts_rvc(args)
    elif engine == "say":
        _voice_say(args)
    elif engine == "ai-tts":
        _voice_ai_tts(args)
    elif engine == "clone-tts":
        _voice_clone_tts(args)
    else:
        _emit(f"ERROR: Unknown voice engine: {engine}", "error")
        sys.exit(1)


# ── Audio Engines ────────────────────────────────────────────────────────────

def _audio_enhance(args):
    """Enhance audio (denoise + super-resolution)."""
    input_paths = [Path(p) for p in args.input]

    for p in input_paths:
        if not p.exists():
            _emit(f"ERROR: File not found: {p}", "error")
            sys.exit(1)

    output_dir = Path(args.output) if args.output else input_paths[0].parent
    output_dir.mkdir(parents=True, exist_ok=True)

    enhance_script = ENHANCE_WORKER_DIR / "enhance.py"
    if not enhance_script.exists():
        _emit("ERROR: worker/enhance/enhance.py not found", "error")
        _emit("  Run: bash worker/enhance/install.sh", "error")
        sys.exit(1)

    cmd = [
        str(CONDA_BIN), "run", "--no-capture-output", "-n", ENHANCE_ENV,
        "python", str(enhance_script),
    ]
    for p in input_paths:
        cmd.append(str(p))
    cmd.extend(["-o", str(output_dir)])

    if args.denoise_only:
        cmd.append("--denoise-only")
    if args.enhance_only:
        cmd.append("--enhance-only")

    mode = "denoise" if args.denoise_only else "enhance-only" if args.enhance_only else "enhance"
    _emit(f"Enhancing {len(input_paths)} file(s) ({mode}) …", "stage")

    result = run_worker(cmd, on_event=_event_handler)
    finish_progress()
    if result.returncode != 0:
        _emit("ERROR: Enhancement failed.", "error")
        sys.exit(1)
    if result.stdout.strip():
        print(result.stdout.strip())


def _audio_demucs(args):
    """Separate audio into stems (vocals, drums, bass, other)."""
    input_paths = [Path(p) for p in args.input]

    for p in input_paths:
        if not p.exists():
            _emit(f"ERROR: File not found: {p}", "error")
            sys.exit(1)

    output_dir = Path(args.output) if args.output else input_paths[0].parent
    output_dir.mkdir(parents=True, exist_ok=True)

    separate_script = SEPARATE_WORKER_DIR / "separate.py"
    if not separate_script.exists():
        _emit("ERROR: worker/separate/separate.py not found", "error")
        _emit("  Run: bash worker/separate/install.sh", "error")
        sys.exit(1)

    for p in input_paths:
        cmd = [
            str(CONDA_BIN), "run", "--no-capture-output", "-n", SEPARATE_ENV,
            "python", str(separate_script),
            str(p),
            "-o", str(output_dir),
        ]
        if args.model:
            cmd.extend(["--model", args.model])

        _emit(f"Separating {p.name} …", "stage")

        result = run_worker(cmd, on_event=_event_handler)
        finish_progress()
        if result.returncode != 0:
            _emit("ERROR: Separation failed.", "error")
            sys.exit(1)
        if result.stdout.strip():
            print(result.stdout.strip())


_ACE_MODEL_MAP = {
    "turbo": "acestep-v15-turbo",
    "sft": "acestep-v15-sft",
    "base": "acestep-v15-base",
}


def split_segments(duration_s: float) -> list:
    """Split song into equal segments for repainting."""
    if duration_s < 60:
        return []

    n = max(2, round(duration_s / 36))
    seg_len = duration_s / n
    segments = [(i * seg_len, (i + 1) * seg_len) for i in range(n)]

    _emit(f"  Duration: {duration_s:.1f}s → {n} segments à {seg_len:.1f}s")
    for i, (s, e) in enumerate(segments):
        _emit(f"    seg {i+1}: {s:.1f}s - {e:.1f}s")

    return segments


def _require_text(args, label="lyrics"):
    """Get text from args.text (already normalized from --text/--lyrics/--*-file)."""
    text = getattr(args, "text", None) or ""
    if not text.strip():
        _emit(f"ERROR: Provide --{label} (-l) or --{label}-file (-f)", "error")
        sys.exit(1)
    return text


def _audio_ace(args):
    """Generate music using ACE-Step 1.5."""

    lyrics = _require_text(args, "lyrics")

    # Output path
    if args.output:
        out_path = Path(args.output)
        if out_path.is_dir() or str(out_path).endswith("/"):
            out_path = Path(args.output) / f"music_{int(time.time())}.mp3"
    else:
        out_path = Path(f"music_{int(time.time())}.mp3")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not ACESTEP_WORKER.exists():
        _emit("ERROR: worker/ace/generate.py not found", "error")
        _emit("  Run: bash worker/ace/install.sh", "error")
        sys.exit(1)

    if not ACESTEP_DIR.exists():
        _emit("ERROR: ACE-Step-1.5 not found", "error")
        _emit(f"  Expected: {ACESTEP_DIR}", "error")
        sys.exit(1)

    # Map --model to ACE config
    ace_model = getattr(args, "model", None) or "turbo"
    ace_config = _ACE_MODEL_MAP.get(ace_model, "acestep-v15-turbo")

    def _ace_base_cmd(lyrics_override=None, task=None, src_audio=None,
                      repaint_start=None, repaint_end=None, output_override=None):
        cmd = [
            str(UV_BIN), "run", "--project", str(ACESTEP_DIR),
            "python", str(ACESTEP_WORKER),
            "--lyrics", lyrics_override or lyrics,
            "--tags", args.tags,
            "-o", output_override or str(out_path),
        ]
        if task:
            cmd.extend(["--task", task])
        if src_audio:
            cmd.extend(["--src-audio", src_audio])
        if repaint_start is not None:
            cmd.extend(["--repaint-start", str(repaint_start)])
        if repaint_end is not None:
            cmd.extend(["--repaint-end", str(repaint_end)])
        if args.cfg_scale is not None:
            cmd.extend(["--guidance-scale", str(args.cfg_scale)])
        if args.temperature is not None:
            cmd.extend(["--lm-temperature", str(args.temperature)])
        if args.top_k is not None:
            cmd.extend(["--lm-top-k", str(args.top_k)])
        if args.seed is not None:
            cmd.extend(["--seed", str(args.seed)])
        if getattr(args, "steps", None) is not None:
            cmd.extend(["--steps", str(args.steps)])
        if getattr(args, "shift", None) is not None:
            cmd.extend(["--shift", str(args.shift)])
        if getattr(args, "no_thinking", False):
            cmd.append("--no-thinking")
        if getattr(args, "infer_method", None) is not None:
            cmd.extend(["--infer-method", str(args.infer_method)])
        if getattr(args, "lm_cfg", None) is not None:
            cmd.extend(["--lm-cfg", str(args.lm_cfg)])
        if getattr(args, "top_p", None) is not None:
            cmd.extend(["--top-p", str(args.top_p)])
        if getattr(args, "batch_size", None) is not None:
            cmd.extend(["--batch-size", str(args.batch_size)])
        if getattr(args, "instrumental", False):
            cmd.append("--instrumental")
        if getattr(args, "language", None):
            cmd.extend(["--language", args.language])
        cmd.extend(["--config-path", ace_config])
        if getattr(args, "bpm", None) is not None:
            cmd.extend(["--bpm", str(args.bpm)])
        if getattr(args, "keyscale", None):
            cmd.extend(["--keyscale", args.keyscale])
        if getattr(args, "timesignature", None):
            cmd.extend(["--timesignature", args.timesignature])
        return cmd

    def _run_ace(cmd, label="Generation"):
        result = run_worker(cmd, on_event=_event_handler)
        finish_progress()
        if result.returncode != 0:
            _emit(f"ERROR: {label} failed (ACE-Step).", "error")
            return False
        return True

    duration_ms = args.duration if args.duration else args.seconds * 1000
    duration_s = duration_ms / 1000

    _emit("Generating music (ACE-Step) …", "stage")
    _emit(f"  Model:    {ace_model} ({ace_config})")
    _emit(f"  Caption:  {args.tags}")
    _emit(f"  Duration: {duration_s:.0f}s")
    _emit(f"  Output:   {out_path}")

    cmd = _ace_base_cmd()
    cmd.extend(["--duration", str(duration_ms)])
    if not _run_ace(cmd, "Music generation"):
        sys.exit(1)

    if out_path.exists():
        print(json.dumps([str(out_path)]))


def _audio_heartmula(args):
    """Generate music using HeartMuLa."""

    lyrics = _require_text(args, "lyrics")

    # Output path
    if args.output:
        out_path = Path(args.output)
        if out_path.is_dir() or str(out_path).endswith("/"):
            out_path = Path(args.output) / f"music_{int(time.time())}.mp3"
    else:
        out_path = Path(f"music_{int(time.time())}.mp3")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    generate_script = MUSIC_WORKER_DIR / "generate.py"
    if not generate_script.exists():
        _emit("ERROR: worker/music/generate.py not found", "error")
        _emit("  Run: bash worker/music/install.sh", "error")
        sys.exit(1)

    ckpt_dir = MUSIC_MODELS_DIR / "ckpt"
    if not ckpt_dir.exists():
        _emit("ERROR: HeartMuLa checkpoints not found", "error")
        _emit(f"  Expected: {ckpt_dir}", "error")
        sys.exit(1)

    # Append bpm/keyscale to caption (HeartMuLa has no native flags)
    tags = args.tags
    if getattr(args, "bpm", None) is not None:
        tags += f", bpm: {args.bpm}"
    if getattr(args, "keyscale", None):
        tags += f", keyscale: {args.keyscale}"
    if getattr(args, "timesignature", None):
        tags += f", timesignature: {args.timesignature}"

    tags = ",".join(t.strip() for t in tags.split(","))

    cmd = [
        str(CONDA_BIN), "run", "--no-capture-output", "-n", HEARTMULA_ENV,
        "python", str(generate_script),
        "--lyrics", lyrics,
        "--tags", tags,
        "-o", str(out_path),
        "--ckpt-dir", str(ckpt_dir),
    ]

    duration_ms = args.duration if args.duration else args.seconds * 1000
    cmd.extend(["--duration", str(duration_ms)])

    if args.top_k:
        cmd.extend(["--topk", str(args.top_k)])
    if args.temperature:
        cmd.extend(["--temperature", str(args.temperature)])
    if args.cfg_scale:
        cmd.extend(["--cfg-scale", str(args.cfg_scale)])
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])

    duration_s = duration_ms / 1000
    _emit("Generating music (HeartMuLa) …", "stage")
    _emit(f"  Tags:     {args.tags}")
    _emit(f"  Duration: {duration_s:.0f}s")
    _emit(f"  Output:   {out_path}")

    result = run_worker(cmd, on_event=_event_handler)
    finish_progress()
    if result.returncode != 0:
        _emit("ERROR: Music generation failed.", "error")
        sys.exit(1)
    if result.stdout.strip():
        print(result.stdout.strip())


def _audio_diarize(args):
    """Split dialogue audio into separate tracks per speaker."""
    input_paths = [Path(p) for p in args.input]

    for p in input_paths:
        if not p.exists():
            _emit(f"ERROR: File not found: {p}", "error")
            sys.exit(1)

    output_dir = Path(args.output) if args.output else input_paths[0].parent
    output_dir.mkdir(parents=True, exist_ok=True)

    diarize_script = DIARIZE_WORKER_DIR / "diarize.py"
    if not diarize_script.exists():
        _emit("ERROR: worker/diarize/diarize.py not found", "error")
        _emit("  Run: bash worker/diarize/install.sh", "error")
        sys.exit(1)

    for p in input_paths:
        cmd = [
            str(CONDA_BIN), "run", "--no-capture-output", "-n", DIARIZE_ENV,
            "python", str(diarize_script),
            str(p),
            "-o", str(output_dir),
        ]
        if args.speakers:
            cmd.extend(["--speakers", str(args.speakers)])
        if args.hf_token:
            cmd.extend(["--hf-token", args.hf_token])
        if args.verify:
            cmd.append("--verify")

        _emit(f"Diarizing {p.name} …", "stage")

        result = run_worker(cmd, on_event=_event_handler)
        finish_progress()
        if result.returncode != 0:
            _emit("ERROR: Diarization failed.", "error")
            sys.exit(1)
        if result.stdout.strip():
            print(result.stdout.strip())


def _audio_sfx(args):
    """Generate sound effects using EzAudio."""

    if not args.text and not getattr(args, "text_file", None):
        _emit("ERROR: --text or --text-file required for sfx engine.", "error")
        sys.exit(1)

    text = args.text
    if not text and args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8").strip()

    # Output path
    if args.output:
        out_path = Path(args.output)
        if out_path.is_dir() or str(out_path).endswith("/"):
            out_path = Path(args.output) / f"sfx_{int(time.time())}.wav"
    else:
        out_path = Path(f"sfx_{int(time.time())}.wav")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    sfx_script = SFX_WORKER_DIR / "generate.py"
    if not sfx_script.exists():
        _emit("ERROR: worker/sfx/generate.py not found", "error")
        _emit("  Run: bash worker/sfx/install.sh", "error")
        sys.exit(1)

    cmd = [
        str(CONDA_BIN), "run", "--no-capture-output", "-n", SFX_ENV,
        "python", str(sfx_script),
        "--text", text,
        "-o", str(out_path),
    ]

    duration_s = args.duration // 1000 if args.duration else args.seconds
    cmd.extend(["--seconds", str(duration_s)])

    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if getattr(args, "steps", None) is not None:
        cmd.extend(["--steps", str(args.steps)])
    if args.cfg_scale is not None:
        cmd.extend(["--cfg-scale", str(args.cfg_scale)])
    if args.model:
        cmd.extend(["--model", args.model])

    _emit("Generating SFX (EzAudio) …", "stage")
    _emit(f"  Prompt:   {text}")
    _emit(f"  Duration: {duration_s}s")
    _emit(f"  Output:   {out_path}")

    result = run_worker(cmd, on_event=_event_handler)
    finish_progress()
    if result.returncode != 0:
        _emit("ERROR: SFX generation failed.", "error")
        sys.exit(1)
    if result.stdout.strip():
        print(result.stdout.strip())


def _audio_voice_removal(args):
    """Remove vocals from audio (demucs → remix non-vocal stems)."""
    input_paths = [Path(p) for p in args.input]
    for p in input_paths:
        if not p.exists():
            _emit(f"ERROR: File not found: {p}", "error")
            sys.exit(1)

    output_dir = Path(args.output) if args.output else input_paths[0].parent
    output_dir.mkdir(parents=True, exist_ok=True)

    separate_script = SEPARATE_WORKER_DIR / "separate.py"
    if not separate_script.exists():
        _emit("ERROR: worker/separate/separate.py not found", "error")
        _emit("  Run: bash worker/separate/install.sh", "error")
        sys.exit(1)

    outputs = []

    for p in input_paths:
        # 1. Separate into stems (tempdir)
        tmp_base = Path(args.tmp_dir) if getattr(args, "tmp_dir", None) else Path(tempfile.gettempdir())
        tmpdir = tmp_base / f"voice_removal_{p.stem}_{os.getpid()}"
        tmpdir.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(CONDA_BIN), "run", "--no-capture-output", "-n", SEPARATE_ENV,
            "python", str(separate_script),
            str(p), "-o", str(tmpdir),
        ]
        if args.model:
            cmd.extend(["--model", args.model])

        _emit(f"Separating {p.name} …", "stage")
        result = run_worker(cmd, on_event=_event_handler)
        finish_progress()
        if result.returncode != 0:
            _emit("ERROR: Separation failed.", "error")
            shutil.rmtree(tmpdir, ignore_errors=True)
            sys.exit(1)

        # 2. Mix non-vocal stems (drums + bass + other)
        stem_name = p.stem
        non_vocal = ["drums", "bass", "other"]
        stem_files = [tmpdir / f"{stem_name}_{s}.wav" for s in non_vocal]
        missing = [str(f) for f in stem_files if not f.exists()]
        if missing:
            _emit(f"ERROR: Missing stems: {missing}", "error")
            shutil.rmtree(tmpdir, ignore_errors=True)
            sys.exit(1)

        out_path = output_dir / f"{stem_name}_no_vocals.wav"

        _emit(f"Mixing {len(non_vocal)} stems → {out_path.name}", "stage")

        mix_cmd = ["ffmpeg", "-y"]
        for sf in stem_files:
            mix_cmd += ["-i", str(sf)]
        n = len(stem_files)
        mix_inputs = "".join(f"[{i}:a]" for i in range(n))
        filter_str = f"{mix_inputs}amix=inputs={n}:normalize=0[mix];[mix]alimiter=limit=0.95[out]"
        mix_cmd += ["-filter_complex", filter_str, "-map", "[out]", str(out_path)]

        mix_result = subprocess.run(mix_cmd, capture_output=True, text=True)
        if mix_result.returncode != 0:
            _emit(f"ERROR: ffmpeg mix failed:\n{mix_result.stderr[-500:]}", "error")
            shutil.rmtree(tmpdir, ignore_errors=True)
            sys.exit(1)

        # 3. Cleanup
        shutil.rmtree(tmpdir, ignore_errors=True)
        outputs.append(str(out_path))

    print(json.dumps(outputs))


def _audio_ltx2(args):
    """Generate audio using LTX-2.3 video engine (virtual audio worker).

    Generates a minimal-resolution video (128x128) and extracts only the audio track.
    Useful for generating ambient sounds, scene audio, or dialog that matches a visual prompt.
    """
    if not LTX2_WORKER.exists():
        _emit("ERROR: worker/ltx2/generate.py not found", "error")
        _emit("  Run: bash worker/ltx2/install.sh", "error")
        sys.exit(1)

    prompt = getattr(args, "text", None) or getattr(args, "prompt", None)
    if not prompt:
        _emit("ERROR: --prompt or --text required for LTX audio generation", "error")
        sys.exit(1)

    # Output path — user decides format (.wav, .mp3, etc.)
    out_path = Path(args.output) if args.output else Path("audio_ltx.wav")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Minimal video resolution (128x128) — overridden by --ratio/--quality or -W/-H
    ratio = getattr(args, "ratio", None)
    quality = getattr(args, "quality", None)
    if ratio and quality:
        width, height = _resolve_video_dims(ratio, quality)
    elif hasattr(args, "width") and args.width and args.width != 768:
        width = args.width
        height = getattr(args, "height", 128) or 128
    else:
        width, height = 192, 192

    # Duration from --seconds (default 5s)
    seconds = getattr(args, "seconds", 5) or 5
    fps = 24
    raw_frames = int(seconds * fps)
    num_frames = ((raw_frames // 8) * 8) + 1

    # Build worker command — reuse video pipeline
    model = getattr(args, "model", None) or "distilled"
    import tempfile
    tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_video.close()

    cmd = [
        str(CONDA_BIN), "run", "--no-capture-output", "-n", LTX2_ENV,
        "python", str(LTX2_WORKER),
        "--model", model,
        "--prompt", prompt,
        "-o", tmp_video.name,
        "-W", str(width),
        "-H", str(height),
        "--frame-rate", str(fps),
        "--num-frames", str(num_frames),
    ]

    seed = getattr(args, "seed", None)
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    # Image conditioning (scene reference)
    image_first = getattr(args, "image_first", None)
    if image_first:
        if not Path(image_first).exists():
            _emit(f"ERROR: Image not found: {image_first}", "error")
            sys.exit(1)
        cmd.extend(["--image-first", image_first])

    if getattr(args, "enhance_prompt", False):
        cmd.append("--enhance-prompt")

    _emit(f"Generating audio via LTX-2.3 ({model}, {seconds}s, {width}x{height}) …", "stage")

    result = run_worker(cmd, on_event=_event_handler)
    finish_progress()
    if result.returncode != 0:
        _emit("ERROR: Audio generation failed.", "error")
        try:
            os.unlink(tmp_video.name)
        except OSError:
            pass
        sys.exit(1)

    # Extract audio from temp video → target format
    _emit("Extracting audio …", "stage")
    extract = subprocess.run(
        ["ffmpeg", "-y", "-i", tmp_video.name, "-vn", "-map", "a", str(out_path)],
        capture_output=True, text=True,
    )
    try:
        os.unlink(tmp_video.name)
    except OSError:
        pass

    if extract.returncode != 0:
        _emit(f"ERROR: ffmpeg audio extraction failed:\n{extract.stderr[-500:]}", "error")
        sys.exit(1)

    print(json.dumps([str(out_path)]))


def cmd_audio(args):
    """Audio processing — dispatch by engine."""
    engine = args.engine
    if engine == "enhance":
        _audio_enhance(args)
    elif engine == "demucs":
        _audio_demucs(args)
    elif engine == "ace-step":
        _audio_ace(args)
    elif engine == "heartmula":
        _audio_heartmula(args)
    elif engine == "diarize":
        _audio_diarize(args)
    elif engine == "sfx":
        _audio_sfx(args)
    elif engine == "voice-removal":
        _audio_voice_removal(args)
    elif engine == "ltx2.3":
        _audio_ltx2(args)
    else:
        _emit(f"ERROR: Unknown audio engine: {engine}", "error")
        sys.exit(1)


# ── Text Engines ─────────────────────────────────────────────────────────────

def _text_whisper(args):
    """Transcribe audio using mlx-whisper."""
    transcribe_script = WHISPER_WORKER_DIR / "transcribe.py"
    if not transcribe_script.exists():
        _emit("ERROR: worker/whisper/transcribe.py not found", "error")
        _emit("  Run: bash worker/whisper/install.sh", "error")
        sys.exit(1)

    input_files = []
    for f in args.input:
        p = Path(f).resolve()
        if not p.exists():
            _emit(f"ERROR: File not found: {p}", "error")
            sys.exit(1)
        input_files.append(str(p))

    cmd = [
        str(CONDA_BIN), "run", "--no-capture-output", "-n", WHISPER_ENV,
        "python", str(transcribe_script),
    ]
    cmd.extend(input_files)

    model = getattr(args, "model", "large-v3-turbo") or "large-v3-turbo"
    cmd.extend(["--model", model])

    if getattr(args, "input_language", None):
        cmd.extend(["--language", args.language])

    if getattr(args, "word_timestamps", False):
        cmd.append("--word-timestamps")

    fmt = getattr(args, "format", "json") or "json"
    cmd.extend(["--format", fmt])

    if args.output:
        out_path = Path(args.output).resolve()
        cmd.extend(["-o", str(out_path)])

    n_files = len(input_files)
    _emit(f"Transcribing {n_files} file{'s' if n_files > 1 else ''} …", "stage")
    _emit(f"  Model: {model}")
    if getattr(args, "input_language", None):
        _emit(f"  Language: {args.language}")
    _emit(f"  Format: {fmt}")

    result = run_worker(cmd, on_event=_event_handler)
    finish_progress()
    if result.returncode != 0:
        _emit("ERROR: Transcription failed", "error")
        sys.exit(1)
    if result.stdout.strip():
        print(result.stdout.strip())


def _text_heartmula_transcribe(args):
    """Transcribe lyrics from audio using HeartTranscriptor."""
    transcribe_script = MUSIC_WORKER_DIR / "transcribe.py"
    if not transcribe_script.exists():
        _emit("ERROR: worker/music/transcribe.py not found", "error")
        sys.exit(1)

    # heartmula-transcribe expects single file as first positional arg
    audio_path = Path(args.input[0]).resolve()
    if not audio_path.exists():
        _emit(f"ERROR: Audio file not found: {audio_path}", "error")
        sys.exit(1)

    ckpt_dir = MUSIC_MODELS_DIR / "ckpt"
    if not ckpt_dir.exists():
        _emit("ERROR: HeartMuLa checkpoints not found", "error")
        _emit(f"  Expected: {ckpt_dir}", "error")
        sys.exit(1)

    cmd = [
        str(CONDA_BIN), "run", "--no-capture-output", "-n", HEARTMULA_ENV,
        "python", str(transcribe_script),
        "--audio", str(audio_path),
        "--ckpt-dir", str(ckpt_dir),
    ]

    if args.output:
        out_path = Path(args.output).resolve()
        cmd.extend(["-o", str(out_path)])

    _emit("Transcribing lyrics …", "stage")
    _emit(f"  Input: {audio_path}")

    result = run_worker(cmd, on_event=_event_handler)
    finish_progress()
    if result.returncode != 0:
        _emit("ERROR: Lyrics transcription failed.", "error")
        sys.exit(1)
    if result.stdout.strip():
        print(result.stdout.strip())


# ── LLM Text — Shared Helpers ────────────────────────────────────────────────
# Inference runs in the "text" conda env via worker/text/inference.py.
# Connection helpers below are shared with models + ps commands.

_LLM_ENGINES = ("ollama",)
TEXT_ENV = "text"
TEXT_INFERENCE_SCRIPT = TEXT_WORKER_DIR / "inference.py"


def _llm_engine_base_url(engine: str, args) -> str:
    """Get base URL for engine: --base-url > engines.json > hardcoded default."""
    if getattr(args, "base_url", None):
        return args.base_url.rstrip("/")
    if TEXT_ENGINES_FILE.exists():
        cfg = json.loads(TEXT_ENGINES_FILE.read_text())
        if engine in cfg and "base_url" in cfg[engine]:
            return cfg[engine]["base_url"].rstrip("/")
    defaults = {"ollama": "http://localhost:11434"}
    return defaults.get(engine, "http://localhost:8000")


def _llm_api_key(engine: str, args) -> str | None:
    """Get API key: --api-key > env var > engines.json > None."""
    if getattr(args, "api_key", None):
        return args.api_key
    env_map = {"ollama": "OLLAMA_API_KEY"}
    env_val = os.environ.get(env_map.get(engine, ""), None)
    if env_val:
        return env_val
    if TEXT_ENGINES_FILE.exists():
        cfg = json.loads(TEXT_ENGINES_FILE.read_text())
        return cfg.get(engine, {}).get("api_key")
    return None


def _llm_model_config_dir(engine: str, model: str) -> Path:
    """Get config directory for a model."""
    safe_name = model.replace("/", "_").replace(":", "_")
    return TEXT_MODELS_DIR / engine / safe_name



def _llm_request(engine: str, base_url: str, path: str, body: dict,
                 api_key: str | None, stream: bool = False):
    """Make HTTP request to LLM engine. Returns response text or streams."""
    url = f"{base_url}{path}"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = requests.post(url, json=body, headers=headers,
                             stream=stream, timeout=300)
        resp.raise_for_status()
    except requests.ConnectionError:
        engine_hints = {
            "ollama": "Is Ollama running? Start with: ollama serve",
        }
        _emit(f"ERROR: Could not connect to {engine} at {base_url}", "error")
        hint = engine_hints.get(engine, "")
        if hint:
            _emit(f"  {hint}")
        sys.exit(1)
    except requests.HTTPError as e:
        _emit(f"ERROR: {engine} returned {resp.status_code}: {resp.text}", "error")
        sys.exit(1)

    if stream:
        return resp
    return resp.json()


def _llm_get(engine: str, base_url: str, path: str,
             api_key: str | None) -> dict | list:
    """GET request to LLM engine. Returns parsed JSON."""
    url = f"{base_url}{path}"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
    except requests.ConnectionError:
        return None  # engine offline
    except requests.HTTPError:
        _emit(f"ERROR: {engine} returned {resp.status_code}: {resp.text}", "error")
        return None
    return resp.json()


def _llm_delete(engine: str, base_url: str, path: str, body: dict,
                api_key: str | None):
    """DELETE request to LLM engine."""
    url = f"{base_url}{path}"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        resp = requests.delete(url, json=body, headers=headers, timeout=30)
        resp.raise_for_status()
    except requests.ConnectionError:
        _emit(f"ERROR: Could not connect to {engine} at {base_url}", "error")
        sys.exit(1)
    except requests.HTTPError:
        _emit(f"ERROR: {engine} returned {resp.status_code}: {resp.text}", "error")
        sys.exit(1)
    return resp.json() if resp.text.strip() else {}


def _text_llm(args):
    """Dispatch LLM inference to worker/text/inference.py via text env."""
    # Daily: 1) update env (pip), 2) pull models + ensure num_ctx
    _ollama_maybe_update_env()
    _ollama_maybe_update_models()

    endpoint = getattr(args, "endpoint", None)
    if not endpoint:
        _emit("ERROR: --endpoint required for LLM engines", "error")
        _emit("  Endpoints: chat, generate, set, show, reset, load, unload")
        sys.exit(1)
    if not TEXT_INFERENCE_SCRIPT.exists():
        _emit(f"ERROR: {TEXT_INFERENCE_SCRIPT} not found", "error")
        _emit("  Run: bash worker/text/install.sh")
        sys.exit(1)

    cmd = [
        str(CONDA_BIN), "run", "--no-capture-output", "-n", TEXT_ENV,
        "python", str(TEXT_INFERENCE_SCRIPT),
        "--engine", args.engine, "--model", args.model, "--endpoint", endpoint,
    ]

    # Forward optional flags
    for flag, attr in [
        ("--prompt", "prompt"), ("--system", "system"),
        ("--messages", "messages"),
        ("--context-length", "context_length"), ("--max-tokens", "max_tokens"),
        ("--temperature", "temperature"), ("--top-p", "top_p"),
        ("--top-k", "top_k"), ("--repeat-penalty", "repeat_penalty"),
        ("--seed", "seed"), ("--stop", "stop"),
        ("--base-url", "base_url"), ("--api-key", "api_key"),
        ("-o", "output"),
    ]:
        val = getattr(args, attr, None)
        if val is not None:
            cmd.extend([flag, str(val)])

    if getattr(args, "stream", False):
        cmd.append("--stream")

    # Forward --thinking
    thinking = getattr(args, "thinking", "False")
    if thinking != "False":
        cmd.extend(["--thinking", thinking])

    # Forward --images
    image_paths = getattr(args, "images", None)
    if image_paths:
        cmd.append("--images")
        cmd.extend(image_paths)

    result = run_worker(cmd, on_event=_event_handler)
    if result.returncode != 0:
        sys.exit(1)


def cmd_text(args):
    """Text — dispatch by engine."""
    engine = args.engine
    if engine == "whisper":
        _text_whisper(args)
    elif engine == "heartmula-transcribe":
        _text_heartmula_transcribe(args)
    elif engine == "ollama":
        _text_llm(args)
    else:
        _emit(f"ERROR: Engine '{engine}' is currently not supported. Use: text ollama", "error")
        sys.exit(1)


# ── PS (Status) ──────────────────────────────────────────────────────────────

def _ps_rvc_models() -> list[dict]:
    """Get active RVC models for ps."""
    models = []
    server_running = check_server()
    if server_running:
        data = api_get("/models")
        for m in data.get("models", []):
            name = m if isinstance(m, str) else m.get("name", str(m))
            config = load_model_config(name)
            target = config.get("target_f0")
            extra = f"target: {target:.0f} Hz" if target else ""
            models.append({"model": name, "engine": "rvc", "status": "loaded",
                           "vram": "-", "ctx": "-", "extra": extra})
    return models


def _ps_ollama_models() -> list[dict]:
    """Get running Ollama models for ps."""
    models = []
    base_url = "http://localhost:11434"
    if TEXT_ENGINES_FILE.exists():
        cfg = json.loads(TEXT_ENGINES_FILE.read_text())
        base_url = cfg.get("ollama", {}).get("base_url", base_url)
    data = _llm_get("ollama", base_url, "/api/ps", None)
    if data is None:
        return models
    for m in data.get("models", []):
        name = m.get("name", "?")
        vram_bytes = m.get("size_vram", m.get("size", 0))
        vram_gb = vram_bytes / (1024**3)
        vram = f"{vram_gb:.1f} GB" if vram_gb > 0 else "-"
        ctx = str(m.get("context_length", "-")) if m.get("context_length") else "-"
        details = m.get("details", {})
        quant = details.get("quantization_level", "")
        family = details.get("family", "")
        extra = f"{family} {quant}".strip() if (family or quant) else ""
        models.append({"model": name, "engine": "ollama", "status": "running",
                       "vram": vram, "ctx": ctx, "extra": extra})
    return models


def cmd_ps(args):
    all_models = []
    all_models.extend(_ps_rvc_models())
    all_models.extend(_ps_ollama_models())

    # JSON mode
    if _event_handler is print_event_json:
        print(json.dumps(all_models))
        return

    # TUI mode
    if not all_models:
        print("No active models.")
        return

    # Header
    print(f"{'MODEL':<30s} {'ENGINE':<13s} {'STATUS':<10s} {'VRAM':<11s} {'CTX':<10s} {'EXTRA'}")
    for m in all_models:
        print(f"{m['model']:<30s} {m['engine']:<13s} {m['status']:<10s} "
              f"{m['vram']:<11s} {m['ctx']:<10s} {m['extra']}")


# ── Output (concatenation, mixing) ────────────────────────────────────────────

def cmd_output(args):
    engine = args.engine

    if engine == "audio-concatenate":
        _output_audio_concatenate(args)
    elif engine == "audio-mucs":
        _output_audio_mucs(args)
    else:
        _emit(f"ERROR: Unknown output engine: {engine}", "error")
        sys.exit(1)


def _parse_clip_opts(clip_args: list[str] | None, n: int) -> list[dict]:
    """Parse --clip arguments into per-input option dicts.

    Each --clip value has the form 'INDEX:key=val,key=val'.
    Returns a list of n dicts with keys: fade_in, fade_out, crossfade, volume,
    start, end.
    """
    opts = [{"fade_in": None, "fade_out": None, "crossfade": None,
             "volume": None, "start": None, "end": None, "pan": None}
            for _ in range(n)]
    if not clip_args:
        return opts

    for spec in clip_args:
        if ":" not in spec:
            _emit(f"ERROR: Invalid --clip format (missing ':'): {spec}", "error")
            sys.exit(1)
        idx_str, kv_str = spec.split(":", 1)
        try:
            idx = int(idx_str)
        except ValueError:
            _emit(f"ERROR: Invalid --clip index: {idx_str}", "error")
            sys.exit(1)
        if idx < 0 or idx >= n:
            _emit(f"ERROR: --clip index {idx} out of range (0..{n - 1})", "error")
            sys.exit(1)
        for pair in kv_str.split(","):
            if "=" not in pair:
                _emit(f"ERROR: Invalid --clip key=value: {pair}", "error")
                sys.exit(1)
            key, val = pair.split("=", 1)
            key = key.strip().replace("-", "_")
            if key not in ("fade_in", "fade_out", "crossfade", "volume", "start", "end", "pan"):
                _emit(f"ERROR: Unknown --clip option: {key}", "error")
                sys.exit(1)
            try:
                opts[idx][key] = float(val)
            except ValueError:
                _emit(f"ERROR: Invalid --clip value for {key}: {val}", "error")
                sys.exit(1)

    # crossfade on index 0 makes no sense
    if opts[0]["crossfade"]:
        _emit("WARNING: crossfade on first clip ignored (no previous clip).", "warning")
        opts[0]["crossfade"] = None

    return opts


def _output_audio_concatenate(args):
    """Concatenate multiple audio files into one via ffmpeg."""
    inputs = args.input
    if not inputs or len(inputs) < 2:
        _emit("ERROR: Need at least 2 input files to concatenate.", "error")
        sys.exit(1)

    # Validate inputs exist
    for f in inputs:
        if not Path(f).exists():
            _emit(f"ERROR: Input file not found: {f}", "error")
            sys.exit(1)

    # Determine output path
    out = args.output
    if not out:
        ext = Path(inputs[0]).suffix
        out = str(Path(inputs[0]).parent / f"concatenated{ext}")

    out_path = Path(out)
    n = len(inputs)
    clip_opts = _parse_clip_opts(getattr(args, "clip", None), n)

    # Build ffmpeg command
    cmd = ["ffmpeg", "-y"]
    for f in inputs:
        cmd += ["-i", f]

    # Build filter chain
    filter_parts = []

    # 1. Normalize each input to 44100 Hz stereo
    for i in range(n):
        filter_parts.append(f"[{i}:a]aresample=44100,aformat=channel_layouts=stereo[norm_{i}]")

    # 2. Per-clip: trim, volume, fade-in, fade-out
    for i in range(n):
        co = clip_opts[i]
        effects = []
        # Trim (start/end)
        if co["start"] is not None or co["end"] is not None:
            trim_args = []
            if co["start"] is not None:
                trim_args.append(f"start={co['start']}")
            if co["end"] is not None:
                trim_args.append(f"end={co['end']}")
            effects.append(f"atrim={':'.join(trim_args)},asetpts=PTS-STARTPTS")
        if co["volume"] is not None:
            effects.append(f"volume={co['volume']}")
        if co["pan"] is not None:
            # Stereo balance: -1.0 = full left, 0 = center, +1.0 = full right
            # pan <= 0: L unchanged, R attenuated (R_gain = 1 + pan)
            # pan >= 0: R unchanged, L attenuated (L_gain = 1 - pan)
            p = max(-1.0, min(1.0, co["pan"]))
            l_gain = max(0.0, 1.0 - p)
            r_gain = max(0.0, 1.0 + p)
            effects.append(f"pan=stereo|c0={l_gain}*c0|c1={r_gain}*c1")
        if co["fade_in"] is not None:
            effects.append(f"afade=t=in:d={co['fade_in']}")
        if co["fade_out"] is not None:
            # areverse trick: fade from the end without knowing duration
            effects.append(f"areverse,afade=t=in:d={co['fade_out']},areverse")

        if effects:
            filter_parts.append(f"[norm_{i}]{','.join(effects)}[a_{i}]")
        else:
            filter_parts.append(f"[norm_{i}]acopy[a_{i}]")

    # 3. Chain clips: crossfade or concat, pairwise L→R
    has_any_crossfade = any(co["crossfade"] for co in clip_opts[1:])

    if has_any_crossfade:
        prev = "a_0"
        for i in range(1, n):
            out_label = f"ch_{i}" if i < n - 1 else "out"
            cf = clip_opts[i]["crossfade"]
            if cf and cf > 0:
                filter_parts.append(
                    f"[{prev}][a_{i}]acrossfade=d={cf}:c1=tri:c2=tri[{out_label}]"
                )
            else:
                filter_parts.append(
                    f"[{prev}][a_{i}]concat=n=2:v=0:a=1[{out_label}]"
                )
            prev = out_label
    else:
        concat_inputs = "".join(f"[a_{i}]" for i in range(n))
        filter_parts.append(f"{concat_inputs}concat=n={n}:v=0:a=1[out]")

    filter_str = ";".join(filter_parts)
    cmd += ["-filter_complex", filter_str, "-map", "[out]"]

    # Bitrate (optional)
    if args.output_bitrate:
        cmd += ["-b:a", args.output_bitrate]

    cmd.append(str(out_path))

    _emit(f"Concatenating {n} files → {out_path.name}", "stage")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        _emit(f"ERROR: ffmpeg failed:\n{result.stderr[-500:]}", "error")
        sys.exit(1)



def _output_audio_mucs(args):
    """Mix (overlay) multiple audio files in parallel into one via ffmpeg."""
    inputs = args.input
    if not inputs or len(inputs) < 2:
        _emit("ERROR: Need at least 2 input files to mix.", "error")
        sys.exit(1)

    for f in inputs:
        if not Path(f).exists():
            _emit(f"ERROR: Input file not found: {f}", "error")
            sys.exit(1)

    out = args.output
    if not out:
        ext = Path(inputs[0]).suffix
        out = str(Path(inputs[0]).parent / f"mixed{ext}")

    out_path = Path(out)
    n = len(inputs)
    clip_opts = _parse_clip_opts(getattr(args, "clip", None), n)

    # Warn about crossfade (not applicable for parallel mix)
    for i, co in enumerate(clip_opts):
        if co["crossfade"]:
            _emit(f"WARNING: crossfade on clip {i} ignored (not applicable for parallel mix).", "warning")

    cmd = ["ffmpeg", "-y"]
    for f in inputs:
        cmd += ["-i", f]

    filter_parts = []

    # 1. Normalize each input to 44100 Hz stereo
    for i in range(n):
        filter_parts.append(f"[{i}:a]aresample=44100,aformat=channel_layouts=stereo[norm_{i}]")

    # 2. Per-clip: trim, volume, pan, fade-in, fade-out
    for i in range(n):
        co = clip_opts[i]
        effects = []
        if co["start"] is not None or co["end"] is not None:
            trim_args = []
            if co["start"] is not None:
                trim_args.append(f"start={co['start']}")
            if co["end"] is not None:
                trim_args.append(f"end={co['end']}")
            effects.append(f"atrim={':'.join(trim_args)},asetpts=PTS-STARTPTS")
        if co["volume"] is not None:
            effects.append(f"volume={co['volume']}")
        if co["pan"] is not None:
            p = max(-1.0, min(1.0, co["pan"]))
            l_gain = max(0.0, 1.0 - p)
            r_gain = max(0.0, 1.0 + p)
            effects.append(f"pan=stereo|c0={l_gain}*c0|c1={r_gain}*c1")
        if co["fade_in"] is not None:
            effects.append(f"afade=t=in:d={co['fade_in']}")
        if co["fade_out"] is not None:
            effects.append(f"areverse,afade=t=in:d={co['fade_out']},areverse")

        if effects:
            filter_parts.append(f"[norm_{i}]{','.join(effects)}[a_{i}]")
        else:
            filter_parts.append(f"[norm_{i}]acopy[a_{i}]")

    # 3. Mix all clips in parallel + limiter to prevent clipping
    mix_inputs = "".join(f"[a_{i}]" for i in range(n))
    filter_parts.append(f"{mix_inputs}amix=inputs={n}:normalize=0[mix]")
    filter_parts.append("[mix]alimiter=limit=0.95[out]")

    filter_str = ";".join(filter_parts)
    cmd += ["-filter_complex", filter_str, "-map", "[out]"]

    if args.output_bitrate:
        cmd += ["-b:a", args.output_bitrate]

    cmd.append(str(out_path))

    _emit(f"Mixing {n} files (parallel) → {out_path.name}", "stage")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        _emit(f"ERROR: ffmpeg failed:\n{result.stderr[-500:]}", "error")
        sys.exit(1)



# ── Image generation (FLUX.2 / SD 1.5) ──────────────────────────────────────

IMAGE_WORKER_DIR = SCRIPT_DIR / "worker" / "image"
IMAGE_WORKER = IMAGE_WORKER_DIR / "generate.py"
FLUX2_DIR = IMAGE_WORKER_DIR / "flux2"
FLUX2_ENV = "flux2"
POSE_WORKER_DIR = SCRIPT_DIR / "worker" / "pose"
POSE_WORKER = POSE_WORKER_DIR / "generate.py"
POSE_ENV = "openpose"
SD15_WORKER_DIR = SCRIPT_DIR / "worker" / "sd15"
SD15_WORKER = SD15_WORKER_DIR / "generate.py"
SD15_ENV = "sd15"
DEPTH_WORKER_DIR = SCRIPT_DIR / "worker" / "depth"
DEPTH_WORKER = DEPTH_WORKER_DIR / "generate.py"
DEPTH_ENV = "depth"
LINEART_WORKER_DIR = SCRIPT_DIR / "worker" / "lineart"
LINEART_WORKER = LINEART_WORKER_DIR / "generate.py"
LINEART_ENV = "lineart"
NORMALMAP_WORKER_DIR = SCRIPT_DIR / "worker" / "normalmap"
NORMALMAP_WORKER = NORMALMAP_WORKER_DIR / "generate.py"
NORMALMAP_ENV = "normalmap"
SKETCH_WORKER_DIR = SCRIPT_DIR / "worker" / "sketch"
SKETCH_WORKER = SKETCH_WORKER_DIR / "generate.py"
SKETCH_ENV = "sketch"
UPSCALE_WORKER_DIR = SCRIPT_DIR / "worker" / "upscale"
UPSCALE_WORKER = UPSCALE_WORKER_DIR / "generate.py"
UPSCALE_ENV = "upscale"
SEGMENT_WORKER_DIR = SCRIPT_DIR / "worker" / "segment"
SEGMENT_WORKER = SEGMENT_WORKER_DIR / "generate.py"
SEGMENT_ENV = "segment"


# ── Video generation (LTX-2.3) ───────────────────────────────────────────────

LTX2_WORKER_DIR = SCRIPT_DIR / "worker" / "ltx2"
LTX2_WORKER = LTX2_WORKER_DIR / "generate.py"
LTX2_ENV = "ltx2"


# Valid controlnet modes and their behavior
_CONTROLNET_PIXEL_ALIGNED = {"depth", "normalmap", "lineart", "sketch"}
_CONTROLNET_REFERENCE = {"pose"}
_CONTROLNET_MODES = _CONTROLNET_PIXEL_ALIGNED | _CONTROLNET_REFERENCE

# Map user-facing model names to FLUX2_MODEL_INFO keys
_IMAGE_MODEL_MAP = {
    "4b":           "flux.2-klein-base-4b",
    "4b-distilled": "flux.2-klein-4b",
    "9b":           "flux.2-klein-base-9b",
    "9b-distilled": "flux.2-klein-9b",
    "9b-kv":        "flux.2-klein-9b-kv",
    "dev":          "flux.2-dev",
}


def _image_openpose(args):
    """Run DWPose pose estimation."""
    images = getattr(args, "images", None)
    if not images:
        _emit("ERROR: --images required for openpose", "error")
        sys.exit(1)

    for img in images:
        if not Path(img).exists():
            _emit(f"ERROR: Image not found: {img}", "error")
            sys.exit(1)

    if not POSE_WORKER.exists():
        _emit("ERROR: worker/pose/generate.py not found", "error")
        _emit("  Run: bash worker/pose/install.sh", "error")
        sys.exit(1)

    out_path = Path(args.output) if args.output else Path("pose.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(CONDA_BIN), "run", "--no-capture-output", "-n", POSE_ENV,
        "python", str(POSE_WORKER),
        "--images", *images,
        "-o", str(out_path),
    ]

    mode = getattr(args, "pose_mode", None)
    if mode:
        cmd.extend(["--mode", mode])

    _emit(f"Extracting pose ({len(images)} image(s)) …", "stage")

    result = run_worker(cmd, on_event=_event_handler)
    finish_progress()
    if result.returncode != 0:
        _emit("ERROR: Pose estimation failed.", "error")
        sys.exit(1)

    if result.stdout.strip():
        print(result.stdout.strip())



def _image_depth(args):
    """Run Apple Depth Pro depth estimation."""
    images = getattr(args, "images", None)
    if not images:
        _emit("ERROR: --images required for depth estimation", "error")
        sys.exit(1)

    for img in images:
        if not Path(img).exists():
            _emit(f"ERROR: Image not found: {img}", "error")
            sys.exit(1)

    if not DEPTH_WORKER.exists():
        _emit("ERROR: worker/depth/generate.py not found", "error")
        _emit("  Run: bash worker/depth/install.sh", "error")
        sys.exit(1)

    out_path = Path(args.output) if args.output else Path("depth.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(CONDA_BIN), "run", "--no-capture-output", "-n", DEPTH_ENV,
        "python", str(DEPTH_WORKER),
        "--images", *images,
        "-o", str(out_path),
    ]

    _emit(f"Estimating depth ({len(images)} image(s)) …", "stage")

    result = run_worker(cmd, on_event=_event_handler)
    finish_progress()
    if result.returncode != 0:
        _emit("ERROR: Depth estimation failed.", "error")
        sys.exit(1)

    if result.stdout.strip():
        print(result.stdout.strip())



def _image_sd15(args, cn_mode=None, cn_path=None):
    """Run SD 1.5 image generation (maturemalemix etc.)."""
    if not SD15_WORKER.exists():
        _emit("ERROR: worker/sd15/generate.py not found", "error")
        _emit("  Run: bash worker/sd15/install.sh", "error")
        sys.exit(1)

    prompt = getattr(args, "prompt", None)
    if not prompt:
        _emit("ERROR: --prompt required for image generation", "error")
        sys.exit(1)

    model = getattr(args, "model", None) or "mm"
    out_path = Path(args.output) if args.output else Path("image.png")
    if out_path.is_dir() or str(out_path).endswith("/"):
        out_path = out_path / f"image_{int(time.time())}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(CONDA_BIN), "run", "--no-capture-output", "-n", SD15_ENV,
        "python", str(SD15_WORKER),
        "--model", model,
        "--prompt", prompt,
        "-o", str(out_path),
        "-W", str(getattr(args, "width", 512)),
        "-H", str(getattr(args, "height", 512)),
    ]

    seed = getattr(args, "seed", None)
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    steps = getattr(args, "steps", None)
    if steps is not None:
        cmd.extend(["--steps", str(steps)])

    cfg = getattr(args, "cfg_scale", None)
    if cfg is not None:
        cmd.extend(["--cfg", str(cfg)])

    negative = getattr(args, "negative_prompt", None)
    if negative is not None:
        cmd.extend(["--negative-prompt", negative])

    loras = getattr(args, "lora", None)
    if loras:
        for lora_spec in loras:
            cmd.extend(["--lora", lora_spec])

    if getattr(args, "no_lora", False):
        cmd.append("--no-lora")

    # ControlNet conditioning
    if cn_mode and cn_path:
        cmd.extend(["--controlnet-image", cn_path, "--controlnet-mode", cn_mode])

    _emit(f"Generating image (sd1.5/{model}) …", "stage")

    result = run_worker(cmd, on_event=_event_handler)
    finish_progress()
    if result.returncode != 0:
        _emit("ERROR: Image generation failed.", "error")
        sys.exit(1)

    if result.stdout.strip():
        print(result.stdout.strip())



def _parse_controlnet(spec):
    """Parse 'mode:filepath' → (mode, filepath). Validates mode and file existence."""
    if ":" not in spec:
        _emit("ERROR: --controlnet must be mode:filepath (e.g. depth:depth.png)", "error")
        sys.exit(1)
    mode, filepath = spec.split(":", 1)
    mode = mode.lower()
    if mode not in _CONTROLNET_MODES:
        valid = ", ".join(sorted(_CONTROLNET_MODES))
        _emit(f"ERROR: Unknown controlnet mode '{mode}'. Valid: {valid}", "error")
        sys.exit(1)
    if not Path(filepath).exists():
        _emit(f"ERROR: ControlNet image not found: {filepath}", "error")
        sys.exit(1)
    return mode, filepath


def _image_lineart(args):
    """Extract line art from images."""
    images = getattr(args, "images", None)
    if not images:
        _emit("ERROR: --images required for lineart extraction", "error")
        sys.exit(1)
    for img in images:
        if not Path(img).exists():
            _emit(f"ERROR: File not found: {img}", "error")
            sys.exit(1)
    if not LINEART_WORKER.exists():
        _emit("ERROR: worker/lineart/generate.py not found", "error")
        _emit("  Run: bash worker/lineart/install.sh", "error")
        sys.exit(1)

    out_path = Path(args.output) if args.output else Path("lineart.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = getattr(args, "model", None) or "teed"
    cmd = [
        str(CONDA_BIN), "run", "--no-capture-output", "-n", LINEART_ENV,
        "python", str(LINEART_WORKER),
        "--images"] + list(images) + ["-o", str(out_path), "--model", model,
    ]

    _emit(f"Extracting lineart ({model}, {len(images)} image(s)) …", "stage")
    result = run_worker(cmd, on_event=_event_handler)
    finish_progress()
    if result.returncode != 0:
        _emit("ERROR: Lineart extraction failed.", "error")
        sys.exit(1)
    if result.stdout.strip():
        print(result.stdout.strip())


def _image_normalmap(args):
    """Estimate surface normals from images."""
    images = getattr(args, "images", None)
    if not images:
        _emit("ERROR: --images required for normalmap estimation", "error")
        sys.exit(1)
    for img in images:
        if not Path(img).exists():
            _emit(f"ERROR: File not found: {img}", "error")
            sys.exit(1)
    if not NORMALMAP_WORKER.exists():
        _emit("ERROR: worker/normalmap/generate.py not found", "error")
        _emit("  Run: bash worker/normalmap/install.sh", "error")
        sys.exit(1)

    out_path = Path(args.output) if args.output else Path("normalmap.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(CONDA_BIN), "run", "--no-capture-output", "-n", NORMALMAP_ENV,
        "python", str(NORMALMAP_WORKER),
        "--images"] + list(images) + ["-o", str(out_path),
    ]
    steps = getattr(args, "steps", None)
    if steps is not None:
        cmd.extend(["--steps", str(steps)])

    _emit(f"Estimating normals ({len(images)} image(s)) …", "stage")
    result = run_worker(cmd, on_event=_event_handler)
    finish_progress()
    if result.returncode != 0:
        _emit("ERROR: Normal map estimation failed.", "error")
        sys.exit(1)
    if result.stdout.strip():
        print(result.stdout.strip())


def _image_sketch(args):
    """Extract sketch/edges from images."""
    images = getattr(args, "images", None)
    if not images:
        _emit("ERROR: --images required for sketch extraction", "error")
        sys.exit(1)
    for img in images:
        if not Path(img).exists():
            _emit(f"ERROR: File not found: {img}", "error")
            sys.exit(1)
    if not SKETCH_WORKER.exists():
        _emit("ERROR: worker/sketch/generate.py not found", "error")
        _emit("  Run: bash worker/sketch/install.sh", "error")
        sys.exit(1)

    out_path = Path(args.output) if args.output else Path("sketch.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(CONDA_BIN), "run", "--no-capture-output", "-n", SKETCH_ENV,
        "python", str(SKETCH_WORKER),
        "--images"] + list(images) + ["-o", str(out_path),
    ]

    _emit(f"Extracting sketch ({len(images)} image(s)) …", "stage")
    result = run_worker(cmd, on_event=_event_handler)
    finish_progress()
    if result.returncode != 0:
        _emit("ERROR: Sketch extraction failed.", "error")
        sys.exit(1)
    if result.stdout.strip():
        print(result.stdout.strip())


def _image_upscale(args):
    """Upscale images using Real-ESRGAN."""
    images = getattr(args, "images", None)
    if not images:
        _emit("ERROR: --images required for upscaling", "error")
        sys.exit(1)
    for img in images:
        if not Path(img).exists():
            _emit(f"ERROR: File not found: {img}", "error")
            sys.exit(1)
    if not UPSCALE_WORKER.exists():
        _emit("ERROR: worker/upscale/generate.py not found", "error")
        _emit("  Run: bash worker/upscale/install.sh", "error")
        sys.exit(1)

    out_path = Path(args.output) if args.output else Path("upscaled.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = getattr(args, "model", None) or "4x"
    cmd = [
        str(CONDA_BIN), "run", "--no-capture-output", "-n", UPSCALE_ENV,
        "python", str(UPSCALE_WORKER),
        "--images"] + list(images) + ["-o", str(out_path), "--model", model,
    ]

    _emit(f"Upscaling ({model}, {len(images)} image(s)) …", "stage")
    result = run_worker(cmd, on_event=_event_handler)
    finish_progress()
    if result.returncode != 0:
        _emit("ERROR: Upscaling failed.", "error")
        sys.exit(1)
    if result.stdout.strip():
        print(result.stdout.strip())


def _image_segment(args):
    """Segment images (background removal / object isolation)."""
    images = getattr(args, "images", None)
    if not images:
        _emit("ERROR: --images required for segmentation", "error")
        sys.exit(1)
    for img in images:
        if not Path(img).exists():
            _emit(f"ERROR: File not found: {img}", "error")
            sys.exit(1)
    if not SEGMENT_WORKER.exists():
        _emit("ERROR: worker/segment/generate.py not found", "error")
        _emit("  Run: bash worker/segment/install.sh", "error")
        sys.exit(1)

    out_path = Path(args.output) if args.output else Path("segment.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output_layer = getattr(args, "output_layer", None) or "foreground"
    cmd = [
        str(CONDA_BIN), "run", "--no-capture-output", "-n", SEGMENT_ENV,
        "python", str(SEGMENT_WORKER),
        "--images"] + list(images) + ["-o", str(out_path),
        "--output-layer", output_layer,
    ]

    _emit(f"Segmenting ({output_layer}, {len(images)} image(s)) …", "stage")
    result = run_worker(cmd, on_event=_event_handler)
    finish_progress()
    if result.returncode != 0:
        _emit("ERROR: Segmentation failed.", "error")
        sys.exit(1)
    if result.stdout.strip():
        print(result.stdout.strip())


def cmd_image(args):
    """Generate or process images."""
    engine = args.engine

    # ── Extraction engines ────────────────────────────────────────────────
    if engine == "openpose":
        return _image_openpose(args)
    if engine == "depth":
        return _image_depth(args)
    if engine == "lineart":
        return _image_lineart(args)
    if engine == "normalmap":
        return _image_normalmap(args)
    if engine == "sketch":
        return _image_sketch(args)
    if engine == "upscale":
        return _image_upscale(args)
    if engine == "segment":
        return _image_segment(args)

    # ── Resolve --ratio + --quality → -W/-H (shared by flux.2 and sd1.5) ──
    ratio = getattr(args, "ratio", None)
    quality = getattr(args, "quality", None)
    if ratio and quality:
        w, h = _resolve_video_dims(ratio, quality)
        args.width = w
        args.height = h
    elif ratio or quality:
        _emit("ERROR: --ratio and --quality must be used together", "error")
        sys.exit(1)

    # ── Parse --controlnet (shared by flux.2 and sd1.5) ───────────────────
    cn_mode, cn_path = None, None
    controlnet = getattr(args, "controlnet", None)
    if controlnet:
        cn_mode, cn_path = _parse_controlnet(controlnet)

    # ── Generation engines ────────────────────────────────────────────────
    if engine == "sd1.5":
        return _image_sd15(args, cn_mode=cn_mode, cn_path=cn_path)

    if engine != "flux.2":
        _emit(f"ERROR: Unknown image engine '{engine}'", "error")
        sys.exit(1)

    # ── FLUX.2 generation ─────────────────────────────────────────────────
    if not IMAGE_WORKER.exists():
        _emit("ERROR: worker/image/generate.py not found", "error")
        _emit("  Run: bash worker/image/install.sh", "error")
        sys.exit(1)

    # Resolve model
    model_key = getattr(args, "model", None) or "4b"
    flux2_model = _IMAGE_MODEL_MAP.get(model_key)
    if not flux2_model:
        valid = ", ".join(_IMAGE_MODEL_MAP.keys())
        _emit(f"ERROR: Unknown model '{model_key}'. Valid: {valid}", "error")
        sys.exit(1)

    prompt = getattr(args, "prompt", None)
    if not prompt:
        _emit("ERROR: --prompt required for image generation", "error")
        sys.exit(1)

    # Output path
    out_path = Path(args.output) if args.output else Path("image.png")
    if out_path.is_dir() or str(out_path).endswith("/"):
        out_path = out_path / f"image_{int(time.time())}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        str(CONDA_BIN), "run", "--no-capture-output", "-n", FLUX2_ENV,
        "python", str(IMAGE_WORKER),
        "--model", flux2_model,
        "--prompt", prompt,
        "-o", str(out_path),
        "-W", str(getattr(args, "width", 1360)),
        "-H", str(getattr(args, "height", 768)),
    ]

    seed = getattr(args, "seed", None)
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    steps = getattr(args, "steps", None)
    if steps is not None:
        cmd.extend(["--steps", str(steps)])

    guidance = getattr(args, "cfg_scale", None)
    if guidance is not None:
        cmd.extend(["--guidance", str(guidance)])

    images = getattr(args, "images", None)
    if images:
        for img in images:
            if not Path(img).exists():
                _emit(f"ERROR: Image not found: {img}", "error")
                sys.exit(1)
        cmd.extend(["--images"] + images)

    # ControlNet conditioning → passed to worker as --controlnet-image/--controlnet-mode
    if cn_mode and cn_path:
        cmd.extend(["--controlnet-image", cn_path, "--controlnet-mode", cn_mode])

    if getattr(args, "no_rescale", False):
        cmd.append("--no-rescale")

    _emit(f"Generating image (flux.2/{model_key}) …", "stage")

    result = run_worker(cmd, on_event=_event_handler)
    finish_progress()
    if result.returncode != 0:
        _emit("ERROR: Image generation failed.", "error")
        sys.exit(1)

    if result.stdout.strip():
        print(result.stdout.strip())



# ── Video: dimension lookup ──────────────────────────────────────────────────

# Vorberechnete Dimensionen für alle ratio+quality Kombinationen.
# Alle Werte sind /128-kompatibel (width//2 und height//2 jeweils /32 und gerade).
# Gilt für alle Worker die --ratio/--quality akzeptieren.
# Generiert von tests/values.py.
# (width//2 und height//2 sind jeweils /32 teilbar und gerade — VAE-Anforderung)
_VIDEO_DIMS: dict[tuple[str, str], tuple[int, int]] = {
    ("16:9", "240p"): (256, 128),
    ("9:16", "240p"): (128, 256),
    ("21:9", "240p"): (256, 128),
    ("9:21", "240p"): (128, 256),
    ("3:2",  "240p"): (384, 256),
    ("2:3",  "240p"): (256, 384),
    ("4:3",  "240p"): (384, 256),
    ("3:4",  "240p"): (256, 384),
    ("4:5",  "240p"): (256, 384),
    ("5:4",  "240p"): (384, 384),
    ("1:1",  "240p"): (384, 384),
    ("1:2",  "240p"): (128, 256),
    ("2:1",  "240p"): (256, 128),
    ("16:9", "360p"): (512, 256),
    ("9:16", "360p"): (256, 512),
    ("21:9", "360p"): (512, 256),
    ("9:21", "360p"): (256, 512),
    ("3:2",  "360p"): (384, 256),
    ("2:3",  "360p"): (256, 384),
    ("4:3",  "360p"): (512, 384),
    ("3:4",  "360p"): (384, 512),
    ("4:5",  "360p"): (384, 512),
    ("5:4",  "360p"): (512, 384),
    ("1:1",  "360p"): (512, 512),
    ("1:2",  "360p"): (256, 512),
    ("2:1",  "360p"): (512, 256),
    ("16:9", "480p"): (640, 384),
    ("9:16", "480p"): (384, 640),
    ("21:9", "480p"): (640, 256),
    ("9:21", "480p"): (256, 640),
    ("3:2",  "480p"): (384, 256),
    ("2:3",  "480p"): (256, 384),
    ("4:3",  "480p"): (512, 384),
    ("3:4",  "480p"): (384, 512),
    ("4:5",  "480p"): (512, 640),
    ("5:4",  "480p"): (640, 512),
    ("1:1",  "480p"): (640, 640),
    ("1:2",  "480p"): (256, 512),
    ("2:1",  "480p"): (512, 256),
    ("16:9", "720p"):  (1152, 640),
    ("9:16", "720p"):  (640, 1152),
    ("21:9", "720p"):  (896, 384),
    ("9:21", "720p"):  (384, 896),
    ("3:2",  "720p"):  (1152, 768),
    ("2:3",  "720p"):  (768, 1152),
    ("4:3",  "720p"):  (1024, 768),
    ("3:4",  "720p"):  (768, 1024),
    ("4:5",  "720p"):  (1024, 1280),
    ("5:4",  "720p"):  (1280, 1024),
    ("1:1",  "720p"):  (1280, 1280),
    ("1:2",  "720p"):  (640, 1280),
    ("2:1",  "720p"):  (1280, 640),
    ("16:9", "1080p"): (1152, 640),
    ("9:16", "1080p"): (640, 1152),
    ("21:9", "1080p"): (1792, 768),
    ("9:21", "1080p"): (768, 1792),
    ("3:2",  "1080p"): (1920, 1280),
    ("2:3",  "1080p"): (1280, 1920),
    ("4:3",  "1080p"): (1536, 1152),
    ("3:4",  "1080p"): (1152, 1536),
    ("4:5",  "1080p"): (1536, 1920),
    ("5:4",  "1080p"): (1920, 1536),
    ("1:1",  "1080p"): (1920, 1920),
    ("1:2",  "1080p"): (896, 1792),
    ("2:1",  "1080p"): (1792, 896),
    ("16:9", "1440p"): (2048, 1152),
    ("9:16", "1440p"): (1152, 2048),
    ("21:9", "1440p"): (1792, 768),
    ("9:21", "1440p"): (768, 1792),
    ("3:2",  "1440p"): (2304, 1536),
    ("2:3",  "1440p"): (1536, 2304),
    ("4:3",  "1440p"): (2560, 1920),
    ("3:4",  "1440p"): (1920, 2560),
    ("4:5",  "1440p"): (2048, 2560),
    ("5:4",  "1440p"): (2560, 2048),
    ("1:1",  "1440p"): (2560, 2560),
    ("1:2",  "1440p"): (1280, 2560),
    ("2:1",  "1440p"): (2560, 1280),
    ("16:9", "2160p"): (2048, 1152),
    ("9:16", "2160p"): (1152, 2048),
    ("21:9", "2160p"): (3584, 1536),
    ("9:21", "2160p"): (1536, 3584),
    ("3:2",  "2160p"): (3840, 2560),
    ("2:3",  "2160p"): (2560, 3840),
    ("4:3",  "2160p"): (3584, 2688),
    ("3:4",  "2160p"): (2688, 3584),
    ("4:5",  "2160p"): (3072, 3840),
    ("5:4",  "2160p"): (3840, 3072),
    ("1:1",  "2160p"): (3840, 3840),
    ("1:2",  "2160p"): (1920, 3840),
    ("2:1",  "2160p"): (3840, 1920),
    ("16:9", "4k"):    (4096, 2304),
    ("9:16", "4k"):    (2304, 4096),
    ("21:9", "4k"):    (3584, 1536),
    ("9:21", "4k"):    (1536, 3584),
    ("3:2",  "4k"):    (3840, 2560),
    ("2:3",  "4k"):    (2560, 3840),
    ("4:3",  "4k"):    (4096, 3072),
    ("3:4",  "4k"):    (3072, 4096),
    ("4:5",  "4k"):    (3072, 3840),
    ("5:4",  "4k"):    (3840, 3072),
    ("1:1",  "4k"):    (4096, 4096),
    ("1:2",  "4k"):    (2048, 4096),
    ("2:1",  "4k"):    (4096, 2048),
}


_VIDEO_RATIOS = list(dict.fromkeys(r for r, _ in _VIDEO_DIMS))
_VIDEO_QUALITIES = list(dict.fromkeys(q for _, q in _VIDEO_DIMS))


def _resolve_video_dims(ratio_str: str, quality_str: str) -> tuple[int, int]:
    """Resolve --ratio + --quality to (width, height) via _VIDEO_DIMS lookup."""
    if (ratio_str, quality_str) not in _VIDEO_DIMS:
        _emit(f"ERROR: Unknown combination '{ratio_str}' + '{quality_str}'. "
              f"Ratios: {', '.join(_VIDEO_RATIOS)}. "
              f"Qualities: {', '.join(_VIDEO_QUALITIES)}", "error")
        sys.exit(1)
    return _VIDEO_DIMS[(ratio_str, quality_str)]


# ── Video generation ──────────────────────────────────────────────────────────

def cmd_video(args):
    """Generate video with LTX-2.3."""
    engine = getattr(args, "engine", "ltx2.3")

    if engine != "ltx2.3":
        _emit(f"ERROR: Unknown video engine '{engine}'", "error")
        sys.exit(1)

    if not LTX2_WORKER.exists():
        _emit("ERROR: worker/ltx2/generate.py not found", "error")
        _emit("  Run: bash worker/ltx2/install.sh", "error")
        sys.exit(1)

    prompt = getattr(args, "prompt", None)
    if not prompt:
        _emit("ERROR: --prompt required for video generation", "error")
        sys.exit(1)

    # Resolve dimensions: --ratio + --quality override -W/-H
    ratio = getattr(args, "ratio", None)
    quality = getattr(args, "quality", None)
    if ratio and quality:
        width, height = _resolve_video_dims(ratio, quality)
    elif ratio or quality:
        _emit("ERROR: --ratio and --quality must be used together", "error")
        sys.exit(1)
    else:
        width = getattr(args, "width", 768)
        height = getattr(args, "height", 512)

    # Output path
    out_path = Path(args.output) if args.output else Path("video.mp4")
    if out_path.is_dir() or str(out_path).endswith("/"):
        out_path = out_path / f"video_{int(time.time())}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Override num_frames from audio length if --audio is set
    audio_path = getattr(args, "audio", None)
    num_frames = getattr(args, "num_frames", 121)
    if audio_path and num_frames == 121:
        import subprocess as _sp, json as _json
        _probe = _sp.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", audio_path],
            capture_output=True, text=True,
        )
        for _s in _json.loads(_probe.stdout).get("streams", []):
            if _s.get("codec_type") == "audio":
                _dur = float(_s["duration"])
                _fps = getattr(args, "frame_rate", 24)
                _raw = int(_dur * _fps)
                num_frames = ((_raw // 8) * 8) + 1
                break

    # Build command
    model = getattr(args, "model", None) or "distilled"
    engine = getattr(args, "engine", "ltx2.3")

    cmd = [
            str(CONDA_BIN), "run", "--no-capture-output", "-n", LTX2_ENV,
            "python", str(LTX2_WORKER),
            "--model", model,
            "--prompt", prompt,
            "-o", str(out_path),
            "-W", str(width),
            "-H", str(height),
            "--frame-rate", str(getattr(args, "frame_rate", 24)),
        ]

    cmd.extend(["--num-frames", str(num_frames)])

    seed = getattr(args, "seed", None)
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    steps = getattr(args, "steps", None)
    if steps is not None:
        cmd.extend(["--steps", str(steps)])

    cfg_scale = getattr(args, "cfg_scale", None)
    if cfg_scale is not None:
        cmd.extend(["--cfg-scale", str(cfg_scale)])

    negative_prompt = getattr(args, "negative_prompt", None)
    if negative_prompt is not None:
        cmd.extend(["--negative-prompt", negative_prompt])

    # Image conditioning: flexible --image PATH FRAME_IDX STRENGTH
    images = getattr(args, "images", None)
    if images:
        for img_args in images:
            path, frame_idx, strength = img_args
            if not Path(path).exists():
                _emit(f"ERROR: Image not found: {path}", "error")
                sys.exit(1)
            cmd.extend(["--image", path, frame_idx, strength])

    # Convenience shortcuts
    image_first = getattr(args, "image_first", None)
    if image_first:
        if not Path(image_first).exists():
            _emit(f"ERROR: Image not found: {image_first}", "error")
            sys.exit(1)
        cmd.extend(["--image-first", image_first])

    image_mid = getattr(args, "image_mid", None)
    if image_mid:
        if not Path(image_mid).exists():
            _emit(f"ERROR: Image not found: {image_mid}", "error")
            sys.exit(1)
        cmd.extend(["--image-mid", image_mid])

    image_last = getattr(args, "image_last", None)
    if image_last:
        if not Path(image_last).exists():
            _emit(f"ERROR: Image not found: {image_last}", "error")
            sys.exit(1)
        cmd.extend(["--image-last", image_last])

    # LoRA
    loras = getattr(args, "lora", None)
    if loras:
        for lora_args in loras:
            lora_path = lora_args[0]
            if not Path(lora_path).exists():
                _emit(f"ERROR: LoRA not found: {lora_path}", "error")
                sys.exit(1)
            cmd.extend(["--lora"] + lora_args)

    # Audio input (A2V pipeline)
    audio = getattr(args, "audio", None)
    if audio:
        if not Path(audio).exists():
            _emit(f"ERROR: Audio file not found: {audio}", "error")
            sys.exit(1)
        cmd.extend(["--audio", audio])

    if getattr(args, "enhance_prompt", False):
        cmd.append("--enhance-prompt")


    # Extend / Retake / Clone modes
    extend = getattr(args, "extend", None)
    retake = getattr(args, "retake", None)
    clone = getattr(args, "clone", None)

    if clone:
        if not Path(clone).exists():
            _emit(f"ERROR: Video not found: {clone}", "error")
            sys.exit(1)
        clone_seconds = str(getattr(args, "seconds", 5.0) or 5.0)
        cmd.extend(["--clone", clone, clone_seconds])
        ref_seconds = getattr(args, "ref_seconds", None)
        if ref_seconds is not None:
            cmd.extend(["--ref-seconds", str(ref_seconds)])

    if extend:
        video_file, seconds = extend
        if not Path(video_file).exists():
            _emit(f"ERROR: Video not found: {video_file}", "error")
            sys.exit(1)
        cmd.extend(["--extend", video_file, seconds])
        ref_seconds = getattr(args, "ref_seconds", None)
        if ref_seconds is not None:
            cmd.extend(["--ref-seconds", str(ref_seconds)])
    if retake:
        video_file, start, end = retake
        if not Path(video_file).exists():
            _emit(f"ERROR: Video not found: {video_file}", "error")
            sys.exit(1)
        cmd.extend(["--retake", video_file, start, end])

    if clone:
        _emit(f"Cloning video ({clone_seconds}s new, ltx2.3/{model}) …", "stage")
    elif extend:
        _emit(f"Extending video by {extend[1]}s (ltx2.3/{model}) …", "stage")
    elif retake:
        _emit(f"Retaking {retake[1]}s–{retake[2]}s (ltx2.3/{model}) …", "stage")
    else:
        _emit(f"Generating video (ltx2.3/{model}) …", "stage")

    result = run_worker(cmd, on_event=_event_handler)
    finish_progress()
    if result.returncode != 0:
        _emit("ERROR: Video generation failed.", "error")
        sys.exit(1)

    if result.stdout.strip():
        print(result.stdout.strip())


# ── Stub handler for future mediums ──────────────────────────────────────────

def _stub_medium(args):
    _emit(f"'{args.medium}' is not yet implemented.", "error")
    sys.exit(1)


# ── CLI Parser ───────────────────────────────────────────────────────────────

class _HelpOnErrorParser(argparse.ArgumentParser):
    """Show full help instead of cryptic error on invalid usage."""
    def error(self, message):
        print(f"Error: {message}\n", file=sys.stderr)
        self.print_help(sys.stderr)
        sys.exit(2)


def build_parser():
    parser = _HelpOnErrorParser(
        prog="generate",
        description="Unified media generation toolkit — audio, text, image, video.",
    )

    sub = parser.add_subparsers(dest="medium")

    # ── voice ─────────────────────────────────────────────────────────────
    p_voice = sub.add_parser("voice", help="Voice conversion and TTS")
    p_voice.add_argument("engine",
                       choices=["rvc", "say", "ai-tts", "clone-tts"],
                       help="Voice engine")
    p_voice.add_argument("input", nargs="*", help="[rvc] Input audio file(s)")
    p_voice.add_argument("--model", dest="voice", help="[rvc] Voice/RVC model name")
    p_voice.add_argument("-o", "--output", help="Output directory")
    # text input (say/ai-tts, shared alias with audio --lyrics)
    voice_text_grp = p_voice.add_mutually_exclusive_group()
    voice_text_grp.add_argument("--text", "--lyrics", "-l", dest="text",
                                help="[say/ai-tts] Inline text to speak")
    voice_text_grp.add_argument("--text-file", "--lyrics-file", "-f",
                                dest="text_file",
                                help="[say/ai-tts] Path to text file")
    voice_text_grp.add_argument("--prompt-file", "-pf", dest="prompt_file",
                                help="[ai-tts] Load text + params from prompt sidecar (.txt)")
    # say/ai-tts shared
    p_voice.add_argument("-v", "--voice", dest="say_voice", default=None,
                       help="[say] macOS voice / [ai-tts] Preset voice (Aiden, Serena, ...)")
    p_voice.add_argument("--rate", type=int, default=None,
                       help="[say] Speaking rate in words per minute")
    # ai-tts specific
    p_voice.add_argument("--tags", "-t", dest="tags", default=None,
                       help="[ai-tts] Style instructions ('dramatic, slow, whispering')")
    p_voice.add_argument("--tts-model", dest="tts_model", default=None,
                       choices=["large", "small"],
                       help="[ai-tts] Model size (default: large = 1.7B)")
    p_voice.add_argument("--language", dest="language", default=None,
                       help="[ai-tts/clone-tts] ISO code: de, en, fr, ja, … (auto-detected if omitted)")
    p_voice.add_argument("--list-voices", action="store_true",
                       help="[ai-tts] Show available voices")
    # rvc-specific
    p_voice.add_argument("--decoder", choices=["rmvpe", "crepe", "harvest", "pm"],
                       help="[rvc] Pitch detection algorithm (default: rmvpe)")
    p_voice.add_argument("--pitch", type=int,
                       help="[rvc] Manual pitch shift in semitones (disables auto-pitch)")
    p_voice.add_argument("--target-hz", type=float,
                       help="[rvc] Target voice pitch in Hz (overrides model config)")
    # clone-tts specific
    p_voice.add_argument("--reference", type=str, action="append", default=None,
                       help="[clone-tts] Reference audio (3-10s WAV of voice to clone, repeatable)")
    p_voice.add_argument("--ref-text", type=str, default=None,
                       help="[clone-tts] Text spoken in reference audio (auto-transcribed if omitted)")
    p_voice.set_defaults(func=cmd_voice)

    # ── audio ─────────────────────────────────────────────────────────────
    p_audio = sub.add_parser("audio",
                             help="Audio processing and generation")
    p_audio.add_argument("engine",
                         choices=["enhance", "demucs", "ace-step", "heartmula", "diarize", "sfx", "voice-removal", "ltx2.3"],
                         help="Audio engine")
    p_audio.add_argument("input", nargs="*", help="Input audio file(s)")
    p_audio.add_argument("--model", default=None,
                         help="Model variant: [demucs] htdemucs/htdemucs_ft, "
                              "[ace-step] turbo/sft/base")
    p_audio.add_argument("-o", "--output", help="Output path or directory")
    # enhance
    p_audio.add_argument("--denoise-only", action="store_true",
                         help="[enhance] Only denoise, skip super-resolution")
    p_audio.add_argument("--enhance-only", action="store_true",
                         help="[enhance] Only super-resolution, skip denoising")
    # music shared (ace-step + heartmula)
    lyrics_grp = p_audio.add_mutually_exclusive_group()
    lyrics_grp.add_argument("--lyrics", "--text", "-l", dest="text",
                            help="[ace-step/heartmula] Inline lyrics/text")
    lyrics_grp.add_argument("--lyrics-file", "--text-file", "-f", dest="text_file",
                            help="[ace-step/heartmula] Path to lyrics/text file")
    p_audio.add_argument("--tags", "-t",
                         help="[ace-step/heartmula] Style tags (e.g. 'disco,happy')")
    p_audio.add_argument("--seconds", "-s", type=int, default=20,
                         help="[ace-step/heartmula] Duration in seconds (default: 20)")
    p_audio.add_argument("--duration", type=int,
                         help="[ace-step/heartmula] Duration in ms (overrides --seconds)")
    p_audio.add_argument("--seed", type=int, default=None,
                         help="[ace-step/heartmula] Random seed")
    p_audio.add_argument("--top-k", type=int, default=None, dest="top_k",
                         help="[ace-step/heartmula] Top-k sampling")
    p_audio.add_argument("--temperature", type=float, default=None,
                         help="[ace-step/heartmula] Sampling temperature")
    p_audio.add_argument("--cfg-scale", type=float, default=None,
                         help="[ace-step/heartmula] CFG scale")
    # ace-step specific
    p_audio.add_argument("--steps", type=int, default=None,
                         help="[ace-step] Inference steps")
    p_audio.add_argument("--shift", type=float, default=None,
                         help="[ace-step] Timestep shift")
    p_audio.add_argument("--no-thinking", action="store_true",
                         help="[ace-step] Disable LM chain-of-thought")
    p_audio.add_argument("--infer-method", choices=["ode", "sde"], default=None,
                         help="[ace-step] Inference method")
    p_audio.add_argument("--lm-cfg", type=float, default=None,
                         help="[ace-step] LM guidance scale")
    p_audio.add_argument("--top-p", type=float, default=None,
                         help="[ace-step] Nucleus sampling")
    p_audio.add_argument("--batch-size", type=int, default=None,
                         help="[ace-step] Parallel samples")
    p_audio.add_argument("--instrumental", action="store_true",
                         help="[ace-step] Force instrumental output")
    p_audio.add_argument("--language", type=str, default=None,
                         help="[ace-step] Vocal language code (e.g. 'en', 'zh', 'ja', 'de')")
    p_audio.add_argument("--bpm", type=int, default=None,
                         help="[ace-step/heartmula] Beats per minute")
    p_audio.add_argument("--keyscale", type=str, default=None,
                         help="[ace-step/heartmula] Musical key (e.g. 'C Major')")
    p_audio.add_argument("--timesignature", type=str, default=None,
                         help="[ace-step/heartmula] Time signature")
    # diarize
    p_audio.add_argument("--speakers", type=int, default=None,
                         help="[diarize] Number of speakers (auto-detect if omitted)")
    p_audio.add_argument("--hf-token", default=None,
                         help="[diarize] HuggingFace token (or HF_TOKEN env var)")
    p_audio.add_argument("--verify", action="store_true",
                         help="[diarize] Verify diarization quality via transcription")
    # voice-removal
    p_audio.add_argument("--tmp-dir", type=str, default=None,
                         help="[voice-removal] Directory for temp stems (default: /tmp)")
    p_audio.set_defaults(func=cmd_audio)

    # ── text ──────────────────────────────────────────────────────────────
    p_text = sub.add_parser("text",
                            help="Text extraction / LLM inference")
    p_text.add_argument("engine",
                        choices=["whisper", "heartmula-transcribe",
                                 "ollama"],
                        help="Text engine")
    p_text.add_argument("input", nargs="*", help="Input audio file(s) [whisper]")
    p_text.add_argument("--model", default=None,
                        help="Model name")
    p_text.add_argument("--language", default=None,
                        help="[whisper] Language hint (e.g. 'en', 'de')")
    p_text.add_argument("--word-timestamps", action="store_true",
                        help="[whisper] Include word-level timestamps")
    p_text.add_argument("--format", default="json",
                        help="[whisper] Output format: json/txt/srt/vtt/tsv/all")
    p_text.add_argument("-o", "--output",
                        help="Output directory or file")
    # LLM-specific flags
    p_text.add_argument("--endpoint",
                        choices=["chat", "generate", "set", "show",
                                 "reset", "load", "unload"],
                        help="[llm] API endpoint")
    p_text.add_argument("--prompt", default=None,
                        help="[llm] Text prompt (for endpoint=generate)")
    p_text.add_argument("--system", default=None,
                        help="[llm] System prompt (for endpoint=generate)")
    p_text.add_argument("--messages", default=None,
                        help="[llm] Messages JSON string or file path (for endpoint=chat)")
    p_text.add_argument("--context-length", type=int, default=None,
                        dest="context_length",
                        help="[llm] Max context window tokens")
    p_text.add_argument("--max-tokens", type=int, default=None,
                        dest="max_tokens",
                        help="[llm] Max output tokens")
    p_text.add_argument("--temperature", type=float, default=None,
                        help="[llm] Sampling temperature")
    p_text.add_argument("--top-p", type=float, default=None, dest="top_p",
                        help="[llm] Top-p (nucleus) sampling")
    p_text.add_argument("--top-k", type=int, default=None, dest="top_k",
                        help="[llm] Top-k sampling")
    p_text.add_argument("--repeat-penalty", type=float, default=None,
                        dest="repeat_penalty",
                        help="[llm] Repetition penalty")
    p_text.add_argument("--seed", type=int, default=None,
                        help="[llm] Random seed")
    p_text.add_argument("--stop", default=None,
                        help="[llm] Stop sequence")
    p_text.add_argument("--images", nargs="+", default=None,
                        help="[llm] Image files for vision models (png, jpg)")
    p_text.add_argument("--thinking", default="False",
                        choices=["True", "False", "low", "medium", "high"],
                        help="[llm] Thinking mode (default: False)")
    p_text.add_argument("--stream", action="store_true",
                        help="[llm] Stream output")
    p_text.add_argument("--base-url", default=None, dest="base_url",
                        help="[llm] Override engine base URL")
    p_text.add_argument("--api-key", default=None, dest="api_key",
                        help="[llm] API key (fallback: env var)")
    p_text.set_defaults(func=cmd_text)

    # ── output ────────────────────────────────────────────────────────────
    p_output = sub.add_parser("output",
                              help="Post-processing: concatenation, mixing")
    p_output.add_argument("engine",
                          choices=["audio-concatenate", "audio-mucs"],
                          help="Output engine")
    p_output.add_argument("input", nargs="*", help="Input files")
    p_output.add_argument("-o", "--output", help="Output file path")
    p_output.add_argument("--output-bitrate", default=None,
                          help="Audio bitrate (e.g. '192k', '320k')")
    p_output.add_argument("--clip", action="append", default=None,
                          help="Per-clip options: INDEX:key=val,key=val "
                               "(keys: fade-in, fade-out, crossfade, volume, start, end, pan)")
    p_output.set_defaults(func=cmd_output)

    # ── Stubs for future mediums ──────────────────────────────────────────
    # ── image ─────────────────────────────────────────────────────────────
    p_image = sub.add_parser("image", help="Image generation & processing")
    p_image.add_argument("engine",
                          choices=["flux.2", "openpose", "sd1.5", "depth",
                                   "lineart", "normalmap", "sketch",
                                   "upscale", "segment"],
                          help="Image engine")
    p_image.add_argument("--model", "-m", default=None,
                          help="Model: flux.2: 4b|4b-distilled|9b|9b-distilled; mm: mm (default)")
    p_image.add_argument("--prompt", "-p", default=None, help="Text prompt")
    p_image.add_argument("-o", "--output", default=None, help="Output file path")
    p_image.add_argument("--seed", type=int, default=None, help="Random seed")
    p_image.add_argument("--steps", type=int, default=None, help="Inference steps")
    p_image.add_argument("--cfg-scale", type=float, default=None, dest="cfg_scale",
                          help="Guidance scale")
    p_image.add_argument("--ratio", default=None,
                          choices=_VIDEO_RATIOS,
                          help="Aspect ratio (e.g. 16:9, 1:1). Requires --quality")
    p_image.add_argument("--quality", default=None,
                          choices=_VIDEO_QUALITIES,
                          help="Quality tier (e.g. 480p, 720p). Requires --ratio")
    p_image.add_argument("-W", "--width", type=int, default=1360, help="Image width (default: 1360)")
    p_image.add_argument("-H", "--height", type=int, default=768, help="Image height (default: 768)")
    p_image.add_argument("--images", nargs="+", default=None,
                          help="Reference image path(s), up to 10")
    p_image.add_argument("--pose-mode", default=None, dest="pose_mode",
                          choices=["wholebody", "body", "bodyhand", "bodyface"],
                          help="[openpose] Detection mode (default: wholebody)")
    p_image.add_argument("--output-layer", default=None, dest="output_layer",
                          choices=["foreground", "background", "both"],
                          help="[segment] Output layer: foreground (default), background, both")
    p_image.add_argument("--controlnet", default=None,
                          help="[flux.2/sd1.5] Conditioning: mode:filepath (e.g. depth:depth.png, pose:pose.png, lineart:lines.png, normalmap:normals.png, sketch:sketch.png)")
    p_image.add_argument("--negative-prompt", default=None, dest="negative_prompt",
                          help="[sd1.5] Negative prompt")
    p_image.add_argument("--lora", action="append", default=None,
                          help="[sd1.5] LoRA: name:intensity (e.g. add_detail:1.2). Repeatable.")
    p_image.add_argument("--no-lora", action="store_true", dest="no_lora",
                          help="[sd1.5] Disable default LoRA")
    p_image.add_argument("--no-rescale", action="store_true", dest="no_rescale",
                          help="[flux.2] Pass reference images in original resolution (skip Pan & Scan)")
    p_image.set_defaults(func=cmd_image)

    # ── video ──────────────────────────────────────────────────────────────
    p_video = sub.add_parser("video", help="Video generation")
    p_video.add_argument("engine", nargs="?", default="ltx2.3",
                          choices=["ltx2.3"],
                          help="Video engine (default: ltx2.3)")
    p_video.add_argument("--model", "-m", default=None,
                          help="Model variant: distilled (default), dev")
    p_video.add_argument("--prompt", "-p", default=None, help="Text prompt")
    p_video.add_argument("-o", "--output", default=None, help="Output file path")
    p_video.add_argument("--seed", type=int, default=None, help="Random seed")
    p_video.add_argument("--steps", type=int, default=None, help="Inference steps")
    p_video.add_argument("--cfg-scale", type=float, default=None, dest="cfg_scale",
                          help="Video CFG guidance scale")
    p_video.add_argument("-W", "--width", type=int, default=768,
                          help="Video width (default: 768)")
    p_video.add_argument("-H", "--height", type=int, default=512,
                          help="Video height (default: 512)")
    p_video.add_argument("--ratio", default=None,
                          choices=_VIDEO_RATIOS,
                          help="Aspect ratio (e.g. 16:9, 1:1). Requires --quality")
    p_video.add_argument("--quality", default=None,
                          choices=_VIDEO_QUALITIES,
                          help="Quality tier (e.g. 480p, 720p). Requires --ratio")
    p_video.add_argument("--num-frames", type=int, default=121, dest="num_frames",
                          help="Frames, must be 8k+1 (default: 121 = ~5s at 24fps)")
    p_video.add_argument("--frame-rate", type=int, default=24, dest="frame_rate",
                          help="FPS (default: 24)")
    p_video.add_argument("--negative-prompt", default=None, dest="negative_prompt",
                          help="Negative prompt")
    p_video.add_argument("--image", dest="images", action="append", nargs=3,
                          metavar=("PATH", "FRAME_IDX", "STRENGTH"),
                          help="Image conditioning: PATH FRAME_IDX STRENGTH (repeatable)")
    p_video.add_argument("--image-first", dest="image_first", default=None,
                          help="Conditioning image for first frame (strength 1.0)")
    p_video.add_argument("--image-mid", dest="image_mid", default=None,
                          help="Conditioning image for middle frame (strength 1.0)")
    p_video.add_argument("--image-last", dest="image_last", default=None,
                          help="Conditioning image for last frame (strength 1.0)")
    p_video.add_argument("--lora", action="append", nargs="+",
                          metavar=("PATH", "STRENGTH"),
                          help="LoRA: PATH [STRENGTH] (repeatable)")
    p_video.add_argument("--audio", default=None,
                          help="Audio file for audio-to-video generation")
    p_video.add_argument("--enhance-prompt", action="store_true", dest="enhance_prompt",
                          help="Auto-enhance prompt via Gemma")
    p_video.add_argument("--extend", nargs=2, metavar=("VIDEO", "SECONDS"),
                          help="Extend an existing video by N seconds")
    p_video.add_argument("--clone", metavar="VIDEO",
                          help="Clone: generate new video using reference as visual context")
    p_video.add_argument("--seconds", "-s", type=float, default=5.0,
                          help="[clone] Output duration in seconds (default: 5)")
    p_video.add_argument("--retake", nargs=3, metavar=("VIDEO", "START", "END"),
                          help="Retake a time region (start/end in seconds)")
    p_video.add_argument("--ref-seconds", type=float, default=None, dest="ref_seconds",
                          help="Context seconds from source (default: 2 for extend, 5 for clone)")
    p_video.add_argument("--fp16", action="store_true",
                          help="[MLX] Use FP16 precision (~50%% less memory)")
    p_video.set_defaults(func=cmd_video)

    # ── Stubs for future mediums ──────────────────────────────────────────
    for stub_name in ["translation", "comparison"]:
        p_stub = sub.add_parser(stub_name, help=f"{stub_name.title()} (coming soon)")
        p_stub.set_defaults(func=_stub_medium, medium=stub_name)

    # ── ps ────────────────────────────────────────────────────────────────
    p_ps = sub.add_parser("ps", help="Show available models and status")
    p_ps.set_defaults(func=cmd_ps)

    # ── server ────────────────────────────────────────────────────────────
    p_server = sub.add_parser("server", help="Manage RVC worker")
    s_sub = p_server.add_subparsers(dest="server_cmd")

    p_ss = s_sub.add_parser("start", help="Start RVC worker")
    p_ss.add_argument("-pt", "--port", type=int, default=5100)
    p_ss.set_defaults(func=cmd_server_start)

    p_st = s_sub.add_parser("stop", help="Stop RVC worker")
    p_st.set_defaults(func=cmd_server_stop)

    p_su = s_sub.add_parser("status", help="Check worker status")
    p_su.set_defaults(func=cmd_server_status)

    # ── models ────────────────────────────────────────────────────────────
    p_models = sub.add_parser("models", help="Manage models")
    p_models.add_argument("engine", nargs="?",
                          choices=["rvc", "ollama", "huggingface"],
                          default=None,
                          help="Engine (required for all except list)")
    m_sub = p_models.add_subparsers(dest="models_cmd")

    p_ml = m_sub.add_parser("list", help="List installed models")
    p_ml.set_defaults(models_func=cmd_models_list)

    p_ms = m_sub.add_parser("search", help="Search for models")
    p_ms.add_argument("query", help="Search terms")
    p_ms.add_argument("--limit", type=int, default=30)
    p_ms.set_defaults(models_func=cmd_models_search)

    p_mi = m_sub.add_parser("install", help="Install RVC model")
    p_mi.add_argument("model_id", help="HuggingFace repo ID or direct URL")
    p_mi.add_argument("--name", help="Local name for the model")
    p_mi.add_argument("--file", help="Specific file from multi-model repos")
    p_mi.set_defaults(models_func=cmd_models_install)

    p_mp = m_sub.add_parser("pull", help="Pull model (ollama, huggingface)")
    p_mp.add_argument("model_id", help="Model name/ID")
    p_mp.set_defaults(models_func=cmd_models_pull)

    p_msh = m_sub.add_parser("show", help="Show model details (ollama)")
    p_msh.add_argument("name", help="Model name")
    p_msh.set_defaults(models_func=cmd_models_show)

    p_mr = m_sub.add_parser("remove", help="Remove an installed model")
    p_mr.add_argument("name", help="Model name")
    p_mr.set_defaults(models_func=cmd_models_remove)

    p_mlo = m_sub.add_parser("load", help="Load model")
    p_mlo.add_argument("name", help="Model name")
    p_mlo.set_defaults(models_func=cmd_models_load)

    p_mu = m_sub.add_parser("unload", help="Unload model (ollama)")
    p_mu.add_argument("name", help="Model name")
    p_mu.set_defaults(models_func=cmd_models_unload)

    p_mc = m_sub.add_parser("calibrate",
                            help="Calibrate target F0 for RVC model")
    p_mc.add_argument("name", help="Model name")
    p_mc.set_defaults(models_func=cmd_models_calibrate)

    p_mf = m_sub.add_parser("set-pitch",
                            help="Set target pitch (Hz) for RVC model")
    p_mf.add_argument("name", help="Model name")
    p_mf.add_argument("hz", type=float,
                      help="Target pitch (male ~120, female ~220, child ~280)")
    p_mf.set_defaults(models_func=cmd_models_set_f0)

    p_models.set_defaults(func=cmd_models)

    return parser


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    global _event_handler

    # Extract --screen-log-format before argparse
    argv = list(sys.argv[1:])
    if "--screen-log-format" in argv:
        idx = argv.index("--screen-log-format")
        if idx + 1 < len(argv) and argv[idx + 1] == "json":
            _event_handler = print_event_json
        argv[idx:idx + 2] = []

    # ── Intercept "models list" at any position ──────────────────────────
    if "models" in argv:
        idx = argv.index("models")
        if idx + 1 < len(argv) and argv[idx + 1] == "list":
            medium = argv[0] if idx > 0 and argv[0] != "models" else None
            engine = argv[1] if idx > 1 else None
            _models_list_all(medium=medium, engine=engine)
            return

    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.medium:
        parser.print_help()
        return

    # ── Normalize text input: --text-file / --prompt-file → args.text ──
    text_file = getattr(args, "text_file", None)
    if text_file:
        p = Path(text_file)
        if not p.exists():
            _emit(f"ERROR: Text file not found: {p}", "error")
            sys.exit(1)
        args.text = p.read_text(encoding="utf-8")

    prompt_file = getattr(args, "prompt_file", None)
    if prompt_file:
        p = Path(prompt_file)
        if not p.exists():
            _emit(f"ERROR: Prompt file not found: {p}", "error")
            sys.exit(1)
        sidecar = _parse_prompt_sidecar(p)
        args.text = sidecar["text"]
        if not getattr(args, "say_voice", None) and sidecar.get("voice"):
            args.say_voice = sidecar["voice"]
        if not getattr(args, "language", None) and sidecar.get("language"):
            args.language = sidecar["language"]
        if not getattr(args, "tags", None) and sidecar.get("tags"):
            args.tags = sidecar["tags"]
        if not getattr(args, "tts_model", None) and sidecar.get("model"):
            args.tts_model = sidecar["model"]

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.parse_args([args.medium, "--help"])


if __name__ == "__main__":
    main()
