#!/usr/bin/env python3
"""
generate — Unified media generation toolkit.

Usage:
  python generate.py ps                                                Show available models & status
  python generate.py voice --engine ai-tts --text "Hello world" -o demos/ Neural TTS (Qwen3-TTS)
  python generate.py voice --engine ai-tts -v Serena --text "Hello"       TTS with preset voice
  python generate.py voice --engine ai-tts -v Aiden -t "dramatic" --text "Silence."  TTS with style
  python generate.py voice --engine ai-tts --tts-model small --text "Hi"  Smaller model (0.6B)
  python generate.py voice --engine ai-tts --text "[Aiden: excited] Hi! [Serena: calm] Hello."  Per-segment style
  python generate.py voice --engine ai-tts --prompt-file demos/speech.txt -o demos/  From sidecar
  python generate.py voice --engine ai-tts --list-voices                  Show available voices
  python generate.py voice --engine say --text "Hello world" -o demos/    macOS TTS
  python generate.py voice --engine say -v Anna --text "Hallo" -o demos/  TTS with specific voice
  python generate.py voice --engine say --model my-voice --text "Hallo"   TTS + RVC voice conversion
  python generate.py voice --engine rvc --model my-voice input.wav       Voice conversion
  python generate.py voice --engine rvc --model my-voice input.wav --pitch 12
  python generate.py voice --engine rvc --model my-voice input.wav --target-hz 280
  python generate.py voice --engine rvc --model my-voice input.wav --decoder crepe
  python generate.py audio --engine enhance input.wav                  Denoise + enhance audio
  python generate.py audio --engine enhance input.wav --denoise-only   Denoise only (faster)
  python generate.py audio --engine enhance input.wav --enhance-only   Super-resolution only
  python generate.py audio --engine demucs input.wav                   Separate into stems
  python generate.py audio --engine demucs input.wav --model htdemucs_ft
  python generate.py audio --engine ace-step -l "lyrics" -t "disco,happy" -o out.mp3
  python generate.py audio --engine ace-step --model sft -f lyrics.txt -t "cinematic"
  python generate.py audio --engine heartmula -l "lyrics" -t "disco,happy" -o out.mp3
  python generate.py audio --engine diarize interview.wav              Split dialogue by speaker
  python generate.py audio --engine diarize interview.wav --speakers 3
  python generate.py audio --engine diarize interview.wav --verify
  python generate.py text --engine whisper audio.wav                   Transcribe audio
  python generate.py text --engine whisper audio.wav --model large-v3 --format srt
  python generate.py text --engine whisper audio.wav --input-language de
  python generate.py text --engine heartmula-transcribe song.mp3       Extract lyrics
  python generate.py output --engine audio-concatenate a.wav b.mp3 -o out.wav  Concatenate audio files
  python generate.py output --engine audio-concatenate a.wav b.wav --output-bitrate 320k -o out.mp3
  python generate.py output --engine audio-concatenate a.wav b.wav c.mp3 --clip 0:fade-in=0.3 --clip 1:crossfade=0.5,volume=1.2 --clip 2:fade-out=0.5 -o out.mp3
  python generate.py models list                           List installed models
  python generate.py models search "neutral male"          Search HuggingFace
  python generate.py models install <id-or-url>            Download from HuggingFace or URL
  python generate.py models remove <name>                  Remove a model
  python generate.py models set-pitch <name> <hz>          Set target pitch manually
  python generate.py models calibrate <name>               Auto-detect target pitch
  python generate.py server start                          Start RVC worker
  python generate.py server stop                           Stop RVC worker
  python generate.py server status                         Check worker status
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
CONDA_BIN = Path(os.environ.get("CONDA_BIN", "/opt/miniconda3/bin/conda"))
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

    _emit("Starting RVC worker ...", "stage")
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
    if PID_FILE.exists():
        pid = int(PID_FILE.read_text().strip())
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            _emit(f"RVC worker stopped (PID {pid})")
        except ProcessLookupError:
            _emit("RVC worker was not running.")
        PID_FILE.unlink(missing_ok=True)
    else:
        _emit("No PID file found. Checking if server is running ...")
        if check_server():
            _emit("Server is running but PID unknown. Kill manually or use:")
            _emit("  lsof -ti:5100 | xargs kill")
        else:
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

def cmd_models_list(args):
    data = api_get("/models")
    models = data.get("models", [])
    if not models:
        _emit("No models installed.")
        _emit("  Search: python generate.py models search \"neutral male\"")
        _emit("  Install: python generate.py models install <hf-model-id>")
        return

    _emit(f"Installed models ({len(models)}):\n")
    for m in models:
        if isinstance(m, str):
            _emit(f"  {m}")
        else:
            name = m.get("name", m.get("model_name", str(m)))
            _emit(f"  {name}")


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
    _emit(f"Extracting {archive_path.name} ...", "stage")
    try:
        extracted = _extract_archive(archive_path, tmp_dir)
    except FileNotFoundError:
        if subprocess.run(["which", "brew"], capture_output=True).returncode != 0:
            _emit("'unar' not found and brew is not installed.", "error")
            sys.exit(1)
        _emit("Installing unar via brew ...", "stage")
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
        _emit("Downloading from URL ...", "stage")
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
            _emit(f"Installing '{name}' from {target_file} ...", "stage")

            if target_file.lower().endswith((".zip", ".rar", ".7z")):
                _emit(f"Downloading {target_file} ...", "stage")
                local_archive = Path(hf_hub_download(model_id, target_file))
                pth_path, idx_path = _install_from_archive(local_archive, name)
            elif target_file.endswith(".pth"):
                _emit(f"Downloading {target_file} ...", "stage")
                pth_path = Path(hf_hub_download(model_id, target_file))
                base = Path(target_file).stem
                matching_idx = [f for f in idx_files if base in f]
                if matching_idx:
                    _emit(f"Downloading {matching_idx[0]} ...", "stage")
                    idx_path = Path(hf_hub_download(model_id, matching_idx[0]))
            else:
                _emit("File must be .pth, .zip, .rar, or .7z", "error")
                sys.exit(1)

        elif len(model_files) > 1:
            total = len(model_files)
            _emit(f"Repo '{model_id}' contains {total} models — installing all ...", "stage")
            installed = []
            for j, mf in enumerate(sorted(model_files), 1):
                mf_name = _sanitize_model_name(Path(mf).stem)
                _emit(f"[{j}/{total}] Installing '{mf_name}' from {mf} ...", "stage")
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
            _emit(f"Installing '{name}' ...", "stage")
            _emit(f"Downloading {pth_file} ...", "stage")
            pth_path = Path(hf_hub_download(model_id, pth_file))
            if idx_files:
                _emit(f"Downloading {idx_files[0]} ...", "stage")
                idx_path = Path(hf_hub_download(model_id, idx_files[0]))

        elif archive_files:
            archive = archive_files[0]
            name = args.name or _sanitize_model_name(Path(archive).stem)
            _emit(f"Installing '{name}' ...", "stage")
            _emit(f"Downloading {archive} ...", "stage")
            local_archive = Path(hf_hub_download(model_id, archive))
            pth_path, idx_path = _install_from_archive(local_archive, name)

        else:
            _emit(f"No .pth or archive files found in {model_id}", "error")
            _emit(f"Files in repo: {', '.join(files[:10])}", "error")
            sys.exit(1)

    _emit("Uploading to RVC worker ...", "stage")
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
    _emit(f"Removing model '{args.name}' ...", "stage")
    try:
        r = requests.delete(f"{RVC_API_URL}/models/{args.name}", timeout=10)
        if r.ok:
            _emit(f"Removed '{args.name}'")
        else:
            _emit(f"API returned {r.status_code}: {r.text}", "error")
    except Exception as e:
        _emit(f"ERROR: {e}", "error")


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
                _emit("No target F0 for this model — calibrating automatically ...", "stage")
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

    _emit(f"Generating speech (Qwen3-TTS {tts_model}) ...", "stage")
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
    if result.stdout.strip():
        print(result.stdout.strip())

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
    _emit(f"  Prompt:   {prompt_path}")


def cmd_voice(args):
    """Voice conversion — dispatch by engine."""
    engine = args.engine
    if engine == "rvc":
        _tts_rvc(args)
    elif engine == "say":
        _voice_say(args)
    elif engine == "ai-tts":
        _voice_ai_tts(args)
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
    _emit(f"Enhancing {len(input_paths)} file(s) ({mode}) ...", "stage")

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

        _emit(f"Separating {p.name} ...", "stage")

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
        if args.topk is not None:
            cmd.extend(["--lm-top-k", str(args.topk)])
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

    _emit("Generating music (ACE-Step) ...", "stage")
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

    if args.topk:
        cmd.extend(["--topk", str(args.topk)])
    if args.temperature:
        cmd.extend(["--temperature", str(args.temperature)])
    if args.cfg_scale:
        cmd.extend(["--cfg-scale", str(args.cfg_scale)])
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])

    duration_s = duration_ms / 1000
    _emit("Generating music (HeartMuLa) ...", "stage")
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

        _emit(f"Diarizing {p.name} ...", "stage")

        result = run_worker(cmd, on_event=_event_handler)
        finish_progress()
        if result.returncode != 0:
            _emit("ERROR: Diarization failed.", "error")
            sys.exit(1)
        if result.stdout.strip():
            print(result.stdout.strip())


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
        cmd.extend(["--language", args.input_language])

    if getattr(args, "word_timestamps", False):
        cmd.append("--word-timestamps")

    fmt = getattr(args, "format", "json") or "json"
    cmd.extend(["--format", fmt])

    if args.output:
        out_path = Path(args.output).resolve()
        cmd.extend(["-o", str(out_path)])

    n_files = len(input_files)
    _emit(f"Transcribing {n_files} file{'s' if n_files > 1 else ''} ...", "stage")
    _emit(f"  Model: {model}")
    if getattr(args, "input_language", None):
        _emit(f"  Language: {args.input_language}")
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

    _emit("Transcribing lyrics ...", "stage")
    _emit(f"  Input: {audio_path}")

    result = run_worker(cmd, on_event=_event_handler)
    finish_progress()
    if result.returncode != 0:
        _emit("ERROR: Lyrics transcription failed.", "error")
        sys.exit(1)
    if result.stdout.strip():
        print(result.stdout.strip())


def cmd_text(args):
    """Text extraction — dispatch by engine."""
    engine = args.engine
    if engine == "whisper":
        _text_whisper(args)
    elif engine == "heartmula-transcribe":
        _text_heartmula_transcribe(args)
    else:
        _emit(f"ERROR: Unknown text engine: {engine}", "error")
        sys.exit(1)


# ── PS (Status) ──────────────────────────────────────────────────────────────

def cmd_ps(args):
    server_running = check_server()

    models_info = []
    if server_running:
        data = api_get("/models")
        for m in data.get("models", []):
            name = m if isinstance(m, str) else m.get("name", str(m))
            config = load_model_config(name)
            target = config.get("target_f0")
            models_info.append({"name": name, "target_f0": target})
    elif RVC_MODELS_DIR.exists():
        for d in sorted(RVC_MODELS_DIR.iterdir()):
            if d.is_dir() and (d / "revoicer.json").exists():
                config = load_model_config(d.name)
                target = config.get("target_f0")
                models_info.append({"name": d.name, "target_f0": target})

    # JSON mode
    if _event_handler is print_event_json:
        print(json.dumps({
            "server": {"running": server_running, "url": RVC_API_URL},
            "models": models_info,
            "formats": ["WAV", "MP3", "FLAC", "OGG", "M4A"],
        }))
        return

    # TUI mode
    print("=== generate — Available Models & Status ===\n")

    if models_info:
        print(f"Installed models ({len(models_info)}):")
        for m in models_info:
            hz_info = f"{m['target_f0']:.0f} Hz" if m["target_f0"] else "not set"
            print(f"  - {m['name']:40s}  target pitch: {hz_info}")
        if not server_running:
            print("\n  (Start server for full model list)")
    else:
        if server_running:
            print("No models installed.")
        else:
            print("(Start server to see installed models)")

    print()
    print("Server:")
    if server_running:
        print(f"  Running on {RVC_API_URL}")
    else:
        print("  Offline — start with: python generate.py server start")

    print()
    print("Supported languages:")
    print("  RVC is language-agnostic — any language works.")
    print()
    print("Supported formats: WAV, MP3, FLAC, OGG, M4A")


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

    _emit(f"Output: {out_path}")


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

    _emit(f"Output: {out_path}")


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
    p_voice.add_argument("input", nargs="*", help="[rvc] Input audio file(s)")
    p_voice.add_argument("--engine", default="rvc", choices=["rvc", "say", "ai-tts"],
                       help="Voice engine (default: rvc)")
    p_voice.add_argument("--model", dest="voice", help="[rvc] Voice/RVC model name")
    p_voice.add_argument("-o", "--output", help="Output directory")
    # text input (say/ai-tts, shared alias with audio --lyrics)
    voice_text_grp = p_voice.add_mutually_exclusive_group()
    voice_text_grp.add_argument("--text", "--lyrics", "-l", dest="text",
                                help="[say/ai-tts] Inline text to speak")
    voice_text_grp.add_argument("--text-file", "--lyrics-file", "-f",
                                dest="text_file",
                                help="[say/ai-tts] Path to text file")
    voice_text_grp.add_argument("--prompt-file", "-p", dest="prompt_file",
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
                       help="[ai-tts] ISO code: de, en, fr, ja, ko, zh, ru, pt, es, it (auto-detected if omitted)")
    p_voice.add_argument("--list-voices", action="store_true",
                       help="[ai-tts] Show available voices")
    # rvc-specific
    p_voice.add_argument("--decoder", choices=["rmvpe", "crepe", "harvest", "pm"],
                       help="[rvc] Pitch detection algorithm (default: rmvpe)")
    p_voice.add_argument("--pitch", type=int,
                       help="[rvc] Manual pitch shift in semitones (disables auto-pitch)")
    p_voice.add_argument("--target-hz", type=float,
                       help="[rvc] Target voice pitch in Hz (overrides model config)")
    p_voice.set_defaults(func=cmd_voice)

    # ── audio ─────────────────────────────────────────────────────────────
    p_audio = sub.add_parser("audio",
                             help="Audio processing and generation")
    p_audio.add_argument("input", nargs="*", help="Input audio file(s)")
    p_audio.add_argument("--engine", required=True,
                         choices=["enhance", "demucs", "ace-step", "heartmula", "diarize"],
                         help="Audio engine")
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
    p_audio.add_argument("--topk", type=int, default=None,
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
    p_audio.set_defaults(func=cmd_audio)

    # ── text ──────────────────────────────────────────────────────────────
    p_text = sub.add_parser("text", help="Text extraction from audio")
    p_text.add_argument("input", nargs="*", help="Input audio file(s)")
    p_text.add_argument("--engine", required=True,
                        choices=["whisper", "heartmula-transcribe"],
                        help="Text engine")
    p_text.add_argument("--model", default=None,
                        help="[whisper] Model: tiny/base/small/medium/large-v3/large-v3-turbo")
    p_text.add_argument("--input-language", default=None,
                        help="[whisper] Language hint (e.g. 'en', 'de')")
    p_text.add_argument("--word-timestamps", action="store_true",
                        help="[whisper] Include word-level timestamps")
    p_text.add_argument("--format", default="json",
                        help="[whisper] Output format: json/txt/srt/vtt/tsv/all")
    p_text.add_argument("-o", "--output",
                        help="Output directory or file")
    p_text.set_defaults(func=cmd_text)

    # ── output ────────────────────────────────────────────────────────────
    p_output = sub.add_parser("output",
                              help="Post-processing: concatenation, mixing")
    p_output.add_argument("input", nargs="*", help="Input files")
    p_output.add_argument("--engine", required=True,
                          choices=["audio-concatenate", "audio-mucs"],
                          help="Output engine")
    p_output.add_argument("-o", "--output", help="Output file path")
    p_output.add_argument("--output-bitrate", default=None,
                          help="Audio bitrate (e.g. '192k', '320k')")
    p_output.add_argument("--clip", action="append", default=None,
                          help="Per-clip options: INDEX:key=val,key=val "
                               "(keys: fade-in, fade-out, crossfade, volume, start, end, pan)")
    p_output.set_defaults(func=cmd_output)

    # ── Stubs for future mediums ──────────────────────────────────────────
    for stub_name in ["image", "video", "vision", "translation", "comparison"]:
        p_stub = sub.add_parser(stub_name, help=f"{stub_name.title()} (coming soon)")
        p_stub.set_defaults(func=_stub_medium, medium=stub_name)

    # ── ps ────────────────────────────────────────────────────────────────
    p_ps = sub.add_parser("ps", help="Show available models and status")
    p_ps.set_defaults(func=cmd_ps)

    # ── server ────────────────────────────────────────────────────────────
    p_server = sub.add_parser("server", help="Manage RVC worker")
    s_sub = p_server.add_subparsers(dest="server_cmd")

    p_ss = s_sub.add_parser("start", help="Start RVC worker")
    p_ss.add_argument("-p", "--port", type=int, default=5100)
    p_ss.set_defaults(func=cmd_server_start)

    p_st = s_sub.add_parser("stop", help="Stop RVC worker")
    p_st.set_defaults(func=cmd_server_stop)

    p_su = s_sub.add_parser("status", help="Check worker status")
    p_su.set_defaults(func=cmd_server_status)

    # ── models ────────────────────────────────────────────────────────────
    p_models = sub.add_parser("models", help="Manage voice models")
    m_sub = p_models.add_subparsers(dest="models_cmd")

    p_ml = m_sub.add_parser("list", help="List installed models")
    p_ml.set_defaults(func=cmd_models_list)

    p_ms = m_sub.add_parser("search", help="Search HuggingFace for RVC models")
    p_ms.add_argument("query", help="Search terms")
    p_ms.add_argument("--limit", type=int, default=30)
    p_ms.set_defaults(func=cmd_models_search)

    p_mi = m_sub.add_parser("install", help="Install model from HuggingFace")
    p_mi.add_argument("model_id", help="HuggingFace repo ID or direct URL")
    p_mi.add_argument("--name", help="Local name for the model")
    p_mi.add_argument("--file", help="Specific file from multi-model repos")
    p_mi.set_defaults(func=cmd_models_install)

    p_mr = m_sub.add_parser("remove", help="Remove an installed model")
    p_mr.add_argument("name", help="Model name")
    p_mr.set_defaults(func=cmd_models_remove)

    p_mc = m_sub.add_parser("calibrate",
                            help="Calibrate target F0 for a model")
    p_mc.add_argument("name", help="Model name")
    p_mc.set_defaults(func=cmd_models_calibrate)

    p_mf = m_sub.add_parser("set-pitch",
                            help="Set target pitch (Hz) for a model")
    p_mf.add_argument("name", help="Model name")
    p_mf.add_argument("hz", type=float,
                      help="Target pitch (male ~120, female ~220, child ~280)")
    p_mf.set_defaults(func=cmd_models_set_f0)

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
