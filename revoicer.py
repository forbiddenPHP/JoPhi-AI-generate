#!/usr/bin/env python3
"""
revoicer — Voice conversion, music generation, and audio processing CLI.

Usage:
  python revoicer.py --PS                                  Show available models & status
  python revoicer.py convert file1.wav file2.wav            Convert files (prints output paths as JSON)
  python revoicer.py convert *.wav -o ./output/             Batch convert to output directory
  python revoicer.py convert *.wav --voice model-name       Use specific voice model
  python revoicer.py convert *.wav --decoder rmvpe          Pitch detection algorithm
  python revoicer.py convert *.wav --pitch 12               Manual pitch shift (semitones)
  python revoicer.py convert *.wav --target-hz 280          Override target voice pitch
  python revoicer.py enhance file1.wav file2.wav -o out/    Denoise + enhance audio
  python revoicer.py enhance file.wav --denoise-only        Denoise only (faster)
  python revoicer.py enhance file.wav --enhance-only        Super-resolution only (no denoise)
  python revoicer.py music -l "lyrics" -t "disco,happy" -o out.mp3   Generate music
  python revoicer.py music -f lyrics.txt -t "rock,guitar" -o out.mp3 Music from file
  python revoicer.py transcribe audio.wav --format all -o out/       Transcribe audio (mlx-whisper)
  python revoicer.py transcribe audio.wav --input-language en        Transcribe with language hint
  python revoicer.py diarize dialog.wav -o out/                      Split dialogue by speaker
  python revoicer.py diarize dialog.wav -o out/ --speakers 3         Diarize with known speaker count
  python revoicer.py diarize dialog.wav -o out/ --verify             Diarize + show stats (gaps, overlaps)
  python revoicer.py separate song.mp3 -o out/                       Separate into stems (vocals/drums/bass/other)
  python revoicer.py transcribe-lyrics song.mp3                      Extract lyrics (HeartTranscriptor)
  python revoicer.py models list                            List installed models
  python revoicer.py models search "neutral male"           Search HuggingFace
  python revoicer.py models install <id-or-url>             Download from HuggingFace or URL
  python revoicer.py models remove <name>                   Remove a model
  python revoicer.py models set-pitch <name> <hz>           Set target pitch manually
  python revoicer.py models calibrate <name>                Auto-detect target pitch
  python revoicer.py server start                           Start RVC worker
  python revoicer.py server stop                            Stop RVC worker
  python revoicer.py server status                          Check worker status
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import math

import requests

from progress import (
    run_worker, print_event_tui, print_event_json, finish_progress,
    ProgressEvent,
)

# Output mode — set by --screen-log-format flag, used by all cmd_* functions
_event_handler = print_event_tui


def _emit(message: str, type: str = "log"):
    """Route a status message through the event handler.

    In TUI mode this prints plain text; in JSON mode it emits a JSON event.
    Use this instead of print(..., file=sys.stderr) for any informational
    output that app.py needs to parse.
    """
    _event_handler(ProgressEvent(type=type, message=message))

# ── Config ───────────────────────────────────────────────────────────────────

RVC_API_URL = os.environ.get("RVC_API_URL", "http://127.0.0.1:5100")
RVC_WORKER_DIR = Path(__file__).parent / "rvc_worker"
RVC_MODELS_DIR = Path(__file__).parent / "rvc_models"
ENHANCE_WORKER_DIR = Path(__file__).parent / "enhance_worker"
MUSIC_WORKER_DIR = Path(__file__).parent / "music_worker"
MUSIC_MODELS_DIR = Path(__file__).parent / "music_models"
CONDA_BIN = Path(os.environ.get("CONDA_BIN", "/opt/miniconda3/bin/conda"))
RVC_ENV = "rvc"
ENHANCE_ENV = "enhance"
HEARTMULA_ENV = "heartmula"
ACESTEP_DIR = Path(__file__).parent / "ace_worker" / "ACE-Step-1.5"
ACESTEP_WORKER = Path(__file__).parent / "ace_worker" / "generate.py"
WHISPER_WORKER_DIR = Path(__file__).parent / "whisper_worker"
WHISPER_ENV = "whisper"
DIARIZE_WORKER_DIR = Path(__file__).parent / "diarize_worker"
DIARIZE_ENV = "diarize"
SEPARATE_WORKER_DIR = Path(__file__).parent / "separate_worker"
SEPARATE_ENV = "separate"
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
    """Detect median fundamental frequency of a WAV file using pyworld (via rvc env).

    Returns median F0 in Hz, or None if detection fails.
    """
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
            [str(CONDA_BIN), "run", "-n", RVC_ENV, "python", "-c", script])
        if result.returncode != 0:
            _emit(f"  WARNING: F0 detection failed: {result.stderr_tail[:200]}", "warning")
            return None
        data = json.loads(result.stdout.strip())
        return data.get("median_f0")
    except Exception as e:
        _emit(f"  WARNING: F0 detection error: {e}", "warning")
        return None


def load_model_config(model_name: str) -> dict:
    """Load per-model config from rvc_models/<model>/revoicer.json."""
    config_path = RVC_MODELS_DIR / model_name / "revoicer.json"
    if config_path.exists():
        return json.loads(config_path.read_text())
    return {}


def save_model_config(model_name: str, config: dict):
    """Save per-model config to rvc_models/<model>/revoicer.json."""
    config_dir = RVC_MODELS_DIR / model_name
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "revoicer.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")


def compute_pitch_shift(input_f0: float, target_f0: float) -> int:
    """Compute pitch shift in semitones to go from input_f0 to target_f0."""
    return round(12 * math.log2(target_f0 / input_f0))


# ── Audio Enhancement (resemble-enhance) ─────────────────────────────────────

def check_enhance_env() -> bool:
    """Check if the enhance conda env is available."""
    try:
        r = subprocess.run(
            [str(CONDA_BIN), "run", "-n", ENHANCE_ENV, "python", "-c",
             "from resemble_enhance.enhancer.inference import denoise; print('ok')"],
            capture_output=True, text=True, timeout=30)
        return r.returncode == 0
    except Exception:
        return False


def enhance_audio(input_path: Path, output_path: Path,
                  denoise_only: bool = False) -> bool:
    """Enhance audio via resemble-enhance (subprocess in enhance env).

    Returns True on success, False on failure (with warning printed).
    """
    enhance_script = ENHANCE_WORKER_DIR / "enhance.py"
    if not enhance_script.exists():
        _emit("  WARNING: enhance_worker/enhance.py not found", "warning")
        return False

    cmd = [
        str(CONDA_BIN), "run", "-n", ENHANCE_ENV,
        "python", str(enhance_script),
        str(input_path),
        "-o", str(output_path.parent),
    ]
    if denoise_only:
        cmd.append("--denoise-only")

    try:
        result = run_worker(cmd, on_event=_event_handler)
        finish_progress()
        if result.returncode != 0:
            _emit(f"  WARNING: Enhancement failed: {result.stderr_tail[:300]}", "warning")
            return False
        return True
    except Exception as e:
        _emit(f"  WARNING: Enhancement error: {e}", "warning")
        return False


# ── API Client ───────────────────────────────────────────────────────────────

def api_get(endpoint: str) -> dict:
    try:
        r = requests.get(f"{RVC_API_URL}{endpoint}", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        _emit("ERROR: RVC worker not running.", "error")
        _emit("  Start with: python revoicer.py server start", "error")
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
        _emit("  Start with: python revoicer.py server start", "error")
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
        print("RVC worker already running.")
        return

    if not CONDA_BIN.exists():
        print(f"ERROR: conda not found at {CONDA_BIN}")
        sys.exit(1)

    print("Starting RVC worker ...")
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

    # Wait for server to be ready
    for i in range(30):
        time.sleep(1)
        if check_server():
            print(f"RVC worker running on port {port} (PID {proc.pid})")
            return

    _emit("ERROR: RVC worker did not start within 30 seconds.", "error")
    proc.kill()
    sys.exit(1)


def cmd_server_stop(args):
    if PID_FILE.exists():
        pid = int(PID_FILE.read_text().strip())
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            print(f"RVC worker stopped (PID {pid})")
        except ProcessLookupError:
            print("RVC worker was not running.")
        PID_FILE.unlink(missing_ok=True)
    else:
        print("No PID file found. Checking if server is running ...")
        if check_server():
            print("Server is running but PID unknown. Kill manually or use:")
            print("  lsof -ti:5100 | xargs kill")
        else:
            print("RVC worker is not running.")


def cmd_server_status(args):
    if check_server():
        models = api_get("/models")
        n = len(models.get("models", []))
        pid = PID_FILE.read_text().strip() if PID_FILE.exists() else "?"
        print(f"RVC worker: running (PID {pid})")
        print(f"  URL:    {RVC_API_URL}")
        print(f"  Models: {n} loaded")
    else:
        print("RVC worker: not running")
        print(f"  Start with: python revoicer.py server start")


# ── Models ───────────────────────────────────────────────────────────────────

def cmd_models_list(args):
    data = api_get("/models")
    models = data.get("models", [])
    if not models:
        print("No models installed.")
        print("  Search: python revoicer.py models search \"neutral male\"")
        print("  Install: python revoicer.py models install <hf-model-id>")
        return

    print(f"Installed models ({len(models)}):\n")
    for m in models:
        if isinstance(m, str):
            print(f"  {m}")
        else:
            name = m.get("name", m.get("model_name", str(m)))
            print(f"  {name}")


def _check_rvc_repo(repo_id: str, files: list[str]) -> tuple[bool, list[str], list[str]]:
    """Check if a repo has usable RVC .pth files.

    Returns (is_rvc, pth_files, index_files).
    Filters out pretrained base models (D_/G_ pairs).
    """
    pth = [f for f in files if f.endswith(".pth")]
    idx = [f for f in files if f.endswith(".index")]

    if not pth:
        return False, [], []

    # Skip pretrained base models (D_*.pth / G_*.pth — discriminator/generator pairs)
    voice_pth = [f for f in pth if not any(
        f.split("/")[-1].startswith(p) for p in ("D_", "G_", "D-", "G-", "f0")
    )]
    if not voice_pth and len(pth) != 1:
        return False, [], []

    return True, voice_pth or pth, idx


def _search_voice_models_com(query: str, limit: int = 25) -> list[dict]:
    """Search voice-models.com for RVC models.

    Returns list of {title, url, hf_repo_id, size}.
    """
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
        # Title: extract everything inside the <a class='fs-5'> tag, then strip HTML
        title_m = re.search(r"class=['\"]fs-5['\"][^>]*>(.*?)</a>", row, re.DOTALL)
        url_m = re.search(r"data-clipboard-text=['\"]([^'\"]+)['\"]", row)
        size_m = re.search(r"badge[^>]*>([^<]+)</span>", row)

        if not (title_m and url_m):
            continue

        # Strip inner HTML tags from title
        title_raw = title_m.group(1)
        title_clean = re.sub(r"<[^>]+>", "", title_raw).strip()

        url = url_m.group(1).strip()
        # Extract HuggingFace repo ID from URL
        # e.g. https://huggingface.co/User/Repo/resolve/main/file.zip?download=true
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
    print(f'Searching for: "{query}" ...\n')

    # ── Source 1: voice-models.com ──────────────────────────────────────
    vm_results = _search_voice_models_com(query, limit=args.limit)
    if vm_results:
        print(f"voice-models.com ({len(vm_results)} results):\n")
        for i, r in enumerate(vm_results, 1):
            install_id = r["hf_repo_id"] or r["url"]
            print(f"  {i:2d}. {r['title']}")
            print(f"      Size: {r['size']}")
            print(f"      Install: python revoicer.py models install {install_id}")
        print()

    # ── Source 2: HuggingFace ───────────────────────────────────────────
    try:
        from huggingface_hub import HfApi
    except ImportError:
        if not vm_results:
            print("ERROR: pip install huggingface-hub")
            sys.exit(1)
        print("(HuggingFace search skipped — pip install huggingface-hub)")
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
        print(f"HuggingFace ({len(results)} results):\n")
        print(f"{'#':>3}  {'Model ID':50s}  {'Downloads':>10s}")
        print("-" * 70)
        for i, m in enumerate(results, 1):
            dl = str(m.downloads) if m.downloads else "?"
            print(f"{i:3d}  {m.id:50s}  {dl:>10s}")
        print()

    if not vm_results and not seen:
        print("No models found. Try broader terms.")
        return

    print("Install: python revoicer.py models install <hf-repo-id or URL>")


def _extract_archive(archive_path: Path, extract_dir: Path) -> list[Path]:
    """Extract .zip, .rar, or .7z archive. Returns list of extracted files."""
    suffix = archive_path.suffix.lower()
    extract_dir.mkdir(parents=True, exist_ok=True)

    if suffix == ".zip":
        import zipfile
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(extract_dir)
    else:
        # unar handles .rar, .7z, .zip and many more
        subprocess.run(["unar", "-o", str(extract_dir), "-f", str(archive_path)],
                       check=True, capture_output=True)

    return list(extract_dir.rglob("*"))


def _upload_model(name: str, pth_path: Path, idx_path: Path | None = None):
    """Pack .pth (+ optional .index) into a .zip and upload to the RVC worker.

    ZIP structure: <name>/<name>.pth (+ <name>/<name>.index)
    RVC expects models in subdirectories of rvc_models/.
    """
    import zipfile
    zip_path = Path(tempfile.mktemp(suffix=".zip", prefix="revoicer_"))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pth_path, f"{name}/{name}.pth")
        if idx_path:
            zf.write(idx_path, f"{name}/{name}.index")

    with open(zip_path, "rb") as f:
        api_post("/upload_model", files={"file": (f"{name}.zip", f)})

    zip_path.unlink(missing_ok=True)


def _download_url(url: str, dest_dir: Path) -> Path:
    """Download a file from a URL to dest_dir. Returns local path."""
    import re as _re
    r = requests.get(url, stream=True, timeout=120, allow_redirects=True)
    r.raise_for_status()

    # Try to get filename from Content-Disposition header
    cd = r.headers.get("Content-Disposition", "")
    fname_m = _re.search(r'filename="?([^";\n]+)"?', cd)
    if fname_m:
        fname = fname_m.group(1).strip()
    else:
        # Fall back to URL path
        from urllib.parse import urlparse, unquote
        fname = unquote(urlparse(url).path.split("/")[-1])
        # Strip query params from filename
        fname = fname.split("?")[0] or "download"

    dest = dest_dir / fname
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    return dest


def _install_from_archive(archive_path: Path, name: str) -> tuple[Path, Path | None]:
    """Extract archive, find .pth + .index, return (pth_path, idx_path)."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="revoicer_"))
    print(f"  Extracting {archive_path.name} ...")
    try:
        extracted = _extract_archive(archive_path, tmp_dir)
    except FileNotFoundError:
        if subprocess.run(["which", "brew"], capture_output=True).returncode != 0:
            print("  ERROR: 'unar' not found and brew is not installed.")
            sys.exit(1)
        print("  Installing unar via brew ...")
        r = subprocess.run(["brew", "install", "unar"],
                           capture_output=True, text=True,
                           env={**os.environ, "HOMEBREW_NO_AUTO_UPDATE": "1"})
        if r.returncode != 0:
            print(f"  ERROR: brew install unar failed: {r.stderr[:200]}")
            sys.exit(1)
        extracted = _extract_archive(archive_path, tmp_dir)
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: Extraction failed: {e.stderr.decode()[:200]}")
        sys.exit(1)

    ex_pth = [f for f in extracted if f.suffix == ".pth"]
    ex_idx = [f for f in extracted if f.suffix == ".index"]

    if not ex_pth:
        print("  ERROR: No .pth files found in archive")
        sys.exit(1)

    print(f"  Found: {ex_pth[0].name}")
    return ex_pth[0], ex_idx[0] if ex_idx else None


def _sanitize_model_name(raw: str) -> str:
    """Turn a filename stem into a clean model name.

    Strips common RVC training metadata like _e300, _s5200, _v2, _rvc, etc.
    """
    import re as _re
    name = raw.lower().replace(" ", "_").replace("-", "_").strip("_")
    # Strip training metadata: _e300, _s5200, _e300_s5200
    name = _re.sub(r"(_e\d+)?(_s\d+)?$", "", name)
    # Remove trailing version markers: _v2, v2
    stripped = _re.sub(r"_?v\d+$", "", name)
    if stripped:
        name = stripped
    # Remove trailing _rvc suffix (technical marker)
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


def cmd_models_install(args):
    model_id = args.model_id
    is_url = _is_url(model_id)

    # Determine HF repo ID (for calibration heuristics)
    hf_repo_id = None
    if is_url:
        hf_repo_id = _hf_repo_from_url(model_id)

    pth_path = None
    idx_path = None
    tmp_dir = None
    name = None  # derived below from actual file

    if is_url:
        # ── Direct URL download ─────────────────────────────────────────
        tmp_dir = Path(tempfile.mkdtemp(prefix="revoicer_"))
        print(f"Downloading from URL ...")
        local_file = _download_url(model_id, tmp_dir)
        print(f"  Downloaded: {local_file.name}")

        name = args.name or _sanitize_model_name(local_file.stem)

        if local_file.suffix.lower() in (".zip", ".rar", ".7z"):
            pth_path, idx_path = _install_from_archive(local_file, name)
        elif local_file.suffix.lower() == ".pth":
            pth_path = local_file
        else:
            print(f"  ERROR: Unexpected file type: {local_file.suffix}")
            print(f"  Expected .zip, .rar, .7z, or .pth")
            sys.exit(1)

    else:
        # ── HuggingFace repo ID ─────────────────────────────────────────
        hf_repo_id = model_id
        try:
            from huggingface_hub import HfApi, hf_hub_download
        except ImportError:
            print("ERROR: pip install huggingface-hub")
            sys.exit(1)

        api = HfApi()
        files = api.list_repo_files(model_id)

        # Filter out pretrained D_/G_ pairs from .pth files
        pth_files = [f for f in files if f.endswith(".pth")
                     and not any(f.split("/")[-1].startswith(p)
                                 for p in ("D_", "G_", "D-", "G-", "f0"))]
        idx_files = [f for f in files if f.endswith(".index")]
        archive_files = [f for f in files
                         if f.lower().endswith((".zip", ".rar", ".7z"))]

        # All installable model files
        model_files = archive_files or pth_files

        # ── User specified --file ──────────────────────────────────────
        target_file = getattr(args, "file", None)
        if target_file:
            # Find exact or partial match
            if target_file not in files:
                matches = [f for f in files if target_file.lower() in f.lower()]
                if len(matches) == 1:
                    target_file = matches[0]
                elif len(matches) > 1:
                    print(f"  ERROR: '{target_file}' is ambiguous. Matches:")
                    for m in matches:
                        print(f"    - {m}")
                    sys.exit(1)
                else:
                    print(f"  ERROR: '{target_file}' not found in {model_id}")
                    sys.exit(1)

            name = args.name or _sanitize_model_name(Path(target_file).stem)
            print(f"Installing '{name}' from {target_file} ...")

            if target_file.lower().endswith((".zip", ".rar", ".7z")):
                print(f"  Downloading {target_file} ...")
                local_archive = Path(hf_hub_download(model_id, target_file))
                pth_path, idx_path = _install_from_archive(local_archive, name)
            elif target_file.endswith(".pth"):
                print(f"  Downloading {target_file} ...")
                pth_path = Path(hf_hub_download(model_id, target_file))
                base = Path(target_file).stem
                matching_idx = [f for f in idx_files if base in f]
                if matching_idx:
                    print(f"  Downloading {matching_idx[0]} ...")
                    idx_path = Path(hf_hub_download(model_id, matching_idx[0]))
            else:
                print(f"  ERROR: File must be .pth, .zip, .rar, or .7z")
                sys.exit(1)

        # ── Multi-model repo: install all ─────────────────────────────
        elif len(model_files) > 1:
            total = len(model_files)
            print(f"Repo '{model_id}' contains {total} models — installing all ...\n")
            installed = []
            for j, mf in enumerate(sorted(model_files), 1):
                mf_name = _sanitize_model_name(Path(mf).stem)
                print(f"[{j}/{total}] Installing '{mf_name}' from {mf} ...")
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
                        print(f"  Skipping (unsupported format)")
                        continue
                    _upload_model(mf_name, mf_pth, mf_idx)
                    sys.stdout.flush()
                    calibrate_model(mf_name, hf_repo_id=hf_repo_id)
                    sys.stderr.flush()
                    installed.append(mf_name)
                    print()
                except Exception as e:
                    _emit(f"ERROR: {e} — skipping", "error")
            print(f"Installed {len(installed)}/{total} models.")
            return

        # ── Single model in repo ───────────────────────────────────────
        elif pth_files:
            pth_file = pth_files[0]
            name = args.name or _sanitize_model_name(Path(pth_file).stem)
            print(f"Installing '{name}' ...")
            print(f"  Downloading {pth_file} ...")
            pth_path = Path(hf_hub_download(model_id, pth_file))
            if idx_files:
                print(f"  Downloading {idx_files[0]} ...")
                idx_path = Path(hf_hub_download(model_id, idx_files[0]))

        elif archive_files:
            archive = archive_files[0]
            name = args.name or _sanitize_model_name(Path(archive).stem)
            print(f"Installing '{name}' ...")
            print(f"  Downloading {archive} ...")
            local_archive = Path(hf_hub_download(model_id, archive))
            pth_path, idx_path = _install_from_archive(local_archive, name)

        else:
            print(f"  ERROR: No .pth or archive files found in {model_id}")
            print(f"  Files in repo: {', '.join(files[:10])}")
            sys.exit(1)

    # Upload to RVC worker
    print(f"  Uploading to RVC worker ...")
    _upload_model(name, pth_path, idx_path)
    print(f"  Uploaded {name}.zip")

    if tmp_dir:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n  Model '{name}' installed.")
    sys.stdout.flush()

    # Auto-calibrate
    calibrate_model(name, hf_repo_id=hf_repo_id)


def _guess_f0_from_name(name: str) -> float | None:
    """Guess target F0 from model name keywords.

    Uses word-boundary matching to avoid false positives
    (e.g. 'german' should NOT match 'man').
    """
    import re as _re
    name_lower = name.lower()

    def _has_word(keywords):
        """Check if any keyword appears as a whole word in name."""
        for kw in keywords:
            if _re.search(r'(?:^|[\s_\-])' + _re.escape(kw) + r'(?:$|[\s_\-])', name_lower):
                return True
        return False

    # Child / young voices
    if _has_word(["child", "kind", "kid", "girl", "mädchen", "boy", "junge",
                  "pippi", "anime", "loli", "young"]):
        return 280.0
    # Female voices
    if _has_word(["female", "frau", "weiblich", "woman",
                  "soprano", "alto", "mezzo"]):
        return 220.0
    # Male voices
    if _has_word(["male", "mann", "männlich", "man",
                  "tenor", "bass", "baritone", "bariton"]):
        return 120.0
    return None


def calibrate_model(model_name: str, hf_repo_id: str = None) -> float | None:
    """Calibrate target F0 for a model using name heuristics.

    1. Infer from model name keywords (male/female/child).
    2. Infer from HuggingFace repo name keywords.
    3. If nothing works, return None — user must set manually via `models set-pitch`.
    """
    target_f0 = None

    # Strategy 1: Infer from model name
    target_f0 = _guess_f0_from_name(model_name)
    if target_f0:
        print(f"  Estimated F0 from model name: {target_f0:.0f} Hz")

    # Strategy 2: Also try the HF repo name if different
    if target_f0 is None and hf_repo_id:
        target_f0 = _guess_f0_from_name(hf_repo_id)
        if target_f0:
            print(f"  Estimated F0 from repo name: {target_f0:.0f} Hz")

    if target_f0 is None:
        _emit("WARNING: Could not determine target F0.", "warning")
        _emit("Set manually:", "log")
        _emit(f"  python revoicer.py models set-pitch {model_name} 120   # Male", "log")
        _emit(f"  python revoicer.py models set-pitch {model_name} 220   # Female", "log")
        _emit(f"  python revoicer.py models set-pitch {model_name} 280   # Child", "log")
        return None

    config = load_model_config(model_name)
    config["target_f0"] = round(target_f0, 1)
    if hf_repo_id:
        config["hf_repo_id"] = hf_repo_id
    save_model_config(model_name, config)

    print(f"  Target F0: {target_f0:.1f} Hz (saved)")
    return target_f0


def cmd_models_calibrate(args):
    """CLI wrapper for calibrate_model."""
    # Try to find HF repo ID from saved config
    config = load_model_config(args.name)
    hf_repo_id = config.get("hf_repo_id")
    result = calibrate_model(args.name, hf_repo_id=hf_repo_id)
    if result is None:
        sys.exit(1)


def cmd_models_set_f0(args):
    """Manually set target F0 for a model."""
    model_name = args.name
    target_f0 = args.hz

    config = load_model_config(model_name)
    config["target_f0"] = target_f0
    save_model_config(model_name, config)

    print(f"Target F0 for '{model_name}': {target_f0} Hz")
    print(f"  Saved to: rvc_models/{model_name}/revoicer.json")


def cmd_models_remove(args):
    print(f"Removing model '{args.name}' ...")
    # rvc-python API may not support DELETE — check
    try:
        r = requests.delete(f"{RVC_API_URL}/models/{args.name}", timeout=10)
        if r.ok:
            print(f"  Removed '{args.name}'")
        else:
            print(f"  API returned {r.status_code}: {r.text}")
    except Exception as e:
        print(f"  ERROR: {e}")


# ── Convert ──────────────────────────────────────────────────────────────────

def cmd_convert(args):
    input_paths = [Path(p) for p in args.input]

    # Validate inputs
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
        # Auto-pitch mode (default)
        if args.target_hz is not None:
            target_f0 = args.target_hz
        elif model_name:
            config = load_model_config(model_name)
            target_f0 = config.get("target_f0")
            if target_f0 is None:
                # Auto-calibrate on first use
                _emit("No target F0 for this model — calibrating automatically ...", "stage")
                target_f0 = calibrate_model(model_name)
                if target_f0 is None:
                    _emit("  Calibration failed. Using pitch=0.", "warning")

    # Set base params (f0method, index_rate)
    base_params = {"index_rate": 0.0}
    if args.decoder is not None:
        base_params["f0method"] = args.decoder
    if manual_pitch:
        base_params["f0up_key"] = args.pitch

    # Convert each file
    output_paths = []
    total = len(input_paths)

    for i, input_path in enumerate(input_paths, 1):
        if output_dir:
            out_path = output_dir / input_path.with_suffix(".wav").name
        else:
            out_path = input_path.with_stem(input_path.stem + "_converted").with_suffix(".wav")

        # Convert non-WAV to WAV via ffmpeg before sending to RVC
        tmp_wav = None
        send_path = input_path
        if input_path.suffix.lower() != ".wav":
            tmp_wav = Path(tempfile.mktemp(suffix=".wav", prefix="revoicer_"))
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

        # Set params (per file, since f0up_key may differ)
        api_post("/params", json={"params": params})

        # Send to RVC API via file upload
        with open(send_path, "rb") as f:
            r = api_post("/convert_file", files={"file": (send_path.name, f, "audio/wav")})

        if tmp_wav:
            tmp_wav.unlink(missing_ok=True)

        # Save output
        out_path.write_bytes(r.content)
        output_paths.append(str(out_path))

    # Output result as JSON array of paths
    print(json.dumps(output_paths))


# ── Enhance ──────────────────────────────────────────────────────────────────

def cmd_enhance(args):
    """Enhance audio files (denoise + super-resolution)."""
    input_paths = [Path(p) for p in args.input]

    # Validate inputs
    for p in input_paths:
        if not p.exists():
            _emit(f"ERROR: File not found: {p}", "error")
            sys.exit(1)

    output_dir = Path(args.output) if args.output else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Build the enhance.py command with all files at once
    # (so models are loaded only once)
    enhance_script = ENHANCE_WORKER_DIR / "enhance.py"
    if not enhance_script.exists():
        _emit("ERROR: enhance_worker/enhance.py not found", "error")
        _emit("  Run: bash enhance_worker/install.sh", "error")
        sys.exit(1)

    # Determine output dir
    if output_dir:
        out_dir = output_dir
    else:
        # Default: same directory as input, with _enhanced suffix
        out_dir = input_paths[0].parent

    cmd = [
        str(CONDA_BIN), "run", "-n", ENHANCE_ENV,
        "python", str(enhance_script),
    ]
    for p in input_paths:
        cmd.append(str(p))
    cmd.extend(["-o", str(out_dir)])

    if args.denoise_only:
        cmd.append("--denoise-only")
    if args.enhance_only:
        cmd.append("--enhance-only")

    mode = "denoise" if args.denoise_only else "enhance-only" if args.enhance_only else "enhance"
    total = len(input_paths)
    _emit(f"Enhancing {total} file(s) ({mode}) ...", "stage")

    result = run_worker(cmd, on_event=_event_handler)
    finish_progress()
    if result.returncode != 0:
        _emit(f"ERROR: Enhancement failed.", "error")
        sys.exit(1)
    # stdout has the JSON output
    if result.stdout.strip():
        print(result.stdout.strip())


# ── Music ─────────────────────────────────────────────────────────────────────


def split_segments(duration_s: float) -> list:
    """Split song into equal segments for repainting based on duration.

    Returns list of (start_s, end_s) tuples.
    Songs under 60s are not split (no refine needed).
    Target ~36s per segment, min 2, scales with duration.
    """
    if duration_s < 60:
        return []

    n = max(2, round(duration_s / 36))
    seg_len = duration_s / n
    segments = [(i * seg_len, (i + 1) * seg_len) for i in range(n)]

    _emit(f"  Duration: {duration_s:.1f}s → {n} segments à {seg_len:.1f}s")
    for i, (s, e) in enumerate(segments):
        _emit(f"    seg {i+1}: {s:.1f}s - {e:.1f}s")

    return segments


def cmd_music(args):
    """Generate music — dispatches to ACE-Step or HeartMuLa based on --engine."""
    engine = getattr(args, "engine", "ace")
    if engine == "heart":
        cmd_music_heart(args)
    else:
        # ace, ace-turbo, ace-sft, ace-base all go through ACE-Step
        cmd_music_ace(args)


_ACE_ENGINE_MAP = {
    "ace": "acestep-v15-turbo",
    "ace-turbo": "acestep-v15-turbo",
    "ace-sft": "acestep-v15-sft",
    "ace-base": "acestep-v15-base",
}


def cmd_music_ace(args):
    """Generate music using ACE-Step 1.5."""

    # ── Read lyrics ──────────────────────────────────────────────────────
    if args.lyrics_file:
        lyrics_path = Path(args.lyrics_file)
        if not lyrics_path.exists():
            _emit(f"ERROR: Lyrics file not found: {lyrics_path}", "error")
            sys.exit(1)
        lyrics = lyrics_path.read_text(encoding="utf-8")
    elif args.lyrics:
        lyrics = args.lyrics
    else:
        _emit("ERROR: Provide --lyrics or --lyrics-file", "error")
        sys.exit(1)

    if not lyrics.strip():
        _emit("ERROR: Lyrics cannot be empty", "error")
        sys.exit(1)

    # ── Determine output path ────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
        if out_path.is_dir() or str(out_path).endswith("/"):
            out_path = Path(args.output) / f"music_{int(time.time())}.mp3"
    else:
        out_path = Path(f"music_{int(time.time())}.mp3")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Check worker script ──────────────────────────────────────────────
    if not ACESTEP_WORKER.exists():
        _emit("ERROR: ace_worker/generate.py not found", "error")
        _emit("  Run: bash ace_worker/install.sh", "error")
        sys.exit(1)

    if not ACESTEP_DIR.exists():
        _emit("ERROR: ACE-Step-1.5 not found", "error")
        _emit(f"  Expected: {ACESTEP_DIR}", "error")
        _emit("  Run: bash ace_worker/install.sh", "error")
        sys.exit(1)

    # ── Build base subprocess command (shared between generate + repaint) ─
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
        # Shared params (mapped names)
        if args.cfg_scale is not None:
            cmd.extend(["--guidance-scale", str(args.cfg_scale)])
        if args.temperature is not None:
            cmd.extend(["--lm-temperature", str(args.temperature)])
        if args.topk is not None:
            cmd.extend(["--lm-top-k", str(args.topk)])
        if args.seed is not None:
            cmd.extend(["--seed", str(args.seed)])
        # ACE-Step specific params
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
        # Map engine name to ACE-Step config path
        ace_config = _ACE_ENGINE_MAP.get(getattr(args, "engine", "ace"), "acestep-v15-turbo")
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

    # --duration (ms) overrides --seconds
    duration_ms = args.duration if args.duration else args.seconds * 1000
    duration_s = duration_ms / 1000

    _emit(f"Generating music (ACE-Step) ...", "stage")
    _emit(f"  Caption: {args.tags}")
    _emit(f"  Duration: {duration_s:.0f}s")
    _emit(f"  Output:   {out_path}")

    cmd = _ace_base_cmd()
    cmd.extend(["--duration", str(duration_ms)])
    if not _run_ace(cmd, "Music generation"):
        sys.exit(1)

    if out_path.exists():
        print(json.dumps([str(out_path)]))


def cmd_music_heart(args):
    """Generate music from lyrics and tags using HeartMuLa."""

    # ── Read lyrics ──────────────────────────────────────────────────────
    if args.lyrics_file:
        lyrics_path = Path(args.lyrics_file)
        if not lyrics_path.exists():
            _emit(f"ERROR: Lyrics file not found: {lyrics_path}", "error")
            sys.exit(1)
        lyrics = lyrics_path.read_text(encoding="utf-8")
    elif args.lyrics:
        lyrics = args.lyrics
    else:
        _emit("ERROR: Provide --lyrics or --lyrics-file", "error")
        sys.exit(1)

    if not lyrics.strip():
        _emit("ERROR: Lyrics cannot be empty", "error")
        sys.exit(1)

    # ── Determine output path ────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
        if out_path.is_dir() or str(out_path).endswith("/"):
            out_path = Path(args.output) / f"music_{int(time.time())}.mp3"
    else:
        out_path = Path(f"music_{int(time.time())}.mp3")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Check worker script ──────────────────────────────────────────────
    generate_script = MUSIC_WORKER_DIR / "generate.py"
    if not generate_script.exists():
        _emit("ERROR: music_worker/generate.py not found", "error")
        _emit("  Run: bash music_worker/install.sh", "error")
        sys.exit(1)

    # ── Check checkpoint directory ───────────────────────────────────────
    ckpt_dir = MUSIC_MODELS_DIR / "ckpt"
    if not ckpt_dir.exists():
        _emit("ERROR: HeartMuLa checkpoints not found", "error")
        _emit(f"  Expected: {ckpt_dir}", "error")
        _emit("  Run: bash music_worker/install.sh", "error")
        sys.exit(1)

    # ── Append bpm/keyscale to caption (HeartMuLa has no native flags) ──
    tags = args.tags
    if getattr(args, "bpm", None) is not None:
        tags += f", bpm: {args.bpm}"
    if getattr(args, "keyscale", None):
        tags += f", keyscale: {args.keyscale}"
    if getattr(args, "timesignature", None):
        tags += f", timesignature: {args.timesignature}"

    # HeartMuLa expects comma-separated tags without spaces
    tags = ",".join(t.strip() for t in tags.split(","))

    # ── Build subprocess command ─────────────────────────────────────────
    cmd = [
        str(CONDA_BIN), "run", "-n", HEARTMULA_ENV,
        "python", str(generate_script),
        "--lyrics", lyrics,
        "--tags", tags,
        "-o", str(out_path),
        "--ckpt-dir", str(ckpt_dir),
    ]

    # --duration (ms) overrides --seconds
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
    _emit(f"Generating music (HeartMuLa) ...", "stage")
    _emit(f"  Tags:     {args.tags}")
    _emit(f"  Duration: {duration_s:.0f}s")
    _emit(f"  Output:   {out_path}")

    result = run_worker(cmd, on_event=_event_handler)
    finish_progress()
    if result.returncode != 0:
        _emit(f"ERROR: Music generation failed.", "error")
        sys.exit(1)
    # stdout has the JSON output
    if result.stdout.strip():
        print(result.stdout.strip())


# ── transcribe (whisper) ───────────────────────────────────────────────────────

_WHISPER_MODEL_MAP = {
    "tiny": "tiny",
    "base": "base",
    "small": "small",
    "medium": "medium",
    "large-v3": "large-v3",
    "large-v3-turbo": "large-v3-turbo",
}


def cmd_transcribe(args):
    """Transcribe audio using mlx-whisper."""
    transcribe_script = WHISPER_WORKER_DIR / "transcribe.py"
    if not transcribe_script.exists():
        _emit("ERROR: whisper_worker/transcribe.py not found", "error")
        _emit("  Run: bash whisper_worker/install.sh", "error")
        sys.exit(1)

    # Resolve input files
    input_files = []
    for f in args.input:
        p = Path(f).resolve()
        if not p.exists():
            _emit(f"ERROR: File not found: {p}", "error")
            sys.exit(1)
        input_files.append(str(p))

    cmd = [
        str(CONDA_BIN), "run", "-n", WHISPER_ENV,
        "python", str(transcribe_script),
    ]
    cmd.extend(input_files)

    # Model
    model = getattr(args, "model", "large-v3-turbo")
    cmd.extend(["--model", model])

    # Language
    if getattr(args, "input_language", None):
        cmd.extend(["--language", args.input_language])

    # Word timestamps
    if getattr(args, "word_timestamps", False):
        cmd.append("--word-timestamps")

    # Format
    fmt = getattr(args, "format", "json") or "json"
    cmd.extend(["--format", fmt])

    # Output directory
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


# ── transcribe-lyrics ──────────────────────────────────────────────────────────

def cmd_transcribe_lyrics(args):
    """Transcribe lyrics from audio using HeartTranscriptor."""
    transcribe_script = MUSIC_WORKER_DIR / "transcribe.py"
    if not transcribe_script.exists():
        _emit("ERROR: music_worker/transcribe.py not found", "error")
        sys.exit(1)

    audio_path = Path(args.input).resolve()
    if not audio_path.exists():
        _emit(f"ERROR: Audio file not found: {audio_path}", "error")
        sys.exit(1)

    ckpt_dir = MUSIC_MODELS_DIR / "ckpt"
    if not ckpt_dir.exists():
        _emit("ERROR: HeartMuLa checkpoints not found", "error")
        _emit(f"  Expected: {ckpt_dir}", "error")
        _emit("  Run: bash music_worker/install.sh", "error")
        sys.exit(1)

    cmd = [
        str(CONDA_BIN), "run", "-n", HEARTMULA_ENV,
        "python", str(transcribe_script),
        "--audio", str(audio_path),
        "--ckpt-dir", str(ckpt_dir),
    ]

    if args.output:
        out_path = Path(args.output).resolve()
        cmd.extend(["-o", str(out_path)])

    _emit(f"Transcribing lyrics ...", "stage")
    _emit(f"  Input: {audio_path}")

    result = run_worker(cmd, on_event=_event_handler)
    finish_progress()
    if result.returncode != 0:
        _emit(f"ERROR: Lyrics transcription failed.", "error")
        sys.exit(1)
    if result.stdout.strip():
        print(result.stdout.strip())


# ── --PS (Print Status / Available Models & Languages) ───────────────────────

def cmd_ps(args):
    server_running = check_server()

    print("=== Revoicer — Available Models & Languages ===\n")

    if server_running:
        data = api_get("/models")
        models = data.get("models", [])
        if models:
            print(f"Installed models ({len(models)}):")
            for m in models:
                name = m if isinstance(m, str) else m.get("name", str(m))
                config = load_model_config(name)
                target = config.get("target_f0")
                hz_info = f"{target:.0f} Hz" if target else "not set"
                print(f"  - {name:40s}  target pitch: {hz_info}")
        else:
            print("No models installed.")
    else:
        # Server not running — show models from disk configs
        if RVC_MODELS_DIR.exists():
            model_dirs = [d for d in RVC_MODELS_DIR.iterdir()
                          if d.is_dir() and (d / "revoicer.json").exists()]
            if model_dirs:
                print(f"Installed models ({len(model_dirs)}):")
                for d in sorted(model_dirs):
                    config = load_model_config(d.name)
                    target = config.get("target_f0")
                    hz_info = f"{target:.0f} Hz" if target else "not set"
                    print(f"  - {d.name:40s}  target pitch: {hz_info}")
                print("\n  (Start server for full model list)")
            else:
                print("No models installed.")
        else:
            print("(Start server to see installed models)")

    print()
    print("Server:")
    if server_running:
        print(f"  Running on {RVC_API_URL}")
    else:
        print("  Offline — start with: python revoicer.py server start")

    print()
    print("Supported languages:")
    print("  RVC is language-agnostic — any language works.")
    print()
    print("Supported formats: WAV, MP3, FLAC, OGG, M4A")


# ── Diarize ──────────────────────────────────────────────────────────────────


def cmd_diarize(args):
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
        _emit("ERROR: diarize_worker/diarize.py not found", "error")
        _emit("  Run: bash diarize_worker/install.sh", "error")
        sys.exit(1)

    for p in input_paths:
        cmd = [
            str(CONDA_BIN), "run", "-n", DIARIZE_ENV,
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
            _emit(f"ERROR: Diarization failed.", "error")
            sys.exit(1)
        if result.stdout.strip():
            print(result.stdout.strip())


# ── Separate ─────────────────────────────────────────────────────────────────


def cmd_separate(args):
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
        _emit("ERROR: separate_worker/separate.py not found", "error")
        _emit("  Run: bash separate_worker/install.sh", "error")
        sys.exit(1)

    for p in input_paths:
        cmd = [
            str(CONDA_BIN), "run", "-n", SEPARATE_ENV,
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
            _emit(f"ERROR: Separation failed.", "error")
            sys.exit(1)
        if result.stdout.strip():
            print(result.stdout.strip())


# ── CLI Parser ───────────────────────────────────────────────────────────────

class _HelpOnErrorParser(argparse.ArgumentParser):
    """Show full help instead of cryptic error on invalid usage."""
    def error(self, message):
        print(f"Error: {message}\n", file=sys.stderr)
        self.print_help(sys.stderr)
        sys.exit(2)


def build_parser():
    parser = _HelpOnErrorParser(
        prog="revoicer",
        description="Voice conversion CLI — consistent AI voices for WAV files.",
    )

    # --PS flag at top level
    parser.add_argument("--PS", action="store_true",
                        help="Show available models and supported languages")

    sub = parser.add_subparsers(dest="command")

    # ── convert ──────────────────────────────────────────────────────────
    p_conv = sub.add_parser("convert", help="Convert audio file(s)")
    p_conv.add_argument("input", nargs="+", help="Input audio file(s)")
    p_conv.add_argument("-o", "--output", help="Output directory")
    p_conv.add_argument("-v", "--voice", help="Voice model name")
    p_conv.add_argument("--decoder", choices=["rmvpe", "crepe", "harvest", "pm"],
                        help="Pitch detection algorithm (default: rmvpe)")
    p_conv.add_argument("--pitch", type=int,
                        help="Manual pitch shift in semitones (disables auto-pitch)")
    p_conv.add_argument("--target-hz", type=float,
                        help="Target voice pitch in Hz (overrides model config)")
    p_conv.set_defaults(func=cmd_convert)

    # ── enhance ─────────────────────────────────────────────────────────
    p_enh = sub.add_parser("enhance",
                           help="Enhance audio (denoise + super-resolution)")
    p_enh.add_argument("input", nargs="+", help="Input audio file(s)")
    p_enh.add_argument("-o", "--output", help="Output directory")
    p_enh.add_argument("--denoise-only", action="store_true",
                        help="Only denoise, skip super-resolution (faster)")
    p_enh.add_argument("--enhance-only", action="store_true",
                        help="Only super-resolution, skip denoising")
    p_enh.set_defaults(func=cmd_enhance)

    # ── music ─────────────────────────────────────────────────────────
    p_music = sub.add_parser("music",
                             help="Generate music (ACE-Step / HeartMuLa)")
    p_music.add_argument("--engine",
                         choices=["ace", "ace-turbo", "ace-sft", "ace-base", "heart"],
                         default="ace",
                         help="Music engine: ace/ace-turbo (8 steps), ace-sft (50 steps, high quality), ace-base (50 steps, all features), heart (default: ace)")
    lyrics_grp = p_music.add_mutually_exclusive_group(required=True)
    lyrics_grp.add_argument("--lyrics", "-l", help="Inline lyrics text")
    lyrics_grp.add_argument("--lyrics-file", "-f",
                            help="Path to lyrics text file")
    p_music.add_argument("--tags", "-t", required=True,
                         help="Style tags or caption (e.g. 'disco,happy,synthesizer')")
    p_music.add_argument("-o", "--output",
                         help="Output file path or directory (default: ./music_<timestamp>.mp3)")
    p_music.add_argument("--seconds", "-s", type=int, default=20,
                         help="Max audio length in seconds (default: 20)")
    p_music.add_argument("--duration", type=int,
                         help="Max audio length in ms (overrides --seconds)")
    p_music.add_argument("--seed", type=int, default=None,
                         help="Random seed for reproducibility")
    p_music.add_argument("--timeout", type=int, default=1800,
                         help="Generation timeout in seconds (default: 1800)")

    # Shared params (mapped to engine-specific names internally)
    p_music.add_argument("--topk", type=int, default=None,
                         help="Top-k sampling (heart: 50, ace: 0=off)")
    p_music.add_argument("--temperature", type=float, default=None,
                         help="Sampling temperature (heart: 1.0, ace: 0.85)")
    p_music.add_argument("--cfg-scale", type=float, default=None,
                         help="CFG scale (heart: 1.5, ace: 7.0)")

    # ACE-Step specific params
    p_music.add_argument("--steps", type=int, default=None,
                         help="[ace] Inference steps (default: 8)")
    p_music.add_argument("--shift", type=float, default=None,
                         help="[ace] Timestep shift (default: 3.0)")
    p_music.add_argument("--no-thinking", action="store_true",
                         help="[ace] Disable LM chain-of-thought")
    p_music.add_argument("--infer-method", choices=["ode", "sde"], default=None,
                         help="[ace] Inference method (default: ode)")
    p_music.add_argument("--lm-cfg", type=float, default=None,
                         help="[ace] LM guidance scale (default: 2.0)")
    p_music.add_argument("--top-p", type=float, default=None,
                         help="[ace] Nucleus sampling (default: 0.9)")
    p_music.add_argument("--batch-size", type=int, default=None,
                         help="[ace] Parallel samples (default: 1)")
    p_music.add_argument("--instrumental", action="store_true",
                         help="[ace] Force instrumental output")
    p_music.add_argument("--bpm", type=int, default=None,
                         help="Beats per minute (default: auto)")
    p_music.add_argument("--keyscale", type=str, default=None,
                         help="Musical key (e.g., 'C Major', 'Am') (default: auto)")
    p_music.add_argument("--timesignature", type=str, default=None,
                         help="Time signature: 2=2/4, 3=3/4, 4=4/4, 6=6/8 (default: auto)")
    p_music.set_defaults(func=cmd_music)

    # ── transcribe ─────────────────────────────────────────────────────
    p_tr = sub.add_parser("transcribe",
                           help="Transcribe audio to text (mlx-whisper)")
    p_tr.add_argument("input", nargs="+", help="Input audio file(s)")
    p_tr.add_argument("--model",
                       choices=["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"],
                       default="large-v3-turbo",
                       help="Whisper model size (default: large-v3-turbo)")
    p_tr.add_argument("--input-language", default=None,
                       help="Input language hint (e.g. 'en', 'de', 'ja'). Auto-detect if omitted.")
    p_tr.add_argument("--word-timestamps", action="store_true",
                       help="Include word-level timestamps")
    p_tr.add_argument("--format", default="json",
                       help="Output format: json, txt, srt, vtt, tsv, all (default: json)")
    p_tr.add_argument("-o", "--output",
                       help="Output directory for transcript files")
    p_tr.add_argument("--timeout", type=int, default=600,
                       help="Timeout in seconds (default: 600)")
    p_tr.set_defaults(func=cmd_transcribe)

    # ── diarize ──────────────────────────────────────────────────────
    p_dia = sub.add_parser("diarize",
                            help="Split dialogue into separate speaker tracks")
    p_dia.add_argument("input", nargs="+", help="Input audio file(s)")
    p_dia.add_argument("-o", "--output", help="Output directory")
    p_dia.add_argument("--speakers", type=int, default=None,
                        help="Number of speakers (auto-detect if not set)")
    p_dia.add_argument("--hf-token", default=None,
                        help="HuggingFace token (or set HF_TOKEN env var)")
    p_dia.add_argument("--verify", action="store_true",
                        help="Show diarization statistics (segments, coverage, gaps, overlaps)")
    p_dia.set_defaults(func=cmd_diarize)

    # ── separate ─────────────────────────────────────────────────────
    p_sep = sub.add_parser("separate",
                            help="Separate audio into stems (vocals/drums/bass/other)")
    p_sep.add_argument("input", nargs="+", help="Input audio file(s)")
    p_sep.add_argument("-o", "--output", help="Output directory")
    p_sep.add_argument("--model", default=None,
                        help="Demucs model (default: htdemucs, alt: htdemucs_ft)")
    p_sep.set_defaults(func=cmd_separate)

    # ── transcribe-lyrics ─────────────────────────────────────────────
    p_tl = sub.add_parser("transcribe-lyrics",
                           help="Transcribe lyrics from audio (HeartTranscriptor)")
    p_tl.add_argument("input", help="Input audio file (MP3, WAV, etc.)")
    p_tl.add_argument("-o", "--output",
                       help="Output file for lyrics (default: print to stdout)")
    p_tl.add_argument("--timeout", type=int, default=600,
                       help="Transcription timeout in seconds (default: 600)")
    p_tl.set_defaults(func=cmd_transcribe_lyrics)

    # ── models ───────────────────────────────────────────────────────────
    p_models = sub.add_parser("models", help="Manage voice models")
    m_sub = p_models.add_subparsers(dest="models_cmd")

    p_ml = m_sub.add_parser("list", help="List installed models")
    p_ml.set_defaults(func=cmd_models_list)

    p_ms = m_sub.add_parser("search", help="Search HuggingFace for RVC models")
    p_ms.add_argument("query", help="Search terms")
    p_ms.add_argument("--limit", type=int, default=30)
    p_ms.set_defaults(func=cmd_models_search)

    p_mi = m_sub.add_parser("install", help="Install model from HuggingFace")
    p_mi.add_argument("model_id", help="HuggingFace repo ID or direct download URL")
    p_mi.add_argument("--name", help="Local name for the model")
    p_mi.add_argument("--file", help="Specific file to install from multi-model repos")
    p_mi.set_defaults(func=cmd_models_install)

    p_mr = m_sub.add_parser("remove", help="Remove an installed model")
    p_mr.add_argument("name", help="Model name")
    p_mr.set_defaults(func=cmd_models_remove)

    p_mc = m_sub.add_parser("calibrate",
                            help="Calibrate target F0 for a model (run once per model)")
    p_mc.add_argument("name", help="Model name")
    p_mc.set_defaults(func=cmd_models_calibrate)

    p_mf = m_sub.add_parser("set-pitch", help="Manually set target pitch (Hz) for a model")
    p_mf.add_argument("name", help="Model name")
    p_mf.add_argument("hz", type=float, help="Target pitch in Hz (male ~120, female ~220, child ~280)")
    p_mf.set_defaults(func=cmd_models_set_f0)

    # ── server ───────────────────────────────────────────────────────────
    p_server = sub.add_parser("server", help="Manage RVC worker")
    s_sub = p_server.add_subparsers(dest="server_cmd")

    p_ss = s_sub.add_parser("start", help="Start RVC worker")
    p_ss.add_argument("-p", "--port", type=int, default=5100)
    p_ss.set_defaults(func=cmd_server_start)

    p_st = s_sub.add_parser("stop", help="Stop RVC worker")
    p_st.set_defaults(func=cmd_server_stop)

    p_su = s_sub.add_parser("status", help="Check worker status")
    p_su.set_defaults(func=cmd_server_status)

    return parser


def main():
    global _event_handler

    # Extract --screen-log-format before argparse (works anywhere in argv)
    argv = list(sys.argv[1:])
    if "--screen-log-format" in argv:
        idx = argv.index("--screen-log-format")
        if idx + 1 < len(argv) and argv[idx + 1] == "json":
            _event_handler = print_event_json
        argv[idx:idx + 2] = []

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.PS:
        cmd_ps(args)
        return

    if not args.command:
        parser.print_help()
        return

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.parse_args([args.command, "--help"])


if __name__ == "__main__":
    main()
