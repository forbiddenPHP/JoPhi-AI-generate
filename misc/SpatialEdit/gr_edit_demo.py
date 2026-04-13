#!/usr/bin/env python3
"""SpatialEdit – Flask Web UI with Flux.2 backend."""

import sys
import os
import json
import subprocess
import threading
import time
import base64
import tempfile

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from flask import Flask, request, jsonify, send_from_directory

# ── Project root ───────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GENERATE_PY = os.path.join(PROJECT_ROOT, "generate.py")


# ── Status tracking + async job ────────────────────────────────────

_status = {"msg": "Idle", "step": 0, "total": 0}
_log = []
_log_lock = threading.Lock()
_job = {"running": False, "result_img": None, "status_msg": None, "error": None}


def set_status(msg, step=0, total=0):
    _status["msg"] = msg
    _status["step"] = step
    _status["total"] = total
    with _log_lock:
        _log.append({"msg": msg, "step": step, "total": total, "ts": time.time()})
    print(f"[status] {msg}" + (f" ({step}/{total})" if total else ""), flush=True)


# ── Generate via Flux.2 ───────────────────────────────────────────

def _run_generate(input_path, prompt, model, ratio, quality, steps, guidance, seed):
    """Run generate.py image flux.2 in subprocess, parse JSON events."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            out_path = tmp.name

        cmd = [
            sys.executable, GENERATE_PY,
            "image", "flux.2",
            "--model", model,
            "--images", input_path,
            "-p", prompt,
            "--ratio", ratio,
            "--quality", quality,
            "--steps", str(steps),
            "--cfg-scale", str(guidance),
            "--seed", str(seed),
            "-o", out_path,
            "--screen-log-format", "json",
        ]

        set_status("Starting Flux.2...")
        print(f"[generate] CMD: {' '.join(cmd)}", flush=True)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=PROJECT_ROOT,
        )

        # Parse stderr for JSON events
        for line in proc.stderr:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                msg = event.get("message", "")
                etype = event.get("type", "")
                if msg:
                    set_status(f"[{etype}] {msg}" if etype else msg)
            except json.JSONDecodeError:
                set_status(line)

        proc.wait()

        if proc.returncode != 0:
            stdout = proc.stdout.read()
            _job["error"] = f"generate.py exited with code {proc.returncode}\n{stdout}"
            _job["running"] = False
            return

        # stdout ends with JSON array of paths
        stdout = proc.stdout.read().strip()
        paths = None
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if line.startswith("["):
                try:
                    paths = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
        if paths and os.path.exists(paths[0]):
            with open(paths[0], "rb") as rf:
                _job["result_img"] = base64.b64encode(rf.read()).decode()
            _job["status_msg"] = f"Done — {paths[0]}"
            set_status("Done.")
        else:
            _job["error"] = f"No output found. stdout: {stdout}"

        if os.path.exists(out_path):
            os.unlink(out_path)

    except Exception as e:
        _job["error"] = str(e)
        set_status(f"ERROR: {e}")

    _job["running"] = False


# ── Flask App ──────────────────────────────────────────────────────

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/models")
def api_models():
    """Read flux.2 models from generate.py."""
    try:
        result = subprocess.run(
            [sys.executable, GENERATE_PY, "models", "list", "--screen-log-format", "json"],
            capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=10)
        all_models = json.loads(result.stdout)
        flux_models = [m for m in all_models if m.get("engine") == "flux.2"]
        return jsonify(flux_models)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/dims")
def api_dims():
    """Read quality/ratio/dimensions table from generate.py."""
    try:
        result = subprocess.run(
            [sys.executable, GENERATE_PY, "output", "quality", "--screen-log-format", "json"],
            capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=10)
        return jsonify(json.loads(result.stdout))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/status")
def api_status():
    after = int(request.args.get("after", 0))
    with _log_lock:
        entries = _log[after:]
        total = len(_log)
    return jsonify({"entries": entries, "total": total, "current": _status})


@app.route("/upload", methods=["POST"])
def api_upload():
    """Save uploaded image as-is, return base64 preview."""
    if "image" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["image"]
    if not f.filename:
        return jsonify({"error": "No file"}), 400
    # Save raw file as-is
    ext = os.path.splitext(f.filename)[1].lower() or ".bin"
    upload_path = os.path.join(STATIC_DIR, "input_temp" + ext)
    f.save(upload_path)
    # Preview via static URL (browser renders if it can)
    return jsonify({"path": upload_path, "preview_url": f"/static/input_temp{ext}",
                    "filename": f.filename})


@app.route("/generate", methods=["POST"])
def api_generate():
    if _job["running"]:
        return jsonify({"error": "Generation already in progress"}), 409

    data = request.form
    prompt = data.get("prompt", "")
    model = data.get("model", "4b-distilled")
    steps = int(data.get("steps", 4))
    guidance = float(data.get("guidance", 1.0))
    seed = int(data.get("seed", 42))
    quality = data.get("quality", "480p")
    ratio = data.get("ratio", "3:4")

    # Handle uploaded image — save as temp PNG (convert if needed)
    if "image" not in request.files or not request.files["image"].filename:
        return jsonify({"error": "No image uploaded"}), 400

    # Use the file saved by /upload
    f = request.files["image"]
    ext = os.path.splitext(f.filename)[1].lower() or ".bin"
    input_path = os.path.join(STATIC_DIR, "input_temp" + ext)
    f.save(input_path)

    # Reset job state and log
    with _log_lock:
        _log.clear()
    _job["running"] = True
    _job["result_img"] = None
    _job["status_msg"] = None
    _job["error"] = None

    t = threading.Thread(target=_run_generate,
                         args=(input_path, prompt, model, ratio, quality, steps, guidance, seed))
    t.start()

    return jsonify({"started": True, "prompt": prompt, "ratio": ratio, "quality": quality})


@app.route("/result")
def api_result():
    if _job["running"]:
        return jsonify({"done": False})
    if _job["error"]:
        return jsonify({"done": True, "error": _job["error"]})
    return jsonify({"done": True, "image": _job["result_img"], "status": _job["status_msg"]})


print("[init] Ready. Starting Flask …")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False, threaded=True)
