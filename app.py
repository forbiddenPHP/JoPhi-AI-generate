#!/usr/bin/env python3
"""
Revoicer Web App — Browser UI for voice conversion.

Features:
  - Dashboard: RVC worker status, installed models
  - Model browser: search & install from HuggingFace
  - Convert: upload WAV, choose voice, convert, listen
  - Batch: convert multiple files at once
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_file

import requests as http_requests

app = Flask(__name__)

RVC_API_URL = os.environ.get("RVC_API_URL", "http://127.0.0.1:5100")
UPLOAD_DIR = Path(tempfile.gettempdir()) / "revoicer_uploads"
OUTPUT_DIR = Path(tempfile.gettempdir()) / "revoicer_output"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def rvc_ok() -> bool:
    try:
        http_requests.get(f"{RVC_API_URL}/models", timeout=3)
        return True
    except (http_requests.ConnectionError, http_requests.Timeout):
        return False


def rvc_get(endpoint: str):
    return http_requests.get(f"{RVC_API_URL}{endpoint}", timeout=10).json()


def rvc_post(endpoint: str, **kwargs):
    return http_requests.post(f"{RVC_API_URL}{endpoint}", timeout=300, **kwargs)


# ── Pages ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── API Endpoints ────────────────────────────────────────────────────────────

@app.route("/api/status")
def api_status():
    if rvc_ok():
        models = rvc_get("/models")
        return jsonify({"running": True, "models": models.get("models", [])})
    return jsonify({"running": False, "models": []})


@app.route("/api/models")
def api_models():
    if not rvc_ok():
        return jsonify({"error": "RVC worker not running"}), 503
    return jsonify(rvc_get("/models"))


def _is_rvc_repo(files: list[str]) -> bool:
    """Check if a repo looks like an RVC voice model (not an LLM)."""
    basenames = {f.split("/")[-1] for f in files}

    pth = [f for f in files if f.endswith(".pth")]
    if not pth:
        return False

    llm_indicators = {
        "tokenizer.json", "tokenizer_config.json", "vocab.txt",
        "merges.txt", "sentencepiece.bpe.model", "special_tokens_map.json",
        "generation_config.json", "vocab.json",
    }
    if llm_indicators & basenames:
        return False

    if any(f.endswith(".safetensors") for f in files):
        return False

    voice_pth = [f for f in pth if not any(
        f.split("/")[-1].startswith(p) for p in ("D_", "G_", "D-", "G-", "f0")
    )]
    if not voice_pth and len(pth) != 1:
        return False

    return True


@app.route("/api/models/search")
def api_models_search():
    query = request.args.get("q", "")
    limit = int(request.args.get("limit", "20"))
    if not query:
        return jsonify({"error": "Missing query parameter 'q'"}), 400

    try:
        from huggingface_hub import HfApi
        api = HfApi()
        words = query.strip().split()

        # One search per term, scoped to "rvc"
        search_queries = {f"rvc {query}"}
        for w in words:
            search_queries.add(f"rvc {w}")

        seen = {}
        for sq in search_queries:
            try:
                for m in api.list_models(search=sq, sort="downloads", limit=limit):
                    if m.id not in seen:
                        seen[m.id] = m
            except Exception:
                pass

        results = sorted(seen.values(), key=lambda m: m.downloads or 0, reverse=True)

        return jsonify({"results": [
            {"id": m.id, "downloads": m.downloads or 0}
            for m in results
        ]})
    except ImportError:
        return jsonify({"error": "huggingface-hub not installed"}), 500


@app.route("/api/models/install", methods=["POST"])
def api_models_install():
    data = request.json
    model_id = data.get("model_id")
    name = data.get("name") or model_id.split("/")[-1].lower().replace(" ", "-")[:40]

    if not model_id:
        return jsonify({"error": "Missing model_id"}), 400

    try:
        import zipfile
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi()
        files = api.list_repo_files(model_id)
        pth_files = [f for f in files if f.endswith(".pth")]
        idx_files = [f for f in files if f.endswith(".index")]

        if not pth_files:
            return jsonify({"error": f"No .pth files in {model_id}"}), 404

        local_pth = hf_hub_download(model_id, pth_files[0])
        local_idx = hf_hub_download(model_id, idx_files[0]) if idx_files else None

        # Pack into .zip for RVC worker API
        zip_path = Path(tempfile.mktemp(suffix=".zip", prefix="revoicer_"))
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(local_pth, f"{name}.pth")
            if local_idx:
                zf.write(local_idx, f"{name}.index")

        with open(zip_path, "rb") as f:
            rvc_post("/upload_model", files={"file": (f"{name}.zip", f)})
        zip_path.unlink(missing_ok=True)

        return jsonify({"ok": True, "name": name})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/models/<name>", methods=["DELETE"])
def api_models_remove(name):
    try:
        r = http_requests.delete(f"{RVC_API_URL}/models/{name}", timeout=10)
        return jsonify({"ok": r.ok})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/convert", methods=["POST"])
def api_convert():
    if not rvc_ok():
        return jsonify({"error": "RVC worker not running"}), 503

    voice = request.form.get("voice")
    f0_method = request.form.get("f0_method", "rmvpe")
    pitch = request.form.get("pitch", "0")

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    input_path = UPLOAD_DIR / f.filename
    f.save(input_path)

    # Set model
    if voice:
        rvc_post(f"/models/{voice}")

    # Set params
    rvc_post("/params", data={"f0_method": f0_method, "pitch": int(pitch)})

    # Convert
    with open(input_path, "rb") as audio_file:
        audio_b64 = base64.b64encode(audio_file.read()).decode()

    r = rvc_post("/convert", data={
        "audio_data": audio_b64,
        "output_format": "wav",
    })

    output_path = OUTPUT_DIR / f"converted_{f.filename}"
    output_path = output_path.with_suffix(".wav")
    output_path.write_bytes(r.content)

    return jsonify({
        "ok": True,
        "output": str(output_path),
        "filename": output_path.name,
    })


@app.route("/api/audio/<filename>")
def api_audio(filename):
    path = OUTPUT_DIR / filename
    if not path.exists():
        path = UPLOAD_DIR / filename
    if not path.exists():
        return jsonify({"error": "File not found"}), 404
    return send_file(path, mimetype="audio/wav")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
