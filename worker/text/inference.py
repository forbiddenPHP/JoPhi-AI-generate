#!/usr/bin/env python3
"""Text Worker — LLM inference via local engines.

Called by generate.py via: conda run -n text python worker/text/inference.py <args>

Uses the ollama Python package for Ollama engine.

Events are emitted to stderr as @inference:{JSON} lines,
parsed by progress.py in the main process.
"""

import argparse
import io
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# ── Imports ──────────────────────────────────────────────────────────────────

try:
    from ollama import Client
except ImportError:
    print("ERROR: ollama not installed. Run: bash worker/text/install.sh",
          file=sys.stderr)
    sys.exit(1)

try:
    import requests
except ImportError:
    requests = None


# ── Constants ────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"
ENGINES_FILE = SCRIPT_DIR / "engines.json"

_LLM_ENGINES = ("ollama",)

_PARAM_MAP_OLLAMA = {
    "context_length": "num_ctx",
    "max_tokens": "num_predict",
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "repeat_penalty": "repeat_penalty",
    "seed": "seed",
    "stop": "stop",
}

_IMAGE_MAX_WIDTH = 500


# ── Event Protocol ───────────────────────────────────────────────────────────

def _event(event: str, **kwargs):
    """Emit a structured inference event to stderr.

    Parsed by progress.py via @inference: prefix detection.
    """
    payload = {"event": event, **kwargs}
    print(f"@inference:{json.dumps(payload, ensure_ascii=False)}",
          file=sys.stderr, flush=True)


# ── Connection Helpers ───────────────────────────────────────────────────────

def _engine_base_url(engine: str, args) -> str:
    if getattr(args, "base_url", None):
        return args.base_url.rstrip("/")
    if ENGINES_FILE.exists():
        cfg = json.loads(ENGINES_FILE.read_text())
        if engine in cfg and "base_url" in cfg[engine]:
            return cfg[engine]["base_url"].rstrip("/")
    defaults = {"ollama": "http://localhost:11434"}
    return defaults.get(engine, "http://localhost:8000")


# ── Config System ────────────────────────────────────────────────────────────

def _model_config_dir(engine: str, model: str) -> Path:
    safe_name = model.replace("/", "_").replace(":", "_")
    return MODELS_DIR / engine / safe_name


def _load_config(engine: str, model: str) -> dict:
    cfg_file = _model_config_dir(engine, model) / "config.json"
    if cfg_file.exists():
        return json.loads(cfg_file.read_text())
    return {}


def _save_config(engine: str, model: str, config: dict):
    cfg_dir = _model_config_dir(engine, model)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_file = cfg_dir / "config.json"
    if cfg_file.exists():
        ts = time.strftime("%Y-%m-%d_%H-%M-%S")
        backup = cfg_dir / f"config.{ts}.json"
        backup.write_text(cfg_file.read_text())
    cfg_file.write_text(json.dumps(config, indent=2) + "\n")


def _reset_config(engine: str, model: str):
    cfg_dir = _model_config_dir(engine, model)
    backups = sorted(cfg_dir.glob("config.*.json"))
    if not backups:
        _event("inference_result", text="No backups found — nothing to reset.")
        return
    oldest = backups[0]
    cfg_file = cfg_dir / "config.json"
    if cfg_file.exists():
        ts = time.strftime("%Y-%m-%d_%H-%M-%S")
        backup = cfg_dir / f"config.{ts}.json"
        backup.write_text(cfg_file.read_text())
    cfg_file.write_text(oldest.read_text())
    msg = f"Config reset to {oldest.name}"
    _event("inference_result", text=msg)


# ── Param Builder ────────────────────────────────────────────────────────────

def _build_options(args) -> dict:
    """Build Ollama options dict from CLI args + config."""
    config = _load_config("ollama", args.model)
    params = {}
    our_params = ["context_length", "max_tokens", "temperature", "top_p",
                  "top_k", "repeat_penalty", "seed", "stop"]
    for p in our_params:
        cli_val = getattr(args, p, None)
        if cli_val is not None:
            engine_key = _PARAM_MAP_OLLAMA.get(p, p)
            params[engine_key] = cli_val
        elif p in config:
            engine_key = _PARAM_MAP_OLLAMA.get(p, p)
            params[engine_key] = config[p]
    return params


# ── Thinking ─────────────────────────────────────────────────────────────────

def _parse_thinking(val: str):
    """Parse --thinking value to ollama think parameter.

    Returns: True, False, or string ("low"/"medium"/"high").
    """
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False
    return val  # "low", "medium", "high"


# ── Image Helpers ────────────────────────────────────────────────────────────

def _is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def _download_image(url: str) -> Path:
    """Download image URL to a temp file, return path."""
    import tempfile
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    ct = resp.headers.get("content-type", "")
    if "jpeg" in ct or "jpg" in ct:
        suffix = ".jpg"
    elif "png" in ct:
        suffix = ".png"
    elif "webp" in ct:
        suffix = ".webp"
    elif "gif" in ct:
        suffix = ".gif"
    else:
        # guess from URL
        from urllib.parse import urlparse
        url_path = urlparse(url).path.lower()
        if url_path.endswith(".png"):
            suffix = ".png"
        elif url_path.endswith((".jpg", ".jpeg")):
            suffix = ".jpg"
        else:
            suffix = ".jpg"  # default
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    os.write(fd, resp.content)
    os.close(fd)
    return Path(tmp)


def _prepare_images(paths: list[str]) -> list[bytes]:
    """Resize images to max 500px width, return as bytes for ollama client.

    Accepts local file paths and URLs (http/https).
    """
    try:
        from PIL import Image as PILImage
    except ImportError:
        print("ERROR: Pillow required for --images. Run: bash worker/text/install.sh",
              file=sys.stderr)
        sys.exit(1)

    tmp_files = []
    results = []
    for p in paths:
        if _is_url(p):
            _event("inference_mode", mode="downloading image")
            path = _download_image(p)
            tmp_files.append(path)
        else:
            path = Path(p).resolve()
            if not path.exists():
                print(f"ERROR: Image not found: {path}", file=sys.stderr)
                sys.exit(1)

        img = PILImage.open(path)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        if img.width > _IMAGE_MAX_WIDTH:
            ratio = _IMAGE_MAX_WIDTH / img.width
            new_h = int(img.height * ratio)
            img = img.resize((_IMAGE_MAX_WIDTH, new_h), PILImage.LANCZOS)

        suffix = path.suffix.lower()
        if suffix in (".jpg", ".jpeg"):
            fmt = "JPEG"
        else:
            fmt = "PNG"

        buf = io.BytesIO()
        img.save(buf, format=fmt, quality=85)
        results.append(buf.getvalue())

    # cleanup temp files
    for tmp in tmp_files:
        try:
            tmp.unlink()
        except OSError:
            pass

    return results


# ── URL Extraction from Text ─────────────────────────────────────────────────

_IMAGE_URL_RE = re.compile(
    r'(https?://\S+\.(?:jpg|jpeg|png|gif|webp|bmp|tiff))\b',
    re.IGNORECASE,
)


def _extract_image_urls(text: str) -> tuple[str, list[str]]:
    """Extract image URLs from text, return (cleaned_text, urls).

    URLs are removed from the text. A PS note is appended referencing them.
    """
    urls = _IMAGE_URL_RE.findall(text)
    if not urls:
        return text, []
    cleaned = _IMAGE_URL_RE.sub("", text).strip()
    ps_lines = [f"\n\nPS: Image from URL {u} as attached" for u in urls]
    return cleaned + "".join(ps_lines), urls


# ── Ollama Endpoints ─────────────────────────────────────────────────────────

def cmd_chat(args, client: Client):
    """Handle --endpoint chat via ollama package."""
    messages_raw = args.messages
    if not messages_raw:
        print("ERROR: --messages required for endpoint 'chat'", file=sys.stderr)
        sys.exit(1)

    if Path(messages_raw).exists():
        messages = json.loads(Path(messages_raw).read_text())
    else:
        messages = json.loads(messages_raw)

    # Extract image URLs from user messages (skip messages that already have images)
    url_images = []
    for msg in messages:
        if msg.get("role") == "user" and msg.get("content") and not msg.get("images"):
            cleaned, urls = _extract_image_urls(msg["content"])
            if urls:
                msg["content"] = cleaned
                url_images.extend(urls)

    options = _build_options(args)
    all_image_paths = list(args.images or []) + url_images
    images = _prepare_images(all_image_paths) if all_image_paths else None
    think_val = _parse_thinking(args.thinking)

    # Inject new images into last user message (preserve existing base64 from history)
    if images:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                existing = msg.get("images", [])
                msg["images"] = list(existing) + images
                break

    _event("inference_gotcha")

    if args.stream:
        _event("inference_mode", mode="stream")
        parts = []
        stream = client.chat(
            model=args.model, messages=messages, stream=True,
            think=think_val, options=options,
        )
        for chunk in stream:
            text = chunk.message.content or ""
            if text:
                _event("inference_token", text=text)
                parts.append(text)
        output = "".join(parts)
    else:
        _event("inference_mode", mode="sync")
        resp = client.chat(
            model=args.model, messages=messages,
            think=think_val, options=options,
        )
        thinking = getattr(resp.message, "thinking", "") or ""
        output = resp.message.content or ""
        if thinking:
            output = f"<thinking>{thinking}</thinking>\n{output}"

    _event("inference_result", text=output)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")


def cmd_generate(args, client: Client):
    """Handle --endpoint generate via ollama package."""
    prompt = args.prompt
    if not prompt:
        print("ERROR: --prompt required for endpoint 'generate'",
              file=sys.stderr)
        sys.exit(1)

    # Extract image URLs from prompt
    prompt, url_images = _extract_image_urls(prompt)

    options = _build_options(args)
    all_image_paths = list(args.images or []) + url_images
    images = _prepare_images(all_image_paths) if all_image_paths else None
    think_val = _parse_thinking(args.thinking)
    system = args.system

    _event("inference_gotcha")

    if args.stream:
        _event("inference_mode", mode="stream")
        parts = []
        kwargs = dict(
            model=args.model, prompt=prompt, stream=True,
            think=think_val, options=options,
        )
        if system:
            kwargs["system"] = system
        if images:
            kwargs["images"] = images
        stream = client.generate(**kwargs)
        for chunk in stream:
            text = chunk.response or ""
            if text:
                _event("inference_token", text=text)
                parts.append(text)
        output = "".join(parts)
    else:
        _event("inference_mode", mode="sync")
        kwargs = dict(
            model=args.model, prompt=prompt,
            think=think_val, options=options,
        )
        if system:
            kwargs["system"] = system
        if images:
            kwargs["images"] = images
        resp = client.generate(**kwargs)
        thinking = getattr(resp, "thinking", "") or ""
        output = resp.response or ""
        if thinking:
            output = f"<thinking>{thinking}</thinking>\n{output}"

    _event("inference_result", text=output)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")


def cmd_set(args, engine):
    """Handle --endpoint set — save default overrides for a model."""
    config = _load_config(engine, args.model)
    our_params = ["context_length", "max_tokens", "temperature", "top_p",
                  "top_k", "repeat_penalty", "seed", "stop"]
    changed = []
    for p in our_params:
        val = getattr(args, p, None)
        if val is not None:
            config[p] = val
            changed.append(f"{p}={val}")
    if not changed:
        msg = "No parameters specified."
        _event("inference_result", text=msg)
        return
    _save_config(engine, args.model, config)
    msg = f"Config saved: {', '.join(changed)}"
    _event("inference_result", text=msg)


def cmd_show(args, engine, client: Client):
    """Handle --endpoint show — display model config."""
    config = _load_config(engine, args.model)
    lines = []
    if config:
        lines.append(json.dumps(config, indent=2))
    else:
        lines.append("No config overrides saved.")

    if engine == "ollama":
        try:
            info = client.show(args.model)
            details = getattr(info, "details", None)
            if details:
                lines.append("\nOllama model info:")
                for k, v in vars(details).items():
                    if v:
                        lines.append(f"  {k}: {v}")
        except Exception:
            pass

    output = "\n".join(lines)
    _event("inference_result", text=output)


def cmd_reset(args, engine):
    """Handle --endpoint reset — restore original config."""
    _reset_config(engine, args.model)


def cmd_load(args, engine, client: Client):
    """Handle --endpoint load — load model into VRAM."""
    if engine == "ollama":
        client.generate(model=args.model, prompt="", keep_alive=-1)
    msg = f"Model {args.model} loaded on {engine}"
    _event("inference_result", text=msg)

    cfg_dir = _model_config_dir(engine, args.model)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_file = cfg_dir / "config.json"
    if not cfg_file.exists():
        cfg_file.write_text("{}\n")
        ts = time.strftime("%Y-%m-%d_%H-%M-%S")
        backup = cfg_dir / f"config.{ts}.json"
        backup.write_text("{}\n")


def cmd_unload(args, engine, client: Client):
    """Handle --endpoint unload — unload model from VRAM."""
    if engine == "ollama":
        client.generate(model=args.model, prompt="", keep_alive=0)
    msg = f"Model {args.model} unloaded from {engine}"
    _event("inference_result", text=msg)


# ── Main ─────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(description="Text Worker — LLM Inference")
    p.add_argument("--engine", required=True, choices=list(_LLM_ENGINES))
    p.add_argument("--model", required=True)
    p.add_argument("--endpoint", required=True,
                   choices=["chat", "generate", "set", "show", "reset",
                            "load", "unload"])
    p.add_argument("--prompt", default=None)
    p.add_argument("--system", default=None)
    p.add_argument("--messages", default=None)
    p.add_argument("--images", nargs="+", default=None)
    p.add_argument("--thinking", default="False",
                   choices=["True", "False", "low", "medium", "high"])
    p.add_argument("--context-length", type=int, default=None,
                   dest="context_length")
    p.add_argument("--max-tokens", type=int, default=None, dest="max_tokens")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None, dest="top_p")
    p.add_argument("--top-k", type=int, default=None, dest="top_k")
    p.add_argument("--repeat-penalty", type=float, default=None,
                   dest="repeat_penalty")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--stop", default=None)
    p.add_argument("--stream", action="store_true")
    p.add_argument("--base-url", default=None, dest="base_url")
    p.add_argument("--api-key", default=None, dest="api_key")
    p.add_argument("-o", "--output", default=None)
    return p


def main():
    args = build_parser().parse_args()
    engine = args.engine

    if engine != "ollama":
        print(f"ERROR: Engine '{engine}' is currently not supported",
              file=sys.stderr)
        sys.exit(1)

    base_url = _engine_base_url(engine, args)
    client = Client(host=base_url)

    endpoint = args.endpoint

    if endpoint == "chat":
        cmd_chat(args, client)
    elif endpoint == "generate":
        cmd_generate(args, client)
    elif endpoint == "set":
        cmd_set(args, engine)
    elif endpoint == "show":
        cmd_show(args, engine, client)
    elif endpoint == "reset":
        cmd_reset(args, engine)
    elif endpoint == "load":
        cmd_load(args, engine, client)
    elif endpoint == "unload":
        cmd_unload(args, engine, client)

    # Final output: array of saved file paths
    output_path = getattr(args, "output", None)
    if output_path and Path(output_path).exists():
        print(json.dumps([str(output_path)]))
    else:
        print(json.dumps([]))


if __name__ == "__main__":
    main()
