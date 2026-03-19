"""Test: Text Worker — Vision (local images, URLs, base64 history)."""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
MODEL = "qwen3.5:latest"
ENGINE = "ollama"
TEST_IMAGE = SCRIPT_DIR / "tests" / "assets" / "test.png"
IMAGE_URL = "https://www.ihpoj.at/~/2/media/0_johannes-in-bulgarien.jpg"

_BASE = [sys.executable, "generate.py", "text", ENGINE, "--model", MODEL]


def _add_tui_json(suite, name, extra_args, output_tui=None, output_json=None):
    """Register a test twice: once TUI, once JSON."""
    suite.add(name=f"Vision: {name} (TUI)", cmd=[*_BASE, *extra_args], output=output_tui)
    json_args = [a for a in extra_args]
    if output_json:
        for i, a in enumerate(json_args):
            if a == "-o" and i + 1 < len(json_args):
                json_args[i + 1] = str(output_json)
                break
    suite.add(name=f"Vision: {name} (JSON)", cmd=[*_BASE, *json_args, "--screen-log-format", "json"], output=output_json)


def register(suite):
    out = suite.out_dir

    # ── Local image ──────────────────────────────────────────────────────

    _add_tui_json(suite, "local image", [
        "--endpoint", "chat",
        "--messages", '[{"role":"user","content":"Describe this image in one sentence."}]',
        "--images", str(TEST_IMAGE),
        "-o", str(out / "vision_local_tui.md"),
    ], output_tui=out / "vision_local_tui.md", output_json=out / "vision_local_json.md")

    _add_tui_json(suite, "local image + thinking", [
        "--endpoint", "chat",
        "--messages", '[{"role":"user","content":"What do you see? Think step by step."}]',
        "--images", str(TEST_IMAGE),
        "--thinking", "True",
        "-o", str(out / "vision_local_thinking_tui.md"),
    ], output_tui=out / "vision_local_thinking_tui.md", output_json=out / "vision_local_thinking_json.md")

    # ── Image URL in prompt ──────────────────────────────────────────────

    _add_tui_json(suite, "image URL", [
        "--endpoint", "chat",
        "--messages", f'[{{"role":"user","content":"Describe this person in one sentence. {IMAGE_URL}"}}]',
        "-o", str(out / "vision_url_tui.md"),
    ], output_tui=out / "vision_url_tui.md", output_json=out / "vision_url_json.md")

    # ── Multi-turn with base64 history ───────────────────────────────────

    b64_file = SCRIPT_DIR / "tests" / "assets" / "johannes.base64"
    history_json = suite.out_dir / "history_b64.json"

    suite.add(
        name="Vision: Prep: build base64 history JSON",
        cmd=[
            sys.executable, "-c",
            f"""import json
b64 = open("{b64_file}").read().strip()
messages = [
    {{"role": "user", "content": "Describe this person.", "images": [b64]}},
    {{"role": "assistant", "content": "A man with dark hair, a beard, and glasses near a pool."}},
    {{"role": "user", "content": "Welche Augenfarbe hat Johannes?"}}
]
open("{history_json}", "w").write(json.dumps(messages))
""",
        ],
        output=history_json,
        prep=True,
    )

    _add_tui_json(suite, "base64 history", [
        "--endpoint", "chat",
        "--messages", str(history_json),
        "-o", str(out / "vision_b64_history_tui.md"),
    ], output_tui=out / "vision_b64_history_tui.md", output_json=out / "vision_b64_history_json.md")
