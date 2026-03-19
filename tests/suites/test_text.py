"""Test: Text Worker — LLM inference via Ollama."""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
MODEL = "qwen3.5:latest"
ENGINE = "ollama"

_BASE = [sys.executable, "generate.py", "text", ENGINE, "--model", MODEL]


def _add_tui_json(suite, name, extra_args, output_tui=None, output_json=None):
    """Register a test twice: once TUI, once JSON."""
    suite.add(name=f"Text: {name} (TUI)", cmd=[*_BASE, *extra_args], output=output_tui)
    json_args = [a for a in extra_args]
    # Replace -o path with JSON variant
    if output_json:
        for i, a in enumerate(json_args):
            if a == "-o" and i + 1 < len(json_args):
                json_args[i + 1] = str(output_json)
                break
    suite.add(name=f"Text: {name} (JSON)", cmd=[*_BASE, *json_args, "--screen-log-format", "json"], output=output_json)


def register(suite):
    out = suite.out_dir

    # ── Generate ─────────────────────────────────────────────────────────

    _add_tui_json(suite, "generate", [
        "--endpoint", "generate",
        "--prompt", "Say hello in one sentence.",
        "-o", str(out / "generate_tui.md"),
    ], output_tui=out / "generate_tui.md", output_json=out / "generate_json.md")

    _add_tui_json(suite, "generate + thinking", [
        "--endpoint", "generate",
        "--prompt", "What is 15 * 17?",
        "--thinking", "True",
        "-o", str(out / "generate_thinking_tui.md"),
    ], output_tui=out / "generate_thinking_tui.md", output_json=out / "generate_thinking_json.md")

    # ── Chat ─────────────────────────────────────────────────────────────

    _add_tui_json(suite, "chat", [
        "--endpoint", "chat",
        "--messages", '[{"role":"user","content":"What is 2+2? Answer in one word."}]',
        "-o", str(out / "chat_tui.md"),
    ], output_tui=out / "chat_tui.md", output_json=out / "chat_json.md")

    _add_tui_json(suite, "chat + thinking", [
        "--endpoint", "chat",
        "--messages", '[{"role":"user","content":"What is 15 * 17?"}]',
        "--thinking", "True",
        "-o", str(out / "chat_thinking_tui.md"),
    ], output_tui=out / "chat_thinking_tui.md", output_json=out / "chat_thinking_json.md")

