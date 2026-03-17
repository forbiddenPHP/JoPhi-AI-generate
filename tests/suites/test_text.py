"""Test: Text Worker — LLM inference via Ollama."""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
MODEL = "qwen3.5:latest"
ENGINE = "ollama"

_BASE = [sys.executable, "generate.py", "text", "--engine", ENGINE, "--model", MODEL]


def _add_tui_json(suite, name, extra_args):
    """Register a test twice: once TUI, once JSON."""
    suite.add(name=f"Text: {name} (TUI)", cmd=[*_BASE, *extra_args])
    suite.add(name=f"Text: {name} (JSON)", cmd=[*_BASE, *extra_args, "--screen-log-format", "json"])


def register(suite):

    # ── Generate ─────────────────────────────────────────────────────────

    _add_tui_json(suite, "generate", [
        "--endpoint", "generate",
        "--prompt", "Say hello in one sentence.",
    ])

    _add_tui_json(suite, "generate + thinking", [
        "--endpoint", "generate",
        "--prompt", "What is 15 * 17?",
        "--thinking", "True",
    ])

    # ── Chat ─────────────────────────────────────────────────────────────

    _add_tui_json(suite, "chat", [
        "--endpoint", "chat",
        "--messages", '[{"role":"user","content":"What is 2+2? Answer in one word."}]',
    ])

    _add_tui_json(suite, "chat + thinking", [
        "--endpoint", "chat",
        "--messages", '[{"role":"user","content":"What is 15 * 17?"}]',
        "--thinking", "True",
    ])

    # ── Config ───────────────────────────────────────────────────────────

    _add_tui_json(suite, "config set", [
        "--endpoint", "set", "--context-length", "128000",
    ])

    _add_tui_json(suite, "config show", [
        "--endpoint", "show",
    ])

    _add_tui_json(suite, "config reset", [
        "--endpoint", "reset",
    ])
