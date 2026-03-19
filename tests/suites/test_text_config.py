"""Test: Text Worker — Config endpoints (set, show, reset)."""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
MODEL = "qwen3.5:latest"
ENGINE = "ollama"

_BASE = [sys.executable, "generate.py", "text", ENGINE, "--model", MODEL]


def _add_tui_json(suite, name, extra_args):
    """Register a test twice: once TUI, once JSON."""
    suite.add(name=f"Config: {name} (TUI)", cmd=[*_BASE, *extra_args])
    suite.add(name=f"Config: {name} (JSON)", cmd=[*_BASE, *extra_args, "--screen-log-format", "json"])


def register(suite):
    _add_tui_json(suite, "set", [
        "--endpoint", "set", "--context-length", "128000",
    ])

    _add_tui_json(suite, "show", [
        "--endpoint", "show",
    ])

    _add_tui_json(suite, "reset", [
        "--endpoint", "reset",
    ])
