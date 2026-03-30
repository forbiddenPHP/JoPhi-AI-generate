"""Test: Text Worker — Config endpoints (set, show, reset)."""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
MODEL = "qwen3.5:latest"
ENGINE = "ollama"

_BASE_STR = f"{sys.executable} generate.py text {ENGINE} --model {MODEL}"


def _add_tui_json(suite, name, extra_args_str, out):
    """Register a test twice: once TUI, once JSON."""
    slug = name.replace(" ", "_")
    out_tui = out / f"config_{slug}_tui.txt"
    out_json = out / f"config_{slug}_json.txt"
    suite.add(
        name=f"Config: {name} (TUI)",
        cmd=["bash", "-c", f"{_BASE_STR} {extra_args_str} 2>&1 | tee {out_tui}"],
        output=out_tui,
    )
    suite.add(
        name=f"Config: {name} (JSON)",
        cmd=["bash", "-c", f"{_BASE_STR} {extra_args_str} --screen-log-format json 2>&1 | tee {out_json}"],
        output=out_json,
    )


def register(suite):
    out = suite.out_dir
    _add_tui_json(suite, "set", "--endpoint set --context-length 128000", out)
    _add_tui_json(suite, "show", "--endpoint show", out)
    _add_tui_json(suite, "reset", "--endpoint reset", out)
