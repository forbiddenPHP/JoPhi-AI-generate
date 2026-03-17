"""Test: Ollama model management — pull (+ auto num_ctx), show, remove."""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
TEST_MODEL = "smollm2:360m"

_GEN = [sys.executable, "generate.py"]
_MODELS = [*_GEN, "models", "--engine", "ollama"]


def _model_already_installed() -> bool:
    """Check if TEST_MODEL is already in Ollama (someone might be using it)."""
    result = subprocess.run(
        [*_MODELS, "list"],
        capture_output=True, text=True, cwd=SCRIPT_DIR,
    )
    return TEST_MODEL in result.stdout or TEST_MODEL in result.stderr


def register(suite):

    if _model_already_installed():
        print(f"  SKIP: {TEST_MODEL} is already installed — skipping model management tests")
        print(f"         (remove it manually first if you want to run these tests)")
        return

    # 1. Pull model (triggers _ollama_set_max_context automatically)
    suite.add(
        name="Models: pull smollm2:360m",
        cmd=[*_MODELS, "pull", TEST_MODEL],
    )

    # 2. Show model details (verify num_ctx appears in parameters)
    suite.add(
        name="Models: show smollm2:360m (TUI)",
        cmd=[*_MODELS, "show", TEST_MODEL],
    )

    suite.add(
        name="Models: show smollm2:360m (JSON)",
        cmd=[*_MODELS, "show", TEST_MODEL, "--screen-log-format", "json"],
    )

    # 3. Quick inference test (model actually works)
    suite.add(
        name="Models: inference smollm2:360m",
        cmd=[*_GEN, "text", "--engine", "ollama", "--model", TEST_MODEL,
             "--endpoint", "generate", "--prompt", "Say hi."],
    )

    # 4. Remove model
    suite.add(
        name="Models: remove smollm2:360m",
        cmd=[*_MODELS, "remove", TEST_MODEL],
    )
