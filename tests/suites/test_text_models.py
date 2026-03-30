"""Test: Ollama model management — pull (+ auto num_ctx), show, remove."""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
TEST_MODEL = "smollm2:360m"

_GEN = f"{sys.executable} generate.py"
_MODELS = f"{_GEN} models ollama"


def _model_already_installed() -> bool:
    """Check if TEST_MODEL is already in Ollama (someone might be using it)."""
    result = subprocess.run(
        [sys.executable, "generate.py", "models", "ollama", "list"],
        capture_output=True, text=True, cwd=SCRIPT_DIR,
    )
    return TEST_MODEL in result.stdout or TEST_MODEL in result.stderr


def register(suite):
    out = suite.out_dir

    if _model_already_installed():
        print(f"  SKIP: {TEST_MODEL} is already installed — skipping model management tests")
        print(f"         (remove it manually first if you want to run these tests)")
        return

    suite.add(
        name="Models: pull smollm2:360m",
        cmd=["bash", "-c", f"{_MODELS} pull {TEST_MODEL} 2>&1 | tee {out / 'models_pull.txt'}"],
        output=out / "models_pull.txt",
    )

    suite.add(
        name="Models: show smollm2:360m (TUI)",
        cmd=["bash", "-c", f"{_MODELS} show {TEST_MODEL} 2>&1 | tee {out / 'models_show_tui.txt'}"],
        output=out / "models_show_tui.txt",
    )

    suite.add(
        name="Models: show smollm2:360m (JSON)",
        cmd=["bash", "-c", f"{_MODELS} show {TEST_MODEL} --screen-log-format json 2>&1 | tee {out / 'models_show_json.txt'}"],
        output=out / "models_show_json.txt",
    )

    suite.add(
        name="Models: inference smollm2:360m",
        cmd=["bash", "-c", f"{_GEN} text ollama --model {TEST_MODEL} --endpoint generate --prompt 'Say hi.' 2>&1 | tee {out / 'models_inference.txt'}"],
        output=out / "models_inference.txt",
    )

    suite.add(
        name="Models: remove smollm2:360m",
        cmd=["bash", "-c", f"{_MODELS} remove {TEST_MODEL} 2>&1 | tee {out / 'models_remove.txt'}"],
        output=out / "models_remove.txt",
    )
