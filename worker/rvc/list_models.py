#!/usr/bin/env python3
"""List installed RVC voice models from worker/rvc/models/.

Scans model directories for .pth files. Each subdirectory with a .pth
file is considered an installed model.

Usage:
    python list_models.py --list-models
"""

import json
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"


def list_models():
    models = []
    if MODELS_DIR.is_dir():
        for d in sorted(MODELS_DIR.iterdir()):
            if d.is_dir() and any(d.glob("*.pth")):
                models.append({"model": d.name, "notice": ""})
    print(json.dumps(models))


if __name__ == "__main__":
    if "--list-models" in sys.argv:
        list_models()
