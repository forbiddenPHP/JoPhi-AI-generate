#!/usr/bin/env python3
"""List installed macOS say voices.

Parses `say -v '?'` output. No conda env needed — runs natively.

Usage:
    python list_models.py --list-models
"""

import json
import re
import subprocess
import sys


def list_models():
    result = subprocess.run(["say", "-v", "?"], capture_output=True, text=True)
    models = []
    for line in result.stdout.splitlines():
        # Format: "Name              lang_CODE  # description"
        # Name can contain spaces/parens, so anchor on lang_CODE pattern
        m = re.match(r"^(.+?)\s+([a-z]{2}_\S+)\s+#", line)
        if m:
            name = m.group(1).strip()
            lang = m.group(2).strip()
            models.append({"model": name, "notice": lang})
    print(json.dumps(models))


if __name__ == "__main__":
    if "--list-models" in sys.argv:
        list_models()
