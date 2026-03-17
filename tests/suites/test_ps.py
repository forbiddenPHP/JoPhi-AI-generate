"""Test: PS — system status."""

import sys


def register(suite):
    suite.add(
        name="System status (ps)",
        cmd=[sys.executable, "generate.py", "ps"],
    )
