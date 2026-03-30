"""Test: PS — system status."""

import sys


def register(suite):
    out = suite.out_dir
    suite.add(
        name="System status (ps)",
        cmd=["bash", "-c", f"{sys.executable} generate.py ps > {out / 'ps.txt'}"],
        output=out / "ps.txt",
    )
