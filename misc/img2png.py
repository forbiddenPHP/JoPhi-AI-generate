#!/usr/bin/env python3
"""Convert any image to PNG via ffmpeg. No Python dependencies."""

import subprocess
import sys
import os


def convert(src, dst=None):
    if dst is None:
        dst = os.path.splitext(src)[0] + ".png"
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", src, dst],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        sys.exit(1)
    return dst


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: img2png.py <input> [output.png]")
        sys.exit(1)
    out = convert(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    print(out)
