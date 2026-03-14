#!/usr/bin/env python3
"""
Language detection worker — detects text language, outputs ISO code.

Usage:
  python langdetect_worker/detect.py --text "Der Fuchs springt"
  → de
"""

from __future__ import annotations

import argparse
import sys

from langdetect import detect

# Supported ISO codes (langdetect may return zh-cn, zh-tw → normalize to zh)
_NORMALIZE = {
    "zh-cn": "zh",
    "zh-tw": "zh",
}


def main():
    parser = argparse.ArgumentParser(description="Detect text language")
    parser.add_argument("--text", required=True, help="Text to detect language for")
    args = parser.parse_args()

    try:
        code = detect(args.text)
    except Exception as e:
        print(f"ERROR: Language detection failed: {e}", file=sys.stderr)
        print("en")  # fallback
        return

    print(_NORMALIZE.get(code, code))


if __name__ == "__main__":
    main()
