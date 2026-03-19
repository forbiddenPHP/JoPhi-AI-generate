#!/usr/bin/env python3
"""Quick test: different prompts for making a person look younger.

Usage:
    python tests/agetest.py
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
JOHANNES = SCRIPT_DIR / "tests" / "assets" / "johannes.png"
OUT_DIR = SCRIPT_DIR / "demos" / "agetest"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPTS = [
    ("abstract", "make the person 20 years younger"),
    ("skin_hair", "remove wrinkles, smooth skin, darken the hair, more youthful face"),
    ("teenager", "turn the man into a 18-year-old teenager version of himself"),
    ("smooth_face", "make the face younger, remove all wrinkles and age spots, fuller hair"),
    ("baby_face", "give the man a baby face, smooth skin, no beard, no wrinkles"),
    ("rejuvenate", "rejuvenate the person, tighter skin, fuller darker hair, brighter eyes"),
    ("age_20", "turn this man to the age of 20"),
]

for name, prompt in PROMPTS:
    out_path = OUT_DIR / f"age_{name}.png"
    print(f"\n{'='*60}")
    print(f"  {name}: {prompt}")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, str(SCRIPT_DIR / "generate.py"), "image",
        "--engine", "flux.2", "--model", "4b-distilled",
        "--images", str(JOHANNES),
        "-p", prompt,
        "-W", "512", "-H", "512", "--seed", "42",
        "-o", str(out_path),
    ]
    subprocess.run(cmd, cwd=str(SCRIPT_DIR))

print(f"\nResults in {OUT_DIR}/")
