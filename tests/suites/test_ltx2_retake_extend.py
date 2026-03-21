"""Test: Video — LTX-2.3 retake + extend (combined).

Base video whisper transcript:
  - 0.0s–0.86s: "Thank you, others." (she)
  - 0.86s–3.48s: silence
  - 3.48s–4.98s: "What do you say or ask inside of the..." (he)

Steps:
  1. Retake passage 1 (her line, 0.0–1.0s) → German dialog
  2. Retake passage 2 (his line) → German dialog
  3. Extend by 5 seconds with continuation
All dev model.
"""

import json
import sys
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
BASE_VIDEO = ASSETS_DIR / "ltx2_retake_base.mp4"
TRANSCRIPT_JSON = ASSETS_DIR / "ltx2_retake_base.json"

# Read timestamps from whisper transcript
with open(TRANSCRIPT_JSON) as f:
    _transcript = json.load(f)
_seg1 = _transcript["segments"][0]  # her line
_seg2 = _transcript["segments"][1]  # his line
SEG1_START = str(_seg1["start"])
SEG1_END = str(_seg1["end"])
SEG2_START = str(_seg2["start"])
SEG2_END = str(_seg2["end"])

RETAKE_PROMPT_1 = (
    "Medium shot of a young couple meeting outside a cozy café on a sunny afternoon. "
    "Warm golden light, shallow depth of field, European city street. "
    "The woman waves and says in German: \"Hallo Peter, wie geht's dir?\" "
    "The man arrives slightly out of breath and apologizes. "
    "They laugh together and discuss whether to sit inside or outside, gesturing toward the sunny terrace."
)

RETAKE_PROMPT_2 = (
    "Medium shot of a young couple meeting outside a cozy café on a sunny afternoon. "
    "Warm golden light, shallow depth of field, European city street. "
    "The woman waves and says in German: \"Hallo Peter, wie geht's dir?\" "
    "The man arrives and says in German: \"Nicht schlecht, und dir Petra?\" "
    "They laugh together and discuss whether to sit inside or outside, gesturing toward the sunny terrace."
)

EXTEND_PROMPT = (
    "The man suddenly says in German: "
    "\"Ich weiß, dass das unverständlich war. Aber egal, bei KI-Videos ist das manchmal so.\""
)


def register(suite):
    out = suite.out_dir
    prep = suite.prep_dir

    # Step 1: Retake passage 1 (her line 0.0–1.0s)
    after_p1 = prep / "retake_p1.mp4"
    suite.add(
        name=f"Retake pass 1 ({SEG1_START}–{SEG1_END}s, she)",
        cmd=[
            sys.executable, "generate.py", "video", "ltx2.3",
            "--model", "dev",
            "-p", RETAKE_PROMPT_1,
            "--retake", str(BASE_VIDEO), SEG1_START, SEG1_END,
            "--ratio", "16:9", "--quality", "360p",
            "--seed", "42",
            "-o", str(after_p1),
        ],
        output=after_p1,
        prep=True,
    )

    # Step 2: Retake passage 2 (his line 3.48–4.98s) on result of step 1
    after_p2 = prep / "retake_p2.mp4"
    suite.add(
        name=f"Retake pass 2 ({SEG2_START}–{SEG2_END}s, he)",
        cmd=[
            sys.executable, "generate.py", "video", "ltx2.3",
            "--model", "dev",
            "-p", RETAKE_PROMPT_2,
            "--retake", str(after_p1), SEG2_START, SEG2_END,
            "--ratio", "16:9", "--quality", "360p",
            "--seed", "42",
            "-o", str(after_p2),
        ],
        output=after_p2,
        prep=True,
    )

    # Step 3: Extend by 5 seconds
    suite.add(
        name="Extend +5s",
        cmd=[
            sys.executable, "generate.py", "video", "ltx2.3",
            "--model", "dev",
            "-p", EXTEND_PROMPT,
            "--extend", str(after_p2), "5",
            "--ratio", "16:9", "--quality", "360p",
            "--seed", "42",
            "-o", str(out / "ltx2_retake_extend.mp4"),
        ],
        output=out / "ltx2_retake_extend.mp4",
    )
