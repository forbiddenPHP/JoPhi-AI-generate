"""Test: Clone-TTS → LTX A2V — clone user voice, generate talking avatar video."""

import subprocess
import sys
from pathlib import Path

_mem = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
VIDEO_QUALITY = "720p" if _mem > 64 * 1024**3 else "480p"

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
REF_IMAGE = ASSETS_DIR / "johannes.png"

CLONE_TEXT = "Actually, I never said this for real. Just AI, ya know?"

VIDEO_PROMPT = (
    "Close-up of a bearded man with glasses speaking directly to camera, "
    "natural indoor lighting, casual setting. "
    "He says in English: \"Actually, I never said this for real. Just AI, ya know?\" "
    "Subtle head movement, natural lip sync, relaxed expression."
)


def register(suite):
    out = suite.out_dir
    prep = suite.prep_dir
    clone_wav = prep / "clone_voice.wav"

    # Prep: Clone-TTS with default reference (user's voice)
    suite.add(
        name="Clone-TTS (user voice EN)",
        cmd=[
            sys.executable, "generate.py", "voice", "clone-tts",
            "--language", "en",
            "--text", CLONE_TEXT,
            "-o", str(clone_wav),
        ],
        output=clone_wav,
        prep=True,
    )

    # LTX A2V with user's face + cloned voice
    suite.add(
        name="LTX A2V avatar (clone voice + johannes.png)",
        cmd=[
            sys.executable, "generate.py", "video", "ltx2.3",
            "-p", VIDEO_PROMPT,
            "--audio", str(clone_wav),
            "--image-first", str(REF_IMAGE),
            "--ratio", "16:9", "--quality", VIDEO_QUALITY,
            "--frame-rate", "24",
            "--seed", "42",
            "-o", str(out / "clone_ltx_avatar.mp4"),
        ],
        output=out / "clone_ltx_avatar.mp4",
    )
