"""Test: Video — LTX-2.3 A2V dev with FLUX-generated reference image."""

import subprocess
import sys
from pathlib import Path

_mem = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
VIDEO_QUALITY = "720p" if _mem > 64 * 1024**3 else "480p"

SCENE_PROMPT = (
    "Medium shot of two men sitting across from each other at a small café table. "
    "Warm afternoon light from large windows, shallow depth of field, clean modern interior."
)

DIALOG = (
    "[Uncle_Fu:german] Hallo Dylan, wie geht es dir heute? "
    "[Dylan:german] Mir geht es bestens, danke der Nachfrage! Und dir? "
    "[Uncle_Fu:german] Kann nicht klagen. Schönes Wetter heute, findest du nicht?"
)

VIDEO_PROMPT = (
    "Medium shot of two men sitting across from each other at a small café table. "
    "Warm afternoon light, shallow depth of field. "
    "The older man leans forward and says in German: \"Hallo Dylan, wie geht es dir heute?\" "
    "The younger man nods, smiles and replies in German: \"Mir geht es bestens, danke der Nachfrage! Und dir?\" "
    "The older man gestures toward the window and says in German: \"Kann nicht klagen. Schönes Wetter heute, findest du nicht?\""
)


def register(suite):
    out = suite.out_dir
    prep = suite.prep_dir
    ref_image = prep / "a2v_ref_scene.png"
    dialog_wav = prep / "a2v_dialog.wav"

    # Prep 1: Generate reference image with FLUX
    suite.add(
        name="FLUX scene image",
        cmd=[
            sys.executable, "generate.py", "image",
            "flux.2", "--model", "4b-distilled",
            "-p", SCENE_PROMPT,
            "--seed", "42",
            "--ratio", "16:9", "--quality", VIDEO_QUALITY,
            "-o", str(ref_image),
        ],
        output=ref_image,
        prep=True,
    )

    # Prep 2: Generate dialog via ai-tts
    suite.add(
        name="AI-TTS dialog",
        cmd=[
            sys.executable, "generate.py", "voice", "ai-tts",
            "--text", DIALOG,
            "-o", str(dialog_wav),
        ],
        output=dialog_wav,
        prep=True,
    )

    # A2V dev with reference image
    suite.add(
        name="Video A2V dev + ref image (dialog)",
        cmd=[
            sys.executable, "generate.py", "video", "ltx2.3",
            "--model", "dev",
            "-p", VIDEO_PROMPT,
            "--audio", str(dialog_wav),
            "--image-first", str(ref_image),
            "--ratio", "16:9", "--quality", VIDEO_QUALITY,
            "--frame-rate", "24",
            "--seed", "42",
            "-o", str(out / "ltx2_a2v_ref_dev.mp4"),
        ],
        output=out / "ltx2_a2v_ref_dev.mp4",
    )
