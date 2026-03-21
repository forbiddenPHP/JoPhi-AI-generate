"""Test: Video — LTX-2.3 audio-to-video (distilled)."""

import sys
from pathlib import Path

DIALOG = (
    "[Uncle_Fu:german] Hallo Dylan, wie geht es dir heute? "
    "[Dylan:german] Mir geht es bestens, danke der Nachfrage! Und dir? "
    "[Uncle_Fu:german] Kann nicht klagen. Schönes Wetter heute, findest du nicht?"
)

PROMPT = (
    "Medium shot of two men sitting across from each other at a small café table. "
    "Warm afternoon light, shallow depth of field. "
    "The older man leans forward and says in German: \"Hallo Dylan, wie geht es dir heute?\" "
    "The younger man nods, smiles and replies in German: \"Mir geht es bestens, danke der Nachfrage! Und dir?\" "
    "The older man gestures toward the window and says in German: \"Kann nicht klagen. Schönes Wetter heute, findest du nicht?\""
)

# num_frames derived from audio length automatically


def register(suite):
    out = suite.out_dir
    prep = suite.prep_dir
    dialog_wav = prep / "a2v_dialog.wav"

    # Prep: Generate dialog via ai-tts
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

    # A2V distilled
    suite.add(
        name="Video A2V distilled (dialog)",
        cmd=[
            sys.executable, "generate.py", "video", "ltx2.3",
            "-p", PROMPT,
            "--audio", str(dialog_wav),
            "--ratio", "16:9", "--quality", "240p",
            "--frame-rate", "24",
            "--seed", "42",
            "-o", str(out / "ltx2_a2v_distilled.mp4"),
        ],
        output=out / "ltx2_a2v_distilled.mp4",
    )
