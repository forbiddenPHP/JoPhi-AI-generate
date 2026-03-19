"""Test: Clone-TTS — zero-shot voice cloning via Qwen3-TTS Base (3 variants)."""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent


def register(suite):
    out = suite.out_dir
    prep = suite.prep_dir

    # Test 1: German with default reference
    suite.add(
        name="Clone-TTS Deutsch (default ref)",
        cmd=[
            sys.executable, "generate.py", "voice", "clone-tts",
            "--language", "de",
            "--text", "Herzlich willkommen! Oh mein Gott, Leute, ich kann es kaum erwarten! "
                      "Heute wird es absolut grandios — ihr werdet nicht glauben, was ich vorbereitet habe!",
            "-o", str(out / "clone_deutsch.wav"),
        ],
        output=out / "clone_deutsch.wav",
    )

    # Test 2: English with default reference
    suite.add(
        name="Clone-TTS English (default ref)",
        cmd=[
            sys.executable, "generate.py", "voice", "clone-tts",
            "--language", "en",
            "--text", "Welcome everyone! Oh my God, you guys, I can barely contain myself! "
                      "Today is going to be absolutely amazing — you won't believe what I've got in store for you!",
            "-o", str(out / "clone_english.wav"),
        ],
        output=out / "clone_english.wav",
    )

    # Test 3: Custom reference (first generate an AI-TTS sample as ref)
    custom_ref = prep / "custom_reference.wav"

    suite.add(
        name="Prep: generate custom reference (AI-TTS)",
        cmd=[
            sys.executable, "generate.py", "voice", "ai-tts",
            "-v", "Serena",
            "--text", "This is my voice. I speak clearly and with confidence.",
            "-o", str(custom_ref),
        ],
        output=custom_ref,
        prep=True,
    )

    suite.add(
        name="Clone-TTS custom reference",
        cmd=[
            sys.executable, "generate.py", "voice", "clone-tts",
            "--reference", str(custom_ref),
            "--language", "en",
            "--text", "Now I am speaking with a completely different voice that was cloned from the reference audio.",
            "-o", str(out / "clone_custom_ref.wav"),
        ],
        output=out / "clone_custom_ref.wav",
    )
