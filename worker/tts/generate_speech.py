#!/usr/bin/env python3
"""
AI-TTS Worker — Qwen3-TTS speech generation via mlx-audio.

Called by generate.py via:
  conda run -n ai-tts python tts_worker/generate_speech.py --text "..." --voice Aiden ...

Outputs JSON on stdout: {"output": "/path/to.wav", "sample_rate": 48000}
Progress on stderr:     [1/3] Generating segment ...
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


# ── Language detection ────────────────────────────────────────────────────────

# ISO code → model lang_code (what codec_language_id expects)
_ISO_TO_MODEL = {
    "de": "german",
    "en": "english",
    "fr": "french",
    "ja": "japanese",
    "ko": "korean",
    "zh": "chinese",
    "ru": "russian",
    "pt": "portuguese",
    "es": "spanish",
    "it": "italian",
}

# langdetect returns zh-cn / zh-tw → normalize
_NORMALIZE_LANG = {
    "zh-cn": "zh",
    "zh-tw": "zh",
}

try:
    from langdetect import detect as _langdetect
except ImportError:
    _langdetect = None


def detect_model_lang(text: str, fallback_iso: str = "en") -> str:
    """Detect language of text. Returns model lang_code (german, english, ...)."""
    if _langdetect is not None:
        try:
            iso = _langdetect(text)
            iso = _NORMALIZE_LANG.get(iso, iso)
            if iso in _ISO_TO_MODEL:
                return _ISO_TO_MODEL[iso]
        except Exception:
            pass
    return _ISO_TO_MODEL.get(fallback_iso, "auto")


def iso_to_model_lang(iso: str) -> str:
    """Convert ISO code (de, en) to model lang_code (german, english)."""
    return _ISO_TO_MODEL.get(iso.lower(), "auto")


# ── Voice/segment parsing ────────────────────────────────────────────────────

_VOICE_MARKER_RE = re.compile(r"\[(\w+)(?::\s*([^\]]*))?\]\s*")

VALID_VOICES = {
    "Aiden", "Dylan", "Eric", "Ryan", "Uncle_Fu",
    "Vivian", "Serena", "Ono_Anna", "Sohee",
}


def parse_dialog(text: str, default_voice: str | None) -> list[tuple[str, str | None, str]]:
    """Parse text with [Voice] or [Voice: instruct] markers into segments.

    Returns list of (voice_name, instruct_or_None, text_segment) tuples.
    If no markers found, returns single segment with default_voice.
    """
    segments = []
    parts = _VOICE_MARKER_RE.split(text)

    # No markers found — single segment
    # parts has only 1 element when no match
    if len(parts) == 1:
        if not default_voice:
            print("ERROR: No voice specified. Use -v or [Voice] markers in text.",
                  file=sys.stderr)
            sys.exit(1)
        return [(default_voice, None, text.strip())]

    # parts = [before, voice1, instruct1, text1, voice2, instruct2, text2, ...]
    # With 2 capture groups, stride is 3
    if parts[0].strip():
        if not default_voice:
            print("ERROR: Text before first [Voice] marker but no -v default set.",
                  file=sys.stderr)
            sys.exit(1)
        segments.append((default_voice, None, parts[0].strip()))

    for i in range(1, len(parts), 3):
        voice = parts[i]
        instruct = parts[i + 1].strip() if i + 1 < len(parts) and parts[i + 1] else None
        text_part = parts[i + 2].strip() if i + 2 < len(parts) else ""
        if not text_part:
            continue
        if voice not in VALID_VOICES:
            print(f"WARNING: Unknown voice '{voice}', using anyway.",
                  file=sys.stderr)
        segments.append((voice, instruct, text_part))

    return segments



# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AI-TTS Worker (Qwen3-TTS)")
    parser.add_argument("--text", required=True, help="Text to speak")
    parser.add_argument("--voice", default=None, help="Preset voice name")
    parser.add_argument("--instruct", default=None, help="Style instructions")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--output", required=True, help="Output WAV path")
    parser.add_argument("--language", default="", help="ISO lang code (de, en, fr, ...). Empty = autodetect.")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Parse dialog segments
    segments = parse_dialog(args.text, args.voice)
    total = len(segments)

    # Load model
    print("Loading model ...", file=sys.stderr)
    from mlx_audio.tts.utils import load_model
    model = load_model(args.model)

    # Language: explicit or autodetect from full text (markers stripped)
    if args.language:
        model_lang = iso_to_model_lang(args.language)
    else:
        plain_text = _VOICE_MARKER_RE.sub("", args.text)
        model_lang = detect_model_lang(plain_text)
        print(f"Detected language: {model_lang}", file=sys.stderr)

    # Silence gap between dialog segments (0.4s at native rate, resampled later)
    native_sr = None
    audio_parts = []

    for i, (voice, seg_instruct, text) in enumerate(segments, 1):
        # Per-segment instruct overrides global --instruct
        instruct = seg_instruct or args.instruct
        tag_info = f" [{seg_instruct}]" if seg_instruct else ""
        print(f"[{i}/{total}] {voice}{tag_info} ({model_lang}): {text[:60]}{'...' if len(text) > 60 else ''}",
              file=sys.stderr)

        gen_kwargs = {
            "text": text,
            "voice": voice,
            "lang_code": model_lang,
        }
        if instruct:
            gen_kwargs["instruct"] = instruct

        results = list(model.generate(**gen_kwargs))

        if not results:
            print(f"ERROR: No audio generated for segment {i}", file=sys.stderr)
            sys.exit(1)

        audio_data = results[0].audio
        if hasattr(audio_data, 'tolist'):
            audio_data = np.array(audio_data, dtype=np.float32)
        else:
            audio_data = np.asarray(audio_data, dtype=np.float32)

        if native_sr is None:
            native_sr = results[0].sample_rate

        # Add silence gap before this segment (except the first)
        if audio_parts:
            gap = np.zeros(int(native_sr * 0.4), dtype=np.float32)
            audio_parts.append(gap)

        audio_parts.append(audio_data)

    # Concatenate all segments
    final_audio = np.concatenate(audio_parts) if len(audio_parts) > 1 else audio_parts[0]

    # Write output at native sample rate
    sf.write(str(output_path), final_audio, native_sr)
    print(f"Saved: {output_path} ({native_sr} Hz)", file=sys.stderr)

    # JSON result on stdout
    print(json.dumps({
        "output": str(output_path),
        "sample_rate": native_sr,
    }))


if __name__ == "__main__":
    main()
