#!/usr/bin/env python3
"""
AI-TTS Worker — Qwen3-TTS speech generation via mlx-audio.

Called by generate.py via:
  conda run -n ai-tts python worker/tts/generate_speech.py --text "..." --voice Aiden ...

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


_MODEL_LANGS = set(_ISO_TO_MODEL.values())  # {"german", "english", ...}


def iso_to_model_lang(code: str) -> str:
    """Convert ISO code (de, en) or model lang name (german, english) to model lang_code."""
    lower = code.lower()
    if lower in _MODEL_LANGS:
        return lower
    return _ISO_TO_MODEL.get(lower, "auto")


# ── Voice/segment parsing ────────────────────────────────────────────────────

_BRACKET_RE = re.compile(r"\[([^\]]+)\]\s*")

VALID_VOICES = {
    "Aiden", "Dylan", "Eric", "Ryan", "Uncle_Fu",
    "Vivian", "Serena", "Ono_Anna", "Sohee",
}
_VOICES_LOWER = {v.lower(): v for v in VALID_VOICES}
_ALL_LANGS = _MODEL_LANGS | set(_ISO_TO_MODEL.keys())


def _classify_bracket(content: str) -> tuple[str | None, str | None, str | None]:
    """Classify bracket content into (voice, instruct, language).

    Splits on : / - separators, then classifies each field:
    - Known voice name (lowercase match) → voice
    - Known language (ISO or model name, lowercase) → language
    - Everything else → instruct (joined with ", ")
    """
    fields = re.split(r'\s*[:\-/]\s*', content)
    voice = None
    lang = None
    instruct_parts = []

    for field in fields:
        field = field.strip()
        if not field:
            continue
        lower = field.lower()
        if voice is None and lower in _VOICES_LOWER:
            voice = _VOICES_LOWER[lower]
        elif lang is None and lower in _ALL_LANGS:
            lang = field
        else:
            instruct_parts.append(field)

    instruct = ", ".join(instruct_parts) if instruct_parts else None
    return voice, instruct, lang


def parse_dialog(text: str, default_voice: str | None) -> list[tuple[str, str | None, str | None, str]]:
    """Parse text with [...] markers into dialog segments.

    Bracket content is split on : / - separators and classified by type:
    - Known voice name → voice (case-insensitive)
    - Known language → language (case-insensitive)
    - Anything else → instruct (style instructions)

    Order inside brackets doesn't matter:
      [Dylan:excited:english] = [english:Dylan:excited] = [excited - english - dylan]

    Returns list of (voice_name, instruct_or_None, language_or_None, text_segment) tuples.
    """
    # Find all bracket markers and their positions
    markers = list(_BRACKET_RE.finditer(text))

    if not markers:
        if not default_voice:
            print("ERROR: No voice specified. Use -v or [Voice] markers in text.",
                  file=sys.stderr)
            sys.exit(1)
        return [(default_voice, None, None, text.strip())]

    segments = []

    # Text before first marker
    before = text[:markers[0].start()].strip()
    if before:
        if not default_voice:
            print("ERROR: Text before first [...] marker but no -v default set.",
                  file=sys.stderr)
            sys.exit(1)
        segments.append((default_voice, None, None, before))

    # Process each marker + text after it
    last_voice = default_voice
    for idx, match in enumerate(markers):
        voice, instruct, lang = _classify_bracket(match.group(1))

        # Use last known voice if bracket had no voice
        if voice is None:
            voice = last_voice
        else:
            last_voice = voice

        if voice is None:
            print(f"WARNING: No voice in [{match.group(1)}] and no default set.",
                  file=sys.stderr)
            continue

        # Text after this marker until next marker (or end)
        text_start = match.end()
        text_end = markers[idx + 1].start() if idx + 1 < len(markers) else len(text)
        text_part = text[text_start:text_end].strip()

        if not text_part:
            continue

        segments.append((voice, instruct, lang, text_part))

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
    parser.add_argument("--ref-audio", default=None, help="Reference audio for voice cloning (Base model only)")
    parser.add_argument("--ref-text", default=None, help="Text spoken in reference audio (required with --ref-audio)")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading model …", file=sys.stderr)
    from mlx_audio.tts.utils import load_model
    model = load_model(args.model)

    # Language: explicit or autodetect from full text (markers stripped)
    if args.language:
        model_lang = iso_to_model_lang(args.language)
    else:
        plain_text = _BRACKET_RE.sub("", args.text)
        model_lang = detect_model_lang(plain_text)
        print(f"Detected language: {model_lang}", file=sys.stderr)

    native_sr = None
    audio_parts = []

    if args.ref_audio:
        # Voice cloning mode (Base model): single segment, no dialog parsing
        print(f"[1/1] Voice clone ({model_lang}): {args.text[:60]}{'...' if len(args.text) > 60 else ''}",
              file=sys.stderr)
        gen_kwargs = {
            "text": args.text,
            "lang_code": model_lang,
            "ref_audio": args.ref_audio,
            "ref_text": args.ref_text or "",
        }
        results = list(model.generate(**gen_kwargs))
        if not results:
            print("ERROR: No audio generated", file=sys.stderr)
            sys.exit(1)
        audio_data = results[0].audio
        if hasattr(audio_data, 'tolist'):
            audio_data = np.array(audio_data, dtype=np.float32)
        else:
            audio_data = np.asarray(audio_data, dtype=np.float32)
        native_sr = results[0].sample_rate
        audio_parts.append(audio_data)
    else:
        # Standard TTS mode: parse dialog segments
        segments = parse_dialog(args.text, args.voice)
        total = len(segments)

        for i, (voice, seg_instruct, seg_lang, text) in enumerate(segments, 1):
            instruct = seg_instruct or args.instruct
            lang = iso_to_model_lang(seg_lang) if seg_lang else model_lang
            tag_info = f" [{seg_instruct}]" if seg_instruct else ""
            print(f"[{i}/{total}] {voice}{tag_info} ({lang}): {text[:60]}{'...' if len(text) > 60 else ''}",
                  file=sys.stderr)

            gen_kwargs = {
                "text": text,
                "voice": voice,
                "lang_code": lang,
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

            if audio_parts:
                gap = np.zeros(int(native_sr * 0.4), dtype=np.float32)
                audio_parts.append(gap)

            audio_parts.append(audio_data)

    # Concatenate all segments
    final_audio = np.concatenate(audio_parts) if len(audio_parts) > 1 else audio_parts[0]

    # Write output at native sample rate
    sf.write(str(output_path), final_audio, native_sr)
    print(f"Saved: {output_path} ({native_sr} Hz)", file=sys.stderr)

    # JSON result on stdout — array of output paths
    print(json.dumps([str(output_path)]))


if __name__ == "__main__":
    main()
