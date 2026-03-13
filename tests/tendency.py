#!/usr/bin/env python3
"""Tendency test: Generate a song, transcribe it, compare section timing.

1. Generate a cheerful English song via ACE-Step
2. Transcribe the result with mlx-whisper (all formats, word timestamps)
3. Match original section tags against transcription segments
   to see if section boundaries are detectable from timestamps.
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DEMOS = PROJECT / "demos"


def parse_duration(s):
    m = re.match(r'^(\d+):(\d{2})$', s.strip())
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    return int(s)


# ── Song ─────────────────────────────────────────────────────────

DURATION = "2:00"
SEED = 34529356981
ENGINE = "ace"
SONG_NAME = "cheerful-tendency"
WAV_OUT = DEMOS / f"{SONG_NAME}.wav"
TRANSCRIPT_DIR = DEMOS / SONG_NAME

CAPTION = "upbeat pop rock, cheerful male voice, acoustic guitar, claps"

SONG = [
    {"tag": "Intro", "lines": []},

    {"tag": "Verse", "lines": [
        "The bus arrived exactly on time,",
        "a stranger gave me his last French fry,",
        "my boss said take the afternoon off,",
        "I did a little dance and I couldn't stop.",
    ]},

    {"tag": "Chorus", "lines": [
        "It's a wonderful ridiculous day,",
        "everything's going my way,",
        "the birds are singing off key,",
        "the birds are singing off key,",
        "but they're singing for me.",
    ]},

    {"tag": "Verse", "lines": [
        "The dog next door started singing along,",
        "the mailman whistled my favorite song,",
        "I found a dollar in my winter coat,",
        "and someone left me a note.",
    ]},

    {"tag": "Chorus", "lines": [
        "It's a wonderful ridiculous day,",
        "everything's going my way,",
        "the birds are singing off key,",
        "the birds are singing off key,",
        "but they're singing for me.",
    ]},

    {"tag": "Outro", "lines": [
        "Oh they're singing for me,",
        "oh they're singing for me,",
        "oh they're singing for me,",
        "yeah they're singing for me.",
    ]},
]


def build_lyrics(song):
    parts = []
    for section in song:
        tag = section["tag"].upper()
        lines = [l for l in section.get("lines", []) if l.strip()]
        if lines:
            parts.append(f"[{tag}]\n" + "\n".join(lines))
        else:
            parts.append(f"[{tag}]")
    return "\n\n".join(parts)


def flat_lines(song):
    """Return list of (section_tag, line_text) for all lines."""
    out = []
    for section in song:
        for line in section.get("lines", []):
            if line.strip():
                out.append((section["tag"], line.strip()))
    return out


# ── Step 1: Generate ─────────────────────────────────────────────

def generate():
    duration_s = parse_duration(DURATION)
    lyrics = build_lyrics(SONG)
    n_lines = sum(len([l for l in s.get("lines", []) if l.strip()]) for s in SONG)
    print(f"=== Generating: {n_lines} lines, {duration_s}s ===")
    print()
    print("Lyrics:")
    print(lyrics)
    print()

    lyrics_file = Path(tempfile.mktemp(suffix=".txt", prefix="lyrics-tendency-"))
    lyrics_file.write_text(lyrics, encoding="utf-8")

    cmd = [
        sys.executable, str(PROJECT / "revoicer.py"), "music",
        "--engine", ENGINE,
        "-f", str(lyrics_file),
        "-t", CAPTION,
        "--seed", str(SEED),
        "-s", str(duration_s),
        "-o", str(WAV_OUT),
    ]

    print(f"Output: {WAV_OUT}")
    r = subprocess.run(cmd)
    lyrics_file.unlink(missing_ok=True)
    if r.returncode != 0:
        print("FAILED: generation", file=sys.stderr)
        sys.exit(1)
    print()


# ── Step 2: Transcribe ───────────────────────────────────────────

def transcribe():
    print("=== Transcribing ===")
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(PROJECT / "revoicer.py"), "transcribe",
        str(WAV_OUT),
        "--input-language", "en",
        "--word-timestamps",
        "--format", "all",
        "-o", str(TRANSCRIPT_DIR),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print("FAILED: transcription", file=sys.stderr)
        if r.stderr:
            print(r.stderr, file=sys.stderr)
        sys.exit(1)

    results = json.loads(r.stdout.strip())
    print(f"  Transcribed: {len(results)} file(s)")
    print(f"  Output: {TRANSCRIPT_DIR}/")
    for f in sorted(TRANSCRIPT_DIR.iterdir()):
        if not f.name.startswith("."):
            print(f"    {f.name} ({f.stat().st_size:,} bytes)")
    print()
    return results[0]


# ── Step 3: Analyse ──────────────────────────────────────────────

def analyse(transcript):
    """Compare original sections against whisper segments."""
    segments = transcript.get("segments", [])

    # Filter out hallucinated segments (near-zero duration or repeated text)
    real_segments = []
    for seg in segments:
        duration = seg["end"] - seg["start"]
        text = seg["text"].strip()
        if duration < 0.5 and not text:
            continue  # empty segment
        if duration < 0.1:
            continue  # hallucination
        real_segments.append(seg)

    original = flat_lines(SONG)

    print("=" * 70)
    print("TENDENCY ANALYSIS: Original sections vs. transcribed segments")
    print("=" * 70)
    print()

    # Show whisper segments with timing
    print(f"Whisper segments ({len(real_segments)} real, {len(segments) - len(real_segments)} filtered):")
    print()
    for seg in real_segments:
        t0 = seg["start"]
        t1 = seg["end"]
        text = seg["text"].strip()
        m0, s0 = divmod(int(t0), 60)
        m1, s1 = divmod(int(t1), 60)
        print(f"  [{m0}:{s0:02d} - {m1}:{s1:02d}]  {text[:80]}")
    print()

    # Show original structure
    print("Original structure:")
    print()
    current_tag = None
    for tag, line in original:
        if tag != current_tag:
            print(f"  [{tag.upper()}]")
            current_tag = tag
        print(f"    {line}")
    print()

    # Try to match first line of each section to a segment
    print("-" * 70)
    print("Section boundary detection:")
    print("-" * 70)
    print()

    sections_with_lines = [s for s in SONG if s.get("lines")]
    section_starts = []

    for section in sections_with_lines:
        first_line = section["lines"][0].strip().lower()
        first_clean = re.sub(r'[^a-z\s]', '', first_line)
        first_words = first_clean.split()[:4]

        best_seg = None
        best_score = 0
        for seg in real_segments:
            seg_clean = re.sub(r'[^a-z\s]', '', seg["text"].strip().lower())
            seg_words = seg_clean.split()
            score = 0
            for j, w in enumerate(first_words):
                if j < len(seg_words) and seg_words[j] == w:
                    score += 1
                else:
                    break
            if score > best_score:
                best_score = score
                best_seg = seg

        if best_seg and best_score >= 2:
            t = best_seg["start"]
            m, s = divmod(int(t), 60)
            section_starts.append({
                "section": section["tag"],
                "expected_first_line": section["lines"][0].strip(),
                "matched_segment": best_seg["text"].strip()[:60],
                "timestamp": t,
                "time_fmt": f"{m}:{s:02d}",
                "confidence": best_score,
            })
            print(f"  {section['tag']:12s} -> {m}:{s:02d}  (matched {best_score}/{len(first_words)} words)")
            print(f"    expected: {section['lines'][0].strip()[:60]}")
            print(f"    got:      {best_seg['text'].strip()[:60]}")
            print()
        else:
            section_starts.append({
                "section": section["tag"],
                "expected_first_line": section["lines"][0].strip(),
                "matched_segment": None,
                "timestamp": None,
                "confidence": best_score,
            })
            print(f"  {section['tag']:12s} -> NOT FOUND (best score: {best_score}/{len(first_words)})")
            print()

    # Summary
    found = sum(1 for s in section_starts if s["timestamp"] is not None)
    total = len(section_starts)
    print("=" * 70)
    print(f"Result: {found}/{total} sections matched to timestamps")

    if found >= 2:
        matched = [s for s in section_starts if s["timestamp"] is not None]
        gaps = []
        for i in range(1, len(matched)):
            gap = matched[i]["timestamp"] - matched[i - 1]["timestamp"]
            gaps.append(gap)
            print(f"  {matched[i-1]['section']} -> {matched[i]['section']}: {gap:.1f}s gap")
        if gaps:
            avg = sum(gaps) / len(gaps)
            print(f"  Average section length: {avg:.1f}s")

    print("=" * 70)
    print()


# ── Main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not WAV_OUT.exists():
        generate()
    else:
        print(f"Using existing: {WAV_OUT}")
        print("  (delete to regenerate)")
        print()

    transcript = transcribe()
    analyse(transcript)
    print("OK")
