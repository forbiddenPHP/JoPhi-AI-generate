#!/usr/bin/env bash
# ── Full Test Suite for generate.py ──────────────────────────────────────────
# Runs all 16 worker tests SEQUENTIALLY.  All output goes to demos/.
# Usage:  conda activate tts-mist && bash tests/full_test_run.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

DEMOS="$SCRIPT_DIR/demos"
PODCAST="$DEMOS/podcast-5min.mp3"
TOTAL=16

passed=0
failed=0
skipped=0
results=()

run_test() {
    local name="$1"; shift
    echo ""
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  TEST: $name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    local start=$SECONDS
    if "$@"; then
        local dur=$(( SECONDS - start ))
        echo ""
        echo "  >>> PASS ($dur s)"
        results+=("PASS  ${dur}s  $name")
        ((passed++)) || true
    else
        local dur=$(( SECONDS - start ))
        echo ""
        echo "  >>> FAIL (exit $?, ${dur}s)"
        results+=("FAIL  ${dur}s  $name")
        ((failed++)) || true
    fi
}

# ── 1. ACE-Step: Musik generieren (2 min) ───────────────────────────────────

LYRICS_ACE='[verse]
Neon lights are burning through the haze
Walking down the boulevard in a purple daze
Every heartbeat echoes off the walls
Dancing shadows answer when the city calls

[chorus]
We are the midnight riders, chasing dreams
Nothing is as broken as it seems
Light it up and let the music flow
We are the midnight riders, steal the show

[verse]
Starlight dripping from the rooftop edge
Balancing our futures on a razor ledge
Every whisper carries through the night
Hold my hand and we will be alright

[chorus]
We are the midnight riders, chasing dreams
Nothing is as broken as it seems
Light it up and let the music flow
We are the midnight riders, steal the show'

echo "  [1/$TOTAL] Generating 120s synthwave with ACE-Step turbo ..."
run_test "ACE-Step (turbo, 120s synthwave)" \
    python generate.py audio --engine ace-step --model turbo \
        -l "$LYRICS_ACE" \
        -t "synthwave,electronic,upbeat,80s,energetic" \
        -s 120 -o "$DEMOS/ace-test-song.wav"

# ── 2. HeartMuLa: Musik generieren (2 min) ──────────────────────────────────

echo "  [2/$TOTAL] Generating 120s synthwave with HeartMuLa ..."
run_test "HeartMuLa (120s synthwave)" \
    python generate.py audio --engine heartmula \
        -l "$LYRICS_ACE" \
        -t "synthwave,electronic,upbeat,80s,energetic" \
        -s 120 -o "$DEMOS/heart-test-song.wav"

# ── 3. Whisper: Transkription EN (ACE-Song) ─────────────────────────────────

echo "  [3/$TOTAL] Transcribing ace-test-song.wav (EN) with Whisper ..."
run_test "Whisper transcribe EN (ace-test-song)" \
    python generate.py text --engine whisper \
        "$DEMOS/ace-test-song.wav" \
        --input-language en --format all \
        -o "$DEMOS/transcribe-ace"

# ── 4. Whisper: Transkription DE (Podcast) ──────────────────────────────────

echo "  [4/$TOTAL] Transcribing podcast-5min.mp3 (DE) with Whisper ..."
run_test "Whisper transcribe DE (podcast)" \
    python generate.py text --engine whisper \
        "$PODCAST" \
        --input-language de --format all \
        -o "$DEMOS/transcribe-podcast"

# ── 5. HeartMuLa Transcribe: Lyrics Extraction ─────────────────────────────

echo "  [5/$TOTAL] Extracting lyrics from ace-test-song.wav with HeartTranscriptor ..."
run_test "HeartMuLa transcribe (lyrics extraction)" \
    python generate.py text --engine heartmula-transcribe \
        "$DEMOS/ace-test-song.wav" \
        -o "$DEMOS/transcribe-ace-lyrics.txt"

# ── 6. Demucs: Separation (ACE-Song) ────────────────────────────────────────

echo "  [6/$TOTAL] Separating ace-test-song.wav into stems with Demucs ..."
run_test "Demucs separate (ace-test-song)" \
    python generate.py audio --engine demucs \
        "$DEMOS/ace-test-song.wav" \
        -o "$DEMOS/separate-ace"

# ── 7. Enhance (ACE-Song vocals) ────────────────────────────────────────────
# Use the vocals stem from demucs separation

VOCALS="$DEMOS/separate-ace/ace-test-song/vocals.wav"
if [ ! -f "$VOCALS" ]; then
    echo "  [7/$TOTAL] SKIP: Vocals stem not found (demucs must pass first)"
    results+=("SKIP  0s  Enhance (vocals)")
    ((skipped++)) || true
else
    echo "  [7/$TOTAL] Enhancing vocals stem with Resemble-Enhance ..."
    run_test "Enhance (vocals from demucs)" \
        python generate.py audio --engine enhance \
            "$VOCALS" \
            -o "$DEMOS/enhance"
fi

# ── 8. Say: macOS TTS (system voice) ────────────────────────────────────────

echo "  [8/$TOTAL] Generating speech with macOS say (system voice) ..."
run_test "Say TTS (system voice)" \
    python generate.py voice --engine say \
        --text "Der Fuchs springt über den Bach und klaut dem Ofen die Tür." \
        -o "$DEMOS/say"

# ── 9. Say: macOS TTS (specific voice) ──────────────────────────────────────

echo "  [9/$TOTAL] Generating speech with macOS say (Anna) ..."
run_test "Say TTS (Anna)" \
    python generate.py voice --engine say \
        -v Anna \
        --text "Petra hat es gesehen und gelacht." \
        -o "$DEMOS/say"

# ── 10-11. RVC: Voice Conversion + Say+RVC Pipeline ─────────────────────────
# Starts server, uses first available model

echo ""
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  RVC Server Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Starting RVC server ..."
python generate.py server start
sleep 5

# Pick first available voice model
VOICE=$(ls -1 "$SCRIPT_DIR/rvc_models" | head -1)
if [ -z "$VOICE" ]; then
    echo "  [10/$TOTAL] SKIP: No RVC voice models installed"
    results+=("SKIP  0s  RVC Convert")
    ((skipped++)) || true
    echo "  [11/$TOTAL] SKIP: No RVC voice models installed"
    results+=("SKIP  0s  Say + RVC Pipeline")
    ((skipped++)) || true
else
    echo "  [10/$TOTAL] Converting ace-test-song.wav with RVC voice: $VOICE ..."
    run_test "RVC convert ($VOICE)" \
        python generate.py voice --engine rvc \
            --model "$VOICE" \
            "$DEMOS/ace-test-song.wav" \
            -o "$DEMOS/convert"

    echo "  [11/$TOTAL] Say + RVC pipeline with voice: $VOICE ..."
    run_test "Say + RVC pipeline ($VOICE)" \
        python generate.py voice --engine say \
            --model "$VOICE" \
            --text "This is a test of the say plus RVC pipeline." \
            -o "$DEMOS/say"
fi

echo ""
echo "  Stopping RVC server ..."
python generate.py server stop || true

# ── 12. Diarize: 1 Speaker (ACE-Song) ───────────────────────────────────────

echo "  [12/$TOTAL] Diarizing ace-test-song.wav (auto speaker count) ..."
run_test "Diarize 1 speaker (ace-test-song)" \
    python generate.py audio --engine diarize \
        "$DEMOS/ace-test-song.wav" \
        -o "$DEMOS/diarize"

# ── 13. Diarize: 3 Speaker Podcast + Verify ─────────────────────────────────

echo "  [13/$TOTAL] Diarizing podcast (3 speakers + verify) ..."
run_test "Diarize 3 speakers + verify (podcast)" \
    python generate.py audio --engine diarize \
        "$PODCAST" \
        --speakers 3 --verify \
        -o "$DEMOS/diarize-podcast"

# ── 14. AI-TTS: Basic ────────────────────────────────────────────────────────

echo "  [14/$TOTAL] Generating speech with AI-TTS (basic) ..."
run_test "AI-TTS basic" \
    python generate.py voice --engine ai-tts \
        --text "Hello world, this is a test of Qwen three TTS." \
        -o "$DEMOS/ai-tts"

# ── 15. AI-TTS: Voice + Tags ────────────────────────────────────────────────

echo "  [15/$TOTAL] Generating speech with AI-TTS (Serena + whispering) ..."
run_test "AI-TTS voice + tags" \
    python generate.py voice --engine ai-tts \
        -v Serena \
        -t "whispering, slow" \
        --text "And then — silence. The kind that swallows you whole." \
        -o "$DEMOS/ai-tts"

# ── 16. PS: System Status ───────────────────────────────────────────────────

echo "  [16/$TOTAL] System status ..."
run_test "PS (system status)" \
    python generate.py ps

# ── Summary ──────────────────────────────────────────────────────────────────

echo ""
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  RESULTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
for r in "${results[@]}"; do
    echo "  $r"
done
echo ""
echo "  Total: $passed passed, $failed failed, $skipped skipped"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

[ "$failed" -eq 0 ]
