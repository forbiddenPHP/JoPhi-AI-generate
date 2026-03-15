#!/usr/bin/env bash
# ── Full Test Suite for generate.py ──────────────────────────────────────────
# Runs all 21 worker tests SEQUENTIALLY.  All output goes to demos/full-test/.
# Existing outputs are skipped automatically (exact file checks).
# Usage:  conda activate tts-mist && bash tests/full_test_run.sh

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

DEMOS="$SCRIPT_DIR/demos/full-test"
mkdir -p "$DEMOS"
PODCAST="$SCRIPT_DIR/demos/test-input-audio-file/podcast-5min.mp3"
TOTAL=18

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

skip() {
    local name="$1"
    echo "  SKIP (exists): $name"
    results+=("SKIP  0s  $name")
    ((skipped++)) || true
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

ACE_WAV="$DEMOS/ace-test-song.wav"

echo "  [1/$TOTAL] ACE-Step (turbo, 120s synthwave)"
if [ -f "$ACE_WAV" ]; then
    skip "ACE-Step (turbo, 120s synthwave)"
else
    run_test "ACE-Step (turbo, 120s synthwave)" \
        python generate.py audio --engine ace-step --model turbo \
            -l "$LYRICS_ACE" \
            -t "synthwave,electronic,upbeat,80s,energetic" \
            -s 120 -o "$ACE_WAV"
fi

# ── 2. HeartMuLa: Musik generieren (2 min) ──────────────────────────────────

HEART_WAV="$DEMOS/heart-test-song.wav"

echo "  [2/$TOTAL] HeartMuLa (120s synthwave)"
if [ -f "$HEART_WAV" ]; then
    skip "HeartMuLa (120s synthwave)"
else
    run_test "HeartMuLa (120s synthwave)" \
        python generate.py audio --engine heartmula \
            -l "$LYRICS_ACE" \
            -t "synthwave,electronic,upbeat,80s,energetic" \
            -s 120 -o "$HEART_WAV"
fi

# ── 3. Whisper: Transkription EN (ACE-Song) ─────────────────────────────────
# Abhängigkeit: ace-test-song.wav (Test 1)

TRANSCRIBE_ACE_JSON="$DEMOS/transcribe-ace/ace-test-song.json"

echo "  [3/$TOTAL] Whisper transcribe EN (ace-test-song)"
if [ -f "$TRANSCRIBE_ACE_JSON" ]; then
    skip "Whisper transcribe EN (ace-test-song)"
elif [ ! -f "$ACE_WAV" ]; then
    echo "  SKIP: ace-test-song.wav nicht vorhanden (Test 1 muss zuerst passen)"
    results+=("SKIP  0s  Whisper transcribe EN (kein Input)")
    ((skipped++)) || true
else
    run_test "Whisper transcribe EN (ace-test-song)" \
        python generate.py text --engine whisper \
            "$ACE_WAV" \
            --input-language en --format all \
            -o "$DEMOS/transcribe-ace"
fi

# ── 4. Whisper: Transkription DE (Podcast) ──────────────────────────────────

TRANSCRIBE_PODCAST_JSON="$DEMOS/transcribe-podcast/podcast-5min.json"

echo "  [4/$TOTAL] Whisper transcribe DE (podcast)"
if [ -f "$TRANSCRIBE_PODCAST_JSON" ]; then
    skip "Whisper transcribe DE (podcast)"
elif [ ! -f "$PODCAST" ]; then
    echo "  SKIP: podcast-5min.mp3 nicht vorhanden"
    results+=("SKIP  0s  Whisper transcribe DE (kein Input)")
    ((skipped++)) || true
else
    run_test "Whisper transcribe DE (podcast)" \
        python generate.py text --engine whisper \
            "$PODCAST" \
            --input-language de --format all \
            -o "$DEMOS/transcribe-podcast"
fi

# ── 5. HeartMuLa Transcribe: Lyrics Extraction ─────────────────────────────
# Abhängigkeit: ace-test-song.wav (Test 1)

LYRICS_TXT="$DEMOS/transcribe-ace-lyrics.txt"

echo "  [5/$TOTAL] HeartMuLa transcribe (lyrics extraction)"
if [ -f "$LYRICS_TXT" ]; then
    skip "HeartMuLa transcribe (lyrics extraction)"
elif [ ! -f "$ACE_WAV" ]; then
    echo "  SKIP: ace-test-song.wav nicht vorhanden (Test 1 muss zuerst passen)"
    results+=("SKIP  0s  HeartMuLa transcribe (kein Input)")
    ((skipped++)) || true
else
    run_test "HeartMuLa transcribe (lyrics extraction)" \
        python generate.py text --engine heartmula-transcribe \
            "$ACE_WAV" \
            -o "$LYRICS_TXT"
fi

# ── 6. Demucs: Separation (ACE-Song) ────────────────────────────────────────
# Abhängigkeit: ace-test-song.wav (Test 1)

VOCALS_WAV="$DEMOS/separate-ace/ace-test-song_vocals.wav"

echo "  [6/$TOTAL] Demucs separate (ace-test-song)"
if [ -f "$VOCALS_WAV" ]; then
    skip "Demucs separate (ace-test-song)"
elif [ ! -f "$ACE_WAV" ]; then
    echo "  SKIP: ace-test-song.wav nicht vorhanden (Test 1 muss zuerst passen)"
    results+=("SKIP  0s  Demucs separate (kein Input)")
    ((skipped++)) || true
else
    run_test "Demucs separate (ace-test-song)" \
        python generate.py audio --engine demucs \
            "$ACE_WAV" \
            -o "$DEMOS/separate-ace"
fi

# ── 7. Enhance (ACE-Song vocals) ────────────────────────────────────────────
# Abhängigkeit: vocals.wav aus Demucs (Test 6)

ENHANCED_VOCALS="$DEMOS/enhance/ace-test-song_vocals.wav"

echo "  [7/$TOTAL] Enhance (vocals from demucs)"
if [ -f "$ENHANCED_VOCALS" ]; then
    skip "Enhance (vocals from demucs)"
elif [ ! -f "$VOCALS_WAV" ]; then
    echo "  SKIP: Vocals stem nicht vorhanden (Test 6 muss zuerst passen)"
    results+=("SKIP  0s  Enhance (kein Input)")
    ((skipped++)) || true
else
    run_test "Enhance (vocals from demucs)" \
        python generate.py audio --engine enhance \
            "$VOCALS_WAV" \
            -o "$DEMOS/enhance"
fi

# ── 8. Say: macOS TTS (system voice) ────────────────────────────────────────

SAY_DEFAULT="$DEMOS/say/say_default_der_fuchs_springt_über_den.wav"

echo "  [8/$TOTAL] Say TTS (system voice)"
if [ -f "$SAY_DEFAULT" ]; then
    skip "Say TTS (system voice)"
else
    run_test "Say TTS (system voice)" \
        python generate.py voice --engine say \
            --text "Der Fuchs springt über den Bach und klaut dem Ofen die Tür." \
            -o "$DEMOS/say"
fi

# ── 9. Say: macOS TTS (specific voice) ──────────────────────────────────────

SAY_ANNA="$DEMOS/say/say_Anna_petra_hat_es_gesehen_und.wav"

echo "  [9/$TOTAL] Say TTS (Anna)"
if [ -f "$SAY_ANNA" ]; then
    skip "Say TTS (Anna)"
else
    run_test "Say TTS (Anna)" \
        python generate.py voice --engine say \
            -v Anna \
            --text "Petra hat es gesehen und gelacht." \
            -o "$DEMOS/say"
fi

# ── 10-11. RVC: Voice Conversion + Say+RVC Pipeline ─────────────────────────
# Abhängigkeit 10: say_default (Test 8)
# Abhängigkeit 11: keine (Say erzeugt eigenen Input)

# Say+RVC Output: say erzeugt wav, dann RVC → say/rvc/*.wav
SAY_RVC_DIR="$DEMOS/say/rvc"

# Server starten wenn convert oder say+rvc noch Arbeit braucht
NEED_RVC_SERVER=false
CONVERT_COUNT=$(ls -1d "$DEMOS/convert"/*/ 2>/dev/null | wc -l | tr -d ' ')
MODEL_COUNT=$(ls -1 "$SCRIPT_DIR/worker/rvc/models" 2>/dev/null | wc -l | tr -d ' ')
if [ "$CONVERT_COUNT" -lt "$MODEL_COUNT" ] || [ ! -d "$SAY_RVC_DIR" ] || [ -z "$(ls -A "$SAY_RVC_DIR" 2>/dev/null)" ]; then
    NEED_RVC_SERVER=true
fi

if $NEED_RVC_SERVER; then
    echo ""
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  RVC Server Setup"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "  Starting RVC server ..."
    python generate.py server start || true
    sleep 5
fi

ALL_VOICES=$(ls -1 "$SCRIPT_DIR/worker/rvc/models" 2>/dev/null)

SAY_DEFAULT_NAME="say_default_der_fuchs_springt_über_den.wav"
SAY_ANNA_NAME="say_Anna_petra_hat_es_gesehen_und.wav"

echo "  [10/$TOTAL] RVC convert (beide Say-Dateien → alle Stimmen)"
if [ -z "$ALL_VOICES" ]; then
    echo "  SKIP: Keine RVC Voice-Models installiert"
    results+=("SKIP  0s  RVC convert (kein Model)")
    ((skipped++)) || true
elif [ ! -f "$SAY_DEFAULT" ] && [ ! -f "$SAY_ANNA" ]; then
    echo "  SKIP: Keine Say-Dateien vorhanden (Tests 8+9 müssen zuerst passen)"
    results+=("SKIP  0s  RVC convert (kein Input)")
    ((skipped++)) || true
else
    ALL_DONE=true
    for VOICE in $ALL_VOICES; do
        for SAY_FILE in "$SAY_DEFAULT" "$SAY_ANNA"; do
            [ ! -f "$SAY_FILE" ] && continue
            SAY_BASENAME="$(basename "$SAY_FILE")"
            CONVERT_OUT="$DEMOS/convert/$VOICE/$SAY_BASENAME"
            if [ -f "$CONVERT_OUT" ]; then
                echo "  SKIP (exists): $VOICE/$SAY_BASENAME"
                continue
            fi
            ALL_DONE=false
            run_test "RVC convert $SAY_BASENAME → $VOICE" \
                python generate.py voice --engine rvc \
                    --model "$VOICE" \
                    "$SAY_FILE" \
                    -o "$DEMOS/convert/$VOICE"
        done
    done
    if $ALL_DONE; then
        skip "RVC convert (alle Stimmen)"
    fi
fi

echo "  [11/$TOTAL] Say + RVC pipeline"
if [ -d "$SAY_RVC_DIR" ] && [ -n "$(ls -A "$SAY_RVC_DIR" 2>/dev/null)" ]; then
    skip "Say + RVC pipeline"
elif [ -z "$VOICE" ]; then
    echo "  SKIP: Keine RVC Voice-Models installiert"
    results+=("SKIP  0s  Say + RVC pipeline (kein Model)")
    ((skipped++)) || true
else
    run_test "Say + RVC pipeline ($VOICE)" \
        python generate.py voice --engine say \
            --model "$VOICE" \
            --text "This is a test of the say plus RVC pipeline." \
            -o "$DEMOS/say"
fi

if $NEED_RVC_SERVER; then
    echo ""
    echo "  Stopping RVC server ..."
    python generate.py server stop 2>/dev/null || true
fi

# ── 12. Diarize: Podcast (3 Speaker + Verify) ───────────────────────────────

DIARIZE_JSON="$DEMOS/diarize/podcast-5min_diarize.json"

echo "  [12/$TOTAL] Diarize podcast (3 speakers + verify)"
if [ -f "$DIARIZE_JSON" ]; then
    skip "Diarize podcast (3 speakers + verify)"
elif [ ! -f "$PODCAST" ]; then
    echo "  SKIP: podcast-5min.mp3 nicht vorhanden"
    results+=("SKIP  0s  Diarize podcast (kein Input)")
    ((skipped++)) || true
else
    run_test "Diarize podcast (3 speakers + verify)" \
        python generate.py audio --engine diarize \
            "$PODCAST" \
            --speakers 3 --verify \
            -o "$DEMOS/diarize"
fi

# ── 13. AI-TTS: Basic (Aiden EN) ─────────────────────────────────────────────
# Output: ai-tts/ai_tts_Aiden_hello_world_this_is_a.wav

AITTS_BASIC="$DEMOS/ai-tts/ai_tts_Aiden_hello_world_this_is_a.wav"

echo "  [13/$TOTAL] AI-TTS basic (Aiden EN)"
if [ -f "$AITTS_BASIC" ]; then
    skip "AI-TTS basic (Aiden EN)"
else
    run_test "AI-TTS basic (Aiden EN)" \
        python generate.py voice --engine ai-tts \
            -v Aiden \
            --text "Hello world, this is a test of Qwen three TTS." \
            -o "$DEMOS/ai-tts"
fi

# ── 14. AI-TTS: Voice + Tags ────────────────────────────────────────────────
# Output: ai-tts/ai_tts_Serena_and_then__silence_the.wav

AITTS_SERENA="$DEMOS/ai-tts/ai_tts_Serena_and_then__silence_the.wav"

echo "  [14/$TOTAL] AI-TTS voice + tags"
if [ -f "$AITTS_SERENA" ]; then
    skip "AI-TTS voice + tags"
else
    run_test "AI-TTS voice + tags" \
        python generate.py voice --engine ai-tts \
            -v Serena \
            -t "whispering, slow" \
            --text "And then — silence. The kind that swallows you whole." \
            -o "$DEMOS/ai-tts"
fi

# ── 15. AI-TTS: Deutsch ────────────────────────────────────────────────────
# Output: ai-tts/ai_tts_Dylan_die_sonne_ging_langsam_unter.wav

AITTS_DYLAN="$DEMOS/ai-tts/ai_tts_Dylan_die_sonne_ging_langsam_unter.wav"

echo "  [15/$TOTAL] AI-TTS Dylan DE"
if [ -f "$AITTS_DYLAN" ]; then
    skip "AI-TTS Dylan DE"
else
    run_test "AI-TTS Dylan DE" \
        python generate.py voice --engine ai-tts \
            -v Dylan --language de \
            --text "Die Sonne ging langsam unter und tauchte die Stadt in ein warmes, goldenes Licht." \
            -o "$DEMOS/ai-tts"
fi

# ── 16. AI-TTS: Uncle_Fu EN ───────────────────────────────────────────────
# Output: ai-tts/ai_tts_Uncle_Fu_listen_carefully_what_i_am.wav

AITTS_UNCLE="$DEMOS/ai-tts/ai_tts_Uncle_Fu_listen_carefully_what_i_am.wav"

echo "  [16/$TOTAL] AI-TTS Uncle_Fu EN"
if [ -f "$AITTS_UNCLE" ]; then
    skip "AI-TTS Uncle_Fu EN"
else
    run_test "AI-TTS Uncle_Fu EN" \
        python generate.py voice --engine ai-tts \
            -v Uncle_Fu --language en \
            --text "Listen carefully. What I am about to tell you will change everything you thought you knew." \
            -o "$DEMOS/ai-tts"
fi

# ── 17. AI-TTS: Talkshow (6 Stimmen, DE) ─────────────────────────────────
# Output: ai-tts/ai_tts_default_dylan_willkommen_zur_sendung_heute.wav

AITTS_TALKSHOW="$DEMOS/ai-tts/ai_tts_default_dylan_willkommen_zur_sendung_heute.wav"

TALKSHOW_TEXT="[Dylan] Willkommen zur Sendung. Heute reden wir über die Zukunft der Stadt. Ono Anna, was denkst du?
[Ono_Anna] Ich finde, wir brauchen mehr Grünflächen. Die Luft in der Innenstadt ist unerträglich.
[Eric] Das stimmt, aber wer soll das bezahlen? Die Stadt hat kein Geld.
[Sohee] Vielleicht sollten wir kreativ werden. In Seoul haben sie alte Bahngleise in Parks verwandelt.
[Uncle_Fu] Gute Idee. Aber hier fehlt der politische Wille dafür.
[Vivian] Ich glaube, das Problem ist ein anderes. Die Leute engagieren sich nicht mehr.
[Dylan] Interessant. Eric, was würdest du vorschlagen?
[Eric] Bürgerinitiativen. Wenn genug Druck da ist, bewegt sich auch die Politik.
[Ono_Anna] Genau. Und Social Media hilft dabei enorm.
[Sohee] Stimmt. Eine virale Kampagne kann mehr bewirken als jede Petition.
[Uncle_Fu] Solange es nicht nur beim Klicken bleibt. Man muss auch rausgehen.
[Vivian] Da bin ich ganz bei dir. Am Ende zählt, was auf der Straße passiert."

echo "  [17/$TOTAL] AI-TTS Talkshow DE (6 Stimmen)"
if [ -f "$AITTS_TALKSHOW" ]; then
    skip "AI-TTS Talkshow DE (6 Stimmen)"
else
    run_test "AI-TTS Talkshow DE (6 Stimmen)" \
        python generate.py voice --engine ai-tts \
            --language de \
            --text "$TALKSHOW_TEXT" \
            -o "$DEMOS/ai-tts"
fi

# ── 18. PS: System Status ────────────────────────────────────────────────────

echo "  [18/$TOTAL] System status ..."
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
