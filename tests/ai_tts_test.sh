#!/usr/bin/env bash
# ── AI-TTS Voice Test Suite (Dylan + Uncle_Fu) ──────────────────────────────
# Tests per-segment instruct via [Voice: tags] markers.
# Monolog, Dialog, Talkshow — plain + refined. DE + EN.
# Usage:  conda activate tts-mist && bash tests/ai_tts_test.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

OUT="$SCRIPT_DIR/demos/ai-tts-tests"
mkdir -p "$OUT"

passed=0
failed=0
results=()

# ── Voices ────────────────────────────────────────────────────────────────────
VOICES=(Dylan Uncle_Fu)

# ── Texts ─────────────────────────────────────────────────────────────────────

MONOLOG_DE="Die Sonne ging langsam unter und tauchte die Stadt in ein warmes, goldenes Licht. \
Auf den Straßen wurde es ruhiger, die letzten Pendler verschwanden in den U-Bahn-Eingängen. \
Ein leichter Wind trug den Duft von frischem Brot aus der Bäckerei an der Ecke herüber. \
Irgendwo in der Ferne spielte jemand Klavier — eine Melodie, die man nicht ganz zuordnen konnte, \
die aber trotzdem vertraut klang. Es war einer dieser Abende, an denen man einfach stehen bleiben \
und durchatmen möchte, bevor der Alltag einen wieder einholt."

MONOLOG_EN="The sun was setting slowly, painting the city in a warm golden glow. \
The streets grew quieter as the last commuters disappeared into the subway entrances. \
A gentle breeze carried the scent of fresh bread from the bakery on the corner. \
Somewhere in the distance, someone was playing the piano — a melody you couldn't quite place, \
yet it still felt familiar. It was one of those evenings where you just want to stop, \
take a deep breath, and let the world slow down before the daily grind catches up with you again."

REFINE_TAGS="dramatic, slow, deep voice, suspenseful pauses"

# ── Dialog texts ──────────────────────────────────────────────────────────────

DIALOG_DE="[Dylan] Hast du das gelesen? Die wollen den ganzen Park abreißen und ein Einkaufszentrum hinbauen.
[Uncle_Fu] Wirklich? Das ist doch der einzige grüne Fleck in der ganzen Gegend.
[Dylan] Genau. Und das Schlimmste ist: Die Entscheidung fällt schon nächste Woche im Stadtrat.
[Uncle_Fu] Dann sollten wir eine Petition starten. Ich kenne ein paar Leute, die das sofort unterschreiben würden.
[Dylan] Gute Idee. Ich schreibe heute Abend den Text, und du kümmerst dich um die Unterschriften?
[Uncle_Fu] Deal. Aber wir brauchen auch Presse. Kennst du jemanden bei der Lokalzeitung?"

DIALOG_EN="[Dylan] Did you read that? They want to tear down the entire park and build a shopping mall.
[Uncle_Fu] Seriously? That is the only green space left in the whole neighborhood.
[Dylan] Exactly. And the worst part is, the city council votes on it next week already.
[Uncle_Fu] Then we should start a petition. I know a few people who would sign immediately.
[Dylan] Good idea. I will write the text tonight, and you handle the signatures?
[Uncle_Fu] Deal. But we also need press coverage. Do you know anyone at the local newspaper?"

# ── Dialog refined texts (per-segment instruct) ──────────────────────────────

DIALOG_REFINED_DE="[Dylan: energetic, fast, young] Hast du das gelesen? Die wollen den ganzen Park abreißen und ein Einkaufszentrum hinbauen.
[Uncle_Fu: calm, wise, deep voice] Wirklich? Das ist doch der einzige grüne Fleck in der ganzen Gegend.
[Dylan: upset, louder] Genau. Und das Schlimmste ist: Die Entscheidung fällt schon nächste Woche im Stadtrat.
[Uncle_Fu: thoughtful, slow] Dann sollten wir eine Petition starten. Ich kenne ein paar Leute, die das sofort unterschreiben würden.
[Dylan: hopeful, lighter tone] Gute Idee. Ich schreibe heute Abend den Text, und du kümmerst dich um die Unterschriften?
[Uncle_Fu: determined, firm] Deal. Aber wir brauchen auch Presse. Kennst du jemanden bei der Lokalzeitung?"

DIALOG_REFINED_EN="[Dylan: energetic, fast, young] Did you read that? They want to tear down the entire park and build a shopping mall.
[Uncle_Fu: calm, wise, deep voice] Seriously? That is the only green space left in the whole neighborhood.
[Dylan: upset, louder] Exactly. And the worst part is, the city council votes on it next week already.
[Uncle_Fu: thoughtful, slow] Then we should start a petition. I know a few people who would sign immediately.
[Dylan: hopeful, lighter tone] Good idea. I will write the text tonight, and you handle the signatures?
[Uncle_Fu: determined, firm] Deal. But we also need press coverage. Do you know anyone at the local newspaper?"

# ── Talkshow texts ────────────────────────────────────────────────────────────

TALKSHOW_DE="[Dylan] Willkommen zur Sendung. Heute reden wir über die Zukunft der Stadt. Ono Anna, was denkst du?
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

TALKSHOW_EN="[Dylan] Welcome to the show. Today we are talking about the future of the city. Ono Anna, what do you think?
[Ono_Anna] I believe we need more green spaces. The air quality downtown is unbearable.
[Eric] That is true, but who is going to pay for it? The city is broke.
[Sohee] Maybe we should get creative. In Seoul, they turned old railway tracks into parks.
[Uncle_Fu] Good idea. But here, the political will is missing.
[Vivian] I think the problem is different. People just don't get involved anymore.
[Dylan] Interesting. Eric, what would you suggest?
[Eric] Grassroots initiatives. When there is enough pressure, politics will follow.
[Ono_Anna] Exactly. And social media helps enormously with that.
[Sohee] True. A viral campaign can achieve more than any petition.
[Uncle_Fu] As long as it doesn't stop at clicking. You have to go out there.
[Vivian] I completely agree. In the end, what happens on the street is what counts."

TALKSHOW_REFINED_DE="[Dylan: enthusiastic, warm, host voice] Willkommen zur Sendung. Heute reden wir über die Zukunft der Stadt. Ono Anna, was denkst du?
[Ono_Anna: passionate, concerned] Ich finde, wir brauchen mehr Grünflächen. Die Luft in der Innenstadt ist unerträglich.
[Eric: skeptical, dry] Das stimmt, aber wer soll das bezahlen? Die Stadt hat kein Geld.
[Sohee: inspired, bright] Vielleicht sollten wir kreativ werden. In Seoul haben sie alte Bahngleise in Parks verwandelt.
[Uncle_Fu: calm, wise, deep voice] Gute Idee. Aber hier fehlt der politische Wille dafür.
[Vivian: thoughtful, slow] Ich glaube, das Problem ist ein anderes. Die Leute engagieren sich nicht mehr.
[Dylan: curious, questioning] Interessant. Eric, was würdest du vorschlagen?
[Eric: determined, firm] Bürgerinitiativen. Wenn genug Druck da ist, bewegt sich auch die Politik.
[Ono_Anna: agreeing, energetic] Genau. Und Social Media hilft dabei enorm.
[Sohee: excited, fast] Stimmt. Eine virale Kampagne kann mehr bewirken als jede Petition.
[Uncle_Fu: serious, grounding] Solange es nicht nur beim Klicken bleibt. Man muss auch rausgehen.
[Vivian: warm, concluding] Da bin ich ganz bei dir. Am Ende zählt, was auf der Straße passiert."

TALKSHOW_REFINED_EN="[Dylan: enthusiastic, warm, host voice] Welcome to the show. Today we are talking about the future of the city. Ono Anna, what do you think?
[Ono_Anna: passionate, concerned] I believe we need more green spaces. The air quality downtown is unbearable.
[Eric: skeptical, dry] That is true, but who is going to pay for it? The city is broke.
[Sohee: inspired, bright] Maybe we should get creative. In Seoul, they turned old railway tracks into parks.
[Uncle_Fu: calm, wise, deep voice] Good idea. But here, the political will is missing.
[Vivian: thoughtful, slow] I think the problem is different. People just don't get involved anymore.
[Dylan: curious, questioning] Interesting. Eric, what would you suggest?
[Eric: determined, firm] Grassroots initiatives. When there is enough pressure, politics will follow.
[Ono_Anna: agreeing, energetic] Exactly. And social media helps enormously with that.
[Sohee: excited, fast] True. A viral campaign can achieve more than any petition.
[Uncle_Fu: serious, grounding] As long as it doesn't stop at clicking. You have to go out there.
[Vivian: warm, concluding] I completely agree. In the end, what happens on the street is what counts."

# ── Count total tests ─────────────────────────────────────────────────────────
# Monolog:          2 voices × 2 langs = 4
# Dialog:           1 pair × 2 langs = 2
# Monolog refined:  2 × 2 = 4
# Dialog refined:   1 × 2 = 2 (per-segment instruct)
# Talkshow:         1 × 2 = 2
# Talkshow refined: 1 × 2 = 2 (per-segment instruct)
TOTAL=16

run_test() {
    local name="$1"; shift
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

n=0

# ══════════════════════════════════════════════════════════════════════════════
#  PART 1: Monolog — jede Stimme, DE + EN
# ══════════════════════════════════════════════════════════════════════════════

for v in "${VOICES[@]}"; do
    ((n++))
    echo "  [$n/$TOTAL] Monolog DE — $v"
    run_test "Monolog DE — $v" \
        python generate.py voice --engine ai-tts \
            -v "$v" --language de \
            --text "$MONOLOG_DE" \
            -o "$OUT/monolog/de/${v}/speech.wav"
done

for v in "${VOICES[@]}"; do
    ((n++))
    echo "  [$n/$TOTAL] Monolog EN — $v"
    run_test "Monolog EN — $v" \
        python generate.py voice --engine ai-tts \
            -v "$v" --language en \
            --text "$MONOLOG_EN" \
            -o "$OUT/monolog/en/${v}/speech.wav"
done

# ══════════════════════════════════════════════════════════════════════════════
#  PART 2: Dialog — DE + EN
# ══════════════════════════════════════════════════════════════════════════════

((n++))
echo "  [$n/$TOTAL] Dialog DE — Dylan + Uncle_Fu"
run_test "Dialog DE — Dylan + Uncle_Fu" \
    python generate.py voice --engine ai-tts \
        --language de \
        --text "$DIALOG_DE" \
        -o "$OUT/dialog/de/speech.wav"

((n++))
echo "  [$n/$TOTAL] Dialog EN — Dylan + Uncle_Fu"
run_test "Dialog EN — Dylan + Uncle_Fu" \
    python generate.py voice --engine ai-tts \
        --language en \
        --text "$DIALOG_EN" \
        -o "$OUT/dialog/en/speech.wav"

# ══════════════════════════════════════════════════════════════════════════════
#  PART 3: Monolog refined — jede Stimme, DE + EN (global --tags)
# ══════════════════════════════════════════════════════════════════════════════

for v in "${VOICES[@]}"; do
    ((n++))
    echo "  [$n/$TOTAL] Monolog refined DE — $v"
    run_test "Monolog refined DE — $v" \
        python generate.py voice --engine ai-tts \
            -v "$v" --language de \
            -t "$REFINE_TAGS" \
            --text "$MONOLOG_DE" \
            -o "$OUT/monolog_refined/de/${v}/speech.wav"
done

for v in "${VOICES[@]}"; do
    ((n++))
    echo "  [$n/$TOTAL] Monolog refined EN — $v"
    run_test "Monolog refined EN — $v" \
        python generate.py voice --engine ai-tts \
            -v "$v" --language en \
            -t "$REFINE_TAGS" \
            --text "$MONOLOG_EN" \
            -o "$OUT/monolog_refined/en/${v}/speech.wav"
done

# ══════════════════════════════════════════════════════════════════════════════
#  PART 4: Dialog refined — per-segment instruct, DE + EN
# ══════════════════════════════════════════════════════════════════════════════

((n++))
echo "  [$n/$TOTAL] Dialog refined DE — Dylan + Uncle_Fu (per-segment instruct)"
run_test "Dialog refined DE — per-segment instruct" \
    python generate.py voice --engine ai-tts \
        --language de \
        --text "$DIALOG_REFINED_DE" \
        -o "$OUT/dialog_refined/de/speech.wav"

((n++))
echo "  [$n/$TOTAL] Dialog refined EN — Dylan + Uncle_Fu (per-segment instruct)"
run_test "Dialog refined EN — per-segment instruct" \
    python generate.py voice --engine ai-tts \
        --language en \
        --text "$DIALOG_REFINED_EN" \
        -o "$OUT/dialog_refined/en/speech.wav"

# ══════════════════════════════════════════════════════════════════════════════
#  PART 5: Talkshow — DE + EN
# ══════════════════════════════════════════════════════════════════════════════

((n++))
echo "  [$n/$TOTAL] Talkshow DE"
run_test "Talkshow DE — alle 6 Stimmen" \
    python generate.py voice --engine ai-tts \
        --language de \
        --text "$TALKSHOW_DE" \
        -o "$OUT/talkshow/de/speech.wav"

((n++))
echo "  [$n/$TOTAL] Talkshow EN"
run_test "Talkshow EN — alle 6 Stimmen" \
    python generate.py voice --engine ai-tts \
        --language en \
        --text "$TALKSHOW_EN" \
        -o "$OUT/talkshow/en/speech.wav"

# ══════════════════════════════════════════════════════════════════════════════
#  PART 6: Talkshow refined — per-segment instruct, DE + EN
# ══════════════════════════════════════════════════════════════════════════════

((n++))
echo "  [$n/$TOTAL] Talkshow refined DE (per-segment instruct)"
run_test "Talkshow refined DE — per-segment instruct" \
    python generate.py voice --engine ai-tts \
        --language de \
        --text "$TALKSHOW_REFINED_DE" \
        -o "$OUT/talkshow_refined/de/speech.wav"

((n++))
echo "  [$n/$TOTAL] Talkshow refined EN (per-segment instruct)"
run_test "Talkshow refined EN — per-segment instruct" \
    python generate.py voice --engine ai-tts \
        --language en \
        --text "$TALKSHOW_REFINED_EN" \
        -o "$OUT/talkshow_refined/en/speech.wav"

# ── Verify prompt sidecars ───────────────────────────────────────────────────

echo ""
echo "  Prompt-Sidecar check:"
PROMPT_COUNT=$(find "$OUT" -name "*.txt" | wc -l | tr -d ' ')
WAV_COUNT=$(find "$OUT" -name "*.wav" | wc -l | tr -d ' ')
echo "  WAV files: $WAV_COUNT"
echo "  TXT files: $PROMPT_COUNT"
if [ "$PROMPT_COUNT" -eq "$WAV_COUNT" ]; then
    echo "  OK: Jede WAV hat einen Prompt-Sidecar."
else
    echo "  WARNING: Nicht jede WAV hat einen Prompt-Sidecar!"
fi

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
echo "  Total: $passed passed, $failed failed (of $TOTAL)"
echo ""
echo "  Output:  $OUT/"
echo "  Prompts: $OUT/*.txt"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

[ "$failed" -eq 0 ]
