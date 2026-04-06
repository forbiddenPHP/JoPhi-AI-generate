#!/usr/bin/env bash
# FPS comparison test: same prompt, same seed, varying fps
set -e
trap 'echo "  ✗ Aborted"; kill 0 2>/dev/null; exit 1' INT TERM

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

CONDA_BIN=$([[ -f "$HOME/.ai-conda-path" ]] && cat "$HOME/.ai-conda-path" || echo "conda")
OUT="./demos"
mkdir -p "$OUT"

PROMPT='The camera opens in a calm, sunlit frog yoga studio. Warm morning light washes over the wooden floor as incense smoke drifts lazily in the air. The senior frog instructor sits cross-legged at the center, eyes closed, voice deep and calm. "We are one with the pond." All the frogs answer softly: "Ommm..." "We are one with the mud." "Ommm..." He smiles faintly. "We are one with the flies." A pause. The camera pans to the side towards one frog who twitches, eyes darting. Suddenly its tongue snaps out, catching a fly mid-air and pulling it into its mouth. The master exhales slowly, still serene. "But we do not chase the flies..." Beat. "not during class." The guilty frog lowers its head in shame, folding its hands back into a meditative pose. The other frogs resume their chant: "Ommm..." Camera holds for a moment on the embarrassed frog, eyes closed too tightly, pretending nothing happened.'

SEED=42
QUALITY=480p
RATIO="16:9"
DURATION=20

for FPS in 1 2 5 10 14 24 25 30 50 60; do
    FRAMES=$((DURATION * FPS))
    OUTFILE="${OUT}/test-fps-${FPS}-seed-${SEED}-${QUALITY}-16x9-${FRAMES}f-${DURATION}s.mp4"
    if [ -f "$OUTFILE" ]; then
        echo "  ⏭ Skipping fps=${FPS} — ${OUTFILE} exists"
        continue
    fi
    echo "  ▶ Generating fps=${FPS}, frames=${FRAMES}, ${DURATION}s → ${OUTFILE}"
    "$CONDA_BIN" run --no-capture-output -n ltx2 python generate.py video ltx2.3 \
        -p "$PROMPT" \
        --num-frames "$FRAMES" \
        --frame-rate "$FPS" \
        --quality "$QUALITY" \
        --ratio "$RATIO" \
        --seed "$SEED" \
        -o "$OUTFILE"
done

echo "  ✓ All done. Files in ${OUT}/"
