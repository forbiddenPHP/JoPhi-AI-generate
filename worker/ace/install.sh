#!/usr/bin/env bash
# ── ACE-Step 1.5 Setup ─────────────────────────────────────────────────────
# Clones ACE-Step repo, removes .git (code committed to project), runs uv sync.
# Models are downloaded automatically on first generation (from HuggingFace).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ACESTEP_DIR="$SCRIPT_DIR/ACE-Step-1.5"
REPO_URL="https://github.com/ace-step/ACE-Step-1.5.git"

# ── Find or install uv (need >= 0.5) ────────────────────────────────────────
UV_MIN_VERSION="0.5"
UV_BIN=""
for _candidate in "$HOME/.local/bin/uv" "$(command -v uv 2>/dev/null)"; do
    if [ -x "$_candidate" ] 2>/dev/null; then
        _ver=$("$_candidate" --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+')
        if [ "$(printf '%s\n' "$UV_MIN_VERSION" "$_ver" | sort -V | head -n1)" = "$UV_MIN_VERSION" ]; then
            UV_BIN="$_candidate"
            break
        fi
    fi
done
if [ -z "$UV_BIN" ]; then
    echo "uv >= $UV_MIN_VERSION not found. Installing ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh 2>/dev/null
    UV_BIN="$HOME/.local/bin/uv"
fi
echo "uv: $("$UV_BIN" --version)"

# ── Clone repo (if not already present) ─────────────────────────────────────
if [ -d "$ACESTEP_DIR" ]; then
    echo "ACE-Step already present at $ACESTEP_DIR"
else
    echo "Cloning ACE-Step 1.5 ..."
    git clone "$REPO_URL" "$ACESTEP_DIR"
    rm -rf "$ACESTEP_DIR/.git"
    echo "Removed .git — code will be committed to project."
fi

# ── Install dependencies ────────────────────────────────────────────────────
echo "Running uv sync ..."
cd "$ACESTEP_DIR"
"$UV_BIN" sync

# ── Verify ──────────────────────────────────────────────────────────────────
echo ""
echo "Verifying ACE-Step installation ..."
"$UV_BIN" run python -c "from acestep.inference import generate_music; print('ACE-Step OK')"

echo ""
echo "ACE-Step 1.5 installed successfully."
echo "Model downloads handled by setup.sh."
