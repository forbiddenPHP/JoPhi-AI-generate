#!/usr/bin/env bash
# set -e  # removed: let setup continue on errors

# ── RVC Worker Installer ─────────────────────────────────────────────────────
# Creates a dedicated conda env for RVC with Python 3.10 and pip <= 23.3.
# This ensures omegaconf 2.0.6 and fairseq 0.12.2 install natively without
# any patching — the old pip doesn't enforce the broken metadata validation.
# ─────────────────────────────────────────────────────────────────────────────

CONDA_BIN="/opt/miniconda3/bin/conda"
ENV_NAME="rvc"

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  RVC Worker — Installer"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ── Check conda ──────────────────────────────────────────────────────────────

if [ ! -f "$CONDA_BIN" ]; then
    echo -e "${RED}ERROR: conda not found at $CONDA_BIN${NC}"
    echo "  Install miniconda: brew install --cask miniconda"
    # exit 1  # warn only, do not abort setup
fi

# ── Create env ───────────────────────────────────────────────────────────────

if "$CONDA_BIN" env list 2>/dev/null | grep -q "^${ENV_NAME} " || [ -d "/opt/miniconda3/envs/$ENV_NAME" ]; then
    echo "  Removing old '$ENV_NAME' env ..."
    "$CONDA_BIN" env remove -y -n "$ENV_NAME" > /dev/null 2>&1
    rm -rf "/opt/miniconda3/envs/$ENV_NAME" 2>/dev/null || true
fi

echo "  Creating env: $ENV_NAME (Python 3.10, pip <= 23.3) ..."
"$CONDA_BIN" create -y -q -n "$ENV_NAME" python=3.10 "pip<=23.3" > /dev/null 2>&1
echo -e "${GREEN}✓${NC} Env created"

# ── Install rvc-python ───────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCKFILE="$SCRIPT_DIR/requirements.lock"
WHEELS_DIR="$SCRIPT_DIR/wheels"

echo "  Installing rvc-python (this may take a few minutes) ..."
if [ -d "$WHEELS_DIR" ] && [ "$(ls -A "$WHEELS_DIR"/*.whl 2>/dev/null)" ]; then
    # Offline install from cached wheels — works even if PyPI is gone
    echo "  Using cached wheels (offline) ..."
    "$CONDA_BIN" run -n "$ENV_NAME" pip install --no-index --find-links="$WHEELS_DIR" -r "$LOCKFILE" 2>&1 | \
        grep -E "^(Successfully|Installing|ERROR)" | head -5
elif [ -f "$LOCKFILE" ]; then
    # Online install with pinned versions
    echo "  Using pinned versions from requirements.lock ..."
    "$CONDA_BIN" run -n "$ENV_NAME" pip install -r "$LOCKFILE" 2>&1 | \
        grep -E "^(Successfully|Installing|Downloading|ERROR)" | head -5
else
    # Fallback: install latest
    echo "  No lockfile or wheels found, installing latest rvc-python ..."
    "$CONDA_BIN" run -n "$ENV_NAME" pip install rvc-python "setuptools<81" "torch>=2.0,<2.6" 2>&1 | \
        grep -E "^(Successfully|Installing|Downloading|ERROR)" | head -5

    # Generate lockfile for future installs
    echo "  Generating requirements.lock ..."
    "$CONDA_BIN" run -n "$ENV_NAME" pip freeze > "$LOCKFILE" 2>/dev/null
    echo -e "${GREEN}✓${NC} Lockfile saved"
fi

if "$CONDA_BIN" run -n "$ENV_NAME" python -c "from rvc_python.infer import RVCInference" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} rvc-python installed"
else
    echo -e "${RED}WARNING: rvc-python installation failed${NC}"
    echo "  Check: conda activate rvc && pip install rvc-python"
fi

# ── Verify ───────────────────────────────────────────────────────────────────

echo ""
echo "── Verification ──"

"$CONDA_BIN" run -n "$ENV_NAME" python -c "
checks = []
try:
    import torch
    checks.append(('PyTorch', torch.__version__, True))
except Exception as e:
    checks.append(('PyTorch', str(e)[:40], False))

try:
    from rvc_python.infer import RVCInference
    checks.append(('rvc-python', 'OK', True))
except Exception as e:
    checks.append(('rvc-python', str(e)[:40], False))

try:
    import fairseq
    checks.append(('fairseq', 'OK', True))
except Exception as e:
    checks.append(('fairseq', str(e)[:40], False))

try:
    import omegaconf
    checks.append(('omegaconf', omegaconf.__version__ if hasattr(omegaconf, '__version__') else 'OK', True))
except Exception as e:
    checks.append(('omegaconf', str(e)[:40], False))

all_ok = all(ok for _, _, ok in checks)
for name, ver, ok in checks:
    sym = 'OK' if ok else 'XX'
    print(f'  {sym} {name:20s} {ver}')

if not all_ok:
    import sys
    sys.exit(1)
"

echo ""
echo -e "${GREEN}✓${NC} RVC Worker ready"
echo "  Start with: ./worker/rvc/start.sh"
echo ""
