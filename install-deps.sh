#!/bin/bash
# Install diffusion-editor into a local virtual environment.
#
# Termin packages are installed from the SDK wheelhouse:
#   $TERMIN_SDK/wheels
#
# SDK discovery:
#   1. $TERMIN_SDK
#   2. /opt/termin

set -euo pipefail

cd "$(dirname "$0")"

VENV="${VENV:-./venv}"
if [ ! -d "$VENV" ]; then
    echo "Creating venv: $VENV"
    python3 -m venv "$VENV"
fi

PY="$VENV/bin/python"
PIP="$VENV/bin/pip"

SDK_ARGS=()
if [ -n "${TERMIN_SDK:-}" ]; then
    SDK_ARGS=(--sdk "$TERMIN_SDK")
fi
export TERMIN_SDK
TERMIN_SDK="$("$PY" -m diffusion_editor.sdk_runtime resolve "${SDK_ARGS[@]}")"

WHEELHOUSE="$TERMIN_SDK/wheels"
echo "Using TERMIN_SDK=$TERMIN_SDK"
echo "Using wheelhouse=$WHEELHOUSE"

TERMIN_REQUIREMENTS_OUTPUT="$(
    "$PY" -m diffusion_editor.sdk_runtime requirements --sdk "$TERMIN_SDK"
)"
mapfile -t TERMIN_REQUIREMENTS <<< "$TERMIN_REQUIREMENTS_OUTPUT"

echo ""
echo "=== Installing diffusion-editor Python requirements ==="
"$PIP" install -r requirements.txt

echo ""
echo "=== Installing exact Termin packages from SDK wheelhouse ==="
"$PIP" install --force-reinstall --no-index --no-deps --find-links "$WHEELHOUSE" \
    "${TERMIN_REQUIREMENTS[@]}"

echo ""
echo "=== Installing diffusion-editor editable package ==="
"$PIP" install --no-build-isolation --no-deps -e .

echo ""
echo "=== Verifying Termin runtime imports ==="
"$PY" -m diffusion_editor.sdk_runtime verify-installed \
    --sdk "$TERMIN_SDK" --imports
"$PIP" check
"$PY" -m diffusion_editor.sdk_runtime write-state --sdk "$TERMIN_SDK"

echo ""
echo "Done. Dependencies installed into $VENV"
