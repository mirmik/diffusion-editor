#!/bin/bash
# Install diffusion-editor into a local virtual environment.
#
# Termin packages are installed from the SDK wheelhouse:
#   $TERMIN_SDK/wheels
#
# SDK discovery:
#   1. $TERMIN_SDK
#   2. the path saved in .termin-sdk
#   3. /opt/termin
#
# The virtual environment is always created by the SDK-owned CPython 3.14t.
# An existing environment is checked before it is modified.

set -euo pipefail

cd "$(dirname "$0")"

VENV="${VENV:-./venv}"
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-python3}"

SDK_ARGS=()
if [ -n "${TERMIN_SDK:-}" ]; then
    SDK_ARGS=(--sdk "$TERMIN_SDK")
fi
export TERMIN_SDK
TERMIN_SDK="$("$BOOTSTRAP_PYTHON" -m diffusion_editor.sdk_runtime resolve "${SDK_ARGS[@]}")"
SDK_PYTHON="$(
    "$BOOTSTRAP_PYTHON" -m diffusion_editor.sdk_runtime python-executable \
        --sdk "$TERMIN_SDK"
)"

PY="$VENV/bin/python"
if [ -e "$VENV" ]; then
    if [ ! -x "$PY" ]; then
        echo "ERROR: Existing VENV=$VENV has no executable bin/python." >&2
        echo "Move it aside or choose a new VENV path; it was not modified." >&2
        exit 2
    fi
    if ! "$BOOTSTRAP_PYTHON" -m diffusion_editor.sdk_runtime \
        verify-python-executable --sdk "$TERMIN_SDK" --python "$PY"; then
        echo "ERROR: Existing VENV=$VENV is not compatible with the selected Termin SDK." >&2
        echo "Move it aside or choose a new VENV path; it was not modified." >&2
        exit 2
    fi
else
    echo "Creating CPython 3.14t venv with $SDK_PYTHON: $VENV"
    "$SDK_PYTHON" -m venv "$VENV"
fi

WHEELHOUSE="$TERMIN_SDK/wheels"
echo "Using TERMIN_SDK=$TERMIN_SDK"
echo "Using Python=$PY"
echo "Using wheelhouse=$WHEELHOUSE"

TERMIN_REQUIREMENTS_OUTPUT="$(
    "$BOOTSTRAP_PYTHON" -m diffusion_editor.sdk_runtime requirements \
        --sdk "$TERMIN_SDK"
)"
mapfile -t TERMIN_REQUIREMENTS <<< "$TERMIN_REQUIREMENTS_OUTPUT"

echo ""
echo "=== Installing diffusion-editor Python requirements ==="
"$PY" -m pip install --only-binary=:all: \
    --find-links "$WHEELHOUSE" -r requirements.txt

echo ""
echo "=== Installing exact Termin packages from SDK wheelhouse ==="
"$PY" -m pip install --force-reinstall --no-index --no-deps --find-links "$WHEELHOUSE" \
    "${TERMIN_REQUIREMENTS[@]}"

echo ""
echo "=== Installing diffusion-editor editable package ==="
"$PY" -m pip install --no-build-isolation --no-deps -e .

echo ""
echo "=== Verifying Termin runtime imports ==="
"$PY" -m diffusion_editor.sdk_runtime verify-installed \
    --sdk "$TERMIN_SDK" --imports
"$PY" scripts/probe_main_process_dependencies.py
"$PY" -m pip check
"$PY" -m diffusion_editor.sdk_runtime write-state --sdk "$TERMIN_SDK"

echo ""
echo "Done. Dependencies installed into $VENV"
