#!/bin/bash
# Install diffusion-editor into a local virtual environment.
#
# Termin packages are installed from the SDK wheelhouse:
#   $TERMIN_SDK/wheels
#
# SDK discovery:
#   1. $TERMIN_SDK
#   2. /opt/termin

set -e

cd "$(dirname "$0")"

VENV="${VENV:-./venv}"
if [ ! -d "$VENV" ]; then
    echo "Creating venv: $VENV"
    python3 -m venv "$VENV"
fi

PY="$VENV/bin/python"
PIP="$VENV/bin/pip"

_sdk_valid() {
    [ -d "$1/lib" ] && [ -d "$1/wheels" ]
}

if [ -n "${TERMIN_SDK:-}" ]; then
    if ! _sdk_valid "$TERMIN_SDK"; then
        echo "ERROR: TERMIN_SDK=$TERMIN_SDK is not a valid Termin SDK." >&2
        echo "Expected both: \$TERMIN_SDK/lib and \$TERMIN_SDK/wheels" >&2
        exit 1
    fi
elif _sdk_valid "/opt/termin"; then
    export TERMIN_SDK="/opt/termin"
else
    echo "ERROR: Termin SDK not found." >&2
    echo "Set TERMIN_SDK or install Termin SDK to /opt/termin." >&2
    echo "Expected SDK layout: lib/ and wheels/" >&2
    exit 1
fi

WHEELHOUSE="$TERMIN_SDK/wheels"
echo "Using TERMIN_SDK=$TERMIN_SDK"
echo "Using wheelhouse=$WHEELHOUSE"

echo ""
echo "=== Installing Termin packages from SDK wheelhouse ==="
"$PIP" install --find-links "$WHEELHOUSE" tcgui termin-display

echo ""
echo "=== Installing diffusion-editor Python requirements ==="
"$PIP" install -r requirements.txt

echo ""
echo "=== Installing diffusion-editor editable package ==="
"$PIP" install --no-build-isolation -e .

echo ""
echo "=== Verifying Termin runtime imports ==="
"$PY" - <<'PY'
from termin.display import SDLBackendWindow

if SDLBackendWindow is None:
    raise RuntimeError("termin.display.SDLBackendWindow is not available in this Termin SDK build")

print("SDLBackendWindow OK")
PY

echo ""
echo "Done. Dependencies installed into $VENV"
