#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

PY="${PYTHON:-./venv/bin/python3}"
if [ ! -x "$PY" ]; then
    echo "ERROR: Python environment not found: $PY" >&2
    echo "Run ./install-deps.sh first." >&2
    exit 1
fi

SDK_ARGS=()
if [ -n "${TERMIN_SDK:-}" ]; then
    SDK_ARGS=(--sdk "$TERMIN_SDK")
fi
export TERMIN_SDK
TERMIN_SDK="$("$PY" -m diffusion_editor.sdk_runtime resolve "${SDK_ARGS[@]}")"
"$PY" -m diffusion_editor.sdk_runtime verify-installed \
    --sdk "$TERMIN_SDK" --imports

exec "$PY" -m diffusion_editor.app.main "$@"
