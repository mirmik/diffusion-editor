#!/bin/bash
cd "$(dirname "$0")"

PY="${PYTHON:-./venv/bin/python3}"
if [ ! -x "$PY" ]; then
    echo "ERROR: Python environment not found: $PY" >&2
    echo "Run ./install-deps.sh first." >&2
    exit 1
fi

exec "$PY" -m diffusion_editor.app.main "$@"
