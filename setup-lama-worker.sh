#!/bin/bash
# Create the isolated regular-CPython environment used by the LaMa worker.

set -euo pipefail

cd "$(dirname "$0")"

LAMA_VENV="${LAMA_VENV:-./.venv-lama}"
LAMA_BOOTSTRAP_PYTHON="${LAMA_BOOTSTRAP_PYTHON:-python3.11}"
LAMA_PYTHON="$LAMA_VENV/bin/python"

verify_python() {
    "$1" -c '
import platform
import sys
if platform.python_implementation() != "CPython":
    raise SystemExit("LaMa worker requires CPython")
if sys.version_info[:2] != (3, 11):
    raise SystemExit(
        f"LaMa worker lock requires Python 3.11, got {platform.python_version()}"
    )
if "t" in getattr(sys, "abiflags", ""):
    raise SystemExit("LaMa worker environment must use regular CPython 3.11")
'
}

verify_python "$LAMA_BOOTSTRAP_PYTHON"
if [ -e "$LAMA_VENV" ]; then
    if [ ! -x "$LAMA_PYTHON" ]; then
        echo "ERROR: Existing LAMA_VENV=$LAMA_VENV has no bin/python." >&2
        exit 2
    fi
    verify_python "$LAMA_PYTHON"
else
    "$LAMA_BOOTSTRAP_PYTHON" -m venv "$LAMA_VENV"
fi

"$LAMA_PYTHON" -m pip install --only-binary=:all: \
    -r requirements-lama-worker-build.txt
"$LAMA_PYTHON" -m pip install --only-binary=:all: \
    --no-binary=fire \
    --no-build-isolation \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements-lama-worker.txt
"$LAMA_PYTHON" -m pip check
"$LAMA_PYTHON" -c '
from simple_lama_inpainting import SimpleLama
import cv2
import torch
import torchvision
print(
    "LaMa worker ready:",
    f"torch={torch.__version__}",
    f"torchvision={torchvision.__version__}",
    f"opencv={cv2.__version__}",
)
'

echo "LaMa worker environment installed at $LAMA_VENV"
