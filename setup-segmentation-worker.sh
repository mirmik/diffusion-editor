#!/bin/bash
# Create the isolated regular-CPython environment used by rembg.

set -euo pipefail

cd "$(dirname "$0")"

SEGMENTATION_VENV="${SEGMENTATION_VENV:-./.venv-segmentation}"
SEGMENTATION_BOOTSTRAP_PYTHON="${SEGMENTATION_BOOTSTRAP_PYTHON:-python3.11}"
SEGMENTATION_PYTHON="$SEGMENTATION_VENV/bin/python"

verify_python() {
    "$1" -c '
import platform
import sys
if platform.python_implementation() != "CPython":
    raise SystemExit("Segmentation worker requires CPython")
if sys.version_info[:2] != (3, 11):
    raise SystemExit(
        "Segmentation worker lock requires Python 3.11, "
        f"got {platform.python_version()}"
    )
if "t" in getattr(sys, "abiflags", ""):
    raise SystemExit(
        "Segmentation worker environment must use regular CPython 3.11"
    )
'
}

verify_python "$SEGMENTATION_BOOTSTRAP_PYTHON"
if [ -e "$SEGMENTATION_VENV" ]; then
    if [ ! -x "$SEGMENTATION_PYTHON" ]; then
        echo "ERROR: Existing SEGMENTATION_VENV=$SEGMENTATION_VENV has no bin/python." >&2
        exit 2
    fi
    verify_python "$SEGMENTATION_PYTHON"
else
    "$SEGMENTATION_BOOTSTRAP_PYTHON" -m venv "$SEGMENTATION_VENV"
fi

"$SEGMENTATION_PYTHON" -m pip install --only-binary=:all: \
    -r requirements-segmentation-worker.txt
"$SEGMENTATION_PYTHON" -m pip check
"$SEGMENTATION_PYTHON" -c '
import cv2
import onnxruntime
import rembg
print(
    "Segmentation worker ready:",
    f"onnxruntime={onnxruntime.__version__}",
    f"opencv={cv2.__version__}",
)
'

echo "Segmentation worker environment installed at $SEGMENTATION_VENV"
