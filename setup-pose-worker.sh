#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POSE_VENV="${POSE_VENV:-$PROJECT_ROOT/.venv-pose}"

source "$PROJECT_ROOT/scripts/resolve_worker_python.sh"
BOOTSTRAP_PYTHON="$(resolve_worker_python "${POSE_BOOTSTRAP_PYTHON:-}")"

if [ ! -x "$POSE_VENV/bin/python" ]; then
    "$BOOTSTRAP_PYTHON" -m venv "$POSE_VENV"
fi

"$POSE_VENV/bin/python" -m pip install --upgrade pip
"$POSE_VENV/bin/python" -m pip install \
    -r "$PROJECT_ROOT/requirements-pose-worker.txt"

"$POSE_VENV/bin/python" -c '
import importlib.metadata
import cv2
import mediapipe
import onnxruntime
import rtmlib
import skimage
print("Pose worker ready")
print("  OpenCV:", cv2.__version__)
print("  MediaPipe:", mediapipe.__version__)
print("  ONNX Runtime:", onnxruntime.__version__)
print("  RTMLib:", importlib.metadata.version("rtmlib"))
print("  scikit-image:", skimage.__version__)
'
