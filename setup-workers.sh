#!/bin/bash
# Create the shared regular-CPython environment used by all model workers.

set -euo pipefail

cd "$(dirname "$0")"

WORKERS_VENV="${WORKERS_VENV:-./.venv-workers}"
source scripts/resolve_worker_python.sh
WORKERS_BOOTSTRAP_PYTHON="$(
    resolve_worker_python "${WORKERS_BOOTSTRAP_PYTHON:-}"
)"
ML_ACCELERATOR="${ML_ACCELERATOR:-auto}"
WORKERS_PYTHON="$WORKERS_VENV/bin/python"

if [ "$ML_ACCELERATOR" = auto ]; then
    if [ -r /proc/driver/nvidia/version ]; then
        ML_ACCELERATOR=cuda
    else
        ML_ACCELERATOR=cpu
    fi
fi

if [ "$ML_ACCELERATOR" = cuda ]; then
    ML_TORCH_INDEX_URL="${ML_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
    ML_TORCH_VERSION="${ML_TORCH_VERSION:-2.10.0+cu128}"
    ML_TORCHVISION_VERSION="${ML_TORCHVISION_VERSION:-0.25.0+cu128}"
fi

verify_python() {
    "$1" -c '
import platform
import sys
if platform.python_implementation() != "CPython":
    raise SystemExit("Model workers require CPython")
if sys.version_info[:2] != (3, 11):
    raise SystemExit(
        "Worker lock requires Python 3.11, "
        f"got {platform.python_version()}"
    )
if "t" in getattr(sys, "abiflags", ""):
    raise SystemExit("Worker environment must use regular CPython 3.11")
'
}

verify_python "$WORKERS_BOOTSTRAP_PYTHON"
if [ -e "$WORKERS_VENV" ]; then
    if [ ! -x "$WORKERS_PYTHON" ]; then
        echo "ERROR: Existing WORKERS_VENV=$WORKERS_VENV has no bin/python." >&2
        exit 2
    fi
    verify_python "$WORKERS_PYTHON"
else
    "$WORKERS_BOOTSTRAP_PYTHON" -m venv "$WORKERS_VENV"
fi

if [ "$ML_ACCELERATOR" = cpu ]; then
    "$WORKERS_PYTHON" -m pip install --only-binary=:all: \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements-workers.txt
else
    if [ "$ML_ACCELERATOR" != cuda ] && [ "$ML_ACCELERATOR" != rocm ]; then
        echo "ERROR: ML_ACCELERATOR must be auto, cpu, cuda or rocm." >&2
        exit 2
    fi
    : "${ML_TORCH_INDEX_URL:?Set the official PyTorch wheel index URL}"
    : "${ML_TORCH_VERSION:?Set an exact torch version, including accelerator suffix}"
    : "${ML_TORCHVISION_VERSION:?Set the matching exact torchvision version}"
    "$WORKERS_PYTHON" -m pip install --only-binary=:all: --no-deps \
        -r requirements-workers-core.txt
    "$WORKERS_PYTHON" -m pip install --only-binary=:all: \
        --index-url "$ML_TORCH_INDEX_URL" \
        "torch==$ML_TORCH_VERSION" \
        "torchvision==$ML_TORCHVISION_VERSION"
fi

"$WORKERS_PYTHON" -m pip check
ML_ACCELERATOR_RESOLVED="$ML_ACCELERATOR" "$WORKERS_PYTHON" -c '
from diffusion_editor.workers.lama_model import LamaModel
import accelerate
import cv2
import diffusers
import onnxruntime
import rembg
import safetensors
import tokenizers
import torch
import torchvision
import transformers
import os
accelerator = os.environ.get("ML_ACCELERATOR_RESOLVED", "cpu")
if accelerator == "cuda":
    if torch.version.cuda is None:
        raise SystemExit("CUDA was selected, but a CPU-only torch was installed")
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA torch was installed, but no usable NVIDIA GPU/driver was found"
        )
print(
    "Shared worker environment ready:",
    f"accelerator={accelerator}",
    f"torch={torch.__version__}",
    f"torchvision={torchvision.__version__}",
    f"opencv={cv2.__version__}",
    f"onnxruntime={onnxruntime.__version__}",
    f"diffusers={diffusers.__version__}",
    f"transformers={transformers.__version__}",
)
'

echo "All model workers installed at $WORKERS_VENV"
