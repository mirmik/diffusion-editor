#!/bin/bash
# Create the isolated regular-CPython environment used by ML model workers.

set -euo pipefail

cd "$(dirname "$0")"

ML_VENV="${ML_VENV:-./.venv-ml}"
ML_BOOTSTRAP_PYTHON="${ML_BOOTSTRAP_PYTHON:-python3.11}"
ML_ACCELERATOR="${ML_ACCELERATOR:-cpu}"
ML_PYTHON="$ML_VENV/bin/python"

verify_python() {
    "$1" -c '
import platform
import sys
if platform.python_implementation() != "CPython":
    raise SystemExit("ML worker requires CPython")
if sys.version_info[:2] != (3, 11):
    raise SystemExit(
        "ML worker lock requires Python 3.11, "
        f"got {platform.python_version()}"
    )
if "t" in getattr(sys, "abiflags", ""):
    raise SystemExit("ML worker must use regular CPython 3.11")
'
}

verify_python "$ML_BOOTSTRAP_PYTHON"
if [ -e "$ML_VENV" ]; then
    if [ ! -x "$ML_PYTHON" ]; then
        echo "ERROR: Existing ML_VENV=$ML_VENV has no bin/python." >&2
        exit 2
    fi
    verify_python "$ML_PYTHON"
else
    "$ML_BOOTSTRAP_PYTHON" -m venv "$ML_VENV"
fi

if [ "$ML_ACCELERATOR" = cpu ]; then
    "$ML_PYTHON" -m pip install --only-binary=:all: \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements-ml-worker.txt
else
    if [ "$ML_ACCELERATOR" != cuda ] && [ "$ML_ACCELERATOR" != rocm ]; then
        echo "ERROR: ML_ACCELERATOR must be cpu, cuda or rocm." >&2
        exit 2
    fi
    : "${ML_TORCH_INDEX_URL:?Set the official PyTorch wheel index URL}"
    : "${ML_TORCH_VERSION:?Set an exact torch version, including accelerator suffix}"
    : "${ML_TORCHVISION_VERSION:?Set the matching exact torchvision version}"
    "$ML_PYTHON" -m pip install --only-binary=:all: \
        -r requirements-ml-worker-core.txt
    "$ML_PYTHON" -m pip install --only-binary=:all: \
        --index-url "$ML_TORCH_INDEX_URL" \
        "torch==$ML_TORCH_VERSION" \
        "torchvision==$ML_TORCHVISION_VERSION"
fi

"$ML_PYTHON" -m pip check
"$ML_PYTHON" -c '
import accelerate
import diffusers
import safetensors
import tokenizers
import torch
import torchvision
import transformers
print(
    "ML worker ready:",
    f"torch={torch.__version__}",
    f"torchvision={torchvision.__version__}",
    f"diffusers={diffusers.__version__}",
    f"transformers={transformers.__version__}",
)
'

echo "ML worker environment installed at $ML_VENV"
