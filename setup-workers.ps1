$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$WorkersVenv = if ($env:WORKERS_VENV) { $env:WORKERS_VENV } else { ".venv-workers" }
$Bootstrap = if ($env:WORKERS_BOOTSTRAP_PYTHON) {
    $env:WORKERS_BOOTSTRAP_PYTHON
} else {
    "py"
}
$BootstrapArgs = if ($env:WORKERS_BOOTSTRAP_PYTHON) { @() } else { @("-3.11") }
$Accelerator = if ($env:ML_ACCELERATOR) { $env:ML_ACCELERATOR } else { "cpu" }
$WorkersPython = Join-Path $WorkersVenv "Scripts/python.exe"

function Assert-WorkerPython {
    param([string]$Python, [string[]]$PrefixArgs = @())
    & $Python @PrefixArgs -c @'
import platform
import sys
if platform.python_implementation() != "CPython":
    raise SystemExit("Model workers require CPython")
if sys.version_info[:2] != (3, 11):
    raise SystemExit(
        f"Worker lock requires Python 3.11, got {platform.python_version()}"
    )
if "t" in getattr(sys, "abiflags", ""):
    raise SystemExit("Worker environment must use regular CPython 3.11")
'@
}

Assert-WorkerPython $Bootstrap $BootstrapArgs
if (Test-Path $WorkersVenv) {
    if (-not (Test-Path $WorkersPython -PathType Leaf)) {
        throw "Existing WORKERS_VENV=$WorkersVenv has no Scripts/python.exe"
    }
    Assert-WorkerPython $WorkersPython
} else {
    & $Bootstrap @BootstrapArgs -m venv $WorkersVenv
}

if ($Accelerator -eq "cpu") {
    & $WorkersPython -m pip install --only-binary=:all: `
        --extra-index-url https://download.pytorch.org/whl/cpu `
        -r requirements-workers.txt
} else {
    if ($Accelerator -notin @("cuda", "rocm")) {
        throw "ML_ACCELERATOR must be cpu, cuda or rocm"
    }
    foreach ($name in @(
        "ML_TORCH_INDEX_URL",
        "ML_TORCH_VERSION",
        "ML_TORCHVISION_VERSION"
    )) {
        if (-not [Environment]::GetEnvironmentVariable($name)) {
            throw "Set $name for a $Accelerator worker environment"
        }
    }
    & $WorkersPython -m pip install --only-binary=:all: --no-deps `
        -r requirements-workers-core.txt
    & $WorkersPython -m pip install --only-binary=:all: `
        --index-url $env:ML_TORCH_INDEX_URL `
        "torch==$($env:ML_TORCH_VERSION)" `
        "torchvision==$($env:ML_TORCHVISION_VERSION)"
}

& $WorkersPython -m pip check
& $WorkersPython -c @'
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
print(
    "Shared worker environment ready:",
    f"torch={torch.__version__}",
    f"torchvision={torchvision.__version__}",
    f"opencv={cv2.__version__}",
    f"onnxruntime={onnxruntime.__version__}",
    f"diffusers={diffusers.__version__}",
    f"transformers={transformers.__version__}",
)
'@

Write-Host "All model workers installed at $WorkersVenv"
