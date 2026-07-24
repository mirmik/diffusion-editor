$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$MlVenv = if ($env:ML_VENV) { $env:ML_VENV } else { ".venv-ml" }
$Bootstrap = if ($env:ML_BOOTSTRAP_PYTHON) {
    $env:ML_BOOTSTRAP_PYTHON
} else {
    "python3.11"
}
$Accelerator = if ($env:ML_ACCELERATOR) { $env:ML_ACCELERATOR } else { "cpu" }
$MlPython = Join-Path $MlVenv "Scripts/python.exe"

function Assert-WorkerPython([string]$Python) {
    & $Python -c @'
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
'@
}

Assert-WorkerPython $Bootstrap
if (Test-Path $MlVenv) {
    if (-not (Test-Path $MlPython)) {
        throw "Existing ML_VENV=$MlVenv has no Scripts/python.exe"
    }
    Assert-WorkerPython $MlPython
} else {
    & $Bootstrap -m venv $MlVenv
}

if ($Accelerator -eq "cpu") {
    & $MlPython -m pip install --only-binary=:all: `
        --extra-index-url https://download.pytorch.org/whl/cpu `
        -r requirements-ml-worker.txt
} elseif ($Accelerator -eq "cuda" -or $Accelerator -eq "rocm") {
    if (-not $env:ML_TORCH_INDEX_URL) {
        throw "Set the official ML_TORCH_INDEX_URL"
    }
    if (-not $env:ML_TORCH_VERSION -or -not $env:ML_TORCHVISION_VERSION) {
        throw "Set exact ML_TORCH_VERSION and ML_TORCHVISION_VERSION"
    }
    & $MlPython -m pip install --only-binary=:all: `
        -r requirements-ml-worker-core.txt
    & $MlPython -m pip install --only-binary=:all: `
        --index-url $env:ML_TORCH_INDEX_URL `
        "torch==$env:ML_TORCH_VERSION" `
        "torchvision==$env:ML_TORCHVISION_VERSION"
} else {
    throw "ML_ACCELERATOR must be cpu, cuda or rocm"
}

& $MlPython -m pip check
& $MlPython -c @'
import diffusers
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
'@

Write-Host "ML worker environment installed at $MlVenv"
