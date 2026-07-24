$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$LamaVenv = if ($env:LAMA_VENV) { $env:LAMA_VENV } else { ".venv-lama" }
$Bootstrap = if ($env:LAMA_BOOTSTRAP_PYTHON) {
    $env:LAMA_BOOTSTRAP_PYTHON
} else {
    "py"
}
$BootstrapArgs = if ($env:LAMA_BOOTSTRAP_PYTHON) { @() } else { @("-3.11") }
$LamaPython = Join-Path $LamaVenv "Scripts/python.exe"

function Assert-LamaPython {
    param([string]$Python, [string[]]$PrefixArgs = @())
    & $Python @PrefixArgs -c @'
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
'@
}

Assert-LamaPython $Bootstrap $BootstrapArgs
if (Test-Path $LamaVenv) {
    if (-not (Test-Path $LamaPython -PathType Leaf)) {
        throw "Existing LAMA_VENV=$LamaVenv has no Scripts/python.exe"
    }
    Assert-LamaPython $LamaPython
} else {
    & $Bootstrap @BootstrapArgs -m venv $LamaVenv
}

& $LamaPython -m pip install --only-binary=:all: `
    -r requirements-lama-worker-build.txt
& $LamaPython -m pip install --only-binary=:all: `
    --no-binary=fire `
    --no-build-isolation `
    --extra-index-url https://download.pytorch.org/whl/cpu `
    -r requirements-lama-worker.txt
& $LamaPython -m pip check
& $LamaPython -c @'
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
'@

Write-Host "LaMa worker environment installed at $LamaVenv"
