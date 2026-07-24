$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$WorkerVenv = if ($env:SEGMENTATION_VENV) {
    $env:SEGMENTATION_VENV
} else {
    ".venv-segmentation"
}
$Bootstrap = if ($env:SEGMENTATION_BOOTSTRAP_PYTHON) {
    $env:SEGMENTATION_BOOTSTRAP_PYTHON
} else {
    "py"
}
$BootstrapArgs = if ($env:SEGMENTATION_BOOTSTRAP_PYTHON) {
    @()
} else {
    @("-3.11")
}
$WorkerPython = Join-Path $WorkerVenv "Scripts/python.exe"

function Assert-SegmentationPython {
    param([string]$Python, [string[]]$PrefixArgs = @())
    & $Python @PrefixArgs -c @'
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
'@
}

Assert-SegmentationPython $Bootstrap $BootstrapArgs
if (Test-Path $WorkerVenv) {
    if (-not (Test-Path $WorkerPython -PathType Leaf)) {
        throw "Existing SEGMENTATION_VENV=$WorkerVenv has no Scripts/python.exe"
    }
    Assert-SegmentationPython $WorkerPython
} else {
    & $Bootstrap @BootstrapArgs -m venv $WorkerVenv
}

& $WorkerPython -m pip install --only-binary=:all: `
    -r requirements-segmentation-worker.txt
& $WorkerPython -m pip check
& $WorkerPython -c @'
import cv2
import onnxruntime
import rembg
print(
    "Segmentation worker ready:",
    f"onnxruntime={onnxruntime.__version__}",
    f"opencv={cv2.__version__}",
)
'@

Write-Host "Segmentation worker environment installed at $WorkerVenv"
