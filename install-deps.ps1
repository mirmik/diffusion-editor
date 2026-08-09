#!/usr/bin/env pwsh
# Install Diffusion Editor into a project-local CPython 3.14t environment.

[CmdletBinding()]
param(
    [string]$Sdk = "",
    [string]$Venv = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList
    )
    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw (
            "Command failed with exit code $LASTEXITCODE`: " +
            "$FilePath $($ArgumentList -join ' ')"
        )
    }
}

function Resolve-SdkRoot {
    param([string]$ExplicitSdk)

    $candidates = [System.Collections.Generic.List[string]]::new()
    if ($ExplicitSdk) {
        $candidates.Add($ExplicitSdk)
    } elseif ($env:TERMIN_SDK) {
        $candidates.Add($env:TERMIN_SDK)
    } else {
        $stateFile = Join-Path $ProjectRoot ".termin-sdk"
        if (Test-Path $stateFile -PathType Leaf) {
            $saved = (Get-Content $stateFile -Raw).Trim()
            if ($saved) {
                $candidates.Add($saved)
            }
        }
        if ($env:LOCALAPPDATA) {
            $candidates.Add((Join-Path $env:LOCALAPPDATA "termin-sdk"))
        }
        if ($env:ProgramFiles) {
            $candidates.Add((Join-Path $env:ProgramFiles "Termin"))
        }
    }

    foreach ($candidate in $candidates) {
        if (-not $candidate) {
            continue
        }
        $root = if ([IO.Path]::IsPathRooted($candidate)) {
            [IO.Path]::GetFullPath($candidate)
        } else {
            [IO.Path]::GetFullPath((Join-Path $ProjectRoot $candidate))
        }
        if (
            (Test-Path (Join-Path $root "lib") -PathType Container) -and
            (Test-Path (Join-Path $root "wheels") -PathType Container) -and
            (Test-Path (Join-Path $root "python-runtime-manifest.json") -PathType Leaf)
        ) {
            return $root
        }
    }
    throw (
        "Termin SDK was not found. Pass -Sdk, set TERMIN_SDK, or install " +
        "the canonical Windows SDK at %LOCALAPPDATA%\termin-sdk."
    )
}

$SdkRoot = Resolve-SdkRoot $Sdk
$env:TERMIN_SDK = $SdkRoot
$sdkPathEntries = @(
    (Join-Path $SdkRoot "bin"),
    (Join-Path $SdkRoot "lib")
) | Where-Object { Test-Path $_ -PathType Container }
if ($sdkPathEntries.Count -gt 0) {
    $env:PATH = (
        ($sdkPathEntries -join [IO.Path]::PathSeparator) +
        [IO.Path]::PathSeparator +
        $env:PATH
    )
}

$SdkPython = Join-Path $SdkRoot "bin\termin_python.exe"
if (-not (Test-Path $SdkPython -PathType Leaf)) {
    throw "Canonical SDK Python launcher is missing: $SdkPython"
}

Invoke-Checked -FilePath $SdkPython -ArgumentList @(
    "-m", "diffusion_editor.sdk_runtime",
    "verify-python-executable", "--sdk", $SdkRoot, "--python", $SdkPython
)

$VenvValue = if ($Venv) {
    $Venv
} elseif ($env:VENV) {
    $env:VENV
} else {
    "venv"
}
$VenvRoot = if ([IO.Path]::IsPathRooted($VenvValue)) {
    [IO.Path]::GetFullPath($VenvValue)
} else {
    [IO.Path]::GetFullPath((Join-Path $ProjectRoot $VenvValue))
}
$VenvPython = Join-Path $VenvRoot "Scripts\python.exe"

if (Test-Path $VenvRoot) {
    if (-not (Test-Path $VenvPython -PathType Leaf)) {
        throw "Existing VENV=$VenvRoot has no Scripts\python.exe; it was not modified."
    }
    Invoke-Checked -FilePath $SdkPython -ArgumentList @(
        "-m", "diffusion_editor.sdk_runtime",
        "verify-python-executable", "--sdk", $SdkRoot, "--python", $VenvPython
    )
} else {
    Write-Host "Creating CPython 3.14t environment: $VenvRoot"
    Invoke-Checked -FilePath $SdkPython -ArgumentList @(
        "-m", "venv", $VenvRoot
    )
}

$TerminRequirements = @(
    & $SdkPython -m diffusion_editor.sdk_runtime requirements --sdk $SdkRoot
)
if ($LASTEXITCODE -ne 0 -or $TerminRequirements.Count -eq 0) {
    throw "Could not resolve the exact Termin wheel closure from $SdkRoot"
}
$Wheelhouse = Join-Path $SdkRoot "wheels"

Invoke-Checked -FilePath $VenvPython -ArgumentList @(
    "-m", "pip", "install", "--only-binary=:all:",
    "--find-links", $Wheelhouse,
    "-r", (Join-Path $ProjectRoot "requirements.txt")
)
$TerminInstallArguments = @(
    "-m", "pip", "install", "--force-reinstall", "--no-index",
    "--no-deps", "--find-links", $Wheelhouse
) + $TerminRequirements
Invoke-Checked -FilePath $VenvPython -ArgumentList @(
    "-m", "pip", "uninstall", "--yes", "tcgui"
)
Invoke-Checked -FilePath $VenvPython -ArgumentList $TerminInstallArguments
Invoke-Checked -FilePath $VenvPython -ArgumentList @(
    "-m", "pip", "install", "--no-build-isolation", "--no-deps",
    "-e", $ProjectRoot
)

Invoke-Checked -FilePath $VenvPython -ArgumentList @(
    "-m", "diffusion_editor.sdk_runtime", "verify-installed",
    "--sdk", $SdkRoot, "--imports"
)
Invoke-Checked -FilePath $VenvPython -ArgumentList @(
    (Join-Path $ProjectRoot "scripts\probe_main_process_dependencies.py")
)
Invoke-Checked -FilePath $VenvPython -ArgumentList @("-m", "pip", "check")
Invoke-Checked -FilePath $VenvPython -ArgumentList @(
    "-m", "diffusion_editor.sdk_runtime", "write-state", "--sdk", $SdkRoot
)

Write-Host "Diffusion Editor CPython 3.14t environment is ready: $VenvRoot"
