#!/usr/bin/env pwsh

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot
$VenvRoot = if ($env:VENV) { $env:VENV } else { "venv" }
$VenvPath = if ([IO.Path]::IsPathRooted($VenvRoot)) {
    $VenvRoot
} else {
    Join-Path $ProjectRoot $VenvRoot
}
$Python = Join-Path $VenvPath "Scripts\python.exe"

if (-not (Test-Path $Python -PathType Leaf)) {
    throw "Python environment not found: $Python. Run .\install-deps.ps1 first."
}

$ResolveArguments = @("-m", "diffusion_editor.sdk_runtime", "resolve")
if ($env:TERMIN_SDK) {
    $ResolveArguments += @("--sdk", $env:TERMIN_SDK)
}
$ResolvedSdk = & $Python @ResolveArguments
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
$env:TERMIN_SDK = $ResolvedSdk.Trim()
$sdkPathEntries = @(
    (Join-Path $env:TERMIN_SDK "bin"),
    (Join-Path $env:TERMIN_SDK "lib")
) | Where-Object { Test-Path $_ -PathType Container }
if ($sdkPathEntries.Count -gt 0) {
    $env:PATH = (
        ($sdkPathEntries -join [IO.Path]::PathSeparator) +
        [IO.Path]::PathSeparator +
        $env:PATH
    )
}

& $Python -m diffusion_editor.sdk_runtime verify-installed `
    --sdk $env:TERMIN_SDK --imports
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
& $Python scripts\run_quality_gates.py @args
exit $LASTEXITCODE
