#!/usr/bin/env pwsh

[CmdletBinding()]
param(
    [ValidateSet("legacy", "native")]
    [string]$Ui = "legacy",
    [ValidateSet("opengl", "vulkan")]
    [string]$Backend = "opengl",
    [int]$Frames = 3
)

$ErrorActionPreference = "Stop"
if ($Frames -le 0) {
    throw "Frames must be positive"
}

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$env:TERMIN_BACKEND = $Backend
$env:DIFFUSION_EDITOR_SMOKE_FRAMES = [string]$Frames

& (Join-Path $ProjectRoot "run.ps1") --ui $Ui
if ($LASTEXITCODE -ne 0) {
    throw "Windows $Ui/$Backend startup smoke failed with exit code $LASTEXITCODE"
}
Write-Host "Windows production editor smoke OK: ui=$Ui backend=$Backend frames=$Frames"
