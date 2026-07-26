$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$venvRoot = if ($env:VENV) { $env:VENV } else { "venv" }
$venvPath = if ([IO.Path]::IsPathRooted($venvRoot)) {
    $venvRoot
} else {
    Join-Path $PSScriptRoot $venvRoot
}
$python = Join-Path $venvPath "Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "Python environment not found: $python. Run .\install-deps.ps1 first."
    exit 1
}

$resolveArgs = @("-m", "diffusion_editor.sdk_runtime", "resolve")
if ($env:TERMIN_SDK) {
    $resolveArgs += @("--sdk", $env:TERMIN_SDK)
}
$resolvedSdk = & $python @resolveArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
$env:TERMIN_SDK = $resolvedSdk.Trim()
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

& $python -m diffusion_editor.sdk_runtime verify-installed `
    --sdk $env:TERMIN_SDK --imports
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
& $python -m diffusion_editor.app.main @args
exit $LASTEXITCODE
