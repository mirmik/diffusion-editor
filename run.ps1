Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot "venv/Scripts/python.exe"
if (-not (Test-Path $python)) {
    Write-Error "Python environment not found: $python. Run install-deps.sh first."
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

& $python -m diffusion_editor.sdk_runtime verify-installed `
    --sdk $env:TERMIN_SDK --imports
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
& $python -m diffusion_editor.app.main @args
exit $LASTEXITCODE
