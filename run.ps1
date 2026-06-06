Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot "venv/Scripts/python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}
& $python -m diffusion_editor.app.main @args
