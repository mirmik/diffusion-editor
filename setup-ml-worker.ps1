# Compatibility entry point; all model workers now share one environment.
& (Join-Path $PSScriptRoot "setup-workers.ps1") @args
exit $LASTEXITCODE
