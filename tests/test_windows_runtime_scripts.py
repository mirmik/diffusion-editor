from pathlib import Path

import pytest

from scripts.smoke_editor_startup import production_run_command


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_windows_startup_smoke_routes_through_powershell_run_contract():
    command = production_run_command(
        "native",
        platform_name="nt",
        find_executable=lambda name: (
            r"C:\Program Files\PowerShell\7\pwsh.exe"
            if name == "pwsh"
            else None
        ),
    )

    assert command[:3] == [
        r"C:\Program Files\PowerShell\7\pwsh.exe",
        "-NoProfile",
        "-File",
    ]
    assert command[-2:] == ["--ui", "native"]
    assert Path(command[3]).name == "run.ps1"


def test_windows_startup_smoke_requires_powershell():
    with pytest.raises(RuntimeError, match="PowerShell is required"):
        production_run_command(
            "legacy",
            platform_name="nt",
            find_executable=lambda _name: None,
        )


def test_windows_installer_enforces_canonical_cp314t_binary_contract():
    installer = _text("install-deps.ps1")

    assert r"bin\termin_python.exe" in installer
    assert "verify-python-executable" in installer
    assert installer.index("verify-python-executable") < installer.index(
        '"-m", "venv"'
    )
    assert "--only-binary=:all:" in installer
    assert '"--force-reinstall", "--no-index"' in installer
    assert "diffusion_editor.sdk_runtime requirements" in installer
    assert "verify-installed" in installer
    assert "probe_main_process_dependencies.py" in installer
    assert '@("-m", "pip", "check")' in installer
    assert "write-state" in installer


def test_windows_run_and_quality_scripts_restore_sdk_dll_paths():
    for relative_path in ("run.ps1", "run-quality-gates.ps1"):
        script = _text(relative_path)
        assert 'Join-Path $env:TERMIN_SDK "bin"' in script
        assert 'Join-Path $env:TERMIN_SDK "lib"' in script
        assert "[IO.Path]::PathSeparator" in script
        assert "verify-installed" in script


def test_windows_ci_is_explicitly_gated_by_published_sdk_asset():
    workflow = _text(".github/workflows/ci.yml")

    assert "windows_sdk_asset:" in workflow
    assert "github.event.client_payload.windows_asset" in workflow
    assert "runs-on: windows-latest" in workflow
    assert r".\install-deps.ps1" in workflow
    assert r".\run-quality-gates.ps1 --skip-resolver --skip-workers" in workflow
    assert r".\scripts\smoke_windows_runtime.ps1" in workflow


def test_linux_ci_uses_sdk_owned_cp314t_environment():
    workflow = _text(".github/workflows/ci.yml")

    assert "termin-sdk-linux-x86_64-py314t-latest-ci.tar.zst" in workflow
    assert "run: ./install-deps.sh" in workflow
    assert "run: ./venv/bin/python -m pytest -q" in workflow
