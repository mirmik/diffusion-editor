from __future__ import annotations

import os
from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESOLVER = PROJECT_ROOT / "scripts" / "resolve_worker_python.sh"


def _executable(path: Path, contents: str) -> Path:
    path.write_text(contents, encoding="utf-8")
    path.chmod(0o755)
    return path


def _resolve(path: Path, configured: str = "") -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PATH"] = str(path)
    return subprocess.run(
        [
            "/bin/bash",
            "-c",
            'source "$1"; resolve_worker_python "$2"',
            "resolver-test",
            str(RESOLVER),
            configured,
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_resolver_uses_working_python311_from_path(tmp_path: Path):
    candidate = _executable(tmp_path / "python3.11", "#!/bin/sh\nexit 0\n")

    result = _resolve(tmp_path)

    assert result.returncode == 0
    assert result.stdout.strip() == str(candidate)


def test_resolver_uses_installed_pyenv_311_when_shim_is_inactive(
    tmp_path: Path,
):
    _executable(tmp_path / "python3.11", "#!/bin/sh\nexit 127\n")
    candidate = _executable(tmp_path / "pyenv-python3.11", "#!/bin/sh\nexit 0\n")
    _executable(
        tmp_path / "pyenv",
        (
            "#!/bin/sh\n"
            '[ "$PYENV_VERSION" = "3.11" ] || exit 3\n'
            '[ "$1 $2" = "which python3.11" ] || exit 4\n'
            f"printf '%s\\n' '{candidate}'\n"
        ),
    )

    result = _resolve(tmp_path)

    assert result.returncode == 0
    assert result.stdout.strip() == str(candidate)


def test_resolver_rejects_invalid_explicit_override(tmp_path: Path):
    invalid = _executable(tmp_path / "not-python311", "#!/bin/sh\nexit 1\n")

    result = _resolve(tmp_path, str(invalid))

    assert result.returncode == 2
    assert "not regular CPython 3.11" in result.stderr
