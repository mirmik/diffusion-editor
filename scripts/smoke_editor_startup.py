#!/usr/bin/env python3
"""Launch the production editor host for a bounded startup/render smoke."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FAILURE_MARKERS = (
    "traceback (most recent call last)",
    "shader is unavailable",
    "missing or dev compile failed",
    "failed to create solid shader",
    "failed to create texture shader",
    "failed to create bitmap shader",
    "failed to create sdf shader",
    "skipping batch",
    "skipping text draw",
    "fatal python error",
    "segmentation fault",
)


def production_run_command(
    *,
    platform_name: str = os.name,
    find_executable=shutil.which,
) -> list[str]:
    if platform_name == "nt":
        powershell = (
            find_executable("pwsh")
            or find_executable("powershell")
            or find_executable("powershell.exe")
        )
        if powershell is None:
            raise RuntimeError(
                "PowerShell is required for the Windows startup smoke"
            )
        return [
            powershell,
            "-NoProfile",
            "-File",
            str(PROJECT_ROOT / "run.ps1"),
        ]
    return [str(PROJECT_ROOT / "run.sh")]


def main() -> int:
    argparse.ArgumentParser().parse_args()
    environment = os.environ.copy()
    environment["DIFFUSION_EDITOR_SMOKE_FRAMES"] = "3"
    with tempfile.TemporaryDirectory(
        prefix="diffusion-editor-startup-smoke-"
    ) as cache_root:
        environment["TERMIN_SDK_SHADER_CACHE_ROOT"] = cache_root
        try:
            result = subprocess.run(
                production_run_command(),
                cwd=PROJECT_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "Production editor startup exceeded 30s"
            ) from exc

    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode:
        raise RuntimeError(
            f"Production editor exited with code {result.returncode}"
        )

    combined = f"{result.stdout}\n{result.stderr}".lower()
    failures = [marker for marker in FAILURE_MARKERS if marker in combined]
    if failures:
        raise RuntimeError(
            "Production editor emitted failure diagnostics: "
            + ", ".join(failures)
        )
    if "[main] window created" not in combined:
        raise RuntimeError("Production editor did not create a window")

    print("Production native editor startup/render smoke OK: 3 frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
