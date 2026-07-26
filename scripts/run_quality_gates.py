#!/usr/bin/env python3
"""Run the bounded CPython 3.14t migration quality gate suite."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

if runtime_site := os.environ.get("DIFFUSION_EDITOR_QA_SITE_PACKAGES"):
    sys.path.insert(0, runtime_site)

from diffusion_editor.quality_gate import verify_runtime_identity


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _python_script(path: str, *arguments: str) -> list[str]:
    runtime_site = os.environ.get("DIFFUSION_EDITOR_QA_SITE_PACKAGES")
    if not runtime_site:
        return [sys.executable, path, *arguments]
    argv = [path, *arguments]
    code = (
        "import os,runpy,sys;"
        "sys.path.insert(0,os.environ['DIFFUSION_EDITOR_QA_SITE_PACKAGES']);"
        f"sys.argv={argv!r};"
        f"runpy.run_path({path!r},run_name='__main__')"
    )
    return [sys.executable, "-c", code]


def _pytest_command() -> list[str]:
    runtime_site = os.environ.get("DIFFUSION_EDITOR_QA_SITE_PACKAGES")
    if not runtime_site:
        return [sys.executable, "-m", "pytest", "-q"]
    code = (
        "import os,sys;"
        "sys.path.insert(0,os.environ['DIFFUSION_EDITOR_QA_SITE_PACKAGES']);"
        "import pytest;"
        "raise SystemExit(pytest.main(['-q']))"
    )
    return [sys.executable, "-c", code]


def _run(label: str, command: list[str], *, env=None, timeout: float) -> None:
    started = time.monotonic()
    print(f"=== {label} ===", flush=True)
    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"{label} exceeded its bounded timeout of {timeout:.0f}s"
        ) from exc
    if result.returncode:
        raise RuntimeError(
            f"{label} failed with exit code {result.returncode}: "
            + " ".join(command)
        )
    print(f"{label} OK in {time.monotonic() - started:.1f}s", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render-backend",
        action="append",
        choices=("opengl", "vulkan"),
        default=[],
        help="run the Termin render smoke for this backend (repeatable)",
    )
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--skip-workers", action="store_true")
    parser.add_argument(
        "--skip-resolver",
        action="store_true",
        help="skip the networked binary-wheel resolver check",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    verify_runtime_identity("quality runner startup")
    _run(
        "main-process import/ABI/resolver gate",
        _python_script("scripts/probe_main_process_dependencies.py"),
        timeout=args.timeout,
    )
    if not args.skip_resolver:
        with tempfile.TemporaryDirectory(
            prefix="diffusion-editor-wheel-gate-"
        ) as directory:
            _run(
                "reviewed binary-wheel resolver",
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "download",
                    "--only-binary=:all:",
                    "--dest",
                    directory,
                    "-r",
                    "requirements-runtime.txt",
                    "-r",
                    "requirements-test.txt",
                ],
                timeout=args.timeout,
            )
    _run(
        "unit and concurrency stress suite",
        _pytest_command(),
        timeout=args.timeout,
    )
    if not args.skip_workers:
        for label, command in (
            (
                "LaMa IPC smoke",
                _python_script(
                    "scripts/smoke_lama_worker.py", "--backend", "identity"
                ),
            ),
            (
                "segmentation IPC smoke",
                _python_script(
                    "scripts/smoke_segmentation_worker.py",
                    "--backend",
                    "threshold",
                ),
            ),
            (
                "Diffusers/Transformers IPC smoke",
                _python_script("scripts/smoke_ml_worker.py"),
            ),
        ):
            _run(label, command, timeout=args.timeout)

    for backend in args.render_backend:
        with tempfile.TemporaryDirectory(
            prefix=f"diffusion-editor-{backend}-shader-cache-"
        ) as shader_cache:
            environment = os.environ.copy()
            environment["TERMIN_BACKEND"] = backend
            environment["TERMIN_SDK_SHADER_CACHE_ROOT"] = shader_cache
            _run(
                f"Termin {backend} render smoke",
                _python_script(
                    "scripts/smoke_termin_runtime.py",
                    "--frames",
                    str(args.frames),
                ),
                env=environment,
                timeout=args.timeout,
            )
            _run(
                f"production editor {backend} startup smoke",
                _python_script("scripts/smoke_editor_startup.py"),
                env=environment,
                timeout=args.timeout,
            )
            if backend == "opengl":
                native_environment = environment.copy()
                native_environment["SDL_VIDEODRIVER"] = "offscreen"
                _run(
                    "native editor opengl windowed smoke",
                    _python_script(
                        "scripts/smoke_native_root.py",
                        "--windowed",
                        "--frames",
                        str(args.frames),
                    ),
                    env=native_environment,
                    timeout=args.timeout,
                )
    verify_runtime_identity("quality runner completion")
    print("All requested CPython 3.14t quality gates passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
