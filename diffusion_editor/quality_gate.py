"""Reusable CPython 3.14t and installed-environment quality gates."""

from __future__ import annotations

from email.parser import Parser
from importlib import metadata
from pathlib import Path
import re
import subprocess
import sys
import sysconfig


WORKER_ONLY_DISTRIBUTIONS = (
    "accelerate",
    "diffusers",
    "opencv-python",
    "opencv-python-headless",
    "onnxruntime",
    "rembg",
    "safetensors",
    "simple-lama-inpainting",
    "tokenizers",
    "torch",
    "torchvision",
    "transformers",
)


class QualityGateError(RuntimeError):
    """One actionable compatibility requirement was not satisfied."""


def _gil_enabled() -> bool:
    probe = getattr(sys, "_is_gil_enabled", None)
    return bool(probe()) if probe is not None else True


def verify_runtime_identity(stage: str = "startup") -> str:
    implementation = sys.implementation.name
    version = sys.version_info[:2]
    soabi = str(sysconfig.get_config_var("SOABI") or "")
    py_gil_disabled = sysconfig.get_config_var("Py_GIL_DISABLED")
    errors = []
    if implementation != "cpython":
        errors.append(f"implementation={implementation}, expected cpython")
    if version != (3, 14):
        errors.append(f"version={version[0]}.{version[1]}, expected 3.14")
    if py_gil_disabled != 1:
        errors.append(
            f"Py_GIL_DISABLED={py_gil_disabled!r}, expected 1"
        )
    if re.match(r"^cpython-314t(?:-|$)", soabi) is None:
        errors.append(f"SOABI={soabi!r}, expected cpython-314t-*")
    if _gil_enabled():
        errors.append("sys._is_gil_enabled() is True")
    if errors:
        raise QualityGateError(
            f"CPython 3.14t gate failed at {stage}: " + "; ".join(errors)
        )
    return soabi


def verify_worker_distributions_absent() -> None:
    installed = []
    for name in WORKER_ONLY_DISTRIBUTIONS:
        try:
            installed.append(f"{name}=={metadata.version(name)}")
        except metadata.PackageNotFoundError:
            pass
    if installed:
        raise QualityGateError(
            "worker-only distributions entered the main environment: "
            + ", ".join(installed)
        )


def verify_installed_wheel_abis() -> tuple[str, ...]:
    """Reject regular cp314 native wheels from the cp314t environment."""

    checked = []
    failures = []
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name") or "<unknown>"
        files = tuple(distribution.files or ())
        native = tuple(
            str(path)
            for path in files
            if str(path).lower().endswith((".so", ".pyd", ".dylib"))
        )
        if not native:
            continue
        wheel_text = distribution.read_text("WHEEL")
        if wheel_text is None:
            failures.append(
                f"{name}: native files have no dist-info/WHEEL metadata"
            )
            continue
        tags = tuple(Parser().parsestr(wheel_text).get_all("Tag") or ())
        regular_cp314 = tuple(
            tag
            for tag in tags
            if len(parts := tag.split("-", 2)) == 3
            and parts[1] == "cp314"
        )
        if regular_cp314:
            failures.append(
                f"{name}: regular CPython wheel tag(s) "
                f"{', '.join(regular_cp314)}; cp314t is required"
            )
            continue
        checked.append(
            f"{name}: {', '.join(tags) or '<no tags>'} "
            f"({len(native)} native file(s))"
        )
    if failures:
        raise QualityGateError(
            "native wheel ABI gate failed:\n- " + "\n- ".join(failures)
        )
    return tuple(sorted(checked, key=str.casefold))


def verify_exact_lock(path: Path) -> tuple[str, ...]:
    requirements = []
    for number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("-", "http:", "https:")) or "==" not in line:
            raise QualityGateError(
                f"{path}:{number}: requirement is not an exact distribution pin: "
                f"{line}"
            )
        requirements.append(line)
    if not requirements:
        raise QualityGateError(f"{path} contains no requirements")
    return tuple(requirements)


def verify_pip_check() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        details = (result.stdout + result.stderr).strip()
        raise QualityGateError(f"pip check failed:\n{details}")


def verify_main_environment(project_root: Path) -> tuple[str, ...]:
    verify_runtime_identity("environment gate startup")
    verify_exact_lock(project_root / "requirements-runtime.txt")
    verify_exact_lock(project_root / "requirements-test.txt")
    verify_worker_distributions_absent()
    wheels = verify_installed_wheel_abis()
    verify_pip_check()
    verify_runtime_identity("environment gate completion")
    return wheels
