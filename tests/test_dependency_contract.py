from __future__ import annotations

from pathlib import Path
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _requirements(name: str) -> tuple[str, ...]:
    return tuple(
        line
        for raw in (PROJECT_ROOT / name).read_text(encoding="utf-8").splitlines()
        if (line := raw.strip()) and not line.startswith("#")
    )


def test_runtime_lock_is_the_pyproject_dependency_source():
    pyproject = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert pyproject["project"]["requires-python"] == ">=3.14,<3.15"
    assert pyproject["tool"]["setuptools"]["dynamic"]["dependencies"] == {
        "file": ["requirements-project.txt"]
    }
    assert pyproject["tool"]["setuptools"]["dynamic"]["optional-dependencies"][
        "test"
    ] == {"file": ["requirements-test.txt"]}


def test_main_process_lock_excludes_unavailable_native_ml_stack():
    runtime = _requirements("requirements-runtime.txt")
    project = _requirements("requirements-project.txt")

    assert "numpy==2.5.1" in runtime
    assert "Pillow==12.3.0" in runtime
    assert "PyYAML==6.0.3" in runtime
    assert "PySDL2==0.9.17" in runtime
    forbidden = (
        "diffusers",
        "transformers",
        "safetensors",
        "tokenizers",
        "simple-lama-inpainting",
        "opencv-python",
        "rembg",
    )
    assert not any(
        requirement.lower().startswith(package)
        for requirement in runtime
        for package in forbidden
    )
    assert project == (
        "tcbase",
        "tcgui",
        "termin-display",
        "tgfx",
        *runtime,
    )


def test_developer_requirements_compose_the_two_authoritative_locks():
    assert _requirements("requirements.txt") == (
        "-r requirements-runtime.txt",
        "-r requirements-test.txt",
    )

    installer = (PROJECT_ROOT / "install-deps.sh").read_text(encoding="utf-8")
    assert "--only-binary=:all:" in installer
