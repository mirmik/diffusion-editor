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


def test_lama_worker_has_a_separate_exact_lock_and_installer():
    runtime = tuple(item.lower() for item in _requirements(
        "requirements-runtime.txt"
    ))
    worker = tuple(item.lower() for item in _requirements(
        "requirements-lama-worker.txt"
    ))
    build = tuple(item.lower() for item in _requirements(
        "requirements-lama-worker-build.txt"
    ))

    assert "simple-lama-inpainting==0.1.2" in worker
    assert "opencv-python==4.11.0.86" in worker
    assert "torch==2.7.1+cpu" in worker
    assert "torchvision==0.22.1+cpu" in worker
    assert all("simple-lama" not in item for item in runtime)
    assert all("opencv-python" not in item for item in runtime)
    assert build == (
        "pip==26.1.2",
        "setuptools==83.0.0",
        "wheel==0.47.0",
        "packaging==26.2",
    )
    assert all("==" in item for item in worker)

    shell_installer = (
        PROJECT_ROOT / "setup-lama-worker.sh"
    ).read_text(encoding="utf-8")
    powershell_installer = (
        PROJECT_ROOT / "setup-lama-worker.ps1"
    ).read_text(encoding="utf-8")
    for installer in (shell_installer, powershell_installer):
        assert "requirements-lama-worker.txt" in installer
        assert "requirements-lama-worker-build.txt" in installer
        assert "--only-binary=:all:" in installer
        assert "--no-binary=fire" in installer
        assert "--no-build-isolation" in installer
        assert "download.pytorch.org/whl/cpu" in installer


def test_segmentation_worker_has_a_separate_binary_only_lock():
    runtime = tuple(item.lower() for item in _requirements(
        "requirements-runtime.txt"
    ))
    worker = tuple(item.lower() for item in _requirements(
        "requirements-segmentation-worker.txt"
    ))

    assert "rembg==2.0.77" in worker
    assert "onnxruntime==1.27.0" in worker
    assert "opencv-python-headless==5.0.0.93" in worker
    assert all("==" in item for item in worker)
    assert all("rembg" not in item for item in runtime)
    assert all("opencv-python" not in item for item in runtime)

    shell_installer = (
        PROJECT_ROOT / "setup-segmentation-worker.sh"
    ).read_text(encoding="utf-8")
    powershell_installer = (
        PROJECT_ROOT / "setup-segmentation-worker.ps1"
    ).read_text(encoding="utf-8")
    for installer in (shell_installer, powershell_installer):
        assert "requirements-segmentation-worker.txt" in installer
        assert "--only-binary=:all:" in installer
        assert "--no-binary" not in installer
