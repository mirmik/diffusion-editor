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


def test_model_workers_share_one_exact_binary_only_lock_and_installer():
    runtime = tuple(item.lower() for item in _requirements(
        "requirements-runtime.txt"
    ))
    worker = tuple(item.lower() for item in _requirements(
        "requirements-workers.txt"
    ))
    core = tuple(item.lower() for item in _requirements(
        "requirements-workers-core.txt"
    ))

    expected = (
        "rembg==2.0.77",
        "onnxruntime==1.27.0",
        "opencv-python-headless==5.0.0.93",
        "torch==2.13.0+cpu",
        "torchvision==0.28.0+cpu",
        "diffusers==0.39.0",
        "transformers==5.14.1",
        "accelerate==1.14.0",
        "safetensors==0.8.0",
        "tokenizers==0.22.2",
    )
    for requirement in expected:
        assert requirement in worker
    assert all("==" in item for item in worker)
    assert all("==" in item for item in core)
    assert not any(
        item.startswith(("torch==", "torchvision=="))
        for item in core
    )
    assert not any("simple-lama-inpainting" in item for item in worker)
    assert all(
        not any(item.startswith(f"{package}==") for item in runtime)
        for package in (
            "torch",
            "torchvision",
            "diffusers",
            "transformers",
            "accelerate",
            "safetensors",
            "tokenizers",
            "rembg",
            "opencv-python",
        )
    )

    for name in ("setup-workers.sh", "setup-workers.ps1"):
        installer = (PROJECT_ROOT / name).read_text(encoding="utf-8")
        assert "requirements-workers.txt" in installer
        assert "requirements-workers-core.txt" in installer
        assert "--only-binary=:all:" in installer
        assert "pip check" in installer
        assert "download.pytorch.org/whl/cpu" in installer
        assert "ML_TORCH_INDEX_URL" in installer

    shell_installer = (PROJECT_ROOT / "setup-workers.sh").read_text(
        encoding="utf-8"
    )
    assert "source scripts/resolve_worker_python.sh" in shell_installer

    for worker_name in ("lama", "segmentation", "ml"):
        for suffix in ("sh", "ps1"):
            wrapper = (
                PROJECT_ROOT / f"setup-{worker_name}-worker.{suffix}"
            ).read_text(encoding="utf-8")
            assert "setup-workers" in wrapper
