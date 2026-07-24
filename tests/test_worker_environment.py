from __future__ import annotations

from pathlib import Path

from diffusion_editor.workers import lama_process, ml_process, segmentation_process


def test_all_model_workers_default_to_the_shared_environment(monkeypatch):
    for name in (
        "DIFFUSION_EDITOR_LAMA_PYTHON",
        "DIFFUSION_EDITOR_SEGMENTATION_PYTHON",
        "DIFFUSION_EDITOR_ML_PYTHON",
    ):
        monkeypatch.delenv(name, raising=False)

    paths = {
        lama_process.default_worker_python(),
        segmentation_process.default_worker_python(),
        ml_process.default_worker_python(),
    }

    assert len(paths) == 1
    assert next(iter(paths)).parts[-3:] == (
        ".venv-workers",
        "bin",
        "python",
    )


def test_worker_python_overrides_preserve_virtualenv_symlinks(
    monkeypatch,
    tmp_path: Path,
):
    virtualenv_python = tmp_path / "venv" / "bin" / "python"
    virtualenv_python.parent.mkdir(parents=True)
    virtualenv_python.symlink_to("/usr/bin/python3")
    variables_and_factories = (
        ("DIFFUSION_EDITOR_LAMA_PYTHON", lama_process.default_worker_python),
        (
            "DIFFUSION_EDITOR_SEGMENTATION_PYTHON",
            segmentation_process.default_worker_python,
        ),
        ("DIFFUSION_EDITOR_ML_PYTHON", ml_process.default_worker_python),
    )

    for variable, factory in variables_and_factories:
        monkeypatch.setenv(variable, str(virtualenv_python))
        assert factory() == virtualenv_python
