from __future__ import annotations

import os
import subprocess
import sys

import pytest

from diffusion_editor.app import main as app_main
from diffusion_editor.app.native_main import _open_path


def test_cli_dispatches_optional_path_to_native_host(monkeypatch):
    calls = []
    verified = []
    monkeypatch.setattr(
        app_main,
        "verify_application_environment",
        lambda: verified.append("runtime"),
    )
    monkeypatch.setattr(
        app_main,
        "_run_native",
        lambda path: calls.append(("native", path)) or 12,
    )

    assert app_main.main([]) == 12
    assert app_main.main(["image.png"]) == 12
    assert app_main.main(["project.deproj"]) == 12
    assert calls == [
        ("native", None),
        ("native", "image.png"),
        ("native", "project.deproj"),
    ]
    assert verified == ["runtime", "runtime", "runtime"]


def test_cli_rejects_removed_ui_selector():
    with pytest.raises(SystemExit) as exc_info:
        app_main.main(["--ui", "legacy"])

    assert exc_info.value.code == 2


def test_cli_verifies_runtime_before_loading_selected_host(monkeypatch):
    calls = []
    monkeypatch.setattr(
        app_main,
        "verify_application_environment",
        lambda: calls.append("runtime"),
    )
    monkeypatch.setattr(
        app_main,
        "_run_native",
        lambda _path: calls.append("native") or 0,
    )

    assert app_main.main([]) == 0
    assert calls == ["runtime", "native"]


def test_cli_and_native_host_import_without_tcgui():
    code = """
import sys
import diffusion_editor.app.main
import diffusion_editor.app.native_main
assert not any(name == 'tcgui' or name.startswith('tcgui.') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.getcwd(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_native_path_uses_application_dialog_workflow():
    calls = []

    class Dialogs:
        def open_project_path(self, path):
            calls.append(("project", path))

        def import_image_path(self, path):
            calls.append(("image", path))

    root = type("Root", (), {"dialog_coordinator": Dialogs()})()
    _open_path(root, "scene.deproj")
    _open_path(root, "source.PNG")

    assert calls == [
        ("project", "scene.deproj"),
        ("image", "source.PNG"),
    ]
