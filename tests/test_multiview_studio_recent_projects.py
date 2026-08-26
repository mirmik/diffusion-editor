from __future__ import annotations

import json
from pathlib import Path

from diffusion_editor.multiview_studio.controller import MultiviewStudioController
from diffusion_editor.multiview_studio.model import MultiviewProject
from diffusion_editor.multiview_studio.native_app import (
    NativeMultiviewStudioApplication,
)
from diffusion_editor.multiview_studio.recent_projects import RecentProjectsStore


def _project(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")
    return path.resolve()


def test_recent_projects_are_mru_deduplicated_and_limited(tmp_path: Path):
    store = RecentProjectsStore(tmp_path / "recent.json", limit=3)
    first = _project(tmp_path / "one.mvstudio.json")
    second = _project(tmp_path / "two.mvstudio.json")
    third = _project(tmp_path / "three.mvstudio.json")
    fourth = _project(tmp_path / "four.mvstudio.json")

    store.record(first)
    store.record(second)
    store.record(third)
    assert store.record(first) == (first, third, second)
    assert store.record(fourth) == (fourth, first, third)

    restored = RecentProjectsStore(store.path, limit=3)
    assert restored.load() == (fourth, first, third)


def test_recent_projects_prune_missing_and_invalid_entries(tmp_path: Path):
    settings_path = tmp_path / "recent.json"
    available = _project(tmp_path / "available.mvstudio.json")
    missing = (tmp_path / "missing.mvstudio.json").resolve()
    settings_path.write_text(
        json.dumps(
            {
                "version": 1,
                "projects": [str(missing), str(available), str(available)],
            }
        ),
        encoding="utf-8",
    )
    store = RecentProjectsStore(settings_path)

    assert store.load() == (available,)
    assert json.loads(settings_path.read_text(encoding="utf-8"))["projects"] == [
        str(available)
    ]

    settings_path.write_text("not json", encoding="utf-8")
    assert store.load() == ()


def test_recent_projects_can_be_removed_and_cleared(tmp_path: Path):
    store = RecentProjectsStore(tmp_path / "recent.json")
    first = _project(tmp_path / "one.mvstudio.json")
    second = _project(tmp_path / "two.mvstudio.json")
    store.record(first)
    store.record(second)

    assert store.remove(second) == (first,)
    store.clear()
    assert store.load() == ()


def test_successful_project_open_is_remembered(tmp_path: Path):
    manifest = MultiviewProject().save(tmp_path / "opened.mvstudio.json")
    store = RecentProjectsStore(tmp_path / "recent.json")

    class View:
        recent = ()

        def set_recent_projects(self, projects):
            self.recent = projects

    application = object.__new__(NativeMultiviewStudioApplication)
    application.controller = MultiviewStudioController()
    application._recent_store = store
    application.view = View()
    application._last_dir = ""

    application._open_project_path(str(manifest))

    assert application.controller.project_path == manifest
    assert application.view.recent == (manifest,)
    assert application._last_dir == str(tmp_path)
