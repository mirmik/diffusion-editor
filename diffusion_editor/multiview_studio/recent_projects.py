"""Persistent most-recently-used project list for Multiview Studio."""

from __future__ import annotations

import json
import os
from pathlib import Path


_DEFAULT_LIMIT = 10
_SETTINGS_VERSION = 1


def default_recent_projects_path() -> Path:
    config_home = os.environ.get("XDG_CONFIG_HOME")
    root = Path(config_home).expanduser() if config_home else Path.home() / ".config"
    return root / "diffusion-editor" / "multiview-studio-recent.json"


class RecentProjectsStore:
    def __init__(
        self,
        path: str | Path | None = None,
        *,
        limit: int = _DEFAULT_LIMIT,
    ) -> None:
        if limit <= 0:
            raise ValueError("recent project limit must be positive")
        self.path = (
            Path(path).expanduser()
            if path is not None
            else default_recent_projects_path()
        )
        self.limit = int(limit)

    def load(self) -> tuple[Path, ...]:
        stored = self._read_paths()
        available: list[Path] = []
        seen: set[Path] = set()
        for item in stored:
            path = Path(item).expanduser().resolve()
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            available.append(path)
            if len(available) >= self.limit:
                break
        result = tuple(available)
        if tuple(stored) and tuple(str(path) for path in result) != tuple(stored):
            self._write(result)
        return result

    def record(self, project_path: str | Path) -> tuple[Path, ...]:
        path = Path(project_path).expanduser().resolve()
        current = [item for item in self.load() if item != path]
        if path.is_file():
            current.insert(0, path)
        result = tuple(current[: self.limit])
        self._write(result)
        return result

    def remove(self, project_path: str | Path) -> tuple[Path, ...]:
        path = Path(project_path).expanduser().resolve()
        result = tuple(item for item in self.load() if item != path)
        self._write(result)
        return result

    def clear(self) -> None:
        self._write(())

    def _read_paths(self) -> tuple[str, ...]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return ()
        if not isinstance(payload, dict) or payload.get("version") != _SETTINGS_VERSION:
            return ()
        projects = payload.get("projects")
        if not isinstance(projects, list):
            return ()
        return tuple(item for item in projects if isinstance(item, str) and item)

    def _write(self, projects: tuple[Path, ...]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _SETTINGS_VERSION,
            "projects": [str(path) for path in projects],
        }
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)
