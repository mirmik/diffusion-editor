"""Discover and navigate PLY files for the native reconstruction viewer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


_NUMBER = re.compile(r"(\d+)")
_MESH_HINTS = ("mesh", "makehuman", "posed", "fitted")
_POINT_HINTS = ("joint", "point", "cloud", "structure", "sparse")


def _natural_key(path: Path) -> tuple:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in _NUMBER.split(path.name)
    )


def discover_ply_files(
    source: Path,
    *,
    pattern: str | None = None,
    recursive: bool = False,
) -> tuple[Path, ...]:
    """Resolve one PLY or a naturally sorted directory sequence.

    With the default pattern, a mixed artifact directory prefers mesh-like PLY
    files over joint/point-cloud companions.  An explicit pattern disables the
    heuristic and returns every match.
    """

    source = Path(source).expanduser().resolve()
    if source.is_file():
        if source.suffix.casefold() != ".ply":
            raise ValueError(f"not a PLY file: {source}")
        return (source,)
    if not source.is_dir():
        raise ValueError(f"path does not exist: {source}")
    selected_pattern = pattern or "*.ply"
    iterator = (
        source.rglob(selected_pattern)
        if recursive else source.glob(selected_pattern)
    )
    candidates = sorted(
        (path.resolve() for path in iterator if path.is_file()),
        key=_natural_key,
    )
    if pattern is None:
        mesh_like = [
            path for path in candidates
            if any(hint in path.stem.casefold() for hint in _MESH_HINTS)
            and not any(hint in path.stem.casefold() for hint in _POINT_HINTS)
        ]
        if mesh_like:
            candidates = mesh_like
    if not candidates:
        scope = "recursively" if recursive else ""
        raise ValueError(
            f"no PLY files matching {selected_pattern!r} {scope} in {source}"
        )
    return tuple(candidates)


@dataclass
class PlySequence:
    paths: tuple[Path, ...]
    index: int = 0

    def __post_init__(self) -> None:
        self.paths = tuple(self.paths)
        if not self.paths:
            raise ValueError("PLY sequence cannot be empty")
        if not 0 <= self.index < len(self.paths):
            raise ValueError("PLY sequence index is out of range")

    @property
    def current(self) -> Path:
        return self.paths[self.index]

    def step(self, delta: int) -> Path:
        self.index = (self.index + int(delta)) % len(self.paths)
        return self.current

    def home(self) -> Path:
        self.index = 0
        return self.current

    def end(self) -> Path:
        self.index = len(self.paths) - 1
        return self.current
