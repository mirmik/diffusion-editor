from pathlib import Path

import pytest

from diffusion_editor.generation.ply_sequence import (
    PlySequence,
    discover_ply_files,
)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ply\n", encoding="ascii")
    return path.resolve()


def test_single_ply_is_a_one_item_sequence(tmp_path: Path):
    path = _touch(tmp_path / "model.ply")
    assert discover_ply_files(path) == (path,)


def test_directory_prefers_mesh_artifacts_and_naturally_sorts(tmp_path: Path):
    tenth = _touch(tmp_path / "view-10-posed-makehuman.ply")
    second = _touch(tmp_path / "view-2-posed-makehuman.ply")
    _touch(tmp_path / "view-2-joints.ply")

    assert discover_ply_files(tmp_path) == (second, tenth)


def test_explicit_pattern_disables_mesh_preference(tmp_path: Path):
    mesh = _touch(tmp_path / "view-posed-makehuman.ply")
    joints = _touch(tmp_path / "view-joints.ply")

    assert discover_ply_files(tmp_path, pattern="*.ply") == (joints, mesh)


def test_recursive_discovery_and_empty_directory_errors(tmp_path: Path):
    nested = _touch(tmp_path / "nested" / "model.ply")
    with pytest.raises(ValueError, match="no PLY files"):
        discover_ply_files(tmp_path)
    assert discover_ply_files(tmp_path, recursive=True) == (nested,)


def test_sequence_wraps_and_supports_home_end(tmp_path: Path):
    paths = tuple(_touch(tmp_path / f"{index}.ply") for index in range(3))
    sequence = PlySequence(paths)

    assert sequence.step(-1) == paths[2]
    assert sequence.step(1) == paths[0]
    assert sequence.end() == paths[2]
    assert sequence.home() == paths[0]
