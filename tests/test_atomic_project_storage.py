from __future__ import annotations

import io
import json
import os
from pathlib import Path
import zipfile

import numpy as np
import pytest

from diffusion_editor.document.layer_stack import LayerStack
from diffusion_editor.document.tool import LamaTool


def _stack() -> LayerStack:
    stack = LayerStack(tile_size=8)
    image = np.zeros((10, 12, 4), dtype=np.uint8)
    image[2:6, 3:8] = (20, 40, 60, 128)
    stack.init_from_image(image)
    stack.selection.data[1:4, 2:5] = 1.0
    return stack


def _temporary_siblings(path: Path) -> list[Path]:
    return list(path.parent.glob(f".{path.name}.*.tmp"))


def test_save_failure_preserves_previous_project_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    destination = tmp_path / "project.deproj"
    destination.write_bytes(b"previous project")
    stack = _stack()

    def fail_after_partial_write(archive):
        archive.writestr("partial", b"incomplete")
        raise RuntimeError("serialization failed")

    monkeypatch.setattr(
        stack,
        "_serialize_manifest_and_layers",
        fail_after_partial_write,
    )

    with pytest.raises(RuntimeError, match="serialization failed"):
        stack.save_project(str(destination))

    assert destination.read_bytes() == b"previous project"
    assert _temporary_siblings(destination) == []


def test_replace_failure_preserves_previous_project_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    destination = tmp_path / "project.deproj"
    destination.write_bytes(b"previous project")
    stack = _stack()

    def fail_replace(_source, _destination):
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        stack.save_project(str(destination))

    assert destination.read_bytes() == b"previous project"
    assert _temporary_siblings(destination) == []


def test_successful_save_replaces_atomically_and_leaves_no_temp(tmp_path: Path):
    destination = tmp_path / "project.deproj"
    destination.write_bytes(b"previous project")
    stack = _stack()

    stack.save_project(str(destination))

    assert destination.read_bytes() != b"previous project"
    assert zipfile.is_zipfile(destination)
    assert _temporary_siblings(destination) == []
    restored = LayerStack()
    restored.load_project(str(destination))
    np.testing.assert_array_equal(restored.composite(), stack.composite())
    np.testing.assert_array_equal(
        restored.selection.data,
        stack.selection.data,
    )


def test_post_commit_directory_fsync_failure_still_reports_saved_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    destination = tmp_path / "project.deproj"
    destination.write_bytes(b"previous project")
    stack = _stack()
    monkeypatch.setattr(
        stack,
        "_fsync_directory",
        lambda _directory: (_ for _ in ()).throw(
            OSError("directory fsync unsupported")),
    )

    stack.save_project(str(destination))

    assert zipfile.is_zipfile(destination)
    assert _temporary_siblings(destination) == []
    restored = LayerStack()
    restored.load_project(str(destination))
    np.testing.assert_array_equal(restored.composite(), stack.composite())


def test_corrupt_load_keeps_open_document_unchanged(tmp_path: Path):
    stack = _stack()
    before = stack.serialize_state()
    active_id = stack.active_layer.id
    destination = tmp_path / "corrupt.deproj"
    with zipfile.ZipFile(destination, "w") as archive:
        archive.writestr("manifest.json", json.dumps({
            "format_version": stack.FORMAT_VERSION,
            "canvas_width": 10,
            "canvas_height": 10,
            "tile_size": 8,
            "layers": [{
                "type": "layer",
                "id": "missing",
                "name": "Missing",
                "visible": True,
                "opacity": 1.0,
                "image_file": "missing.npy",
                "children": [],
            }],
        }))

    with pytest.raises(ValueError, match="image entry is missing"):
        stack.load_project(str(destination))

    assert stack.active_layer.id == active_id
    # Loading the pre-failure snapshot into a comparison aggregate avoids ZIP
    # timestamp metadata when checking that every document field survived.
    expected = LayerStack()
    expected.load_state(before)
    np.testing.assert_array_equal(stack.composite(), expected.composite())
    np.testing.assert_array_equal(
        stack.selection.data,
        expected.selection.data,
    )


def test_invalid_selection_shape_keeps_open_document_unchanged():
    source = _stack()
    snapshot = source.serialize_state()
    with zipfile.ZipFile(io.BytesIO(snapshot), "r") as archive:
        manifest = json.loads(archive.read("manifest.json"))
        entries = {
            info.filename: archive.read(info)
            for info in archive.infolist()
            if info.filename not in {"manifest.json", "selection.npy"}
        }
    selection_buffer = io.BytesIO()
    np.save(selection_buffer, np.ones((2, 3), dtype=np.float32))
    entries["selection.npy"] = selection_buffer.getvalue()
    manifest["selection_file"] = "selection.npy"

    bad = io.BytesIO()
    with zipfile.ZipFile(bad, "w", zipfile.ZIP_STORED) as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
        archive.writestr("manifest.json", json.dumps(manifest))

    target = _stack()
    before = target.composite().copy()
    active_id = target.active_layer.id

    with pytest.raises(
            ValueError,
            match="unexpected dimensions"):
        target.load_state(bad.getvalue())

    assert target.active_layer.id == active_id
    np.testing.assert_array_equal(target.composite(), before)


def test_huge_declared_npy_shape_is_rejected_before_array_allocation():
    source = _stack()
    snapshot = source.serialize_state()
    with zipfile.ZipFile(io.BytesIO(snapshot), "r") as archive:
        manifest = json.loads(archive.read("manifest.json"))
        image_file = manifest["layers"][0]["image_file"]
        entries = {
            info.filename: archive.read(info)
            for info in archive.infolist()
            if info.filename not in {"manifest.json", image_file}
        }
    false_header = io.BytesIO()
    np.lib.format.write_array_header_1_0(false_header, {
        "descr": "|u1",
        "fortran_order": False,
        # 400 MB is below the configured array budget, but the entry contains
        # no payload. A raw np.load would attempt the allocation before EOF.
        "shape": (100_000_000, 4),
    })
    entries[image_file] = false_header.getvalue()

    bad = io.BytesIO()
    with zipfile.ZipFile(bad, "w", zipfile.ZIP_STORED) as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
        archive.writestr("manifest.json", json.dumps(manifest))

    target = _stack()
    active_id = target.active_layer.id
    before = target.composite().copy()
    with pytest.raises(
            ValueError,
            match="declared array payload exceeds archive entry size"):
        target.load_state(bad.getvalue())

    assert target.active_layer.id == active_id
    np.testing.assert_array_equal(target.composite(), before)


def test_zero_itemsize_selection_cannot_bypass_preallocation_checks():
    source = _stack()
    snapshot = source.serialize_state()
    with zipfile.ZipFile(io.BytesIO(snapshot), "r") as archive:
        manifest = json.loads(archive.read("manifest.json"))
        selection_file = manifest["selection_file"]
        entries = {
            info.filename: archive.read(info)
            for info in archive.infolist()
            if info.filename not in {"manifest.json", selection_file}
        }
    false_header = io.BytesIO()
    np.lib.format.write_array_header_1_0(false_header, {
        "descr": "|V0",
        "fortran_order": False,
        "shape": (10**12, 10**12),
    })
    entries[selection_file] = false_header.getvalue()

    bad = io.BytesIO()
    with zipfile.ZipFile(bad, "w", zipfile.ZIP_STORED) as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
        archive.writestr("manifest.json", json.dumps(manifest))

    target = _stack()
    active_id = target.active_layer.id
    with pytest.raises(ValueError, match="zero-size array dtypes"):
        target.load_state(bad.getvalue())
    assert target.active_layer.id == active_id


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("canvas_width", 0, "canvas_width must be >= 1"),
        (
            "canvas_height",
            LayerStack.MAX_PROJECT_CANVAS_DIMENSION + 1,
            "canvas_height must be <=",
        ),
        ("tile_size", True, "tile_size must be an integer"),
    ),
)
def test_manifest_resource_validation_happens_before_commit(
    field,
    value,
    message,
):
    source = _stack()
    snapshot = source.serialize_state()
    with zipfile.ZipFile(io.BytesIO(snapshot), "r") as archive:
        manifest = json.loads(archive.read("manifest.json"))
        entries = {
            info.filename: archive.read(info)
            for info in archive.infolist()
            if info.filename != "manifest.json"
        }
    manifest[field] = value
    bad = io.BytesIO()
    with zipfile.ZipFile(bad, "w", zipfile.ZIP_STORED) as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
        archive.writestr("manifest.json", json.dumps(manifest))

    target = _stack()
    active_id = target.active_layer.id
    with pytest.raises(ValueError, match=message):
        target.load_state(bad.getvalue())
    assert target.active_layer.id == active_id


def test_invalid_rgba_initialization_is_rejected_before_replacing_document():
    stack = _stack()
    active_id = stack.active_layer.id
    before = stack.composite().copy()

    with pytest.raises(ValueError, match="uint8 RGBA"):
        stack.init_from_image(np.zeros((2, 2, 3), dtype=np.uint8))

    assert stack.active_layer.id == active_id
    np.testing.assert_array_equal(stack.composite(), before)


def test_empty_or_malformed_state_cannot_be_serialized():
    with pytest.raises(ValueError, match="at least one root layer"):
        LayerStack().serialize_state()

    stack = _stack()
    stack.active_layer.image = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="uint8 RGBA"):
        stack.serialize_state()


def test_invalid_tool_source_patch_is_rejected_before_serialization():
    stack = _stack()
    stack.active_layer.tool = LamaTool(
        source_patch=np.zeros((2, 2, 5), dtype=np.uint8),
        patch_x=0,
        patch_y=0,
        patch_w=2,
        patch_h=2,
    )

    with pytest.raises(ValueError, match="source_patch"):
        stack.serialize_state()


def test_single_channel_tool_source_patch_roundtrips_as_rgb():
    stack = _stack()
    stack.active_layer.tool = LamaTool(
        source_patch=np.full((2, 3, 1), 91, dtype=np.uint8),
        patch_x=0,
        patch_y=0,
        patch_w=3,
        patch_h=2,
    )

    restored = LayerStack()
    restored.load_state(stack.serialize_state())

    source = np.asarray(restored.active_layer.tool.source_patch)
    assert source.shape == (2, 3, 3)
    assert np.all(source == 91)


def test_current_format_duplicate_ids_are_rejected_without_commit():
    source = _stack()
    source.add_layer("Second")
    snapshot = source.serialize_state()
    with zipfile.ZipFile(io.BytesIO(snapshot), "r") as archive:
        manifest = json.loads(archive.read("manifest.json"))
        entries = {
            info.filename: archive.read(info)
            for info in archive.infolist()
            if info.filename != "manifest.json"
        }
    manifest["layers"][1]["id"] = manifest["layers"][0]["id"]
    bad = io.BytesIO()
    with zipfile.ZipFile(bad, "w", zipfile.ZIP_STORED) as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
        archive.writestr("manifest.json", json.dumps(manifest))

    target = _stack()
    active_id = target.active_layer.id
    before = target.composite().copy()
    with pytest.raises(ValueError, match="IDs must be unique"):
        target.load_state(bad.getvalue())

    assert target.active_layer.id == active_id
    np.testing.assert_array_equal(target.composite(), before)


def test_current_format_missing_id_and_identity_reference_are_rejected():
    source = _stack()
    source.solo_layer_id = source.active_layer.id
    snapshot = source.serialize_state()
    with zipfile.ZipFile(io.BytesIO(snapshot), "r") as archive:
        manifest = json.loads(archive.read("manifest.json"))
        entries = {
            info.filename: archive.read(info)
            for info in archive.infolist()
            if info.filename != "manifest.json"
        }
    removed_id = manifest["layers"][0].pop("id")
    assert manifest["solo_layer_id"] == removed_id
    bad = io.BytesIO()
    with zipfile.ZipFile(bad, "w", zipfile.ZIP_STORED) as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
        archive.writestr("manifest.json", json.dumps(manifest))

    target = _stack()
    active_id = target.active_layer.id
    before = target.composite().copy()
    with pytest.raises(ValueError, match="layer ID"):
        target.load_state(bad.getvalue())

    assert target.active_layer.id == active_id
    np.testing.assert_array_equal(target.composite(), before)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("active_layer_path", "99", "active_layer_path"),
        ("solo_layer_id", "missing-layer", "solo_layer_id"),
    ),
)
def test_current_format_invalid_identity_paths_are_rejected_without_commit(
    field,
    value,
    message,
):
    source = _stack()
    snapshot = source.serialize_state()
    with zipfile.ZipFile(io.BytesIO(snapshot), "r") as archive:
        manifest = json.loads(archive.read("manifest.json"))
        entries = {
            info.filename: archive.read(info)
            for info in archive.infolist()
            if info.filename != "manifest.json"
        }
    manifest[field] = value
    bad = io.BytesIO()
    with zipfile.ZipFile(bad, "w", zipfile.ZIP_STORED) as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
        archive.writestr("manifest.json", json.dumps(manifest))

    target = _stack()
    active_id = target.active_layer.id
    with pytest.raises(ValueError, match=message):
        target.load_state(bad.getvalue())
    assert target.active_layer.id == active_id


def test_observer_failure_after_load_does_not_turn_commit_into_failure():
    source = _stack()
    source.add_layer("Loaded")
    snapshot = source.serialize_state()
    target = _stack()
    target.on_changed = lambda: (_ for _ in ()).throw(
        RuntimeError("observer failed"))

    target.load_state(snapshot)

    assert len(target.layers) == 2
    assert target.active_layer.name == "Loaded"
    np.testing.assert_array_equal(target.composite(), source.composite())
