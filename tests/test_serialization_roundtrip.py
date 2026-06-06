"""Roundtrip tests for project/snapshot serialization."""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np

from diffusion_editor.document.layer import Layer
from diffusion_editor.document.tool import DiffusionTool
from diffusion_editor.document.layer_stack import LayerStack


def _solid_rgba(w: int, h: int, rgba: tuple[int, int, int, int]) -> np.ndarray:
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[:, :] = rgba
    return arr


def _build_stack() -> tuple[LayerStack, Layer]:
    stack = LayerStack(tile_size=64)
    stack.on_changed = lambda: None
    stack.init_from_image(_solid_rgba(32, 24, (255, 0, 0, 255)))
    stack.add_layer("Paint", _solid_rgba(32, 24, (0, 255, 0, 128)))

    # Make a nested diffusion layer to verify typed layer fields and path restore.
    tool = DiffusionTool(
        source_patch=None,
        patch_x=1, patch_y=2, patch_w=16, patch_h=12,
        prompt="p", negative_prompt="np",
        strength=0.55, guidance_scale=8.0, steps=20, seed=123,
    )
    diff = Layer("Diff", 32, 24)
    diff.x = 3
    diff.y = 4
    diff.patch_rect = (2, 3, 12, 10)
    diff.mask.data[3:6, 4:8] = 1.0
    top = stack.layers[0]
    tool.ip_adapter_layer_id = top.id
    tool.ip_adapter_layer_name_hint = top.name
    diff.tool = tool
    top.add_child(diff)
    stack.mark_layer_dirty(top)
    stack.active_layer = diff
    return stack, diff


def test_snapshot_roundtrip_restores_active_layer_and_content():
    stack, _ = _build_stack()
    before = stack.composite().copy()
    active_path = stack.get_layer_path(stack.active_layer)
    active_id = stack.active_layer.id
    snapshot = stack.serialize_state()

    restored = LayerStack(tile_size=16)
    restored.on_changed = lambda: None
    restored.load_state(snapshot)

    assert restored.width == 32
    assert restored.height == 24
    assert restored.tile_size == 64
    assert restored.get_layer_path(restored.active_layer) == active_path
    assert restored.active_layer.id == active_id
    np.testing.assert_array_equal(restored.composite(), before)

    active = restored.active_layer
    assert isinstance(active.tool, DiffusionTool)
    assert active.x == 3
    assert active.y == 4
    assert active.patch_rect == (2, 3, 12, 10)
    assert active.tool.ip_adapter_layer_id == restored.layers[0].id
    assert active.tool.ip_adapter_layer_name_hint == "Paint"
    assert active.has_mask()


def test_project_file_roundtrip(tmp_path):
    stack, _ = _build_stack()
    path = tmp_path / "roundtrip.deproj"
    stack.save_project(str(path))

    restored = LayerStack()
    restored.on_changed = lambda: None
    restored.load_project(str(path))

    assert len(restored.layers) == 2
    assert isinstance(restored.layers[0], Layer)
    assert isinstance(restored.layers[0].children[0].tool, DiffusionTool)
    assert restored.get_layer_path(restored.active_layer) == "0/0"


def test_selection_roundtrip():
    stack, _ = _build_stack()
    stack.selection.data[7:10, 11:15] = 0.5
    snapshot = stack.serialize_state()

    restored = LayerStack()
    restored.on_changed = lambda: None
    restored.load_state(snapshot)

    assert restored.selection.data.shape == (24, 32)
    np.testing.assert_array_equal(restored.selection.data, stack.selection.data)


def test_tool_nested_mask_file_migrates_to_layer_mask():
    stack, _ = _build_stack()
    original = stack.serialize_state()

    src = zipfile.ZipFile(io.BytesIO(original), "r")
    manifest = json.loads(src.read("manifest.json"))
    diffusion_layer = manifest["layers"][0]["children"][0]
    diffusion_layer["tool"]["mask_file"] = diffusion_layer.pop("mask_file")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as dst:
        for name in src.namelist():
            if name == "manifest.json":
                continue
            dst.writestr(name, src.read(name))
        dst.writestr("manifest.json", json.dumps(manifest))
    src.close()

    restored = LayerStack()
    restored.on_changed = lambda: None
    restored.load_state(buf.getvalue())

    assert restored.active_layer is not None
    assert restored.active_layer.has_mask()
    np.testing.assert_array_equal(
        restored.active_layer.mask.data,
        stack.active_layer.mask.data,
    )


def test_tool_manual_patch_rect_migrates_to_layer_patch_rect():
    stack, _ = _build_stack()
    original = stack.serialize_state()

    src = zipfile.ZipFile(io.BytesIO(original), "r")
    manifest = json.loads(src.read("manifest.json"))
    manifest["format_version"] = 7
    diffusion_layer = manifest["layers"][0]["children"][0]
    diffusion_layer.pop("patch_rect", None)
    diffusion_layer["tool"]["manual_patch_rect"] = [4, 5, 14, 15]

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as dst:
        for name in src.namelist():
            if name == "manifest.json":
                continue
            dst.writestr(name, src.read(name))
        dst.writestr("manifest.json", json.dumps(manifest))
    src.close()

    restored = LayerStack()
    restored.on_changed = lambda: None
    restored.load_state(buf.getvalue())

    assert restored.active_layer.patch_rect == (1, 1, 11, 11)
    assert "manual_patch_rect" not in restored.active_layer.tool.__dict__


def test_legacy_project_without_layer_ids_gets_unique_ids():
    stack, _ = _build_stack()
    original = stack.serialize_state()

    src = zipfile.ZipFile(io.BytesIO(original), "r")
    manifest = json.loads(src.read("manifest.json"))

    def strip_ids(layer_dict):
        layer_dict.pop("id", None)
        for child in layer_dict.get("children", []):
            strip_ids(child)

    for layer_dict in manifest["layers"]:
        strip_ids(layer_dict)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as dst:
        for name in src.namelist():
            if name == "manifest.json":
                continue
            dst.writestr(name, src.read(name))
        dst.writestr("manifest.json", json.dumps(manifest))
    src.close()

    restored = LayerStack()
    restored.on_changed = lambda: None
    restored.load_state(buf.getvalue())

    ids = [layer.id for layer in restored._all_layers_flat()]
    assert all(ids)
    assert len(ids) == len(set(ids))


def test_solo_composite_uses_hidden_layer_without_changing_visible():
    stack = LayerStack(tile_size=64)
    stack.on_changed = lambda: None
    stack.init_from_image(_solid_rgba(4, 4, (255, 0, 0, 255)))
    stack.insert_image_layer("Reference", _solid_rgba(2, 2, (0, 255, 0, 255)), x=1, y=1)
    ref = stack.active_layer
    stack.set_visibility(ref, False)

    normal = stack.composite()
    np.testing.assert_array_equal(normal[1, 1], np.array([255, 0, 0, 255], dtype=np.uint8))

    stack.toggle_solo_layer(ref)
    assert ref.visible is False
    solo = stack.composite()
    np.testing.assert_array_equal(solo[1, 1], np.array([0, 255, 0, 255], dtype=np.uint8))
    np.testing.assert_array_equal(solo[0, 0], np.array([0, 0, 0, 0], dtype=np.uint8))


def test_solo_parent_does_not_include_children():
    stack = LayerStack(tile_size=64)
    stack.on_changed = lambda: None
    stack.init_from_image(_solid_rgba(4, 4, (0, 0, 0, 0)))

    parent = Layer("Parent", 4, 4, _solid_rgba(4, 4, (255, 0, 0, 255)))
    child = Layer("Child", 4, 4, _solid_rgba(4, 4, (0, 0, 255, 255)))
    parent.add_child(child)
    stack.insert_layer(parent)

    stack.toggle_solo_layer(parent)
    parent_solo = stack.composite()
    np.testing.assert_array_equal(parent_solo[0, 0], np.array([255, 0, 0, 255], dtype=np.uint8))

    stack.toggle_solo_layer(child)
    child_solo = stack.composite()
    np.testing.assert_array_equal(child_solo[0, 0], np.array([0, 0, 255, 255], dtype=np.uint8))
