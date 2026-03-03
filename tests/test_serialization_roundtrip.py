"""Roundtrip tests for project/snapshot serialization."""

from __future__ import annotations

import numpy as np

from diffusion_editor.layer import DiffusionLayer, Layer
from diffusion_editor.layer_stack import LayerStack


def _solid_rgba(w: int, h: int, rgba: tuple[int, int, int, int]) -> np.ndarray:
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[:, :] = rgba
    return arr


def _build_stack() -> tuple[LayerStack, DiffusionLayer]:
    stack = LayerStack(tile_size=64)
    stack.on_changed = lambda: None
    stack.init_from_image(_solid_rgba(32, 24, (255, 0, 0, 255)))
    stack.add_layer("Paint", _solid_rgba(32, 24, (0, 255, 0, 128)))

    # Make a nested diffusion layer to verify typed layer fields and path restore.
    diff = DiffusionLayer(
        name="Diff",
        width=32,
        height=24,
        source_patch=None,
        patch_x=1,
        patch_y=2,
        patch_w=16,
        patch_h=12,
        prompt="p",
        negative_prompt="np",
        strength=0.55,
        guidance_scale=8.0,
        steps=20,
        seed=123,
    )
    diff.mask[3:6, 4:8] = 255
    diff.manual_patch_rect = (2, 3, 12, 10)
    top = stack.layers[0]
    top.add_child(diff)
    stack.mark_layer_dirty(top)
    stack.active_layer = diff
    return stack, diff


def test_snapshot_roundtrip_restores_active_layer_and_content():
    stack, _ = _build_stack()
    before = stack.composite().copy()
    active_path = stack.get_layer_path(stack.active_layer)
    snapshot = stack.serialize_state()

    restored = LayerStack(tile_size=16)
    restored.on_changed = lambda: None
    restored.load_state(snapshot)

    assert restored.width == 32
    assert restored.height == 24
    assert restored.tile_size == 64
    assert restored.get_layer_path(restored.active_layer) == active_path
    np.testing.assert_array_equal(restored.composite(), before)

    active = restored.active_layer
    assert isinstance(active, DiffusionLayer)
    assert active.manual_patch_rect == (2, 3, 12, 10)
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
    assert isinstance(restored.layers[0].children[0], DiffusionLayer)
    assert restored.get_layer_path(restored.active_layer) == "0/0"
