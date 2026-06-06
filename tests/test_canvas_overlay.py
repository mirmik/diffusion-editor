import numpy as np

from diffusion_editor.canvas.canvas_overlay import (
    MASK_OPACITY,
    SELECTION_OPACITY,
    CanvasOverlayBridge,
    compose_overlay_pixels,
)
from diffusion_editor.document.layer_stack import LayerStack


def _rgba(width, height, color=(0, 0, 0, 0)):
    arr = np.zeros((height, width, 4), dtype=np.uint8)
    arr[:] = color
    return arr


def test_compose_overlay_pixels_combines_selection_and_mask():
    selection_alpha = np.array([[255.0 * SELECTION_OPACITY]], dtype=np.float32)
    mask_alpha = np.array([[255.0 * MASK_OPACITY]], dtype=np.float32)

    overlay = compose_overlay_pixels(selection_alpha, mask_alpha)

    assert overlay[0, 0, 0] > overlay[0, 0, 2]
    assert overlay[0, 0, 3] > selection_alpha[0, 0]
    assert overlay[0, 0, 3] > mask_alpha[0, 0]


def test_overlay_bridge_rebuilds_selection_overlay():
    stack = LayerStack(tile_size=8)
    stack.init_from_image(_rgba(8, 8))
    stack.selection.data[2, 3] = 1.0
    overlays = []
    bridge = CanvasOverlayBridge(stack, set_overlay=overlays.append)

    bridge.rebuild()

    assert overlays[-1] is bridge.overlay
    assert bridge.overlay[2, 3, 2] > bridge.overlay[2, 3, 0]
    assert bridge.overlay[2, 3, 3] == int(255.0 * SELECTION_OPACITY)


def test_overlay_bridge_rebuilds_offset_layer_mask_overlay():
    stack = LayerStack(tile_size=8)
    stack.init_from_image(_rgba(10, 10))
    stack.insert_image_layer("mask", _rgba(4, 4), x=2, y=3)
    layer = stack.active_layer
    layer.mask.data[1, 1] = 1.0
    bridge = CanvasOverlayBridge(stack, set_overlay=lambda _overlay: None)

    bridge.rebuild()

    assert bridge.overlay[4, 3, 0] == 255
    assert bridge.overlay[4, 3, 3] == int(255.0 * MASK_OPACITY)
    assert bridge.overlay[1, 1, 3] == 0


def test_overlay_bridge_updates_mask_region_with_selection_underneath():
    stack = LayerStack(tile_size=8)
    stack.init_from_image(_rgba(8, 8))
    stack.selection.data[2, 2] = 1.0
    layer = stack.active_layer
    bridge = CanvasOverlayBridge(stack, set_overlay=lambda _overlay: None)
    bridge.rebuild()

    layer.mask.data[2, 2] = 1.0
    bridge.update_mask_region(layer, (2, 2, 3, 3))

    assert bridge.overlay[2, 2, 0] > bridge.overlay[2, 2, 2]
    assert bridge.overlay[2, 2, 3] > int(255.0 * MASK_OPACITY)


def test_overlay_bridge_updates_preview_region_from_dirty_local_mask():
    stack = LayerStack(tile_size=8)
    stack.init_from_image(_rgba(8, 8))
    layer = stack.active_layer
    bridge = CanvasOverlayBridge(stack, set_overlay=lambda _overlay: None)
    layer.mask.data[1, 1] = 1.0
    bridge.rebuild()
    preview = np.zeros((3, 3), dtype=np.float32)
    preview[1, 1] = 0.25

    bridge.update_mask_region_preview(layer, (0, 0, 3, 3), preview)

    assert bridge.overlay[1, 1, 3] == int(255.0 * MASK_OPACITY * 0.25)


def test_overlay_bridge_honors_hidden_mask_overlay_flag():
    stack = LayerStack(tile_size=8)
    stack.init_from_image(_rgba(8, 8))
    stack.active_layer.mask.data[2, 2] = 1.0
    overlays = []
    bridge = CanvasOverlayBridge(stack, set_overlay=overlays.append)
    bridge.show_mask = False

    bridge.rebuild()

    assert overlays[-1] is None
    assert bridge.overlay is None
