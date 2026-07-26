import numpy as np

from diffusion_editor.canvas.canvas_composite import CanvasCompositeBridge
from diffusion_editor.document.layer import Layer
from diffusion_editor.document.layer_stack import LayerStack


def _rgba(width, height, color=(0, 0, 0, 0)):
    arr = np.zeros((height, width, 4), dtype=np.uint8)
    arr[:] = color
    return arr


def test_bridge_updates_cpu_composite_and_set_image():
    stack = LayerStack(tile_size=8)
    image = _rgba(8, 8)
    image[2, 3] = (10, 20, 30, 255)
    stack.init_from_image(image)
    images = []
    bridge = CanvasCompositeBridge(
        stack,
        gpu_compositing=False,
        set_image=images.append,
    )

    composite = bridge.update_composite()

    assert composite is images[-1]
    assert composite[2, 3].tolist() == [10, 20, 30, 255]


def test_bridge_refresh_modified_layer_rect_updates_cpu_composite_region():
    stack = LayerStack(tile_size=8)
    stack.init_from_image(_rgba(8, 8))
    layer = stack.active_layer
    images = []
    bridge = CanvasCompositeBridge(
        stack,
        gpu_compositing=False,
        set_image=images.append,
    )
    bridge.update_composite()
    layer.image[2, 3] = (100, 50, 25, 255)

    bridge.refresh_modified_layer_rect(
        layer,
        (3, 2, 4, 3),
        (3, 2, 4, 3),
    )

    assert bridge.composite[2, 3].tolist() == [100, 50, 25, 255]
    assert images[-1] is bridge.composite


def test_cpu_partial_composite_matches_full_with_upper_layer_and_group_opacity():
    stack = LayerStack(tile_size=4)
    stack._width = 8
    stack._height = 8
    bottom = Layer("bottom", 8, 8, _rgba(8, 8, (5, 20, 180, 255)))
    group = Layer("group", 8, 8, _rgba(8, 8, (170, 10, 20, 96)))
    target = Layer("target", 8, 8, _rgba(8, 8, (20, 190, 40, 128)))
    group.add_child(target)
    group.opacity = 0.35
    upper = Layer("upper", 8, 8, _rgba(8, 8, (230, 210, 10, 80)))
    stack._layers = [upper, group, bottom]
    stack._active_layer = target
    stack._rebuild_caches()

    updated_regions = []
    bridge = CanvasCompositeBridge(
        stack,
        gpu_compositing=False,
        set_image=lambda _image: None,
        update_image_region=lambda x, y, image: updated_regions.append(
            (x, y, image.copy())),
    )
    bridge.update_composite()

    target.image[2:4, 3:5] = (180, 30, 210, 72)
    bridge.refresh_modified_layer_rect(
        target,
        (3, 2, 5, 4),
        (3, 2, 5, 4),
    )

    np.testing.assert_array_equal(bridge.composite, stack.composite())
    assert updated_regions[-1][0:2] == (3, 2)
    np.testing.assert_array_equal(
        updated_regions[-1][2],
        stack.composite()[2:4, 3:5],
    )


def test_cpu_mask_erase_preview_uses_canonical_composition_and_restores_layer():
    stack = LayerStack(tile_size=4)
    stack.init_from_image(_rgba(8, 8, (10, 30, 200, 255)))
    target_image = _rgba(8, 8, (220, 30, 20, 180))
    stack.add_layer("target", target_image)
    target = stack.active_layer
    target.opacity = 0.5
    stack.add_layer("upper", _rgba(8, 8, (20, 220, 40, 96)))
    original = target.image.copy()

    bridge = CanvasCompositeBridge(
        stack,
        gpu_compositing=False,
        set_image=lambda _image: None,
    )
    bridge.update_composite()

    erase = np.full((2, 2), 0.75, dtype=np.float32)
    bridge.preview_erased_layer_rect(
        target,
        (3, 2, 5, 4),
        (3, 2, 5, 4),
        erase,
    )
    preview_result = bridge.composite.copy()

    # Build the canonical expected result using the same temporary source
    # pixels; the bridge itself must leave the document unchanged.
    expected_alpha = np.clip(
        original[2:4, 3:5, 3].astype(np.float32) * (1.0 - erase),
        0,
        255,
    ).astype(np.uint8)
    target.image[2:4, 3:5, 3] = expected_alpha
    stack.mark_layer_dirty(target, (3, 2, 5, 4), pixels_changed=False)
    expected = stack.composite()
    target.image[:] = original
    stack.mark_layer_dirty(target, (3, 2, 5, 4), pixels_changed=False)

    np.testing.assert_array_equal(preview_result, expected)
    np.testing.assert_array_equal(target.image, original)


def test_bridge_refresh_layer_transform_rebuilds_cpu_composite():
    stack = LayerStack(tile_size=8)
    image = _rgba(8, 8)
    image[1, 1] = (255, 0, 0, 255)
    stack.init_from_image(image)
    layer = stack.active_layer
    bridge = CanvasCompositeBridge(
        stack,
        gpu_compositing=False,
        set_image=lambda _image: None,
    )
    bridge.update_composite()

    old_bounds = layer.bounds
    layer.x = 2
    layer.y = 1
    bridge.refresh_layer_transform(layer, old_bounds)

    assert bridge.composite[1, 1, 3] == 0
    assert bridge.composite[2, 3, 3] == 255


def test_bridge_refresh_modified_layer_rect_uses_gpu_compositor_when_enabled():
    class FakeCompositor:
        def __init__(self):
            self.marked_layer = None
            self.marked_rect = None
            self.composite_calls = 0

        def mark_dirty(self, layer, rect=None):
            self.marked_layer = layer
            self.marked_rect = rect

        def composite(self):
            self.composite_calls += 1

    stack = LayerStack(tile_size=8)
    stack.init_from_image(_rgba(8, 8))
    layer = stack.active_layer
    bridge = CanvasCompositeBridge(
        stack,
        gpu_compositing=False,
        set_image=lambda _image: None,
    )
    fake = FakeCompositor()
    bridge.gpu_compositing = True
    bridge.gpu_compositor = fake

    bridge.refresh_modified_layer_rect(
        layer,
        (0, 0, 1, 1),
        (0, 0, 1, 1),
    )

    assert fake.marked_layer is layer
    assert fake.marked_rect == (0, 0, 1, 1)
    assert fake.composite_calls == 1
    assert bridge.composite_stale is True


def test_bridge_layer_transform_recomposites_without_pixel_upload():
    class FakeCompositor:
        def __init__(self):
            self.composite_dirty_calls = 0
            self.composite_calls = 0

        def mark_composite_dirty(self):
            self.composite_dirty_calls += 1

        def composite(self):
            self.composite_calls += 1

    stack = LayerStack(tile_size=8)
    stack.init_from_image(_rgba(8, 8))
    layer = stack.active_layer
    bridge = CanvasCompositeBridge(
        stack,
        gpu_compositing=False,
        set_image=lambda _image: None,
    )
    fake = FakeCompositor()
    bridge.gpu_compositing = True
    bridge.gpu_compositor = fake

    bridge.refresh_layer_transform(layer, layer.bounds)

    assert fake.composite_dirty_calls == 1
    assert fake.composite_calls == 1
    assert bridge.composite_stale is True


def test_bridge_hidden_layer_refresh_falls_back_to_cpu_without_gpu_compositor():
    stack = LayerStack(tile_size=8)
    stack.init_from_image(_rgba(8, 8, (1, 2, 3, 255)))
    stack.add_layer("hidden", _rgba(8, 8, (255, 0, 0, 255)))
    layer = stack.active_layer
    layer.visible = False
    images = []
    bridge = CanvasCompositeBridge(
        stack,
        gpu_compositing=False,
        set_image=images.append,
    )
    bridge.update_composite()
    bridge.gpu_compositing = True
    bridge.gpu_compositor = None

    layer.image[0, 0] = (0, 255, 0, 255)
    bridge.refresh_modified_layer_rect(
        layer,
        (0, 0, 1, 1),
        (0, 0, 1, 1),
    )

    assert images[-1] is bridge.composite
    assert bridge.composite[0, 0].tolist() == [1, 2, 3, 255]


def test_bridge_hidden_gpu_layer_accumulates_region_without_compositing():
    class FakeCompositor:
        def __init__(self):
            self.marked = []
            self.composite_calls = 0

        def mark_dirty(self, layer, rect=None):
            self.marked.append((layer, rect))

        def composite(self):
            self.composite_calls += 1

    stack = LayerStack(tile_size=8)
    stack.init_from_image(_rgba(8, 8))
    layer = stack.active_layer
    layer.visible = False
    bridge = CanvasCompositeBridge(
        stack,
        gpu_compositing=False,
        set_image=lambda _image: None,
    )
    fake = FakeCompositor()
    bridge.gpu_compositing = True
    bridge.gpu_compositor = fake

    bridge.refresh_modified_layer_rect(
        layer,
        (2, 1, 4, 3),
        (2, 1, 4, 3),
    )

    assert fake.marked == [(layer, (2, 1, 4, 3))]
    assert fake.composite_calls == 0
