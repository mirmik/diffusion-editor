import numpy as np

from diffusion_editor.canvas.canvas_composite import CanvasCompositeBridge
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
            self.composite_calls = 0

        def mark_dirty(self, layer):
            self.marked_layer = layer

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
