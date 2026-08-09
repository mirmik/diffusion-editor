import numpy as np

from diffusion_editor.canvas.brush import BrushToolMode
from diffusion_editor.canvas.editor_canvas_controller import EditorCanvasController
from diffusion_editor.document.layer_stack import LayerStack
from diffusion_editor.document.change_event import DocumentChangeKind


def _controller(image):
    stack = LayerStack(tile_size=8)
    stack.init_from_image(image)
    images = []
    overlays = []
    image_regions = []
    overlay_regions = []
    controller = EditorCanvasController(
        stack,
        gpu_compositing=False,
        set_image=images.append,
        set_overlay=overlays.append,
        update_image_region=lambda x, y, data: image_regions.append(
            (x, y, data.copy())),
        update_overlay_region=lambda x, y, data: overlay_regions.append(
            (x, y, data.copy())),
    )
    controller.refresh()
    return (
        stack,
        controller,
        images,
        overlays,
        image_regions,
        overlay_regions,
    )


def test_controller_paints_through_partial_texture_updates():
    image = np.zeros((32, 32, 4), dtype=np.uint8)
    stack, controller, images, _overlays, regions, _overlay_regions = (
        _controller(image))
    controller.brush.set_size(5)
    controller.brush.set_hardness(1.0)
    controller.brush.set_color(255, 0, 0, 255)

    controller.pointer_down(8, 8, controller.LEFT_BUTTON)
    controller.pointer_move(16, 8)
    controller.pointer_up(16, 8)

    assert stack.active_layer.image[8, 8, 3] > 0
    assert stack.active_layer.image[8, 16, 3] > 0
    assert len(images) == 1
    assert regions
    assert any(x <= 8 < x + data.shape[1] for x, _y, data in regions)


def test_completed_paint_stroke_rebuilds_overlay_once():
    image = np.zeros((32, 32, 4), dtype=np.uint8)
    _stack, controller, *_rest = _controller(image)
    rebuilds = []
    original_rebuild = controller._overlay_bridge.rebuild

    def rebuild():
        rebuilds.append(True)
        original_rebuild()

    controller._overlay_bridge.rebuild = rebuild
    controller.pointer_down(8, 8, controller.LEFT_BUTTON)
    controller.pointer_up(8, 8)

    assert rebuilds == [True]


def test_controller_mask_overlay_uses_partial_updates():
    image = np.zeros((24, 24, 4), dtype=np.uint8)
    stack, controller, _images, overlays, _regions, overlay_regions = (
        _controller(image))
    controller.set_brush_tool(BrushToolMode.MASK)
    controller.set_mask_brush(5, 1.0, 1.0)

    controller.pointer_down(7, 9, controller.LEFT_BUTTON)
    controller.pointer_move(14, 9)

    assert stack.active_layer.mask.data[9, 7] > 0.0
    assert stack.active_layer.mask.data[9, 14] > 0.0
    assert overlays
    assert overlay_regions


def test_controller_preserves_rect_modes_annotations_and_callbacks():
    image = np.zeros((20, 30, 4), dtype=np.uint8)
    _stack, controller, *_rest = _controller(image)
    selections = []
    controller.on_selection_rect_drawn = lambda *rect: selections.append(rect)
    controller.set_selection_rect_mode(True)

    controller.pointer_down(12, 10, controller.LEFT_BUTTON)
    controller.pointer_move(4, 3)

    annotations = {item.kind: item.rect for item in controller.annotations()}
    assert annotations["selection"] == (12, 10, 4, 3)
    assert "active-layer" in annotations

    controller.pointer_up(4, 3)
    assert selections == [(4, 3, 13, 11)]


def test_controller_eyedropper_and_brush_adjustment_are_toolkit_neutral():
    image = np.zeros((8, 8, 4), dtype=np.uint8)
    image[3, 4] = (12, 34, 56, 255)
    _stack, controller, *_rest = _controller(image)
    picked = []
    controller.on_color_picked = lambda *rgba: picked.append(rgba)
    initial_size = controller.brush.size

    controller.pointer_down(
        4,
        3,
        controller.LEFT_BUTTON,
        controller.CTRL_MODIFIER,
    )
    controller.adjust_brush_size(5)

    assert picked == [(12, 34, 56, 255)]
    assert controller.brush.size == initial_size + 5


def test_typed_pixel_event_uses_regional_canvas_refresh():
    image = np.zeros((12, 12, 4), dtype=np.uint8)
    stack, controller, images, _overlays, regions, _overlay_regions = (
        _controller(image))
    layer = stack.active_layer
    layer.image[2:5, 3:7] = (255, 0, 0, 255)
    stack.mark_layer_dirty(layer, layer.local_rect_to_canvas((3, 2, 7, 5)))

    event = stack.publish_change(
        DocumentChangeKind.PIXELS,
        layers=(layer,),
        dirty_rect=(3, 2, 7, 5),
    )
    controller.handle_document_change(event)

    assert len(images) == 1
    assert [(x, y, data.shape[:2]) for x, y, data in regions] == [
        (3, 2, (3, 4))]


def test_metadata_event_does_not_upload_canvas_pixels():
    image = np.zeros((12, 12, 4), dtype=np.uint8)
    stack, controller, images, _overlays, regions, _overlay_regions = (
        _controller(image))
    layer = stack.active_layer

    event = stack.publish_change(
        DocumentChangeKind.METADATA, layers=(layer,))
    controller.handle_document_change(event)

    assert len(images) == 1
    assert regions == []
