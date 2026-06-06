import numpy as np

from diffusion_editor.canvas.brush import Brush
from diffusion_editor.canvas.canvas_composite import CanvasCompositeBridge
from diffusion_editor.canvas.canvas_mask_erase import MaskEraseStrokeBuffer
from diffusion_editor.canvas.canvas_mask_paint import CanvasMaskPainter
from diffusion_editor.canvas.canvas_overlay import CanvasOverlayBridge
from diffusion_editor.canvas.canvas_paint_stroke import PaintStrokeBuffer
from diffusion_editor.canvas.canvas_selection_paint import CanvasSelectionPainter
from diffusion_editor.canvas.canvas_smudge import SmudgeStrokeBuffer
from diffusion_editor.canvas.canvas_tool_context import CanvasToolContext
from diffusion_editor.document.layer_stack import LayerStack


def _rgba(width, height, color=(0, 0, 0, 0)):
    image = np.zeros((height, width, 4), dtype=np.uint8)
    image[:] = color
    return image


def _context(stack, brush, mask_painter):
    composite = CanvasCompositeBridge(
        stack,
        gpu_compositing=False,
        set_image=lambda _image: None,
    )
    composite.update_composite()
    overlay = CanvasOverlayBridge(
        stack,
        set_overlay=lambda _overlay: None,
    )
    return CanvasToolContext(
        stack,
        brush,
        composite,
        overlay,
        PaintStrokeBuffer(),
        CanvasSelectionPainter(),
        mask_painter,
        MaskEraseStrokeBuffer(),
        SmudgeStrokeBuffer(),
    )


def test_mask_erase_finish_clips_alpha_erase_to_visible_layer_rect():
    stack = LayerStack(tile_size=8)
    stack.init_from_image(_rgba(8, 8))
    stack.insert_image_layer("Offset", _rgba(6, 6, (255, 0, 0, 255)), x=-2, y=-2)
    layer = stack.active_layer

    brush = Brush()
    brush.set_size(5)
    brush.set_hardness(1.0)
    mask_painter = CanvasMaskPainter()
    mask_painter.set_brush(5, 1.0, 1.0)
    context = _context(stack, brush, mask_painter)

    context.begin_mask_erase(layer)
    dirty = context.mask_erase_dab(1, 1)
    context.preview_mask_erase(layer, dirty)
    context.finish_mask_erase(layer)

    assert dirty == (0, 0, 4, 4)
    assert layer.mask.data[1, 1] < 1.0
    assert layer.image[1, 1, 3] == 255
    assert layer.image[2, 2, 3] == 0
