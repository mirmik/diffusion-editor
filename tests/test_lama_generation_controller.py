import numpy as np
from PIL import Image

from diffusion_editor.generation.types import EnginePollEvent, LamaResult
from diffusion_editor.generation.lama_controller import LamaGenerationController
from diffusion_editor.document.layer import Layer
from diffusion_editor.document.layer_stack import LayerStack
from diffusion_editor.document.tool import LamaTool


def _rgba(width, height, color):
    arr = np.zeros((height, width, 4), dtype=np.uint8)
    arr[:] = color
    return arr


class _Engine:
    def __init__(self):
        self.is_busy = False
        self.calls = []
        self.poll_result = None

    def submit_request(self, request):
        self.calls.append((request.image, request.mask_image))
        return True

    def poll_event(self):
        result = self.poll_result
        self.poll_result = None
        return result


def _stack_with_lama_layer():
    stack = LayerStack()
    stack.init_from_image(_rgba(16, 16, (0, 0, 0, 255)))
    layer = Layer("LaMa", 16, 16, _rgba(16, 16, (0, 0, 0, 0)))
    layer.tool = LamaTool(
        source_patch=None,
        patch_x=2, patch_y=2, patch_w=8, patch_h=8,
    )
    layer.mask.data[4:8, 4:8] = 1.0
    stack.insert_layer(layer)
    return stack, layer


def test_start_remove_submits_patch_and_mask():
    _stack, layer = _stack_with_lama_layer()
    engine = _Engine()
    controller = LamaGenerationController(
        engine=engine,
        composite_below=lambda _layer: _rgba(16, 16, (10, 20, 30, 255)),
    )

    event = controller.start_remove(layer)

    assert event.status == "Removing objects (LaMa)..."
    assert controller.pending_layer is layer
    assert len(engine.calls) == 1
    image, mask = engine.calls[0]
    assert image.size == mask.size
    assert layer.tool.source_patch is image


def test_poll_returns_pending_lama_layer_and_clears_pending():
    _stack, layer = _stack_with_lama_layer()
    engine = _Engine()
    controller = LamaGenerationController(
        engine=engine,
        composite_below=lambda _layer: _rgba(16, 16, (10, 20, 30, 255)),
    )
    controller.start_remove(layer)
    result = Image.fromarray(_rgba(8, 8, (1, 2, 3, 255)), "RGBA")
    engine.poll_result = EnginePollEvent(
        task_type="inference",
        result=LamaResult(image=result),
    )

    event = controller.poll()

    assert event.inference_result == (layer, result)
    assert controller.pending_layer is None
