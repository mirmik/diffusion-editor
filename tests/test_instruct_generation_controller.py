import numpy as np
from PIL import Image

from diffusion_editor.generation.instruct_controller import (
    InstructGenerationController,
)
from diffusion_editor.generation.types import (
    EnginePollEvent,
    InstructInferenceResult,
)
from diffusion_editor.document.layer import Layer
from diffusion_editor.document.layer_stack import LayerStack
from diffusion_editor.document.tool import InstructTool


def _rgba(width, height, color):
    arr = np.zeros((height, width, 4), dtype=np.uint8)
    arr[:] = color
    return arr


class _Engine:
    def __init__(self):
        self.is_busy = False
        self.is_loaded = True
        self.calls = []
        self.poll_result = None

    def submit_load(self):
        self.calls.append(("load",))
        return True

    def submit_request(self, request):
        self.calls.append(("submit_request", request))
        return True

    def poll_event(self):
        result = self.poll_result
        self.poll_result = None
        return result


def _stack_with_instruct_layer():
    stack = LayerStack()
    stack.init_from_image(_rgba(16, 16, (0, 0, 0, 255)))
    layer = Layer("Instruct", 16, 16, _rgba(16, 16, (0, 0, 0, 0)))
    layer.tool = InstructTool(
        source_patch=None,
        patch_x=2, patch_y=2, patch_w=8, patch_h=8,
        instruction="make it red",
        image_guidance_scale=1.5,
        guidance_scale=7.0,
        steps=20,
        seed=123,
    )
    stack.insert_layer(layer)
    return stack, layer


def test_start_apply_loads_model_when_needed():
    _stack, layer = _stack_with_instruct_layer()
    engine = _Engine()
    engine.is_loaded = False
    controller = InstructGenerationController(
        engine=engine,
        composite_below=lambda _layer: _rgba(16, 16, (10, 20, 30, 255)),
    )

    event = controller.start_apply(layer)

    assert event.model_loading is True
    assert event.status == "Loading InstructPix2Pix model..."
    assert controller.pending_layer is layer
    assert engine.calls == [("load",)]


def test_poll_model_load_resumes_pending_instruction():
    _stack, layer = _stack_with_instruct_layer()
    engine = _Engine()
    engine.is_loaded = False
    controller = InstructGenerationController(
        engine=engine,
        composite_below=lambda _layer: _rgba(16, 16, (10, 20, 30, 255)),
    )
    controller.start_apply(layer)
    engine.is_loaded = True
    engine.calls.clear()
    engine.poll_result = EnginePollEvent(task_type="load", result=True)

    event = controller.poll()

    assert event.model_loaded is True
    assert event.status == "Applying instruction..."
    assert engine.calls[0][0] == "submit_request"


def test_poll_inference_returns_pending_layer_and_clears_pending():
    _stack, layer = _stack_with_instruct_layer()
    engine = _Engine()
    controller = InstructGenerationController(
        engine=engine,
        composite_below=lambda _layer: _rgba(16, 16, (10, 20, 30, 255)),
    )
    controller.start_apply(layer)
    result = Image.fromarray(_rgba(8, 8, (1, 2, 3, 255)), "RGBA")
    engine.poll_result = EnginePollEvent(
        task_type="inference",
        result=InstructInferenceResult(image=result, seed=456),
    )

    event = controller.poll()

    assert event.inference_result == (layer, result, 456)
    assert controller.pending_layer is None
