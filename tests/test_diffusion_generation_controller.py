import numpy as np
from PIL import Image

from diffusion_editor.generation.diffusion_controller import (
    DiffusionGenerationController,
)
from diffusion_editor.generation.types import (
    DiffusionInferenceResult,
    EnginePollEvent,
)
from diffusion_editor.document.layer import Layer
from diffusion_editor.document.layer_stack import LayerStack
from diffusion_editor.document.tool import DiffusionTool


def _rgba(width, height, color):
    arr = np.zeros((height, width, 4), dtype=np.uint8)
    arr[:] = color
    return arr


def _tool(mode="txt2img"):
    return DiffusionTool(
        source_patch=None,
        patch_x=0, patch_y=0, patch_w=4, patch_h=4,
        prompt="p", negative_prompt="np",
        strength=0.5, guidance_scale=7.0, steps=20, seed=1,
        mode=mode,
    )


class _Engine:
    def __init__(self):
        self.is_busy = False
        self.is_loaded = True
        self.ip_adapter_loaded = False
        self.model_path = "loaded.safetensors"
        self.model_info = {}
        self.calls = []
        self.poll_result = None

    def submit_load(self, path, prediction_type=None):
        self.calls.append(("load", path, prediction_type))
        return True

    def submit_load_ip_adapter(self):
        self.calls.append(("load_ip_adapter",))
        return True

    def submit_request(self, request):
        self.calls.append(("submit_request", request))
        return True

    def poll_event(self):
        result = self.poll_result
        self.poll_result = None
        return result


def _stack_with_diff_layer(mode="txt2img"):
    stack = LayerStack()
    stack.init_from_image(_rgba(8, 8, (0, 0, 0, 255)))
    layer = Layer("Diff", 8, 8, _rgba(8, 8, (0, 0, 0, 0)))
    layer.tool = _tool(mode=mode)
    stack.insert_layer(layer)
    return stack, layer


def test_start_regeneration_loads_requested_model_before_inference():
    stack, layer = _stack_with_diff_layer()
    layer.tool.model_path = "next.safetensors"
    layer.tool.prediction_type = "v_prediction"
    engine = _Engine()
    controller = DiffusionGenerationController(
        engine=engine,
        layer_stack=stack,
        composite_below=lambda _layer: None,
    )

    event = controller.start_regeneration(layer)

    assert event.status == "Loading model for regeneration..."
    assert controller.pending_layer is layer
    assert engine.calls == [("load", "next.safetensors", "v_prediction")]


def test_poll_model_load_resumes_pending_regeneration_and_reports_loaded():
    stack, layer = _stack_with_diff_layer()
    engine = _Engine()
    controller = DiffusionGenerationController(
        engine=engine,
        layer_stack=stack,
        composite_below=lambda _layer: None,
    )
    controller.start_regeneration(layer)
    engine.calls.clear()
    engine.poll_result = EnginePollEvent(
        task_type="load",
        result="loaded.safetensors",
    )

    event = controller.poll()

    assert event.model_loaded_path == "loaded.safetensors"
    assert event.status == "Regenerating (4x4)..."
    assert engine.calls[0][0] == "submit_request"


def test_start_regeneration_loads_ip_adapter_when_reference_is_present():
    stack, layer = _stack_with_diff_layer()
    ref = Layer("Reference", 2, 2, _rgba(2, 2, (10, 20, 30, 255)))
    stack.insert_layer(ref)
    layer.tool.ip_adapter_layer_id = ref.id
    layer.tool.ip_adapter_layer_name_hint = ref.name
    engine = _Engine()
    controller = DiffusionGenerationController(
        engine=engine,
        layer_stack=stack,
        composite_below=lambda _layer: None,
    )

    event = controller.start_regeneration(layer)

    assert event.status == "Loading IP-Adapter..."
    assert controller.pending_layer is layer
    assert engine.calls == [("load_ip_adapter",)]


def test_poll_inference_returns_pending_layer_and_clears_pending():
    stack, layer = _stack_with_diff_layer()
    engine = _Engine()
    controller = DiffusionGenerationController(
        engine=engine,
        layer_stack=stack,
        composite_below=lambda _layer: None,
    )
    controller.start_regeneration(layer)
    result_image = Image.fromarray(_rgba(4, 4, (1, 2, 3, 255)), "RGBA")
    engine.poll_result = EnginePollEvent(
        task_type="inference",
        result=DiffusionInferenceResult(image=result_image, seed=123),
    )

    event = controller.poll()

    assert event.inference_result == (layer, result_image, 123)
    assert controller.pending_layer is None
