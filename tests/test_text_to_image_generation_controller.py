from __future__ import annotations

import numpy as np
from PIL import Image

from diffusion_editor.document.layer import Layer
from diffusion_editor.document.tool import TextToImageTool
from diffusion_editor.generation.text_to_image_controller import (
    TextToImageGenerationController,
)
from diffusion_editor.generation.types import (
    EnginePollEvent,
    TextToImageInferenceResult,
)


class _Engine:
    def __init__(self, *, loaded: bool = True):
        self.is_busy = False
        self.is_loaded = loaded
        self.loaded_profile_id = "qwen-image-2512" if loaded else None
        self.calls = []
        self.poll_result = None

    def loaded_configuration_matches(
            self, profile_id, _parameters, _lora_adapters):
        return self.is_loaded and self.loaded_profile_id == profile_id

    def submit_load(self, profile_id, parameters, lora_adapters):
        self.calls.append(("load", profile_id, parameters, lora_adapters))
        return True

    def submit_request(self, request):
        self.calls.append(("request", request))
        return True

    def poll_event(self):
        event, self.poll_result = self.poll_result, None
        return event


def _layer() -> Layer:
    pixels = np.zeros((11, 23, 4), dtype=np.uint8)
    layer = Layer("Generated", 23, 11, pixels)
    layer.x = 7
    layer.y = 13
    layer.tool = TextToImageTool()
    layer.tool.set_parameter("prompt", "a tiny observatory")
    layer.tool.set_parameter("seed", 123)
    return layer


def test_request_has_no_source_image_and_uses_exact_layer_dimensions():
    layer = _layer()
    engine = _Engine()
    controller = TextToImageGenerationController(engine=engine)

    event = controller.start(layer)

    assert event.status == "Generating 23x11..."
    request = engine.calls[-1][1]
    assert request.width == 23
    assert request.height == 11
    assert request.parameters["prompt"] == "a tiny observatory"
    assert not hasattr(request, "image")
    context = controller.pending_context
    assert context.input_image is None
    assert context.input_array is None
    assert context.paste.canvas_rect == (7, 13, 30, 24)
    assert context.paste.layer_local_rect == (0, 0, 23, 11)
    assert context.paste.geometry_source == "layer"


def test_load_completion_resumes_pending_generation():
    layer = _layer()
    engine = _Engine(loaded=False)
    controller = TextToImageGenerationController(engine=engine)

    event = controller.start(layer)
    assert event.model_loading
    assert engine.calls[-1][0] == "load"

    engine.is_loaded = True
    engine.loaded_profile_id = "qwen-image-2512"
    engine.poll_result = EnginePollEvent(task_type="load", result=True)
    event = controller.poll()

    assert event.model_loaded
    assert event.status == "Generating 23x11..."
    assert engine.calls[-1][0] == "request"


def test_result_must_match_the_frozen_layer_size():
    layer = _layer()
    engine = _Engine()
    controller = TextToImageGenerationController(engine=engine)
    controller.start(layer)
    engine.poll_result = EnginePollEvent(
        task_type="inference",
        result=TextToImageInferenceResult(
            Image.new("RGB", (23, 11), "teal"), 123),
    )

    event = controller.poll()

    context, image, seed = event.inference_result
    assert context.layer_id == layer.id
    assert image.size == (23, 11)
    assert seed == 123
    assert controller.pending_context is None



def test_reference_is_frozen_before_model_load_and_identified_in_provenance():
    from diffusion_editor.generation.provenance import capture_tool_state
    layer = _layer()
    layer.tool.set_profile("qwen-image-2.1")
    layer.tool.reference_image = Image.new("RGBA", (9, 7), (10, 20, 30, 64))
    engine = _Engine(loaded=False)
    controller = TextToImageGenerationController(engine=engine)
    assert controller.start(layer).model_loading
    context = controller.pending_context
    assert context.reference_image.to_image().getpixel((0, 0)) == (10, 20, 30, 64)
    assert context.request_provenance.parameters.to_dict()["reference_image_hash"] == context.reference_image.content_hash
    before = capture_tool_state(layer.tool).fingerprint
    layer.tool.reference_image.paste((0, 0, 0, 0), (0, 0, 9, 7))
    assert capture_tool_state(layer.tool).fingerprint != before
    engine.is_loaded = True
    engine.loaded_profile_id = "qwen-image-2.1"
    engine.poll_result = EnginePollEvent(task_type="load", result=True)
    controller.poll()
    request = engine.calls[-1][1]
    assert request.reference_image.getpixel((0, 0)) == (10, 20, 30, 64)
    assert not hasattr(request, "image")


def test_missing_reference_fails_before_submitting_and_old_profiles_ignore_reference():
    layer = _layer()
    layer.tool.reference_layer_id = "missing"
    layer.tool.set_profile("qwen-image-2.1")
    engine = _Engine()
    controller = TextToImageGenerationController(engine=engine)
    assert "unavailable" in controller.start(layer).status
    assert not engine.calls
    layer.tool.set_profile("qwen-image-2512")
    controller.start(layer)
    assert engine.calls[-1][1].reference_image is None
