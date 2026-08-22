import numpy as np
from PIL import Image
import pytest

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
from diffusion_editor.generation.image_edit_profiles import (
    FLUX2_KLEIN_PROFILE_ID,
    QWEN_IMAGE_EDIT_PROFILE_ID,
    SENSENOVA_U15_PROFILE_ID,
)


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
    assert controller.pending_context.layer_id == layer.id
    assert engine.calls == [("load",)]


def test_start_apply_without_mask_or_manual_patch_uses_full_composite():
    _stack, layer = _stack_with_instruct_layer()
    engine = _Engine()
    composite = _rgba(20, 14, (10, 20, 30, 255))
    controller = InstructGenerationController(
        engine=engine,
        composite_below=lambda _layer: composite,
    )

    event = controller.start_apply(layer)

    assert event.status == "Applying instruction..."
    request = engine.calls[-1][1]
    assert request.image.size == (20, 14)
    assert layer.tool.source_patch.size == (20, 14)
    assert (
        layer.tool.patch_x,
        layer.tool.patch_y,
        layer.tool.patch_w,
        layer.tool.patch_h,
    ) == (0, 0, 20, 14)


@pytest.mark.parametrize("profile_id", [
    QWEN_IMAGE_EDIT_PROFILE_ID,
    FLUX2_KLEIN_PROFILE_ID,
    SENSENOVA_U15_PROFILE_ID,
])
def test_multi_image_profile_resolves_second_image_from_layer(profile_id):
    stack, layer = _stack_with_instruct_layer()
    reference = next(item for item in stack.all_layers() if item is not layer)
    reference.image[:, :] = (90, 80, 70, 255)
    layer.tool.set_profile(profile_id)
    layer.tool.reference_layer_id = reference.id
    layer.tool.reference_layer_name_hint = reference.name
    engine = _Engine()
    controller = InstructGenerationController(
        engine=engine,
        layer_stack=stack,
        composite_below=lambda _layer: _rgba(
            16, 16, (10, 20, 30, 255)),
    )

    event = controller.start_apply(layer)

    assert event.status == "Applying instruction..."
    request = engine.calls[-1][1]
    assert request.reference_image is not None
    assert request.reference_image.getpixel((0, 0)) == (90, 80, 70)
    assert controller.pending_context.reference_image is not None


def test_qwen_start_apply_reports_missing_second_image_layer():
    stack, layer = _stack_with_instruct_layer()
    layer.tool.set_profile(QWEN_IMAGE_EDIT_PROFILE_ID)
    layer.tool.reference_layer_id = "missing-layer"
    layer.tool.reference_layer_name_hint = "Old portrait"
    controller = InstructGenerationController(
        engine=_Engine(),
        layer_stack=stack,
        composite_below=lambda _layer: _rgba(
            16, 16, (10, 20, 30, 255)),
    )

    event = controller.start_apply(layer)

    assert event.status == "AI Edit reference missing: Old portrait"


def test_qwen_request_captures_effective_lora_stack():
    _stack, layer = _stack_with_instruct_layer()
    layer.tool.set_profile(QWEN_IMAGE_EDIT_PROFILE_ID)
    adapters = [adapter.to_dict() for adapter in layer.tool.lora_adapters]
    adapters[0]["weight"] = 0.6
    layer.tool.set_lora_adapters(adapters)
    engine = _Engine()
    controller = InstructGenerationController(
        engine=engine,
        composite_below=lambda _layer: _rgba(
            16, 16, (10, 20, 30, 255)),
    )

    event = controller.start_apply(layer)

    assert event.status == "Applying instruction..."
    request = engine.calls[-1][1]
    assert request.lora_adapters[0]["stable_id"] == "lightning"
    assert request.lora_adapters[0]["weight"] == 0.6


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

    context, image, seed = event.inference_result
    assert (context.layer_id, image, seed) == (layer.id, result, 456)
    assert controller.pending_context is None
