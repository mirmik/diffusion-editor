from __future__ import annotations

from dataclasses import replace

import numpy as np
from PIL import Image
import pytest
from termin.gui_native import (
    PointerEvent,
    PointerEventType,
    Rect,
    tc_ui_document_create,
    tc_ui_document_destroy,
)

from diffusion_editor.app.application import EditorApplication, EngineSet
from diffusion_editor.app.generation_panels import (
    GenerationAction,
    GenerationIntent,
    GenerationPanelKind,
    GenerationPanelsCoordinator,
    GenerationPhase,
)
from diffusion_editor.app.native_generation_panels import (
    NativeGenerationPanels,
)
from diffusion_editor.app.presentation import PanelUpdate, ViewPorts
from diffusion_editor.document.layer import Layer
from diffusion_editor.document.change_event import DocumentChangeKind
from diffusion_editor.document.tool import (
    DiffusionTool,
    InstructTool,
    LamaTool,
)
from diffusion_editor.generation.types import (
    DiffusionInferenceResult,
    EnginePollEvent,
)
from diffusion_editor.generation.image_edit_profiles import (
    FLUX2_KLEIN_PROFILE_ID,
    QWEN_IMAGE_EDIT_PROFILE_ID,
    image_edit_profile,
)


class _Settings:
    def __init__(self, models_dir):
        self.values = {"models_dir": str(models_dir)}

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, value):
        self.values[key] = value


class _Engine:
    def __init__(self, *, loaded=True, model_path="model.safetensors"):
        self.is_busy = False
        self.is_loaded = loaded
        self.model_path = model_path if loaded else None
        self.ip_adapter_loaded = False
        self.model_info = {}
        self.calls = []
        self.poll_result = None

    def submit_load(self, *args):
        self.calls.append(("load", *args))
        return True

    def submit_load_ip_adapter(self):
        self.calls.append(("load_ip_adapter",))
        return True

    def submit_request(self, request):
        self.calls.append(("submit_request", request))
        return True

    def poll_event(self):
        result, self.poll_result = self.poll_result, None
        return result

    def shutdown(self):
        pass


class _Canvas:
    def __init__(self):
        self.mask_brush = None
        self.mask_eraser = False
        self.show_mask = True

    def set_mask_brush(self, size, hardness, flow):
        self.mask_brush = (size, hardness, flow)

    def set_mask_eraser(self, value):
        self.mask_eraser = value

    def set_show_mask(self, value):
        self.show_mask = value


def _rgba(size=16, alpha=255):
    image = np.zeros((size, size, 4), dtype=np.uint8)
    image[:, :, 3] = alpha
    return image


def _diffusion_tool():
    return DiffusionTool(
        source_patch=None,
        patch_x=0,
        patch_y=0,
        patch_w=16,
        patch_h=16,
        prompt="initial",
        negative_prompt="negative",
        strength=0.3,
        guidance_scale=7.0,
        steps=20,
        seed=1,
        model_path="model.safetensors",
        mode="txt2img",
    )


def _instruct_tool():
    return InstructTool(
        source_patch=Image.fromarray(_rgba()[:, :, :3], "RGB"),
        patch_x=0,
        patch_y=0,
        patch_w=16,
        patch_h=16,
        instruction="initial instruction",
        image_guidance_scale=1.5,
        guidance_scale=7.0,
        steps=20,
        seed=2,
    )


def _qwen_edit_tool():
    return InstructTool(
        source_patch=Image.fromarray(_rgba()[:, :, :3], "RGB"),
        patch_x=0,
        patch_y=0,
        patch_w=16,
        patch_h=16,
        model_profile_id=QWEN_IMAGE_EDIT_PROFILE_ID,
    )


def _application(tmp_path):
    diffusion = _Engine()
    instruct = _Engine()
    lama = _Engine()
    segmentation = _Engine()
    grounding = _Engine()
    application = EditorApplication(
        settings=_Settings(tmp_path),
        engines=EngineSet(
            diffusion,
            segmentation,
            lama,
            instruct,
            grounding,
        ),
    )
    application.layer_stack.init_from_image(_rgba())
    return application, diffusion, instruct, lama


def _insert_tool_layer(application, name, tool):
    layer = Layer(name, 16, 16, _rgba(alpha=0))
    layer.tool = tool
    application.layer_stack.insert_layer(layer)
    return layer


def test_generation_coordinator_projects_tools_and_preserves_layer_drafts(
    tmp_path,
):
    (tmp_path / "a.safetensors").write_bytes(b"")
    (tmp_path / "flux-ignore.safetensors").write_bytes(b"")
    application, _diffusion, _instruct, _lama = _application(tmp_path)
    canvas = _Canvas()
    diff = _insert_tool_layer(application, "Diff", _diffusion_tool())
    other = _insert_tool_layer(application, "Reference", None)
    application.layer_stack.active_layer = diff
    coordinator = GenerationPanelsCoordinator(
        application, canvas, random_seed=lambda: 77)

    assert coordinator.state.active_kind == GenerationPanelKind.DIFFUSION
    assert [item.name for item in coordinator.state.models] == [
        "a.safetensors"]
    assert coordinator.state.reference_layers[0].stable_id == other.id
    coordinator.handle_intent(GenerationIntent(
        GenerationAction.SELECT_MODEL,
        str(tmp_path / "a.safetensors"),
    ))
    coordinator.handle_intent(GenerationIntent(
        GenerationAction.LOAD_MODEL))
    assert _diffusion.calls[-1][:2] == (
        "load", str(tmp_path / "a.safetensors"))

    coordinator.handle_intent(GenerationIntent(
        GenerationAction.SET_PROMPT, "draft"))
    coordinator.handle_intent(GenerationIntent(
        GenerationAction.SET_STRENGTH, 0.8))
    application.layer_stack.mark_layer_dirty(diff)

    assert diff.tool.prompt == "initial"
    assert coordinator.state.diffusion.prompt == "draft"
    assert coordinator.state.diffusion.strength == 0.8

    application.layer_stack.active_layer = other
    assert coordinator.state.active_kind == GenerationPanelKind.NONE
    application.layer_stack.active_layer = diff
    assert coordinator.state.diffusion.prompt == "draft"


def test_ai_edit_selects_layer_or_embeds_external_reference(tmp_path):
    application, _diffusion, _instruct, _lama = _application(tmp_path)
    reference = application.layer_stack.active_layer
    layer = _insert_tool_layer(application, "AI Edit", _qwen_edit_tool())
    coordinator = GenerationPanelsCoordinator(application, _Canvas())

    assert coordinator.state.instruct.supports_reference_image
    coordinator.handle_intent(GenerationIntent(
        GenerationAction.SELECT_IMAGE_EDIT_REFERENCE,
        reference.id,
    ))
    assert layer.tool.reference_layer_id == reference.id
    assert layer.tool.reference_image is None
    assert coordinator.state.instruct.reference_label == (
        f"Layer: {reference.name}")

    path = tmp_path / "second.png"
    Image.new("RGB", (9, 7), (11, 22, 33)).save(path)
    coordinator.set_reference_file_picker(
        lambda selected: selected(str(path)))
    coordinator.handle_intent(GenerationIntent(
        GenerationAction.PICK_IMAGE_EDIT_REFERENCE))
    assert layer.tool.reference_layer_id is None
    assert layer.tool.reference_image.size == (9, 7)
    assert layer.tool.reference_image.getpixel((0, 0)) == (11, 22, 33)
    assert coordinator.state.instruct.reference_label == "External: second.png"

    coordinator.handle_intent(GenerationIntent(
        GenerationAction.CLEAR_IMAGE_EDIT_REFERENCE))
    assert layer.tool.reference_image is None
    assert coordinator.state.instruct.reference_label == "None"


def test_generation_intents_update_mask_reference_and_run_settings(tmp_path):
    application, diffusion, _instruct, _lama = _application(tmp_path)
    canvas = _Canvas()
    reference = application.layer_stack.active_layer
    layer = _insert_tool_layer(application, "Diff", _diffusion_tool())
    coordinator = GenerationPanelsCoordinator(
        application, canvas, random_seed=lambda: 1234)

    changes = (
        (GenerationAction.SET_PROMPT, "new prompt"),
        (GenerationAction.SET_NEGATIVE_PROMPT, "new negative"),
        (GenerationAction.SET_MODE, "txt2img"),
        (GenerationAction.SET_MASKED_CONTENT, "latent_noise"),
        (GenerationAction.SET_STRENGTH, 0.55),
        (GenerationAction.SET_STEPS, 31),
        (GenerationAction.SET_GUIDANCE, 8.5),
        (GenerationAction.SET_RESIZE, True),
        (GenerationAction.SET_SEED, "bad seed"),
        (GenerationAction.SELECT_PREDICTION, "epsilon"),
        (GenerationAction.SET_IP_SCALE, 0.7),
    )
    for action, value in changes:
        coordinator.handle_intent(GenerationIntent(action, value))
    coordinator.handle_intent(GenerationIntent(
        GenerationAction.SELECT_IP_REFERENCE, reference.id))
    coordinator.handle_intent(GenerationIntent(
        GenerationAction.SET_MASK_SIZE, 73))
    coordinator.handle_intent(GenerationIntent(
        GenerationAction.SET_MASK_HARDNESS, 0.6))
    coordinator.handle_intent(GenerationIntent(
        GenerationAction.SET_MASK_FLOW, 0.4))
    coordinator.handle_intent(GenerationIntent(
        GenerationAction.SET_MASK_ERASER, True))
    coordinator.handle_intent(GenerationIntent(
        GenerationAction.SET_SHOW_MASK, False))

    assert layer.tool.ip_adapter_layer_id == reference.id
    assert canvas.mask_brush == (73, 0.6, 0.4)
    assert canvas.mask_eraser is True
    assert canvas.show_mask is False

    diffusion.ip_adapter_loaded = True
    coordinator.handle_intent(GenerationIntent(GenerationAction.RUN))
    tool = layer.tool
    assert tool.prompt == "new prompt"
    assert tool.negative_prompt == "new negative"
    assert tool.masked_content == "latent_noise"
    assert tool.strength == 0.55
    assert tool.steps == 31
    assert tool.guidance_scale == 8.5
    assert tool.resize_to_model_resolution is True
    assert tool.seed == 1234
    assert tool.prediction_type == "epsilon"
    assert tool.ip_adapter_scale == 0.7
    assert diffusion.calls[-1][0] == "submit_request"
    assert coordinator.state.diffusion.phase == GenerationPhase.RUNNING
    coordinator.handle_intent(GenerationIntent(
        GenerationAction.SELECT_BACKGROUND))
    assert application.engines.segmentation.calls[-1][0] == (
        "submit_request")


def test_generation_switches_lama_and_instruct_and_clears_mask(tmp_path):
    application, _diffusion, _instruct, lama = _application(tmp_path)
    canvas = _Canvas()
    lama_layer = _insert_tool_layer(
        application,
        "LaMa",
        LamaTool(None, 0, 0, 16, 16),
    )
    lama_layer.mask.data[4:8, 4:8] = 1.0
    coordinator = GenerationPanelsCoordinator(application, canvas)

    assert coordinator.state.active_kind == GenerationPanelKind.LAMA
    coordinator.handle_intent(GenerationIntent(GenerationAction.RUN))
    assert lama.calls[-1][0] == "submit_request"
    assert coordinator.state.lama.phase == GenerationPhase.RUNNING
    coordinator.handle_intent(GenerationIntent(
        GenerationAction.CLEAR_MASK))
    assert not lama_layer.has_mask()

    instruct_layer = _insert_tool_layer(
        application, "Instruct", _instruct_tool())
    coordinator.handle_intent(GenerationIntent(
        GenerationAction.SET_INSTRUCTION, "make it blue"))
    coordinator.handle_intent(GenerationIntent(
        GenerationAction.SET_IMAGE_GUIDANCE, 2.1))
    coordinator.handle_intent(GenerationIntent(
        GenerationAction.SET_GUIDANCE, 9.0))
    coordinator.handle_intent(GenerationIntent(
        GenerationAction.SET_STEPS, 27))
    coordinator.handle_intent(GenerationIntent(
        GenerationAction.SET_SEED, "44"))
    coordinator.handle_intent(GenerationIntent(GenerationAction.RUN))

    tool = instruct_layer.tool
    assert tool.instruction == "make it blue"
    assert tool.image_guidance_scale == 2.1
    assert tool.guidance_scale == 9.0
    assert tool.steps == 27
    assert tool.seed == 44
    assert coordinator.state.active_kind == GenerationPanelKind.INSTRUCT


def test_generation_panel_state_machine_accepts_async_updates(tmp_path):
    application, diffusion, _instruct, _lama = _application(tmp_path)
    diffusion.is_loaded = False
    diffusion.model_path = None
    layer = _insert_tool_layer(application, "Diff", _diffusion_tool())
    coordinator = GenerationPanelsCoordinator(application, _Canvas())
    assert coordinator.state.diffusion.phase == GenerationPhase.IDLE
    assert coordinator.state.lama.phase == GenerationPhase.IDLE

    updates = (
        ("model-loading", GenerationPhase.LOADING),
        ("model-loaded", GenerationPhase.READY),
        ("running", GenerationPhase.RUNNING),
        ("result", GenerationPhase.RESULT),
        ("model-error", GenerationPhase.ERROR),
    )
    for name, expected in updates:
        coordinator.update_panel(PanelUpdate(
            "diffusion",
            name,
            {"path": "x.safetensors", "status": "working", "error": "bad"},
        ))
        assert coordinator.state.diffusion.phase == expected
    assert coordinator.state.active_layer_id == layer.id

    coordinator.update_panel(PanelUpdate(
        "diffusion",
        "model-loaded",
        {
            "path": "x.safetensors",
            "info": {
                "scheduler": "Euler",
                "prediction_type": "epsilon",
                "algorithm_type": "dpmsolver",
            },
        },
    ))
    assert "scheduler: Euler" in (
        coordinator.state.diffusion.model_diagnostics)

    coordinator.update_panel(PanelUpdate(
        "instruct", "model-loading"))
    assert coordinator.state.instruct.phase == GenerationPhase.LOADING
    coordinator.update_panel(PanelUpdate(
        "instruct", "model-loaded"))
    assert coordinator.state.instruct.phase == GenerationPhase.READY
    coordinator.update_panel(PanelUpdate(
        "instruct", "running", {"status": "Applying"}))
    assert coordinator.state.instruct.phase == GenerationPhase.RUNNING
    coordinator.update_panel(PanelUpdate(
        "instruct", "result"))
    assert coordinator.state.instruct.phase == GenerationPhase.RESULT
    coordinator.update_panel(PanelUpdate(
        "instruct", "inference-error", {"error": "failed"}))
    assert coordinator.state.instruct.phase == GenerationPhase.ERROR

    coordinator.update_panel(PanelUpdate(
        "lama", "running", {"status": "Removing"}))
    assert coordinator.state.lama.phase == GenerationPhase.RUNNING
    coordinator.update_panel(PanelUpdate("lama", "result"))
    assert coordinator.state.lama.phase == GenerationPhase.RESULT
    coordinator.update_panel(PanelUpdate(
        "lama", "error", {"error": "failed"}))
    assert coordinator.state.lama.phase == GenerationPhase.ERROR


def test_fake_diffusion_engine_result_flows_through_application_poll(tmp_path):
    application, diffusion, _instruct, _lama = _application(tmp_path)
    layer = _insert_tool_layer(application, "Diff", _diffusion_tool())
    coordinator = GenerationPanelsCoordinator(application, _Canvas())
    application.bind_view(ViewPorts(panels=coordinator))
    coordinator.handle_intent(GenerationIntent(GenerationAction.RUN))

    result = Image.fromarray(
        np.full((16, 16, 4), (12, 34, 56, 255), dtype=np.uint8),
        "RGBA",
    )
    diffusion.poll_result = EnginePollEvent(
        task_type="inference",
        result=DiffusionInferenceResult(result, 99),
    )
    application.poll()

    restored = application.layer_stack.find_layer_by_id(layer.id)
    assert tuple(restored.image[0, 0]) == (12, 34, 56, 255)
    assert application.status_text == "Regenerated (seed=99)"
    assert coordinator.state.diffusion.phase == GenerationPhase.RESULT


def test_native_generation_panels_sync_without_feedback_and_edit_all_fields(
    tmp_path,
):
    (tmp_path / "model.safetensors").write_bytes(b"")
    application, _diffusion, _instruct, _lama = _application(tmp_path)
    reference = application.layer_stack.active_layer
    layer = _insert_tool_layer(application, "Diff", _diffusion_tool())
    coordinator = GenerationPanelsCoordinator(application, _Canvas())
    document = tc_ui_document_create()
    panel = NativeGenerationPanels(
        document,
        coordinator.state,
        coordinator.handle_intent,
        lambda: None,
    )
    coordinator.bind_view(panel)
    assert document.add_root(panel.widget.handle)

    panel.diffusion_prompt.text = "native prompt"
    panel.diffusion_negative_prompt.text = "native negative"
    panel.mode_combo.selected_index = 1
    panel.masked_content_combo.selected_index = 2
    panel.diffusion_strength.value = 0.65
    panel.diffusion_steps.value = 26
    panel.diffusion_guidance.value = 8.0
    panel.diffusion_seed.text = "55"
    panel.diffusion_resize.checked = True
    panel.ip_scale.value = 0.72
    panel.ip_reference_combo.selected_index = 1

    state = coordinator.state.diffusion
    assert state.prompt == "native prompt"
    assert state.negative_prompt == "native negative"
    assert state.mode == "img2img"
    assert state.masked_content == "latent_noise"
    assert state.strength == pytest.approx(0.65)
    assert state.steps == 26
    assert state.guidance_scale == 8.0
    assert state.seed_text == "55"
    assert state.resize_to_model_resolution
    assert state.ip_adapter_scale == pytest.approx(0.72)
    assert layer.tool.ip_adapter_layer_id == reference.id
    coordinator.update_panel(PanelUpdate(
        "diffusion", "running", {"status": "Working"}))
    assert not panel.regenerate_button.widget.enabled
    coordinator.update_panel(PanelUpdate(
        "diffusion", "inference-error", {"error": "failed"}))
    assert panel.regenerate_button.widget.enabled

    instruct_layer = _insert_tool_layer(
        application, "Instruct", _instruct_tool())
    assert panel.instruct_group.widget.visible
    assert not panel.diffusion_group.widget.visible
    panel.instruct_text.text = "native instruction"
    panel.instruct_image_guidance.value = 2.2
    panel.instruct_guidance.value = 10.0
    panel.instruct_steps.value = 33
    panel.instruct_seed.text = "66"
    state = coordinator.state.instruct
    assert state.instruction == "native instruction"
    assert state.image_guidance_scale == pytest.approx(2.2)
    assert state.guidance_scale == 10.0
    assert state.steps == 33
    assert state.seed_text == "66"

    application.layer_stack.active_layer = reference
    assert panel.empty_label.visible
    application.layer_stack.active_layer = instruct_layer
    assert panel.instruct_group.widget.visible
    instruct_layer.tool = None
    application.layer_stack.publish_change(
        DocumentChangeKind.METADATA, layers=(instruct_layer,))
    assert panel.empty_label.visible

    panel.close()
    coordinator.close()
    tc_ui_document_destroy(document)


def test_native_generation_panel_expands_inside_shell_owned_scroll(tmp_path):
    application, _diffusion, _instruct, _lama = _application(tmp_path)
    _insert_tool_layer(application, "Diff", _diffusion_tool())
    coordinator = GenerationPanelsCoordinator(application, _Canvas())
    document = tc_ui_document_create()
    panel = NativeGenerationPanels(
        document,
        coordinator.state,
        coordinator.handle_intent,
        lambda: None,
    )
    left_content = document.create_vstack("TestLeftPanelContent")
    left_content.add_preferred_child(panel.widget)
    left_scroll = document.create_scroll_area("TestLeftPanelScroll")
    left_scroll.set_scroll_axes(False, True)
    left_scroll.set_content(left_content)
    assert document.add_root(left_scroll.handle)

    try:
        document.layout_roots(Rect(0.0, 0.0, 500.0, 700.0))
        widgets = {
            item["stable_id"]: item
            for item in document.inspect_snapshot()["widgets"]
            if item["stable_id"]
        }
        panel_bounds = widgets[
            "diffusion-editor.generation-panels"]["bounds"]
        clear_bounds = widgets[
            "diffusion-editor.generation.diffusion.clear-mask"]["bounds"]

        assert clear_bounds.y + clear_bounds.height <= (
            panel_bounds.y + panel_bounds.height
        )
        assert left_scroll.content_size.height > left_scroll.widget.bounds.height

        prompt_bounds = widgets[
            "diffusion-editor.generation.diffusion.prompt"]["bounds"]
        wheel = PointerEvent()
        wheel.type = PointerEventType.Wheel
        wheel.x = prompt_bounds.x + prompt_bounds.width * 0.5
        wheel.y = prompt_bounds.y + prompt_bounds.height * 0.5
        wheel.wheel_y = -1.0
        document.dispatch_pointer_event(wheel)

        assert left_scroll.scroll_y > 0.0
    finally:
        panel.close()
        coordinator.close()
        tc_ui_document_destroy(document)


def test_native_generation_programmatic_apply_is_feedback_free(tmp_path):
    application, _diffusion, _instruct, _lama = _application(tmp_path)
    _insert_tool_layer(application, "Diff", _diffusion_tool())
    coordinator = GenerationPanelsCoordinator(application, _Canvas())
    document = tc_ui_document_create()
    intents = []
    panel = NativeGenerationPanels(
        document, coordinator.state, intents.append, lambda: None)
    assert document.add_root(panel.widget.handle)

    panel.apply_generation_panels_state(replace(
        coordinator.state,
        diffusion=replace(
            coordinator.state.diffusion,
            prompt="programmatic",
            steps=42,
        ),
    ))

    assert intents == []
    assert panel.diffusion_prompt.text == "programmatic"
    assert panel.diffusion_steps.value == 42
    panel.close()
    coordinator.close()
    tc_ui_document_destroy(document)


def test_ai_edit_panel_exposes_complete_schema_without_advanced_sections(
    tmp_path,
):
    application, _diffusion, _instruct, _lama = _application(tmp_path)
    _insert_tool_layer(application, "AI Edit", _qwen_edit_tool())
    coordinator = GenerationPanelsCoordinator(application, _Canvas())
    document = tc_ui_document_create()
    panel = NativeGenerationPanels(
        document,
        coordinator.state,
        coordinator.handle_intent,
        lambda: None,
    )
    coordinator.bind_view(panel)
    assert document.add_root(panel.widget.handle)
    try:
        qwen_ids = {
            parameter.stable_id
            for parameter in image_edit_profile(
                QWEN_IMAGE_EDIT_PROFILE_ID).parameters
        }
        assert set(panel._image_edit_widgets) >= qwen_ids
        assert all(
            panel._image_edit_widgets[parameter_id].widget.visible
            for parameter_id in panel._image_edit_widgets
        )
        assert all(
            panel._image_edit_widgets[parameter_id].widget.enabled
            for parameter_id in qwen_ids
        )
        assert panel.image_edit_reference_combo.widget.enabled
        assert panel.image_edit_reference_browse.widget.enabled
        assert not panel._image_edit_widgets[
            "image_guidance_scale"].widget.enabled

        coordinator.handle_intent(GenerationIntent(
            GenerationAction.SET_IMAGE_EDIT_PARAMETER,
            ("prompt", "qwen prompt"),
        ))
        coordinator.handle_intent(GenerationIntent(
            GenerationAction.SELECT_IMAGE_EDIT_PROFILE,
            FLUX2_KLEIN_PROFILE_ID,
        ))
        coordinator.handle_intent(GenerationIntent(
            GenerationAction.SET_IMAGE_EDIT_PARAMETER,
            ("prompt", "flux prompt"),
        ))
        assert coordinator.state.instruct.model_profile_id == (
            FLUX2_KLEIN_PROFILE_ID)
        assert coordinator.state.instruct.instruction == "flux prompt"
        assert not panel.image_edit_reference_combo.widget.enabled
        assert not panel.image_edit_reference_browse.widget.enabled

        coordinator.handle_intent(GenerationIntent(
            GenerationAction.SELECT_IMAGE_EDIT_PROFILE,
            QWEN_IMAGE_EDIT_PROFILE_ID,
        ))
        assert coordinator.state.instruct.instruction == "qwen prompt"
        assert "Advanced" not in {
            item.get("text", "")
            for item in document.inspect_snapshot()["widgets"]
        }
    finally:
        panel.close()
        coordinator.close()
        tc_ui_document_destroy(document)
