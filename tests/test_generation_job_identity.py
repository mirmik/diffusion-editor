from __future__ import annotations

from collections import deque

import numpy as np
from PIL import Image
import pytest

from diffusion_editor.app.application import EditorApplication, EngineSet
from diffusion_editor.app.dialogs import ApplicationDialogCoordinator
from diffusion_editor.document.commands import (
    DetachLayerToolCommand,
    MoveLayerCommand,
    RemoveLayerCommand,
    SetLayerNameCommand,
)
from diffusion_editor.document.layer import Layer
from diffusion_editor.document.tool import DiffusionTool
from diffusion_editor.generation.job_context import (
    ApplyFrozenGeneratedResultCommand,
)
from diffusion_editor.generation.provenance import (
    FrozenJsonObject,
    GenerationProvenance,
    ModelIdentity,
    ModelIdentityPolicy,
    ModelIdentityStatus,
    RequestProvenance,
)
from diffusion_editor.generation.types import (
    DiffusionInferenceResult,
    EnginePollEvent,
)


def _rgba(width: int, height: int, color) -> np.ndarray:
    array = np.zeros((height, width, 4), dtype=np.uint8)
    array[:] = color
    return array


class _Settings:
    def get(self, _key, default=None):
        return default

    def set(self, _key, _value):
        pass


class _DelayedEngine:
    supports_job_ids = True
    supports_model_identity_policy = True

    def __init__(self):
        self.is_busy = False
        self.is_loaded = True
        self.model_path = "model.safetensors"
        self.ip_adapter_loaded = True
        self.model_info = {}
        self.submissions = []
        self.events = deque()
        self.cancel_calls = 0
        self.shutdown_calls = 0

    def submit_load(
            self,
            path,
            prediction_type=None,
            *,
            expected_content_hash=None,
            model_identity_policy="warn",
            job_id=None):
        self.submissions.append(
            (
                "load",
                job_id,
                path,
                prediction_type,
                expected_content_hash,
                model_identity_policy,
            )
        )
        return True

    def submit_load_ip_adapter(self, *, job_id=None):
        self.submissions.append(("load_ip_adapter", job_id))
        return True

    def submit_request(self, request, *, job_id=None):
        self.submissions.append(("request", job_id, request))
        return True

    def poll_event(self):
        return self.events.popleft() if self.events else None

    def cancel(self):
        self.cancel_calls += 1
        return True

    def shutdown(self):
        self.shutdown_calls += 1

    def gpu_available(self):
        return False


class _Canvas:
    def __init__(self):
        self.fit_calls = 0

    def fit_in_view(self):
        self.fit_calls += 1


def _application() -> tuple[EditorApplication, _DelayedEngine]:
    diffusion = _DelayedEngine()
    application = EditorApplication(
        settings=_Settings(),
        engines=EngineSet(
            diffusion=diffusion,
            segmentation=_DelayedEngine(),
            lama=_DelayedEngine(),
            instruct=_DelayedEngine(),
            grounding=_DelayedEngine(),
        ),
    )
    application.layer_stack.init_from_image(
        _rgba(8, 8, (20, 30, 40, 255)))
    return application, diffusion


def _insert_diffusion_layer(
        application: EditorApplication,
        *,
        patch=(0, 0, 8, 8),
) -> Layer:
    x, y, width, height = patch
    layer = Layer(
        "Generated",
        8,
        8,
        _rgba(8, 8, (0, 0, 0, 0)),
    )
    layer.tool = DiffusionTool(
        source_patch=None,
        patch_x=x,
        patch_y=y,
        patch_w=width,
        patch_h=height,
        prompt="prompt",
        negative_prompt="",
        strength=0.5,
        guidance_scale=7.0,
        steps=10,
        seed=1,
        model_path="model.safetensors",
        mode="txt2img",
    )
    application.layer_stack.insert_layer(layer)
    return layer


def _start(
        application: EditorApplication,
        engine: _DelayedEngine,
        layer: Layer,
):
    event = application.diffusion_controller.start_regeneration(layer)
    assert event.status.startswith("Regenerating")
    context = application.diffusion_controller.pending_context
    assert context is not None
    assert context.application_policy.value == "reject_stale"
    assert engine.submissions[-1][0:2] == ("request", context.job_id)
    return context


def _complete(
        engine: _DelayedEngine,
        job_id: str | None,
        *,
        color=(200, 10, 20, 255),
        provenance: GenerationProvenance | None = None,
) -> None:
    engine.events.append(EnginePollEvent(
        task_type="inference",
        result=DiffusionInferenceResult(
            Image.fromarray(_rgba(8, 8, color), "RGBA"),
            123,
            provenance,
        ),
        job_id=job_id,
    ))


def test_completed_result_attaches_model_runtime_and_frozen_request_provenance():
    application, engine = _application()
    layer = _insert_diffusion_layer(application)
    context = _start(application, engine, layer)
    model = ModelIdentity(
        provider="local",
        repository="model.safetensors",
        revision=None,
        content_hash=f"sha256:{'a' * 64}",
        local_override="/models/model.safetensors",
        status=ModelIdentityStatus.CONFIRMED_IMMUTABLE,
    )
    worker_provenance = GenerationProvenance(
        operation="diffusion",
        model=model,
        request=RequestProvenance.capture(
            "diffusion",
            {"prompt": "worker request without input hashes"},
        ),
        seed=123,
        width=8,
        height=8,
        runtime=FrozenJsonObject.capture({
            "pipeline": "FakePipeline",
            "device": "cpu",
        }),
        warnings=("IP-Adapter identity is floating",),
    )

    _complete(
        engine,
        context.job_id,
        provenance=worker_provenance,
    )
    application.poll()

    assert layer.tool.generation_provenance is not None
    assert layer.tool.generation_provenance.model == model
    assert layer.tool.model_identity == model
    assert layer.tool.generation_provenance.request == (
        context.request_provenance
    )
    parameters = (
        layer.tool.generation_provenance.request.parameters.to_dict()
    )
    assert parameters["prompt"] == "prompt"
    assert "input_image_hash" in parameters
    assert "reproducibility warning" in application.status_text


def test_confirmed_model_hash_and_strict_policy_reach_model_load_request():
    application, engine = _application()
    layer = _insert_diffusion_layer(application)
    layer.tool.model_identity = ModelIdentity(
        provider="local",
        repository="model.safetensors",
        revision=None,
        content_hash=f"sha256:{'c' * 64}",
        local_override="/models/model.safetensors",
        status=ModelIdentityStatus.CONFIRMED_IMMUTABLE,
    )
    layer.tool.model_identity_policy = (
        ModelIdentityPolicy.REQUIRE_IMMUTABLE
    )

    event = application.diffusion_controller.start_regeneration(layer)

    assert event.status == "Loading model for regeneration..."
    assert engine.submissions[-1][0] == "load"
    assert engine.submissions[-1][4:] == (
        f"sha256:{'c' * 64}",
        "require_immutable",
    )


def test_strict_policy_rejects_unpinned_ip_adapter_before_inference():
    application, engine = _application()
    layer = _insert_diffusion_layer(application)
    background = next(
        candidate
        for candidate in application.layer_stack.all_layers()
        if candidate is not layer
    )
    layer.tool.ip_adapter_layer_id = background.id
    layer.tool.model_identity_policy = (
        ModelIdentityPolicy.REQUIRE_IMMUTABLE
    )
    engine.model_info = {
        "model_identity": {
            "provider": "local",
            "repository": "model.safetensors",
            "revision": None,
            "content_hash": f"sha256:{'d' * 64}",
            "local_override": "/models/model.safetensors",
            "status": "confirmed_immutable",
            "warning": None,
        },
    }

    event = application.diffusion_controller.start_regeneration(layer)

    assert event.ip_adapter_error is not None
    assert "strict reproducibility" in event.status
    assert not engine.submissions
    assert application.diffusion_controller.pending_context is None


def test_success_resolves_stable_layer_and_creates_one_undoable_command():
    application, engine = _application()
    layer = _insert_diffusion_layer(application)
    context = _start(application, engine, layer)
    revision = application.document_revision

    _complete(engine, context.job_id)
    application.poll()

    resolved = application.layer_stack.find_layer_by_id(context.layer_id)
    assert resolved is layer
    assert tuple(resolved.image[0, 0]) == (200, 10, 20, 255)
    assert application.document_revision == revision + 1
    assert application.document.undo() == "Apply Diffusion Result"
    restored = application.layer_stack.find_layer_by_id(context.layer_id)
    assert np.count_nonzero(restored.image) == 0


@pytest.mark.parametrize("history_action", ["undo", "redo"])
def test_undo_or_redo_after_submit_rejects_delayed_result(history_action):
    application, engine = _application()
    layer = _insert_diffusion_layer(application)
    application.document.execute(SetLayerNameCommand(layer, "Changed"))
    if history_action == "redo":
        assert application.document.undo() == "Rename Layer"
    context = _start(
        application,
        engine,
        application.layer_stack.find_layer_by_id(layer.id),
    )

    if history_action == "undo":
        assert application.document.undo() == "Rename Layer"
    else:
        assert application.document.redo() == "Rename Layer"
    revision_after_history = application.document_revision
    _complete(engine, context.job_id)
    application.poll()

    resolved = application.layer_stack.find_layer_by_id(layer.id)
    assert np.count_nonzero(resolved.image) == 0
    assert application.document_revision == revision_after_history
    assert "revision changed" in application.status_text


def test_new_and_open_rotate_session_even_when_stable_layer_ids_repeat(tmp_path):
    application, engine = _application()
    layer = _insert_diffusion_layer(application)
    project_path = tmp_path / "same.deproj"
    application.layer_stack.save_project(str(project_path))
    coordinator = ApplicationDialogCoordinator(application, _Canvas())

    first = _start(application, engine, layer)
    first_session = application.document_session_id
    coordinator.open_project_path(str(project_path))
    assert application.document_session_id != first_session
    assert application.layer_stack.find_layer_by_id(first.layer_id) is not None
    revision_after_open = application.document_revision
    _complete(engine, first.job_id)
    application.poll()
    reopened = application.layer_stack.find_layer_by_id(first.layer_id)
    assert np.count_nonzero(reopened.image) == 0
    assert application.document_revision == revision_after_open
    assert "document was replaced" in application.status_text

    second = _start(application, engine, reopened)
    second_session = application.document_session_id
    coordinator.new_project()
    assert application.document_session_id != second_session
    revision_after_new = application.document_revision
    _complete(engine, second.job_id)
    application.poll()
    assert application.document_revision == revision_after_new
    assert "document was replaced" in application.status_text


@pytest.mark.parametrize("mutation", ["delete", "reorder", "detach"])
def test_layer_lifecycle_changes_reject_delayed_result_without_extra_history(
        mutation,
):
    application, engine = _application()
    target = _insert_diffusion_layer(application)
    other = Layer(
        "Other", 8, 8, _rgba(8, 8, (0, 0, 0, 0)))
    application.layer_stack.insert_layer(other)
    context = _start(application, engine, target)

    if mutation == "delete":
        application.document.execute(RemoveLayerCommand(target))
    elif mutation == "reorder":
        application.document.execute(MoveLayerCommand(
            layer=target,
            new_parent=None,
            index=len(application.layer_stack.layers),
        ))
    else:
        application.document.execute(DetachLayerToolCommand(target))
    revision_after_mutation = application.document_revision
    _complete(engine, context.job_id)
    application.poll()

    resolved = application.layer_stack.find_layer_by_id(target.id)
    if mutation == "delete":
        assert resolved is None
        expected = "no longer exists"
    elif mutation == "detach":
        assert resolved.tool is None
        assert np.count_nonzero(resolved.image) == 0
        expected = "tool changed"
    else:
        assert np.count_nonzero(resolved.image) == 0
        expected = "revision changed"
    assert application.document_revision == revision_after_mutation
    assert expected in application.status_text


@pytest.mark.parametrize("mutation", ["patch", "mask"])
def test_untracked_patch_or_mask_edit_is_detected_by_frozen_context(mutation):
    application, engine = _application()
    layer = _insert_diffusion_layer(application)
    context = _start(application, engine, layer)
    revision = application.document_revision

    if mutation == "patch":
        layer.tool.patch_x = 1
    else:
        layer.mask.data[2, 2] = 1.0
    _complete(engine, context.job_id)
    application.poll()

    assert np.count_nonzero(layer.image) == 0
    assert application.document_revision == revision
    expected = "geometry" if mutation == "patch" else "mask"
    assert expected in application.status_text


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prompt", "changed prompt"),
        ("seed", 999),
        ("strength", 0.9),
        ("model_path", "different.safetensors"),
    ],
)
def test_untracked_tool_request_edit_rejects_delayed_result(field, value):
    application, engine = _application()
    layer = _insert_diffusion_layer(application)
    context = _start(application, engine, layer)
    revision = application.document_revision

    setattr(layer.tool, field, value)
    _complete(engine, context.job_id)
    application.poll()

    assert np.count_nonzero(layer.image) == 0
    assert application.document_revision == revision
    assert "request settings changed" in application.status_text


@pytest.mark.parametrize("mutation", ["pixel_revision", "external_revision"])
def test_in_flight_external_canvas_mutation_rejects_delayed_result(mutation):
    application, engine = _application()
    layer = _insert_diffusion_layer(application)
    context = _start(application, engine, layer)
    base_revision = application.document_revision

    if mutation == "pixel_revision":
        layer.image[0, 0] = (1, 2, 3, 4)
        layer.mark_pixels_changed()
        expected = "pixels changed"
    else:
        assert application.mark_external_mutation() == base_revision + 1
        expected = "revision changed"
    revision_after_mutation = application.document_revision
    _complete(engine, context.job_id)
    application.poll()

    assert application.document_revision == revision_after_mutation
    assert expected in application.status_text
    assert not application.history.can_undo


def test_wrong_or_missing_terminal_job_id_cannot_consume_active_job():
    application, engine = _application()
    layer = _insert_diffusion_layer(application)
    context = _start(application, engine, layer)
    revision = application.document_revision

    _complete(engine, "job_from_an_older_request")
    application.poll()
    assert application.diffusion_controller.pending_context == context
    assert application.document_revision == revision
    assert "stale job identity" in application.status_text

    _complete(engine, None)
    application.poll()
    assert application.diffusion_controller.pending_context == context
    assert application.document_revision == revision
    assert "stale job identity" in application.status_text

    _complete(engine, context.job_id)
    application.poll()
    assert tuple(layer.image[0, 0]) == (200, 10, 20, 255)


def test_frozen_result_command_never_rereads_mutable_tool_geometry():
    application, engine = _application()
    layer = _insert_diffusion_layer(
        application, patch=(1, 2, 3, 2))
    context = _start(application, engine, layer)
    assert context.paste is not None

    layer.tool.patch_x = 5
    layer.tool.patch_y = 5
    layer.tool.patch_w = 1
    layer.tool.patch_h = 1
    command = ApplyFrozenGeneratedResultCommand(
        layer=layer,
        result_image=Image.new("RGBA", (3, 2), (9, 8, 7, 255)),
        paste=context.paste,
        label="Apply Frozen Test Result",
    )
    application.document.execute(command)

    assert tuple(layer.image[2, 1]) == (9, 8, 7, 255)
    assert tuple(layer.image[5, 5]) == (0, 0, 0, 0)


def test_shutdown_invalidates_job_and_poll_cannot_apply_to_closed_document():
    application, engine = _application()
    layer = _insert_diffusion_layer(application)
    context = _start(application, engine, layer)
    revision = application.document_revision

    application.close()
    _complete(engine, context.job_id)
    application.poll()

    assert engine.cancel_calls == 1
    assert np.count_nonzero(layer.image) == 0
    assert application.document_revision == revision
