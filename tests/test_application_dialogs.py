from __future__ import annotations

from dataclasses import replace

import numpy as np
from PIL import Image
import pytest
from termin.gui_native import (
    EventResult,
    KeyCode,
    KeyEvent,
    KeyEventType,
    PointerEvent,
    PointerEventType,
    Rect,
    tc_ui_document_create,
    tc_ui_document_destroy,
)

from diffusion_editor.app.application import EditorApplication, EngineSet
from diffusion_editor.app.dialogs import (
    ApplicationDialogCoordinator,
    FileDialogKind,
    NewDocumentState,
    SettingsState,
    UnsavedDecision,
)
from diffusion_editor.app.native_dialogs import NativeApplicationDialogs
from diffusion_editor.document.commands import AddLayerCommand
from diffusion_editor.document.session import RecoveryRecord


class _Settings:
    def __init__(self, tmp_path):
        self.values = {
            "models_dir": str(tmp_path),
            "last_dir": str(tmp_path),
        }

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, value):
        self.values[key] = value


class _Engine:
    model_info = {}
    is_busy = False

    def __init__(self):
        self.requests = []

    def poll_event(self):
        return None

    def shutdown(self):
        pass

    def gpu_available(self):
        return False

    def submit_request(self, request):
        self.requests.append(request)
        return True


class _Canvas:
    def __init__(self):
        self.fit_calls = 0

    def fit_in_view(self):
        self.fit_calls += 1


class _Dialogs:
    def __init__(self):
        self.files = []
        self.new_documents = []
        self.settings = []
        self.grounding = []
        self.errors = []
        self.unsaved = []
        self.recoveries = []

    def show_new_document_dialog(self, state, callback):
        self.new_documents.append((state, callback))

    def show_file_dialog(self, spec, callback):
        self.files.append((spec, callback))

    def show_settings_dialog(self, state, callback):
        self.settings.append((state, callback))

    def show_grounding_dialog(self, gpu_available, callback):
        self.grounding.append((gpu_available, callback))

    def show_error(self, title, message):
        self.errors.append((title, message))

    def show_unsaved_changes(self, action, callback):
        self.unsaved.append((action, callback))

    def show_recovery(self, record, callback):
        self.recoveries.append((record, callback))


def _application(tmp_path):
    engines = [_Engine() for _ in range(5)]
    app = EditorApplication(
        settings=_Settings(tmp_path),
        engines=EngineSet(*engines),
    )
    image = np.zeros((8, 10, 4), dtype=np.uint8)
    image[:, :, 3] = 255
    app.layer_stack.init_from_image(image)
    app.reset_document_session(None)
    return app, engines[-1]


def test_dialog_coordinator_file_specs_cancel_and_last_directory(tmp_path):
    app, _grounding = _application(tmp_path)
    canvas = _Canvas()
    view = _Dialogs()
    coordinator = ApplicationDialogCoordinator(app, canvas)
    coordinator.bind_view(view)

    coordinator.open_project()
    spec, callback = view.files[-1]
    assert spec.kind == FileDialogKind.OPEN_FILE
    assert spec.directory == str(tmp_path)
    assert "*.deproj" in spec.filters
    callback(None)
    assert app.project_path is None

    selected_references = []
    coordinator.pick_ai_edit_reference(selected_references.append)
    reference_spec, callback = view.files[-1]
    assert reference_spec.kind == FileDialogKind.OPEN_FILE
    assert reference_spec.title == "Select AI Edit Reference"
    assert "*.png" in reference_spec.filters
    callback(str(tmp_path / "reference.png"))
    assert selected_references == [str(tmp_path / "reference.png")]

    image_path = tmp_path / "inputs" / "source.png"
    image_path.parent.mkdir()
    Image.fromarray(
        np.full((5, 7, 4), 128, dtype=np.uint8), "RGBA",
    ).save(image_path)
    coordinator.import_image()
    import_spec, callback = view.files[-1]
    assert import_spec.kind == FileDialogKind.OPEN_FILE
    assert "*.webp" in import_spec.filters
    callback(str(image_path))

    assert (app.layer_stack.width, app.layer_stack.height) == (7, 5)
    assert app.last_dir == str(image_path.parent)
    assert app.settings.values["last_dir"] == str(image_path.parent)
    assert canvas.fit_calls == 1
    assert app.status_text == "Imported: source.png"
    coordinator.close()
    app.close()


def test_new_document_selects_resolution_and_validates_pixel_budget(tmp_path):
    app, _grounding = _application(tmp_path)
    canvas = _Canvas()
    view = _Dialogs()
    coordinator = ApplicationDialogCoordinator(app, canvas)
    coordinator.bind_view(view)
    original_session = app.document_session_id
    original_image = app.layer_stack.active_layer.image.copy()

    coordinator.new_project()
    state, callback = view.new_documents[-1]
    assert state == NewDocumentState(1024, 1024)
    callback(None)
    assert app.document_session_id == original_session
    assert np.array_equal(app.layer_stack.active_layer.image, original_image)

    coordinator.new_project()
    _state, callback = view.new_documents[-1]
    callback(NewDocumentState(640, 480))
    assert (app.layer_stack.width, app.layer_stack.height) == (640, 480)
    assert tuple(app.layer_stack.active_layer.image[0, 0]) == (
        255, 255, 255, 255)
    assert app.status_text == "New 640x480 document"
    assert canvas.fit_calls == 1

    current_session = app.document_session_id
    coordinator.new_project()
    _state, callback = view.new_documents[-1]
    callback(NewDocumentState(65536, 65536))
    assert app.document_session_id == current_session
    assert (app.layer_stack.width, app.layer_stack.height) == (640, 480)
    assert view.errors[-1][0] == "New Document"
    assert "pixel" in view.errors[-1][1]

    coordinator.close()
    app.close()


def test_dialog_coordinator_save_export_and_errors(tmp_path):
    app, _grounding = _application(tmp_path)
    view = _Dialogs()
    coordinator = ApplicationDialogCoordinator(app, _Canvas())
    coordinator.bind_view(view)

    coordinator.save_project_as()
    spec, callback = view.files[-1]
    assert spec.kind == FileDialogKind.SAVE_FILE
    assert spec.file_name == "project.deproj"
    callback(str(tmp_path / "scene"))
    project_path = tmp_path / "scene.deproj"
    assert project_path.is_file()
    assert app.project_path == str(project_path)

    coordinator.export_image()
    spec, callback = view.files[-1]
    assert "*.jpeg" in spec.filters
    callback(str(tmp_path / "render"))
    assert (tmp_path / "render.png").is_file()
    assert app.status_text == "Exported: render.png"

    coordinator.export_image_path(str(tmp_path / "bad.gif"))
    assert view.errors[-1][0] == "Export Image"
    assert "Unknown export extension" in view.errors[-1][1]
    coordinator.close()
    app.close()


def test_open_commit_remains_coherent_when_remembering_directory_fails(
    tmp_path,
):
    app, _grounding = _application(tmp_path)
    project_path = tmp_path / "source.deproj"
    app.layer_stack.save_project(str(project_path))
    app.document.execute(AddLayerCommand(name="Unsaved"))
    old_session = app.document_session_id
    canvas = _Canvas()
    view = _Dialogs()
    coordinator = ApplicationDialogCoordinator(app, canvas)
    coordinator.bind_view(view)
    app.set_last_dir = lambda _directory: (_ for _ in ()).throw(
        RuntimeError("settings write failed"))

    coordinator.open_project_path(str(project_path))
    _action, callback = view.unsaved[-1]
    callback(UnsavedDecision.DISCARD)

    assert app.project_path == str(project_path)
    assert app.document_session_id != old_session
    assert not app.history.can_undo
    assert len(app.layer_stack.layers) == 1
    assert canvas.fit_calls == 1
    assert app.status_text == "Opened: source.deproj"
    assert view.errors == []
    coordinator.close()
    app.close()


def test_dialog_coordinator_settings_and_grounding(tmp_path):
    app, grounding = _application(tmp_path)
    refreshed = []
    view = _Dialogs()
    coordinator = ApplicationDialogCoordinator(
        app, _Canvas(), on_models_dir_changed=lambda: refreshed.append(True))
    coordinator.bind_view(view)

    coordinator.show_settings()
    state, callback = view.settings[-1]
    assert state.models_dir == str(tmp_path)
    assert state.mcp_server_enabled is False
    callback(replace(
        state,
        models_dir=str(tmp_path / "models"),
        history_limit_gib=1.5,
        mcp_server_enabled=True,
        agent_model="local",
        agent_temperature=0.2,
        agent_stream=False,
    ))
    assert app.models_dir == str(tmp_path / "models")
    assert app.last_dir == str(tmp_path / "models")
    assert app.history_memory_limit_bytes == int(1.5 * 1024**3)
    assert app.settings.values["agent_model"] == "local"
    assert app.settings.values["agent_temperature"] == 0.2
    assert app.settings.values["agent_stream"] is False
    assert app.settings.values["mcp_server_enabled"] is True
    assert "restart Diffusion Editor" in app.status_text
    assert refreshed == [True]

    coordinator.show_grounding()
    available, callback = view.grounding[-1]
    assert available is False
    from diffusion_editor.grounding.types import GroundingParams
    params = GroundingParams(
        prompt="cat.",
        model_id="dino",
        box_threshold=0.4,
        text_threshold=0.3,
        use_gpu=False,
        sam2_model_id=None,
        sam2_mask_channel=0,
        mask_threshold=0.0,
        max_hole_area=0,
        max_sprinkle_area=0,
        multimask=True,
        non_overlap=False,
    )
    callback(params)
    assert grounding.requests[-1].params == params
    assert app.status_text == "Grounding: detecting..."
    coordinator.close()
    app.close()


def _settings_state(tmp_path):
    return SettingsState(
        models_dir=str(tmp_path),
        history_limit_gib=5.0,
        mcp_server_enabled=False,
        agent_base_url="http://localhost:8080",
        agent_api_key="",
        agent_model="default",
        agent_temperature=0.7,
        agent_max_tokens=1024,
        agent_timeout_seconds=60,
        agent_stream=True,
    )


def test_native_settings_accept_cancel_reopen_and_destroy(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    document = tc_ui_document_create()
    service = NativeApplicationDialogs(
        document,
        lambda: Rect(0.0, 0.0, 800.0, 700.0),
        lambda: None,
    )
    results = []
    try:
        state = replace(
            _settings_state(tmp_path), models_dir=str(models_dir))
        service.show_settings_dialog(state, results.append)
        assert service.settings_dialog.open
        assert document.overlay_count == 1
        service._browse_models_directory()
        directory_dialog = service._file_dialogs[
            FileDialogKind.OPEN_DIRECTORY]
        assert directory_dialog.open
        assert document.overlay_count == 2
        assert directory_dialog.activate("accept")
        assert service.settings_models_dir.text == str(models_dir)
        assert document.overlay_count == 1
        assert service.settings_dialog.activate("cancel")
        assert results == [None]
        assert document.overlay_count == 0

        service.show_settings_dialog(state, results.append)
        service.settings_agent_model.text = "native-model"
        service.settings_history.value = 2.0
        service.settings_mcp.checked = True
        assert service.settings_dialog.activate("ok")
        assert results[-1].agent_model == "native-model"
        assert results[-1].history_limit_gib == 2.0
        assert results[-1].mcp_server_enabled is True

        service.show_settings_dialog(state, results.append)
        service.close()
        assert not service.settings_dialog.open
        assert document.overlay_count == 0
    finally:
        tc_ui_document_destroy(document)


def test_native_new_document_accept_cancel_and_reopen():
    document = tc_ui_document_create()
    service = NativeApplicationDialogs(
        document,
        lambda: Rect(0.0, 0.0, 800.0, 600.0),
        lambda: None,
    )
    results = []
    try:
        service.show_new_document_dialog(
            NewDocumentState(1024, 768), results.append)
        assert service.new_document_dialog.open
        assert service.new_document_width.value == 1024
        assert service.new_document_height.value == 768
        assert service.new_document_dialog.activate("cancel")
        assert results == [None]

        service.show_new_document_dialog(
            NewDocumentState(800, 600), results.append)
        service.new_document_width.value = 640
        service.new_document_height.value = 480
        assert service.new_document_dialog.activate("create")
        assert results[-1] == NewDocumentState(640, 480)
        assert document.overlay_count == 0
    finally:
        service.close()
        tc_ui_document_destroy(document)


def test_native_dialogs_preserve_legacy_labels_and_compact_settings(tmp_path):
    document = tc_ui_document_create()
    service = NativeApplicationDialogs(
        document,
        lambda: Rect(0.0, 0.0, 800.0, 700.0),
        lambda: None,
    )
    try:
        assert service.settings_content.preferred_size.width == 520.0
        assert service.settings_content.preferred_size.height == 0.0
        assert service.settings_history_note.text == (
            "Older history entries are removed when the limit is exceeded.")
        assert service.settings_mcp_label.text == (
            "Enable local editor MCP server on startup")
        assert service.settings_mcp_note.text == (
            "Applies after restart. Allows local scripts to control the editor.")
        assert service.settings_temperature_caption.text == "Temperature"
        assert service.settings_max_tokens_caption.text == "Max tokens"
        assert service.settings_timeout_caption.text == "Timeout sec"

        assert service.grounding_sam_caption.text == (
            "SAM 2.1 segmentation (experimental)")
        assert service.grounding_mask_threshold.label == (
            "Mask threshold (higher = tighter)")
        assert service.grounding_max_hole.label == (
            "Max hole area (0 = off, px)")
        assert service.grounding_max_sprinkle.label == (
            "Max sprinkle area (0 = off, px)")
        assert service.grounding_multimask_label.text == (
            "Multimask output (3 candidates per box)")
    finally:
        service.close()
        tc_ui_document_destroy(document)


def test_native_file_dialog_modal_routing_cancel_accept_and_reopen(tmp_path):
    text_path = tmp_path / "readme.txt"
    text_path.write_text("hello", encoding="utf-8")
    document = tc_ui_document_create()
    root = document.create_vstack("UnderlyingRoot")
    button = document.create_button("Under")
    activations = []
    button.connect_clicked(lambda: activations.append(True))
    root.add_flex_child(button.widget, 1.0)
    assert document.add_root(root.handle)
    document.layout_roots(Rect(0.0, 0.0, 800.0, 600.0))
    service = NativeApplicationDialogs(
        document,
        lambda: Rect(0.0, 0.0, 800.0, 600.0),
        lambda: None,
    )
    results = []
    from diffusion_editor.app.dialogs import FileDialogSpec
    spec = FileDialogSpec(
        FileDialogKind.OPEN_FILE,
        "Open",
        str(tmp_path),
        "Text | *.txt",
    )
    try:
        service.show_file_dialog(spec, results.append)
        assert document.overlay_count == 1
        event = PointerEvent()
        event.type = PointerEventType.Down
        event.x = 10.0
        event.y = 10.0
        assert document.dispatch_pointer_event(event) == EventResult.Handled
        event.type = PointerEventType.Up
        assert document.dispatch_pointer_event(event) == EventResult.Handled
        assert activations == []

        escape = KeyEvent()
        escape.type = KeyEventType.Down
        escape.key = KeyCode.Escape
        assert document.dispatch_key_event(escape) == EventResult.Handled
        assert results == [None]

        service.show_file_dialog(spec, results.append)
        dialog = service._file_dialogs[FileDialogKind.OPEN_FILE]
        index = next(
            index for index, entry in enumerate(dialog.model.entries)
            if entry.name == "readme.txt")
        assert dialog.model.select(index)
        assert dialog.activate("accept")
        assert results[-1] == str(text_path)
    finally:
        service.close()
        assert document.overlay_count == 0
        tc_ui_document_destroy(document)


def test_native_grounding_accept_cancel_validation_and_reopen(tmp_path):
    document = tc_ui_document_create()
    service = NativeApplicationDialogs(
        document,
        lambda: Rect(0.0, 0.0, 800.0, 700.0),
        lambda: None,
    )
    results = []
    try:
        service.show_grounding_dialog(False, results.append)
        assert not service.grounding_gpu.widget.enabled
        service.grounding_prompt.text = "cat"
        assert service.grounding_dialog.activate("detect")
        assert results[-1].prompt == "cat."
        assert not results[-1].use_gpu

        service.show_grounding_dialog(True, results.append)
        assert service.grounding_gpu.widget.enabled
        assert service.grounding_dialog.activate("cancel")
        assert results[-1] is None

        service.show_grounding_dialog(True, results.append)
        assert service.grounding_dialog.activate("detect")
        assert results[-1] is None
        assert document.overlay_count == 1
    finally:
        service.close()
        assert document.overlay_count == 0
        tc_ui_document_destroy(document)


def test_native_unsaved_and_recovery_dialogs_report_typed_decisions(tmp_path):
    document = tc_ui_document_create()
    service = NativeApplicationDialogs(
        document,
        lambda: Rect(0.0, 0.0, 800.0, 600.0),
        lambda: None,
    )
    unsaved = []
    recovered = []
    record = RecoveryRecord(
        session_id="document_test",
        created_at=1.0,
        project_path=None,
        snapshot_path=tmp_path / "document_test.deproj",
        snapshot_bytes=10,
    )
    try:
        service.show_unsaved_changes("quit the editor", unsaved.append)
        dialog = service._lifecycle_dialogs[-1]
        assert dialog.activate("discard")
        assert unsaved == [UnsavedDecision.DISCARD]

        service.show_recovery(record, recovered.append)
        recovery_dialog = service._lifecycle_dialogs[-1]
        assert recovery_dialog.activate("yes")
        assert recovered == [True]
        assert document.overlay_count == 0
    finally:
        service.close()
        tc_ui_document_destroy(document)
