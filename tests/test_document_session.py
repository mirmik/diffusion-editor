from __future__ import annotations

import numpy as np

from diffusion_editor.app.application import EditorApplication, EngineSet
from diffusion_editor.app.dialogs import (
    ApplicationDialogCoordinator,
    UnsavedDecision,
)
from diffusion_editor.document.commands import (
    AddLayerCommand,
    SetLayerNameCommand,
)
from diffusion_editor.document.session import DocumentSession, RecoveryStore


class _Settings:
    def __init__(self, tmp_path):
        self.values = {
            "recovery_dir": str(tmp_path / "recovery"),
            "recovery_interval_seconds": 5,
        }

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, value):
        self.values[key] = value


class _Engine:
    model_info = {}

    def poll_event(self):
        return None

    def shutdown(self):
        pass

    def gpu_available(self):
        return False


class _Canvas:
    def fit_in_view(self):
        pass


class _Dialogs:
    def __init__(self):
        self.unsaved = []
        self.files = []
        self.errors = []
        self.recoveries = []

    def show_unsaved_changes(self, action, callback):
        self.unsaved.append((action, callback))

    def show_file_dialog(self, spec, callback):
        self.files.append((spec, callback))

    def show_error(self, title, message):
        self.errors.append((title, message))

    def show_recovery(self, record, callback):
        self.recoveries.append((record, callback))

    def show_settings_dialog(self, _state, _callback):
        raise AssertionError("unexpected settings dialog")

    def show_grounding_dialog(self, _available, _callback):
        raise AssertionError("unexpected grounding dialog")


def _application(tmp_path) -> EditorApplication:
    engine = _Engine()
    application = EditorApplication(
        settings=_Settings(tmp_path),
        engines=EngineSet(engine, engine, engine, engine, engine),
    )
    application.layer_stack.init_from_image(
        np.zeros((6, 8, 4), dtype=np.uint8))
    application.clear_history()
    application.reset_document_session(None)
    return application


def test_dirty_save_and_undo_back_to_saved_revision(tmp_path):
    application = _application(tmp_path)
    application.document.execute(AddLayerCommand(name="Second"))
    assert application.document_dirty

    path = tmp_path / "saved.deproj"
    application.layer_stack.save_project(str(path))
    application.mark_document_saved(str(path))
    saved_revision = application.session.saved_revision
    assert not application.document_dirty

    layer = application.layer_stack.active_layer
    application.document.execute(SetLayerNameCommand(layer, "Changed"))
    assert application.document_dirty
    assert application.document.undo() == "Rename Layer"
    assert not application.document_dirty
    assert application.session.saved_revision >= saved_revision
    application.close()


def test_unsaved_cancel_discard_and_save_failure_are_non_destructive(tmp_path):
    application = _application(tmp_path)
    coordinator = ApplicationDialogCoordinator(application, _Canvas())
    dialogs = _Dialogs()
    coordinator.bind_view(dialogs)
    application.document.execute(AddLayerCommand(name="Unsaved"))
    original_session = application.document_session_id

    coordinator.new_project()
    _action, finish = dialogs.unsaved.pop()
    finish(UnsavedDecision.CANCEL)
    assert application.document_session_id == original_session
    assert len(application.layer_stack.layers) == 2

    coordinator.new_project()
    _action, finish = dialogs.unsaved.pop()
    finish(UnsavedDecision.DISCARD)
    assert application.document_session_id != original_session
    assert len(application.layer_stack.layers) == 1

    application.document.execute(AddLayerCommand(name="Still unsaved"))
    application.session.path = str(tmp_path / "cannot-save.deproj")
    original_save = application.layer_stack.save_project
    application.layer_stack.save_project = lambda _path: (_ for _ in ()).throw(
        OSError("disk full"))
    current_session = application.document_session_id
    coordinator.request_quit()
    _action, finish = dialogs.unsaved.pop()
    finish(UnsavedDecision.SAVE)
    assert application.running
    assert application.document_session_id == current_session
    assert application.document_dirty
    assert dialogs.errors[-1][0] == "Save Project"
    application.layer_stack.save_project = original_save
    coordinator.close()
    application.close()


def test_periodic_recovery_is_complete_bounded_and_restored_as_unsaved(tmp_path):
    application = _application(tmp_path)
    application.document.execute(AddLayerCommand(name="Recovered"))
    dirty_session = application.document_session_id
    application._next_recovery_at = 0
    application.poll()
    record = application.available_recovery
    assert record is not None
    assert record.session_id == dirty_session

    restored = _application(tmp_path)
    discovered = restored.available_recovery
    assert discovered is not None
    restored.restore_recovery(discovered)
    assert [layer.name for layer in restored.layer_stack.layers][0] == "Recovered"
    assert restored.project_path is None
    assert restored.document_dirty
    assert restored.document_session_id != dirty_session
    assert restored.available_recovery is None
    application.close()
    restored.close()


def test_recovery_store_rejects_oversized_and_prunes_old_records(tmp_path):
    store = RecoveryStore(tmp_path, max_snapshot_bytes=4, max_records=1)
    first = DocumentSession()
    second = DocumentSession()
    store.write(first, b"one")
    store.write(second, b"two")
    assert [record.session_id for record in store.records()] == [
        second.session_id]
    try:
        store.write(second, b"large")
    except ValueError as exc:
        assert "size limit" in str(exc)
    else:
        raise AssertionError("oversized recovery snapshot was accepted")
