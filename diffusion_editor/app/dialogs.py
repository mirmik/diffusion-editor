"""Toolkit-neutral application dialog workflows."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
from PIL import Image
from tcbase import log

from ..agent.config import DEFAULT_AGENT_BASE_URL, DEFAULT_AGENT_MODEL
from ..document.layer_stack import LayerStack
from ..document.session import RecoveryRecord
from ..grounding.types import GroundingParams
from .application import (
    MAX_HISTORY_MEMORY_LIMIT_GIB,
    MIN_HISTORY_MEMORY_LIMIT_GIB,
    EditorApplication,
)


_BYTES_PER_GIB = 1024 * 1024 * 1024
_IMAGE_FILTER = "Images | *.png *.jpg *.jpeg *.bmp *.tiff *.webp"
_PROJECT_FILTER = "Diffusion Editor Project | *.deproj"
_EXPORT_FILTER = "PNG | *.png;;JPEG | *.jpg *.jpeg"
_EXPORT_EXTENSIONS = {".png", ".jpg", ".jpeg"}
DEFAULT_NEW_DOCUMENT_WIDTH = 1024
DEFAULT_NEW_DOCUMENT_HEIGHT = 1024
MIN_NEW_DOCUMENT_DIMENSION = 1
MAX_NEW_DOCUMENT_DIMENSION = LayerStack.MAX_PROJECT_CANVAS_DIMENSION
MAX_NEW_DOCUMENT_PIXELS = LayerStack.MAX_PROJECT_PIXELS


class FileDialogKind(str, Enum):
    OPEN_FILE = "open_file"
    SAVE_FILE = "save_file"
    OPEN_DIRECTORY = "open_directory"


class UnsavedDecision(str, Enum):
    SAVE = "save"
    DISCARD = "discard"
    CANCEL = "cancel"


@dataclass(frozen=True)
class FileDialogSpec:
    kind: FileDialogKind
    title: str
    directory: str = ""
    filters: str = ""
    file_name: str = ""


@dataclass(frozen=True)
class NewDocumentState:
    width: int = DEFAULT_NEW_DOCUMENT_WIDTH
    height: int = DEFAULT_NEW_DOCUMENT_HEIGHT


@dataclass(frozen=True)
class SettingsState:
    models_dir: str
    history_limit_gib: float
    mcp_server_enabled: bool
    agent_base_url: str
    agent_api_key: str
    agent_model: str
    agent_temperature: float
    agent_max_tokens: int
    agent_timeout_seconds: float
    agent_stream: bool


class ApplicationDialogPresentation(Protocol):
    def show_new_document_dialog(
            self,
            state: NewDocumentState,
            on_finished: Callable[[NewDocumentState | None], None]) -> None: ...

    def show_file_dialog(
            self,
            spec: FileDialogSpec,
            on_finished: Callable[[str | None], None]) -> None: ...

    def show_settings_dialog(
            self,
            state: SettingsState,
            on_finished: Callable[[SettingsState | None], None]) -> None: ...

    def show_grounding_dialog(
            self,
            gpu_available: bool,
            on_finished: Callable[[GroundingParams | None], None]) -> None: ...

    def show_error(self, title: str, message: str) -> None: ...

    def show_unsaved_changes(
            self,
            action: str,
            on_finished: Callable[[UnsavedDecision], None]) -> None: ...

    def show_recovery(
            self,
            record: RecoveryRecord,
            on_finished: Callable[[bool], None]) -> None: ...


class ApplicationDialogCoordinator:
    """Bridge native dialog results to application-owned operations."""

    def __init__(
            self,
            application: EditorApplication,
            canvas,
            *,
            on_models_dir_changed: Callable[[], None] | None = None) -> None:
        self._application = application
        self._canvas = canvas
        self._on_models_dir_changed = on_models_dir_changed or (lambda: None)
        self._view: ApplicationDialogPresentation | None = None
        self._closed = False
        self._new_document_state = NewDocumentState()

    @property
    def command_handlers(self) -> dict[str, Callable[[], None]]:
        return {
            "file.new": self.new_project,
            "file.new_from_image": self.new_project_from_image,
            "file.open": self.open_project,
            "file.save": self.save_project,
            "file.save_as": self.save_project_as,
            "file.import": self.import_image,
            "file.export": self.export_image,
            "app.quit": self.request_quit,
            "edit.settings": self.show_settings,
            "layer.detect": self.show_grounding,
        }

    def bind_view(self, view: ApplicationDialogPresentation) -> None:
        self._require_open()
        self._view = view
        record = self._application.available_recovery
        if record is not None:
            view.show_recovery(record, lambda restore: self._finish_recovery(
                record, restore))

    def new_project(self) -> None:
        self._require_open()
        if self._view is None:
            self._confirm_destructive(
                "create a new document",
                lambda: self._new_project_unchecked(
                    self._new_document_state),
            )
            return
        self._view.show_new_document_dialog(
            self._new_document_state,
            self._select_new_document,
        )

    def _select_new_document(
            self, state: NewDocumentState | None) -> None:
        if state is None or self._closed:
            return
        error = self.validate_new_document_state(state)
        if error is not None:
            self._show_error("New Document", error)
            return
        self._confirm_destructive(
            "create a new document",
            lambda: self._new_project_unchecked(state),
        )

    def _new_project_unchecked(self, state: NewDocumentState) -> None:
        self._require_open()
        try:
            white = np.full(
                (state.height, state.width, 4), 255, dtype=np.uint8)
        except (MemoryError, ValueError) as exc:
            self._show_error(
                "New Document",
                f"Could not allocate {state.width}×{state.height} pixels: "
                f"{exc}",
            )
            return
        self._application.document.prepare_mutation()
        self._application.layer_stack.init_from_image(white)
        self._new_document_state = state
        self._commit_document_reset(None)
        self._present_document_reset(None)
        self._best_effort(
            lambda: self._application.set_status(
                f"New {state.width}x{state.height} document"),
            "update status after creating a document",
        )

    @staticmethod
    def validate_new_document_state(
            state: NewDocumentState) -> str | None:
        width = state.width
        height = state.height
        if (
                isinstance(width, bool)
                or isinstance(height, bool)
                or not isinstance(width, int)
                or not isinstance(height, int)):
            return "Width and height must be whole numbers."
        if (
                width < MIN_NEW_DOCUMENT_DIMENSION
                or height < MIN_NEW_DOCUMENT_DIMENSION):
            return "Width and height must be at least 1 pixel."
        if (
                width > MAX_NEW_DOCUMENT_DIMENSION
                or height > MAX_NEW_DOCUMENT_DIMENSION):
            return (
                "Width and height must not exceed "
                f"{MAX_NEW_DOCUMENT_DIMENSION} pixels."
            )
        if width * height > MAX_NEW_DOCUMENT_PIXELS:
            return (
                f"The document contains {width * height:,} pixels; "
                f"the limit is {MAX_NEW_DOCUMENT_PIXELS:,}."
            )
        return None

    def new_project_from_image(self) -> None:
        self._show_file(
            FileDialogSpec(
                FileDialogKind.OPEN_FILE,
                "New From Image",
                self._application.last_dir,
                _IMAGE_FILTER,
            ),
            self._new_from_image_path,
        )

    def open_project(self) -> None:
        self._show_file(
            FileDialogSpec(
                FileDialogKind.OPEN_FILE,
                "Open Project",
                self._application.last_dir,
                _PROJECT_FILTER,
            ),
            self.open_project_path,
        )

    def import_image(self) -> None:
        self._show_file(
            FileDialogSpec(
                FileDialogKind.OPEN_FILE,
                "Import Image",
                self._application.last_dir,
                _IMAGE_FILTER,
            ),
            self.import_image_path,
        )

    def pick_ai_edit_reference(
            self, on_selected: Callable[[str], None]) -> None:
        self._show_file(
            FileDialogSpec(
                FileDialogKind.OPEN_FILE,
                "Select AI Edit Reference",
                self._application.last_dir,
                _IMAGE_FILTER,
            ),
            on_selected,
        )

    def save_project(self) -> None:
        self._require_open()
        if not self._application.project_path:
            self.save_project_as()
            return
        self._save_project_path(self._application.project_path)

    def save_project_as(self) -> None:
        self._show_file(
            FileDialogSpec(
                FileDialogKind.SAVE_FILE,
                "Save Project",
                self._application.last_dir,
                _PROJECT_FILTER,
                self._project_file_name(),
            ),
            self._save_project_path,
        )

    def export_image(self) -> None:
        self._show_file(
            FileDialogSpec(
                FileDialogKind.SAVE_FILE,
                "Export Image",
                self._application.last_dir,
                _EXPORT_FILTER,
                "image.png",
            ),
            self.export_image_path,
        )

    def show_settings(self) -> None:
        self._require_open()
        if self._view is None:
            return
        settings = self._application.settings
        state = SettingsState(
            models_dir=self._application.models_dir,
            history_limit_gib=(
                self._application.history_memory_limit_bytes / _BYTES_PER_GIB),
            mcp_server_enabled=bool(settings.get(
                "mcp_server_enabled", False)),
            agent_base_url=str(settings.get(
                "agent_api_base_url", DEFAULT_AGENT_BASE_URL)),
            agent_api_key=str(settings.get("agent_api_key", "")),
            agent_model=str(settings.get(
                "agent_model", DEFAULT_AGENT_MODEL)),
            agent_temperature=float(settings.get("agent_temperature", 0.7)),
            agent_max_tokens=int(settings.get("agent_max_tokens", 1024)),
            agent_timeout_seconds=float(settings.get(
                "agent_timeout_seconds", 60)),
            agent_stream=bool(settings.get("agent_stream", True)),
        )
        self._view.show_settings_dialog(state, self._apply_settings)

    def show_grounding(self) -> None:
        self._require_open()
        if self._view is None:
            return
        self._view.show_grounding_dialog(
            self._application.grounding_controller.gpu_available(),
            self._submit_grounding,
        )

    def open_project_path(self, path: str) -> None:
        self._confirm_destructive(
            "open another project",
            lambda: self._open_project_path_unchecked(path),
        )

    def _open_project_path_unchecked(self, path: str) -> None:
        self._require_open()
        try:
            self._application.document.prepare_mutation()
            self._application.layer_stack.load_project(path)
            self._commit_document_reset(path)
        except Exception as exc:
            self._operation_error("Open Project", path, exc)
            return
        self._best_effort(
            lambda: self._remember_parent(path),
            "remember project directory",
        )
        self._present_document_reset(path)
        self._best_effort(
            lambda: self._application.set_status(
                f"Opened: {os.path.basename(path)}"),
            "update status after opening a project",
        )

    def import_image_path(self, path: str) -> None:
        self._confirm_destructive(
            "replace the document with an image",
            lambda: self._import_image_path_unchecked(path),
        )

    def _import_image_path_unchecked(self, path: str) -> None:
        self._require_open()
        try:
            image = Image.open(path).convert("RGBA")
            self._application.document.prepare_mutation()
            self._application.layer_stack.init_from_image(
                np.array(image, dtype=np.uint8))
            self._commit_document_reset(None, clean=False)
        except Exception as exc:
            self._operation_error("Import Image", path, exc)
            return
        self._best_effort(
            lambda: self._remember_parent(path),
            "remember imported image directory",
        )
        self._present_document_reset(None)
        self._best_effort(
            lambda: self._application.set_status(
                f"Imported: {os.path.basename(path)}"),
            "update status after importing an image",
        )

    def export_image_path(self, path: str) -> None:
        self._require_open()
        normalized, error = self.normalize_export_path(path)
        if error is not None:
            self._show_error("Export Image", error)
            return
        assert normalized is not None
        try:
            array = np.ascontiguousarray(
                self._application.layer_stack.composite())
            if array.size == 0:
                raise ValueError("nothing to export")
            image = Image.fromarray(array, "RGBA")
            if normalized.lower().endswith((".jpg", ".jpeg")):
                image = image.convert("RGB")
            image.save(normalized)
            self._remember_parent(normalized)
            self._application.set_status(
                f"Exported: {os.path.basename(normalized)}")
        except Exception as exc:
            self._operation_error("Export Image", normalized, exc)

    @staticmethod
    def normalize_export_path(path: str) -> tuple[str | None, str | None]:
        extension = Path(path).suffix.lower()
        if not extension:
            return path + ".png", None
        if extension not in _EXPORT_EXTENSIONS:
            known = ", ".join(sorted(_EXPORT_EXTENSIONS))
            return None, (
                f"Unknown export extension '{extension}'. Use {known}.")
        return path, None

    def close(self) -> None:
        self._closed = True
        self._view = None

    def request_quit(self) -> None:
        self._confirm_destructive(
            "quit the editor", self._application.request_stop)

    def _new_from_image_path(self, path: str) -> None:
        self.import_image_path(path)
        if self._application.project_path is None:
            self._application.set_status(
                f"New from image: {os.path.basename(path)}")

    def _save_project_path(self, path: str) -> bool:
        self._require_open()
        if not path.lower().endswith(".deproj"):
            path += ".deproj"
        try:
            self._application.layer_stack.save_project(path)
        except Exception as exc:
            self._operation_error("Save Project", path, exc)
            return False
        self._application.mark_document_saved(path)
        self._best_effort(
            lambda: self._remember_parent(path),
            "remember saved project directory",
        )
        self._best_effort(
            lambda: self._application.set_window_title(
                f"{os.path.basename(path)} — Diffusion Editor"),
            "update title after saving a project",
        )
        self._best_effort(
            lambda: self._application.set_status(
                f"Saved: {os.path.basename(path)}"),
            "update status after saving a project",
        )
        return True

    def _apply_settings(self, state: SettingsState | None) -> None:
        if state is None or self._closed:
            return
        models_dir_changed = state.models_dir != self._application.models_dir
        self._application.set_models_dir(state.models_dir)
        if models_dir_changed:
            self._application.set_last_dir(self._application.models_dir)
        self._application.set_history_memory_limit_bytes(
            int(state.history_limit_gib * _BYTES_PER_GIB))
        settings = self._application.settings
        mcp_setting_changed = (
            bool(settings.get("mcp_server_enabled", False))
            != state.mcp_server_enabled
        )
        settings.set("mcp_server_enabled", state.mcp_server_enabled)
        settings.set("agent_api_base_url", state.agent_base_url.strip())
        settings.set("agent_api_key", state.agent_api_key.strip())
        settings.set("agent_model", state.agent_model.strip())
        settings.set("agent_temperature", float(state.agent_temperature))
        settings.set("agent_max_tokens", int(state.agent_max_tokens))
        settings.set(
            "agent_timeout_seconds", float(state.agent_timeout_seconds))
        settings.set("agent_stream", bool(state.agent_stream))
        self._on_models_dir_changed()
        if mcp_setting_changed:
            self._application.set_status(
                "Saved settings; restart Diffusion Editor to apply the "
                "Editor MCP change")
        else:
            self._application.set_status(
                "Saved settings: models, history, automation and Agent Chat")

    def _submit_grounding(self, params: GroundingParams | None) -> None:
        if params is None or self._closed:
            return
        layer = self._application.layer_stack.active_layer
        if layer is None or not layer.accepts_pixel_edits:
            self._application.set_status("Grounding: no active layer")
            return
        event = self._application.grounding_controller.start_detection(
            layer, params)
        if event.status:
            self._application.set_status(event.status)

    def _show_file(
            self,
            spec: FileDialogSpec,
            operation: Callable[[str], None]) -> None:
        self._require_open()
        if self._view is None:
            return

        def finished(path: str | None) -> None:
            if path and not self._closed:
                operation(path)

        self._view.show_file_dialog(spec, finished)

    def _commit_document_reset(
            self,
            project_path: str | None,
            *,
            clean: bool = True) -> None:
        self._application.clear_history()
        self._application.reset_document_session(
            project_path, clean=clean)

    def _confirm_destructive(
            self,
            action: str,
            continuation: Callable[[], None]) -> None:
        self._require_open()
        if not self._application.document_dirty:
            continuation()
            return
        if self._view is None:
            return
        self._view.show_unsaved_changes(
            action,
            lambda decision: self._finish_destructive(
                decision, continuation),
        )

    def _finish_destructive(
            self,
            decision: UnsavedDecision,
            continuation: Callable[[], None]) -> None:
        if self._closed or decision == UnsavedDecision.CANCEL:
            return
        if decision == UnsavedDecision.DISCARD:
            self._application.discard_unsaved_changes()
            continuation()
            return
        path = self._application.project_path
        if path is not None:
            if self._save_project_path(path):
                continuation()
            return
        self._show_file(
            FileDialogSpec(
                FileDialogKind.SAVE_FILE,
                "Save Project",
                self._application.last_dir,
                _PROJECT_FILTER,
                self._project_file_name(),
            ),
            lambda selected: (
                continuation() if self._save_project_path(selected) else None),
        )

    def _finish_recovery(
            self, record: RecoveryRecord, restore: bool) -> None:
        if self._closed:
            return
        try:
            if restore:
                self._application.restore_recovery(record)
                self._present_document_reset(None)
            else:
                self._application.discard_recovery(record)
        except Exception as exc:
            self._operation_error(
                "Recover Document", str(record.snapshot_path), exc)

    def _present_document_reset(self, project_path: str | None) -> None:
        self._best_effort(
            lambda: self._application.set_window_title(
                "Diffusion Editor"
                if project_path is None
                else (
                    f"{os.path.basename(project_path)} "
                    "— Diffusion Editor")),
            "update title after replacing the document",
        )
        self._best_effort(
            self._canvas.fit_in_view,
            "fit the replacement document in view",
        )

    def _remember_parent(self, path: str) -> None:
        self._application.set_last_dir(os.path.dirname(path))

    @staticmethod
    def _best_effort(callback: Callable[[], None], description: str) -> None:
        try:
            callback()
        except Exception:
            log.exception(f"Failed to {description}")

    def _project_file_name(self) -> str:
        path = self._application.project_path
        return os.path.basename(path) if path else "project.deproj"

    def _operation_error(
            self, title: str, path: str, exc: Exception) -> None:
        log.exception(f"{title} failed: {path}")
        message = str(exc)
        self._application.set_status(f"{title} error: {message}")
        self._show_error(title, message)

    def _show_error(self, title: str, message: str) -> None:
        if self._view is not None:
            self._view.show_error(title, message)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("application dialog coordinator is closed")
