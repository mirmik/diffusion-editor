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


class FileDialogKind(str, Enum):
    OPEN_FILE = "open_file"
    SAVE_FILE = "save_file"
    OPEN_DIRECTORY = "open_directory"


@dataclass(frozen=True)
class FileDialogSpec:
    kind: FileDialogKind
    title: str
    directory: str = ""
    filters: str = ""
    file_name: str = ""


@dataclass(frozen=True)
class SettingsState:
    models_dir: str
    history_limit_gib: float
    agent_base_url: str
    agent_api_key: str
    agent_model: str
    agent_temperature: float
    agent_max_tokens: int
    agent_timeout_seconds: float
    agent_stream: bool


class ApplicationDialogPresentation(Protocol):
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
            "edit.settings": self.show_settings,
            "layer.detect": self.show_grounding,
        }

    def bind_view(self, view: ApplicationDialogPresentation) -> None:
        self._require_open()
        self._view = view

    def new_project(self) -> None:
        self._require_open()
        white = np.full((1024, 1024, 4), 255, dtype=np.uint8)
        self._application.layer_stack.init_from_image(white)
        self._document_reset(None)
        self._application.set_status("New 1024x1024 document")

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
        self._require_open()
        try:
            self._application.layer_stack.load_project(path)
            self._remember_parent(path)
            self._document_reset(path)
            self._application.set_status(f"Opened: {os.path.basename(path)}")
        except Exception as exc:
            self._operation_error("Open Project", path, exc)

    def import_image_path(self, path: str) -> None:
        self._require_open()
        try:
            image = Image.open(path).convert("RGBA")
            self._application.layer_stack.init_from_image(
                np.array(image, dtype=np.uint8))
            self._remember_parent(path)
            self._document_reset(None)
            self._application.set_status(
                f"Imported: {os.path.basename(path)}")
        except Exception as exc:
            self._operation_error("Import Image", path, exc)

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

    def _new_from_image_path(self, path: str) -> None:
        self.import_image_path(path)
        if self._application.project_path is None:
            self._application.set_status(
                f"New from image: {os.path.basename(path)}")

    def _save_project_path(self, path: str) -> None:
        self._require_open()
        if not path.lower().endswith(".deproj"):
            path += ".deproj"
        try:
            self._application.layer_stack.save_project(path)
            self._application.project_path = path
            self._remember_parent(path)
            self._application.set_window_title(
                f"{os.path.basename(path)} — Diffusion Editor")
            self._application.set_status(
                f"Saved: {os.path.basename(path)}")
        except Exception as exc:
            self._operation_error("Save Project", path, exc)

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
        settings.set("agent_api_base_url", state.agent_base_url.strip())
        settings.set("agent_api_key", state.agent_api_key.strip())
        settings.set("agent_model", state.agent_model.strip())
        settings.set("agent_temperature", float(state.agent_temperature))
        settings.set("agent_max_tokens", int(state.agent_max_tokens))
        settings.set(
            "agent_timeout_seconds", float(state.agent_timeout_seconds))
        settings.set("agent_stream", bool(state.agent_stream))
        self._on_models_dir_changed()
        self._application.set_status(
            "Saved settings: models directory, history and Agent Chat")

    def _submit_grounding(self, params: GroundingParams | None) -> None:
        if params is None or self._closed:
            return
        layer = self._application.layer_stack.active_layer
        if layer is None:
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

    def _document_reset(self, project_path: str | None) -> None:
        self._application.clear_history()
        self._application.project_path = project_path
        self._application.set_window_title(
            "Diffusion Editor"
            if project_path is None
            else f"{os.path.basename(project_path)} — Diffusion Editor")
        self._canvas.fit_in_view()

    def _remember_parent(self, path: str) -> None:
        self._application.set_last_dir(os.path.dirname(path))

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
