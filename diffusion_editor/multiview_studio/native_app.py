"""Windowed native application for the multiview studio prototype."""

from __future__ import annotations

import faulthandler
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import threading
import time
from typing import Callable
from concurrent.futures import Future, ThreadPoolExecutor

from tcbase import log
from termin.gui_native import (
    FileDialogMode,
    FileDialogModel,
    MessageBoxKind,
    Rect,
)

from ..app.native_root import WindowedNativeComposition
from ..sdk_runtime import verify_application_environment
from .controller import MultiviewStudioController
from .model import ViewKey
from .native_view import NativeMultiviewStudioView
from .qwen_generation import QwenViewGenerator, generated_view_keys
from .trellis_generation import TrellisShapeGenerator


_IMAGE_FILTER = "Images | *.png *.jpg *.jpeg *.bmp *.tiff *.webp"
_PROJECT_FILTER = "Multiview Studio Project | *.mvstudio.json *.json"
_DEFAULT_PROJECT_NAME = "project.mvstudio.json"


class NativeMultiviewStudioApplication:
    def __init__(
        self,
        composition: WindowedNativeComposition,
        project_path: str | None = None,
    ) -> None:
        self.composition = composition
        self.document = composition.document
        self.controller = MultiviewStudioController()
        self.view = NativeMultiviewStudioView(
            self.document,
            self,
            request_repaint=composition.request_repaint,
            texture_lease_factory=composition.create_texture_lease,
        )
        self._dialogs: list[object] = []
        self._connections: list[object] = []
        self._message_boxes: list[object] = []
        self._last_dir = str(Path.cwd())
        self._closed = False
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="multiview-studio"
        )
        self._pending: Queue[Callable[[], None]] = Queue()
        self._active_future: Future | None = None
        self._cancel = threading.Event()
        self._qwen = QwenViewGenerator()
        self._trellis = TrellisShapeGenerator()
        self.controller.connect(self._apply_project)
        if project_path:
            self._safe(lambda: self.controller.open_project(project_path))

    def new_project(self) -> None:
        if self._job_active():
            self.view.set_status("Cancel the active operation before creating a project")
            return
        self.controller.new_project()
        self.view.set_status("New multiview project")

    def open_project(self) -> None:
        if self._job_active():
            self.view.set_status("Cancel the active operation before opening a project")
            return
        self._show_file_dialog(
            FileDialogMode.OpenFile,
            _PROJECT_FILTER,
            self._open_project_path,
        )

    def save_project(self) -> None:
        if self.controller.project_path is None:
            self.save_project_as()
            return
        self._safe(self.controller.save)

    def save_project_as(self) -> None:
        self._show_file_dialog(
            FileDialogMode.SaveFile,
            _PROJECT_FILTER,
            self._save_project_path,
            file_name=_DEFAULT_PROJECT_NAME,
        )

    def pick_source(self, source: str) -> None:
        if self._job_active():
            return
        self._pick_image(lambda path: self.controller.set_source(source, path))

    def pick_slot(self, key: ViewKey) -> None:
        if self._job_active():
            return
        self._pick_image(lambda path: self.controller.set_slot_image(key, path))

    def clear_slot(self, key: ViewKey) -> None:
        self.controller.clear_slot(key)

    def set_slot_included(self, key: ViewKey, included: bool) -> None:
        self.controller.set_slot_included(key, included)

    def set_qwen_seed(self, seed: int) -> None:
        self._safe(lambda: self.controller.set_qwen_seed(seed))

    def set_trellis_setting(self, field: str, value: int) -> None:
        self._safe(lambda: self.controller.set_trellis_setting(field, value))

    def generate_view(self, key: ViewKey) -> None:
        self._start_qwen_generation((key,))

    def generate_four(self) -> None:
        self._start_qwen_generation(
            generated_view_keys(self.controller.project, "four")
        )

    def generate_missing(self) -> None:
        self._start_qwen_generation(
            generated_view_keys(self.controller.project, "missing")
        )

    def generate_all(self) -> None:
        self._start_qwen_generation(
            generated_view_keys(self.controller.project, "all")
        )

    def build_shape(self) -> None:
        if self._job_active():
            self.view.set_status("Another operation is already running")
            return
        errors = self.controller.project.validate_shape_request()
        if errors:
            self._show_error("Cannot build shape", "\n".join(errors))
            return
        project_path = self.controller.project_path
        if project_path is None:
            self.view.set_status("Save the project before building a shape")
            return
        snapshot = self.controller.project
        settings = snapshot.trellis
        count = len(snapshot.included_slots())
        self._cancel = threading.Event()
        self.view.set_busy(True)
        self.view.apply_project(snapshot, project_path, self.controller.dirty)
        self.view.set_status(
            f"Preparing TRELLIS.2: {count} view(s), "
            f"{settings.total_steps} steps, {settings.warmup_steps} warmup"
        )
        # Qwen and TRELLIS.2 cannot coexist comfortably on the target GPU.
        self._qwen.shutdown()

        def progress(message: str) -> None:
            self._post(lambda text=message: self.view.set_status(text))

        future = self._executor.submit(
            self._trellis.generate,
            snapshot,
            project_path,
            self._cancel,
            progress,
        )
        self._active_future = future
        future.add_done_callback(
            lambda completed, expected=project_path: self._post(
                lambda: self._finish_shape_generation(completed, expected)
            )
        )

    def open_shape(self) -> None:
        path = Path(self.controller.project.shape_path)
        if not path.is_file():
            self.view.set_status("No generated shape to open")
            return
        self._safe(
            lambda: subprocess.Popen(
                ["termin", "show", str(path)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        )

    def cancel_job(self) -> None:
        if self._job_active():
            self._cancel.set()
            self._trellis.cancel()
            self.view.set_status("Cancelling active operation...")

    def process_pending(self) -> int:
        processed = 0
        while True:
            try:
                callback = self._pending.get_nowait()
            except Empty:
                return processed
            callback()
            processed += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._cancel.set()
        self._trellis.cancel()
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._qwen.shutdown()
        for dialog in self._dialogs:
            if getattr(dialog, "open", False):
                dialog.activate("cancel")
        for box in self._message_boxes:
            if getattr(box, "open", False):
                self.document.dismiss_overlay(box.handle)
        self.view.close()
        self._dialogs.clear()
        self._message_boxes.clear()
        self._connections.clear()

    def _apply_project(self, project, path, dirty: bool) -> None:
        self.view.apply_project(project, path, dirty)
        name = path.name if path else "Untitled"
        marker = " *" if dirty else ""
        self.composition.set_window_title(
            f"Multiview Shape Studio — {name}{marker}"
        )

    def _pick_image(self, callback: Callable[[str], None]) -> None:
        self._show_file_dialog(
            FileDialogMode.OpenFile,
            _IMAGE_FILTER,
            callback,
        )

    def _show_file_dialog(
        self,
        mode: FileDialogMode,
        filters: str,
        callback: Callable[[str], None],
        *,
        file_name: str = "",
    ) -> None:
        dialog = self.document.create_file_dialog(mode)
        dialog.set_initial_directory(self._last_dir)
        dialog.set_filters(FileDialogModel.parse_filter_string(filters))
        dialog.set_file_name(file_name)

        def finished(path: str | None) -> None:
            if not path or self._closed:
                return
            self._last_dir = str(Path(path).expanduser().resolve().parent)
            self._safe(lambda: callback(path))

        self._dialogs.append(dialog)
        self._connections.append(dialog.connect_path_finished(finished))
        dialog.show(Rect(0, 0, 1700, 950))
        self.composition.request_repaint()

    def _open_project_path(self, path: str) -> None:
        self.controller.open_project(path)

    def _save_project_path(self, path: str) -> None:
        target = Path(path).expanduser()
        if not target.name.endswith(".json"):
            target = target.with_name(target.name + ".mvstudio.json")
        self.controller.save(target)
        self.view.set_status(f"Saved {target.name}")

    def _start_qwen_generation(self, keys: tuple[ViewKey, ...]) -> None:
        if self._job_active():
            self.view.set_status("Another operation is already running")
            return
        if not keys:
            self.view.set_status("No view slots need generation")
            return
        project_path = self.controller.project_path
        if project_path is None:
            self.view.set_status("Save the project before generating views")
            return
        snapshot = self.controller.project
        self._cancel = threading.Event()
        self.view.set_busy(True)
        self.view.apply_project(snapshot, project_path, self.controller.dirty)
        self.view.set_status(f"Starting Qwen for {len(keys)} view(s)...")

        def progress(message: str) -> None:
            self._post(lambda text=message: self.view.set_status(text))

        future = self._executor.submit(
            self._qwen.generate,
            snapshot,
            project_path,
            keys,
            self._cancel,
            progress,
        )
        self._active_future = future
        future.add_done_callback(
            lambda completed, expected=project_path: self._post(
                lambda: self._finish_qwen_generation(completed, expected)
            )
        )

    def _finish_qwen_generation(
        self, future: Future, expected_project_path: Path
    ) -> None:
        self._active_future = None
        self.view.set_busy(False)
        try:
            generated = future.result()
        except Exception as error:
            message = str(error)
            if "cancel" in message.lower():
                self.view.set_status("View generation cancelled")
            else:
                self._show_error("Qwen view generation failed", message)
            self.view.apply_project(
                self.controller.project,
                self.controller.project_path,
                self.controller.dirty,
            )
            return
        if self.controller.project_path != expected_project_path:
            self.view.set_status("Ignored Qwen result for a different project")
            return
        self.controller.set_slot_images(generated)
        self.controller.save()
        self.view.set_status(f"Generated {len(generated)} view(s)")

    def _finish_shape_generation(
        self, future: Future, expected_project_path: Path
    ) -> None:
        self._active_future = None
        self.view.set_busy(False)
        try:
            shape_path = future.result()
        except Exception as error:
            message = str(error)
            if "cancel" in message.lower():
                self.view.set_status("TRELLIS.2 shape generation cancelled")
            else:
                self._show_error("TRELLIS.2 shape generation failed", message)
            self.view.apply_project(
                self.controller.project,
                self.controller.project_path,
                self.controller.dirty,
            )
            return
        if self.controller.project_path != expected_project_path:
            self.view.set_status("Ignored TRELLIS.2 result for a different project")
            return
        self.controller.set_shape_path(shape_path)
        self.controller.save()
        self.view.set_status(f"Shape ready: {shape_path}")

    def _job_active(self) -> bool:
        return self._active_future is not None

    def _post(self, callback: Callable[[], None]) -> None:
        self._pending.put(callback)
        self.composition.request_repaint()

    def _safe(self, operation: Callable[[], object]) -> None:
        try:
            operation()
        except Exception as error:
            self._show_error("Multiview Studio", str(error))
            self._apply_project(
                self.controller.project,
                self.controller.project_path,
                self.controller.dirty,
            )

    def _show_error(self, title: str, message: str) -> None:
        box = self.document.create_message_box(
            title, message, MessageBoxKind.Error
        )
        self._message_boxes.append(box)
        self._connections.append(
            box.connect_finished(lambda _result: self._release_box(box))
        )
        box.show(Rect(0, 0, 1700, 950))
        self.composition.request_repaint()

    def _release_box(self, box) -> None:
        if box in self._message_boxes:
            self._message_boxes.remove(box)


def run_native(project_path: str | None = None) -> int:
    verify_application_environment()
    faulthandler.enable()
    log.set_level(log.Level.INFO)
    smoke_frames = int(os.environ.get("MULTIVIEW_STUDIO_SMOKE_FRAMES", "0"))
    rendered = 0
    composition = WindowedNativeComposition(
        title="Multiview Shape Studio",
        width=1700,
        height=950,
    )
    application = NativeMultiviewStudioApplication(composition, project_path)
    composition.request_repaint()
    try:
        while not composition.should_close:
            events = composition.pump_events()
            pending = application.process_pending()
            if smoke_frames:
                composition.request_repaint()
            painted = composition.render_frame()
            rendered += int(painted)
            if smoke_frames and rendered >= smoke_frames:
                break
            if not painted and not events and not pending:
                time.sleep(0.01)
    finally:
        application.close()
        composition.close()
    return 0
