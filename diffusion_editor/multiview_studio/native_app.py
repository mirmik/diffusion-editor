"""Windowed native application for the multiview studio prototype."""

from __future__ import annotations

import faulthandler
import os
from pathlib import Path
from queue import Empty, Queue
import shutil
import tempfile
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
from ..app.native_reconstruction_viewport import NativeReconstructionViewport
from ..sdk_runtime import verify_application_environment
from .controller import MultiviewStudioController
from .model import MultiviewProject, RefineCube, ViewKey
from .native_view import NativeMultiviewStudioView
from .qwen_generation import QwenViewGenerator, generated_view_keys
from .recent_projects import RecentProjectsStore
from .trellis_generation import TrellisShapeGenerator
from .trellis_texture_generation import TrellisTextureGenerator


_IMAGE_FILTER = "Images | *.png *.jpg *.jpeg *.bmp *.tiff *.webp"
_PROJECT_FILTER = "Multiview Studio Project | *.mvstudio.json *.json"
_DEFAULT_PROJECT_NAME = "project.mvstudio.json"


def _gltf_point_to_viewport(point: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = map(float, point)
    return x, -z, y


def _viewport_point_to_gltf(point: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = map(float, point)
    return x, z, -y


class NativeMultiviewStudioApplication:
    def __init__(
        self,
        composition: WindowedNativeComposition,
        project_path: str | None = None,
        *,
        recent_store: RecentProjectsStore | None = None,
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
        self.reconstruction_viewport = NativeReconstructionViewport(
            self.document,
            graphics_owner=composition.graphics,
            request_repaint=composition.request_repaint,
            resource_namespace="multiview-studio",
            # TRELLIS.2's exported front is opposite to the Pixal3D default
            # used by the shared viewport.
            front_azimuth=0.0,
        )
        self.reconstruction_viewport.bind_refine_cube_edit(
            self._edit_refine_cube_from_viewport
        )
        self.view.mount_model_viewport(self.reconstruction_viewport)
        self._recent_store = recent_store or RecentProjectsStore()
        self.view.set_recent_projects(self._recent_store.load())
        self._loaded_shape_path = ""
        self._refine_cube_before_edit: RefineCube | None = None
        self._refine_cube_editing = False
        self._unsaved_workspace = tempfile.TemporaryDirectory(
            prefix="diffusion-editor-multiview-"
        )
        self._unsaved_project_path = (
            Path(self._unsaved_workspace.name) / "untitled.mvstudio.json"
        )
        self._dialogs: list[object] = []
        self._connections: list[object] = []
        self._message_boxes: list[object] = []
        self._last_dir = str(Path.cwd())
        self._closed = False
        self._quit_requested = False
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="multiview-studio"
        )
        self._pending: Queue[Callable[[], None]] = Queue()
        self._active_future: Future | None = None
        self._cancel = threading.Event()
        self._qwen = QwenViewGenerator()
        self._trellis = TrellisShapeGenerator()
        self._texture = TrellisTextureGenerator()
        composition.set_unhandled_key_handler(self.view.dispatch_shortcut)
        self.controller.connect(self._apply_project)
        if project_path:
            self._safe(lambda: self._open_project_path(project_path))

    def new_project(self) -> None:
        if self._job_active():
            self.view.set_status("Cancel the active operation before creating a project")
            return
        self._reset_refine_editing()
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
        self._safe(self._save_current_project)

    def save_project_as(self) -> None:
        self._show_file_dialog(
            FileDialogMode.SaveFile,
            _PROJECT_FILTER,
            self._save_project_path,
            file_name=_DEFAULT_PROJECT_NAME,
        )

    def open_recent_project(self, path: str) -> None:
        if self._job_active():
            self.view.set_status("Cancel the active operation before opening a project")
            return
        target = Path(path).expanduser().resolve()
        if not target.is_file():
            self.view.set_recent_projects(self._recent_store.remove(target))
            self.view.set_status(f"Recent project is no longer available: {target.name}")
            return
        self._safe(lambda: self._open_project_path(str(target)))

    def clear_recent_projects(self) -> None:
        self._recent_store.clear()
        self.view.set_recent_projects(())
        self.view.set_status("Recent projects cleared")

    def quit(self) -> None:
        self._quit_requested = True

    @property
    def quit_requested(self) -> bool:
        return self._quit_requested

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

    def set_qwen_seed(self, seed: int) -> None:
        self._safe(lambda: self.controller.set_qwen_seed(seed))

    def set_trellis_setting(self, field: str, value: int) -> None:
        self._safe(lambda: self.controller.set_trellis_setting(field, value))

    def set_texture_setting(self, field: str, value: int) -> None:
        self._safe(lambda: self.controller.set_texture_setting(field, value))

    def set_mesh_postprocess(self, field: str, value: bool | float) -> None:
        self._safe(lambda: self.controller.set_mesh_postprocess(field, value))

    def begin_refine_cube(self) -> None:
        if self._job_active():
            return
        if not Path(self.controller.project.geometry_path).is_file():
            self.view.set_status("Build a model before selecting a refine cube")
            return
        if not self._refine_cube_editing:
            self._refine_cube_before_edit = self.controller.project.refine_cube
        self._refine_cube_editing = True
        self.reconstruction_viewport.begin_refine_cube_pick(
            self._accept_refine_cube_pick
        )
        self.view.set_refine_editing(True, picking=True)
        self.view.set_status("Click the model to place a refine cube")

    def _accept_refine_cube_pick(
        self, center: tuple[float, float, float], side: float
    ) -> None:
        self._safe(lambda: self.controller.set_refine_cube(
            _viewport_point_to_gltf(center), side, confirmed=False
        ))
        self.view.set_refine_editing(True, picking=False)
        self.view.set_status("Refine cube placed; adjust it or confirm")

    def set_refine_cube_value(self, field: str, value: float) -> None:
        if not self._refine_cube_editing:
            self._refine_cube_before_edit = self.controller.project.refine_cube
            self._refine_cube_editing = True
        self._safe(lambda: self.controller.set_refine_cube_value(field, value))
        self.view.set_refine_editing(True, picking=False)

    def _edit_refine_cube_from_viewport(
        self, center: tuple[float, float, float], side: float
    ) -> None:
        if self._job_active():
            return
        if not self._refine_cube_editing:
            self._refine_cube_before_edit = self.controller.project.refine_cube
            self._refine_cube_editing = True
        self._safe(lambda: self.controller.update_refine_cube(
            _viewport_point_to_gltf(center), side
        ))
        self.view.set_refine_editing(True, picking=False)

    def confirm_refine_cube(self) -> None:
        self._safe(self.controller.confirm_refine_cube)
        self._reset_refine_editing()
        self.view.set_status("Refine cube confirmed")

    def cancel_refine_cube(self) -> None:
        previous = self._refine_cube_before_edit
        self.reconstruction_viewport.cancel_refine_cube_pick()
        self.controller.replace_refine_cube(previous)
        self._reset_refine_editing()
        self.view.set_status("Refine cube edit cancelled")

    def clear_refine_cube(self) -> None:
        self._reset_refine_editing()
        self.controller.clear_refine_cube()
        self.view.set_status("Refine cube cleared")

    def _reset_refine_editing(self) -> None:
        viewport = getattr(self, "reconstruction_viewport", None)
        if viewport is not None:
            viewport.cancel_refine_cube_pick()
        self._refine_cube_before_edit = None
        self._refine_cube_editing = False
        if hasattr(self.view, "set_refine_editing"):
            self.view.set_refine_editing(False)

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
        snapshot = self.controller.project
        generation_path = project_path or self._unsaved_project_path
        settings = snapshot.trellis
        count = len(snapshot.populated_slots())
        self._cancel = threading.Event()
        self._set_busy(True)
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
            generation_path,
            self._cancel,
            progress,
        )
        self._active_future = future
        future.add_done_callback(
            lambda completed, expected=snapshot: self._post(
                lambda: self._finish_shape_generation(completed, expected)
            )
        )

    def reprocess_shape(self) -> None:
        if self._job_active():
            self.view.set_status("Another operation is already running")
            return
        snapshot = self.controller.project
        if not self._trellis.has_reusable_cache(snapshot):
            self.view.set_status(
                "Cached decoded mesh is unavailable or generation inputs changed"
            )
            return
        project_path = self.controller.project_path
        self._cancel = threading.Event()
        self._set_busy(True)
        self.view.apply_project(snapshot, project_path, self.controller.dirty)
        self.view.set_status("Reprocessing cached decoded mesh...")
        self._qwen.shutdown()

        def progress(message: str) -> None:
            self._post(lambda text=message: self.view.set_status(text))

        future = self._executor.submit(
            self._trellis.reprocess,
            snapshot,
            self._cancel,
            progress,
        )
        self._active_future = future
        future.add_done_callback(
            lambda completed, expected=snapshot: self._post(
                lambda: self._finish_shape_reprocess(completed, expected)
            )
        )

    def texture_model(self) -> None:
        if self._job_active():
            self.view.set_status("Another operation is already running")
            return
        snapshot = self.controller.project
        errors = snapshot.validate_texture_request()
        if errors:
            self._show_error("Cannot texture model", "\n".join(errors))
            return
        geometry = Path(snapshot.geometry_path).expanduser().resolve()
        if not geometry.is_file():
            self._show_error(
                "Cannot texture model",
                f"Current geometry is missing: {geometry}",
            )
            return
        project_path = self.controller.project_path
        generation_path = project_path or self._unsaved_project_path
        settings = snapshot.texture
        count = len(snapshot.populated_slots())
        self._cancel = threading.Event()
        self._set_busy(True)
        self.view.apply_project(snapshot, project_path, self.controller.dirty)
        self.view.set_status(
            f"Preparing texture: repaired mesh, {count} view(s), "
            f"{settings.total_steps} steps"
        )
        self._qwen.shutdown()

        def progress(message: str) -> None:
            self._post(lambda text=message: self.view.set_status(text))

        future = self._executor.submit(
            self._texture.generate,
            snapshot,
            generation_path,
            self._cancel,
            progress,
        )
        self._active_future = future
        future.add_done_callback(
            lambda completed, expected=snapshot: self._post(
                lambda: self._finish_texture_generation(completed, expected)
            )
        )

    def open_shape(self) -> None:
        path = Path(self.controller.project.shape_path)
        if not path.is_file():
            self.view.set_status("No generated shape to show")
            return
        self._safe(lambda: self._load_shape_in_workspace(str(path), force=True))
        self.view.set_status(f"Model shown in workspace: {path.name}")

    def cancel_job(self) -> None:
        if self._job_active():
            self._cancel.set()
            self._trellis.cancel()
            self._texture.cancel()
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

    def render_viewports(self) -> bool:
        return self.reconstruction_viewport.render_if_dirty()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.reconstruction_viewport.cancel_refine_cube_pick()
        self.composition.set_unhandled_key_handler(None)
        self._cancel.set()
        self._trellis.cancel()
        self._texture.cancel()
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._qwen.shutdown()
        for dialog in self._dialogs:
            if getattr(dialog, "open", False):
                dialog.activate("cancel")
        for box in self._message_boxes:
            if getattr(box, "open", False):
                self.document.dismiss_overlay(box.handle)
        self.reconstruction_viewport.close()
        self.view.close()
        self._unsaved_workspace.cleanup()
        self._dialogs.clear()
        self._message_boxes.clear()
        self._connections.clear()

    def _apply_project(self, project, path, dirty: bool) -> None:
        self.view.apply_project(project, path, dirty)
        self.view.set_reprocess_available(
            self._trellis.has_reusable_cache(project)
        )
        self._sync_shape_workspace(project.shape_path)
        self._sync_refine_cube(project.refine_cube)
        name = path.name if path else "Untitled"
        marker = " *" if dirty else ""
        self.composition.set_window_title(
            f"Multiview Shape Studio — {name}{marker}"
        )

    def _sync_shape_workspace(self, shape_path: str) -> None:
        path = Path(shape_path) if shape_path else None
        if path is not None and path.is_file():
            self._load_shape_in_workspace(str(path))
        elif self._loaded_shape_path:
            self.reconstruction_viewport.clear_model()
            self._loaded_shape_path = ""

    def _sync_refine_cube(self, cube: RefineCube | None) -> None:
        if cube is None:
            self.reconstruction_viewport.clear_refine_cube()
            return
        self.reconstruction_viewport.set_refine_cube(
            _gltf_point_to_viewport(cube.center), cube.side, cube.confirmed
        )

    def _load_shape_in_workspace(self, shape_path: str, *, force: bool = False) -> None:
        resolved = str(Path(shape_path).expanduser().resolve())
        if not force and resolved == self._loaded_shape_path:
            return
        self.reconstruction_viewport.load_glb(resolved, fit_camera=True)
        self._loaded_shape_path = resolved

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
        self._reset_refine_editing()
        self.controller.open_project(path)
        opened = self.controller.project_path
        if opened is not None:
            self._remember_project(opened)

    def _save_current_project(self) -> None:
        saved = self.controller.save()
        self._remember_project(saved)
        self.view.set_status(f"Saved {saved.name}")

    def _save_project_path(self, path: str) -> None:
        target = Path(path).expanduser()
        if not target.name.endswith(".json"):
            target = target.with_name(target.name + ".mvstudio.json")
        self._adopt_unsaved_artifacts(target)
        saved = self.controller.save(target)
        self._remember_project(saved)
        self.view.set_status(f"Saved {saved.name}")

    def _remember_project(self, path: str | Path) -> None:
        self.view.set_recent_projects(self._recent_store.record(path))
        self._last_dir = str(Path(path).expanduser().resolve().parent)

    def _adopt_unsaved_artifacts(self, manifest_path: Path) -> None:
        """Copy session-owned images and shape output beside the manifest."""
        session_root = Path(self._unsaved_workspace.name).resolve()
        destination_root = manifest_path.expanduser().resolve().parent / "views"
        adopted: dict[ViewKey, Path] = {}
        for slot in self.controller.project.slots:
            if not slot.image_path:
                continue
            source = Path(slot.image_path).expanduser().resolve()
            if not source.is_file() or not source.is_relative_to(session_root):
                continue
            destination_root.mkdir(parents=True, exist_ok=True)
            destination = destination_root / source.name
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            shutil.copy2(source, temporary)
            temporary.replace(destination)
            adopted[slot.key] = destination
        if adopted:
            self.controller.set_slot_images(adopted)

        project = self.controller.project
        artifact_paths = {
            "geometry": project.geometry_path,
            "shape": project.shape_path,
        }
        copied_runs: dict[Path, Path] = {}
        adopted_paths: dict[str, Path] = {}
        project_root = manifest_path.expanduser().resolve().parent
        for role, value in artifact_paths.items():
            if not value:
                continue
            source = Path(value).expanduser().resolve()
            if not source.is_file() or not source.is_relative_to(session_root):
                adopted_paths[role] = source
                continue
            source_run = source.parent
            destination_run = copied_runs.get(source_run)
            if destination_run is None:
                category = (
                    "texture-runs"
                    if source_run.parent.name == "texture-runs"
                    else "shape-runs"
                )
                destination_root = project_root / category
                destination_root.mkdir(parents=True, exist_ok=True)
                destination_run = destination_root / source_run.name
                suffix = 2
                while destination_run.exists():
                    destination_run = destination_root / (
                        f"{source_run.name}-{suffix}"
                    )
                    suffix += 1
                shutil.copytree(source_run, destination_run)
                copied_runs[source_run] = destination_run
            adopted_paths[role] = destination_run / source.name
        geometry = adopted_paths.get("geometry")
        shape = adopted_paths.get("shape")
        if geometry is not None and shape is not None:
            self.controller.set_model_paths(geometry, shape)

    def _start_qwen_generation(self, keys: tuple[ViewKey, ...]) -> None:
        if self._job_active():
            self.view.set_status("Another operation is already running")
            return
        if not keys:
            self.view.set_status("No view slots need generation")
            return
        snapshot = self.controller.project
        project_path = self.controller.project_path
        generation_path = project_path or self._unsaved_project_path
        self._cancel = threading.Event()
        self._set_busy(True)
        self.view.apply_project(snapshot, project_path, self.controller.dirty)
        self.view.set_status(f"Starting Qwen for {len(keys)} view(s)...")

        def progress(message: str) -> None:
            self._post(lambda text=message: self.view.set_status(text))

        future = self._executor.submit(
            self._qwen.generate,
            snapshot,
            generation_path,
            keys,
            self._cancel,
            progress,
        )
        self._active_future = future
        future.add_done_callback(
            lambda completed, expected=snapshot: self._post(
                lambda: self._finish_qwen_generation(completed, expected)
            )
        )

    def _finish_qwen_generation(
        self, future: Future, expected_project: MultiviewProject
    ) -> None:
        self._active_future = None
        self._set_busy(False)
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
        if self.controller.project != expected_project:
            self.view.set_status("Ignored Qwen result for a different project")
            return
        self.controller.set_slot_images(generated)
        if self.controller.project_path is not None:
            self._adopt_unsaved_artifacts(self.controller.project_path)
            self.controller.save()
            self.view.set_status(f"Generated {len(generated)} view(s)")
        else:
            self.view.set_status(
                f"Generated {len(generated)} view(s) in unsaved project"
            )

    def _finish_shape_generation(
        self, future: Future, expected_project: MultiviewProject
    ) -> None:
        self._active_future = None
        self._set_busy(False)
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
        if self.controller.project != expected_project:
            self.view.set_status("Ignored TRELLIS.2 result for a different project")
            return
        self.controller.set_shape_path(shape_path)
        if self.controller.project_path is not None:
            self._adopt_unsaved_artifacts(self.controller.project_path)
            self.controller.save()
            self.view.set_status(f"Shape ready: {shape_path}")
        else:
            self.view.set_status(f"Shape ready in unsaved project: {shape_path}")

    def _finish_shape_reprocess(
        self, future: Future, expected_project: MultiviewProject
    ) -> None:
        self._active_future = None
        self._set_busy(False)
        try:
            shape_path = future.result()
        except Exception as error:
            message = str(error)
            if "cancel" in message.lower():
                self.view.set_status("Cached mesh postprocess cancelled")
            else:
                self._show_error("Mesh postprocess failed", message)
            self.view.apply_project(
                self.controller.project,
                self.controller.project_path,
                self.controller.dirty,
            )
            return
        if self.controller.project != expected_project:
            self.view.set_status("Ignored postprocess result for a different project")
            return
        self.controller.set_shape_path(shape_path)
        if self.controller.project_path is not None:
            self._adopt_unsaved_artifacts(self.controller.project_path)
            self.controller.save()
            self.view.set_status(f"Cached mesh reprocessed: {shape_path.name}")
        else:
            self.view.set_status(
                f"Cached mesh reprocessed in unsaved project: {shape_path.name}"
            )

    def _finish_texture_generation(
        self, future: Future, expected_project: MultiviewProject
    ) -> None:
        self._active_future = None
        self._set_busy(False)
        try:
            shape_path = future.result()
        except Exception as error:
            message = str(error)
            if "cancel" in message.lower():
                self.view.set_status("TRELLIS.2 texturing cancelled")
            else:
                self._show_error("TRELLIS.2 texturing failed", message)
            self.view.apply_project(
                self.controller.project,
                self.controller.project_path,
                self.controller.dirty,
            )
            return
        if self.controller.project != expected_project:
            self.view.set_status("Ignored texture result for a different project")
            return
        self.controller.set_textured_shape_path(shape_path)
        if self.controller.project_path is not None:
            self._adopt_unsaved_artifacts(self.controller.project_path)
            self.controller.save()
            self.view.set_status(f"Textured model ready: {shape_path.name}")
        else:
            self.view.set_status(
                f"Textured model ready in unsaved project: {shape_path.name}"
            )

    def _job_active(self) -> bool:
        return self._active_future is not None

    def _set_busy(self, busy: bool) -> None:
        self.view.set_busy(busy)
        viewport = getattr(self, "reconstruction_viewport", None)
        if viewport is not None:
            viewport.set_refine_cube_edit_enabled(not busy)

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
        while not composition.should_close and not application.quit_requested:
            events = composition.pump_events()
            pending = application.process_pending()
            viewport_rendered = application.render_viewports()
            if smoke_frames:
                composition.request_repaint()
            painted = composition.render_frame()
            rendered += int(painted)
            if smoke_frames and rendered >= smoke_frames:
                break
            if not painted and not viewport_rendered and not events and not pending:
                time.sleep(0.01)
    finally:
        application.close()
        composition.close()
    return 0
