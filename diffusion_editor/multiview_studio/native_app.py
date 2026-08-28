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
from .model import (
    MAX_REFINE_REGIONS,
    MultiviewProject,
    RefineCube,
    RefineShapeResult,
    ViewKey,
)
from .native_view import NativeMultiviewStudioView
from .qwen_generation import QwenViewGenerator, generated_view_keys
from .recent_projects import RecentProjectsStore
from .trellis_generation import TrellisShapeGenerator
from .trellis_refine_generation import TrellisRegionRefineGenerator
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
        self.reconstruction_viewport.bind_refine_mask_changed(
            self._commit_refine_mask
        )
        self.view.mount_model_viewport(self.reconstruction_viewport)
        self._recent_store = recent_store or RecentProjectsStore()
        self.view.set_recent_projects(self._recent_store.load())
        self._loaded_shape_path = ""
        self._displayed_mesh_signature = None
        self._selected_mesh_index = 0
        self._displayed_mesh_index = 0
        self._displayed_refine_result: tuple[int, int] | None = None
        self._refine_mask_visible = False
        self._refine_mask_painting = False
        self._refine_cube_before_edit: RefineCube | None = None
        self._refine_cube_visible = False
        self._refine_cube_visible_before_edit = False
        self._editing_refine_region_index: int | None = None
        self._refine_region_before_edit: int | None = None
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
        self._refine = TrellisRegionRefineGenerator()
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

    def set_refine_shape_setting(
        self, region_index: int, field: str, value: int | float
    ) -> None:
        self._safe(
            lambda: self.controller.set_refine_shape_setting(
                region_index, field, value
            )
        )

    def set_refine_postprocess(
        self, region_index: int, field: str, value: bool | float
    ) -> None:
        self._safe(
            lambda: self.controller.set_refine_postprocess(
                region_index, field, value
            )
        )

    def set_mesh_postprocess(self, field: str, value: bool | float) -> None:
        self._safe(lambda: self.controller.set_mesh_postprocess(field, value))

    def begin_refine_cube(self) -> None:
        if self._job_active():
            return
        if not Path(self.controller.project.geometry_path).is_file():
            self.view.set_status("Build a model before selecting a refine cube")
            return
        if len(self.controller.project.refine_regions) >= MAX_REFINE_REGIONS:
            self.view.set_status(
                f"The project already has {MAX_REFINE_REGIONS} refine regions"
            )
            return
        self._begin_refine_editing()
        self._selected_mesh_index = 0
        self._displayed_mesh_index = 0
        self._editing_refine_region_index = None
        self._refine_cube_visible = True
        self.view.set_selected_mesh(0)
        self.view.set_refine_cube_visible(True)
        self._sync_displayed_mesh(self.controller.project)
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
        self._begin_refine_editing()
        self._safe(lambda: self.controller.set_refine_cube_value(field, value))
        self.view.set_refine_editing(True, picking=False)

    def _edit_refine_cube_from_viewport(
        self, center: tuple[float, float, float], side: float
    ) -> None:
        if self._job_active():
            return
        self._begin_refine_editing()
        self._safe(lambda: self.controller.update_refine_cube(
            _viewport_point_to_gltf(center), side
        ))
        self.view.set_refine_editing(True, picking=False)

    def confirm_refine_cube(self) -> None:
        cube = self.controller.project.refine_cube
        if cube is None:
            self.view.set_status("Place a refine cube before confirming it")
            return
        if not self._refine_cube_visible:
            self.view.set_status("Select a refine cube before confirming it")
            return
        self._displayed_mesh_index = 0
        self._sync_displayed_mesh(self.controller.project)
        triangle_count = self.reconstruction_viewport.cube_subset_triangle_count(
            _gltf_point_to_viewport(cube.center), cube.side
        )
        if not triangle_count:
            self.view.set_status("The refine cube contains no model surface")
            return
        before = len(self.controller.project.refine_regions)
        edited_index = self._editing_refine_region_index
        try:
            self.controller.confirm_refine_cube(edited_index)
        except Exception as error:
            self._show_error("Multiview Studio", str(error))
            return
        region_index = (
            edited_index + 1 if edited_index is not None else before + 1
        )
        self._reset_refine_editing()
        self._selected_mesh_index = region_index
        self._displayed_mesh_index = 0
        self._editing_refine_region_index = region_index - 1
        self._refine_cube_visible = True
        self.view.set_selected_mesh(region_index)
        self.view.set_refine_cube_visible(True)
        self._sync_displayed_mesh(self.controller.project)
        action = "updated" if edited_index is not None else "confirmed"
        self.view.set_status(
            f"Refine region {region_index} {action} · "
            f"{triangle_count:,} triangles"
        )

    def cancel_refine_cube(self) -> None:
        previous = self._refine_cube_before_edit
        previous_visible = self._refine_cube_visible_before_edit
        previous_region = self._refine_region_before_edit
        self.reconstruction_viewport.cancel_refine_cube_pick()
        self._displayed_mesh_index = 0
        self._editing_refine_region_index = previous_region
        self._refine_cube_visible = previous_visible and previous is not None
        self.controller.replace_refine_cube(previous)
        self._reset_refine_editing()
        self.view.set_refine_cube_visible(self._refine_cube_visible)
        self._sync_displayed_mesh(self.controller.project)
        self.view.set_status("Refine cube edit cancelled")

    def clear_refine_cube(self) -> None:
        self._reset_refine_editing()
        self._editing_refine_region_index = None
        self._refine_cube_visible = False
        self.view.set_refine_cube_visible(False)
        self.controller.clear_refine_cube()
        self.view.set_status("Refine cube cleared")

    def set_model_shading(self, mode: str) -> None:
        normalized = str(mode).strip().lower()
        self.view._set_model_shading(normalized)

    def toggle_refine_mask_visible(self) -> None:
        if self._displayed_mesh_index == 0:
            self.view.set_status(
                "Double-click a refine region before showing its protection"
            )
            return
        self._refine_mask_visible = not self._refine_mask_visible
        self._apply_refine_mask_controls()

    def toggle_refine_mask_painting(self) -> None:
        if self._displayed_mesh_index == 0:
            self.view.set_status(
                "Double-click a refine region before painting its protection"
            )
            return
        self._refine_mask_painting = not self._refine_mask_painting
        if self._refine_mask_painting:
            self._refine_mask_visible = True
            self.view.set_status(
                "Protection brush: red geometry stays unchanged; drag paints "
                "through the mesh; Shift+drag unlocks; wheel changes radius"
            )
        else:
            self.view.set_status(
                "Protection painting off; orbit controls are active and "
                "protection stays visible"
            )
        self._apply_refine_mask_controls()

    def clear_refine_protection(self) -> None:
        index = self._displayed_mesh_index - 1
        if index < 0 or index >= len(self.controller.project.refine_regions):
            self.view.set_status(
                "Double-click a refine region before clearing protection"
            )
            return
        self.controller.set_refine_mask_weights(index, ())
        protection = self.controller.project.refine_mask(index)
        self._set_viewport_refine_mask(protection)
        self._refine_mask_visible = True
        self._apply_refine_mask_controls()
        self.view.set_status(
            f"Refine region {index + 1} protection cleared; all latents editable"
        )

    def _apply_refine_mask_controls(self) -> None:
        available = self._displayed_mesh_index > 0
        if not available:
            self._refine_mask_visible = False
            self._refine_mask_painting = False
        visible = bool(getattr(self, "_refine_mask_visible", False))
        painting = bool(getattr(self, "_refine_mask_painting", False))
        set_visible = getattr(
            self.reconstruction_viewport, "set_refine_mask_visible", None
        )
        if set_visible is not None:
            set_visible(available and visible)
        set_painting = getattr(
            self.reconstruction_viewport, "set_refine_mask_edit_enabled", None
        )
        if set_painting is not None:
            set_painting(available and painting)
        setter = getattr(self.view, "set_model_mask_state", None)
        if setter is not None:
            setter(
                available=available,
                visible=visible,
                painting=painting,
            )

    def _commit_refine_mask(
        self,
        mesh_vertex_weights: tuple[
            tuple[tuple[int, float], ...], ...
        ],
    ) -> None:
        index = self._displayed_mesh_index - 1
        if index < 0 or index >= len(self.controller.project.refine_regions):
            return
        self.controller.set_refine_mask_weights(index, mesh_vertex_weights)
        vertex_count = sum(len(weights) for weights in mesh_vertex_weights)
        self.view.set_status(
            f"Refine region {index + 1} protection · "
            f"{vertex_count:,} weighted vertices"
        )

    def _set_viewport_refine_mask(self, refine_mask) -> None:
        set_weighted = getattr(
            self.reconstruction_viewport, "set_refine_vertex_mask", None
        )
        if set_weighted is not None:
            set_weighted(
                refine_mask.mesh_vertex_weights,
                legacy_mesh_faces=refine_mask.mesh_faces,
            )
            return
        set_legacy = getattr(
            self.reconstruction_viewport, "set_refine_face_mask", None
        )
        if set_legacy is not None:
            set_legacy(refine_mask.mesh_faces)

    def select_refine_region(self, index: int) -> None:
        self._reset_refine_editing()
        self._displayed_refine_result = None
        maximum = len(self.controller.project.refine_regions)
        selected = max(0, min(int(index), maximum))
        self._selected_mesh_index = selected
        self._displayed_mesh_index = 0
        self._editing_refine_region_index = selected - 1 if selected else None
        self._refine_cube_visible = selected > 0
        self.view.set_selected_mesh(selected)
        self.view.set_refine_cube_visible(self._refine_cube_visible)
        if selected:
            self.controller.select_refine_region(selected - 1)
        self._sync_displayed_mesh(self.controller.project)

    def select_mesh(self, index: int) -> None:
        self._reset_refine_editing()
        self._displayed_refine_result = None
        maximum = len(self.controller.project.refine_regions)
        self._selected_mesh_index = max(0, min(int(index), maximum))
        self._displayed_mesh_index = self._selected_mesh_index
        self._editing_refine_region_index = None
        self._refine_cube_visible = False
        self.view.set_selected_mesh(self._selected_mesh_index)
        self.view.set_refine_cube_visible(False)
        self._sync_displayed_mesh(self.controller.project)

    def select_refine_result(self, region_index: int, result_index: int) -> None:
        project = self.controller.project
        results = project.region_refine_results(region_index)
        if not 0 <= result_index < len(results):
            self.view.set_status("Refined region result is unavailable")
            return
        result = results[result_index]
        path = Path(result.refined_region_path).expanduser().resolve()
        if not path.is_file():
            self.view.set_status(f"Refined region is missing: {path.name}")
            return
        self._reset_refine_editing()
        self._selected_mesh_index = region_index + 1
        self._displayed_mesh_index = 0
        self._displayed_refine_result = (region_index, result_index)
        self._refine_cube_visible = False
        self._refine_mask_visible = False
        self._refine_mask_painting = False
        self.view.set_selected_refine_result(region_index, result_index)
        self.view.set_refine_cube_visible(False)
        self._sync_displayed_mesh(project)
        self.view.set_status(
            f"Standalone refined region {region_index + 1}.{result_index + 1}"
        )

    def set_refine_view_patch(
        self,
        region_index: int,
        key: ViewKey,
        bounds: tuple[float, float, float, float] | None,
    ) -> None:
        self._safe(
            lambda: self.controller.set_refine_view_patch(
                region_index,
                key,
                bounds,
            )
        )

    def _begin_refine_editing(self) -> None:
        if self._refine_cube_editing:
            return
        self._refine_cube_before_edit = self.controller.project.refine_cube
        self._refine_cube_visible_before_edit = self._refine_cube_visible
        self._refine_region_before_edit = self._editing_refine_region_index
        self._refine_cube_editing = True

    def _reset_refine_editing(self) -> None:
        viewport = getattr(self, "reconstruction_viewport", None)
        if viewport is not None:
            viewport.cancel_refine_cube_pick()
        self._refine_cube_before_edit = None
        self._refine_cube_visible_before_edit = False
        self._refine_region_before_edit = None
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

    def texture_refine_result(
        self, region_index: int, result_index: int
    ) -> None:
        if self._job_active():
            self.view.set_status("Another operation is already running")
            return
        snapshot = self.controller.project
        errors = snapshot.validate_refine_texture_request(
            region_index, result_index
        )
        if errors:
            self._show_error("Cannot texture refined region", "\n".join(errors))
            return
        source = snapshot.region_refine_results(region_index)[result_index]
        settings = snapshot.texture
        patches = snapshot.view_patches(region_index)
        project_path = self.controller.project_path
        generation_path = project_path or self._unsaved_project_path
        self._cancel = threading.Event()
        self._set_busy(True)
        self.view.apply_project(snapshot, project_path, self.controller.dirty)
        self.view.set_status(
            f"Texturing refined region {region_index + 1}.{result_index + 1} · "
            f"{len(patches)} patch(es), {settings.total_steps} steps"
        )
        self._qwen.shutdown()

        def progress(message: str) -> None:
            self._post(lambda text=message: self.view.set_status(text))

        future = self._executor.submit(
            self._texture.generate_refined_region,
            snapshot,
            region_index,
            result_index,
            generation_path,
            self._cancel,
            progress,
        )
        self._active_future = future
        future.add_done_callback(
            lambda completed, expected=snapshot, region=region_index,
            result=result_index, source_key=source.request_key: self._post(
                lambda: self._finish_refine_texture_generation(
                    completed,
                    expected,
                    region,
                    result,
                    source_key,
                )
            )
        )

    def refine_region(self, region_index: int) -> None:
        if self._job_active():
            self.view.set_status("Another operation is already running")
            return
        snapshot = self.controller.project
        errors = snapshot.validate_refine_request(region_index)
        if errors:
            self._show_error("Cannot refine region", "\n".join(errors))
            return
        region = snapshot.refine_regions[region_index]
        mask = snapshot.refine_mask(region_index)
        source = Path(snapshot.shape_path or snapshot.geometry_path).resolve()
        try:
            self._load_shape_in_workspace(str(source), force=True)
            geometry = self.reconstruction_viewport.cube_mesh_subset_arrays(
                _gltf_point_to_viewport(region.center),
                region.side,
                mesh_vertex_weights=mask.mesh_vertex_weights,
                legacy_mesh_faces=mask.mesh_faces,
            )
        except Exception as error:
            self._show_error("Cannot extract refine region", str(error))
            return
        finally:
            self._displayed_mesh_signature = None
            self._sync_displayed_mesh(snapshot)
        vertices, faces, weights = geometry
        if not len(faces):
            self.view.set_status("The refine region contains no model surface")
            return
        project_path = self.controller.project_path
        generation_path = project_path or self._unsaved_project_path
        settings = snapshot.region_refine_settings(region_index)
        self._cancel = threading.Event()
        self._set_busy(True)
        self.view.apply_project(snapshot, project_path, self.controller.dirty)
        self.view.set_status(
            f"Encoding standalone region {region_index + 1} to Shape-SLat · "
            f"{settings.steps} refine steps · "
            f"{settings.warmup_steps} front warmup / stage · "
            f"structure strength {settings.strength:.2f} · full detail rebuild"
        )
        self._qwen.shutdown()

        def progress(message: str) -> None:
            self._post(lambda text=message: self.view.set_status(text))

        future = self._executor.submit(
            self._refine.generate,
            snapshot,
            region_index,
            generation_path,
            geometry,
            self._cancel,
            progress,
        )
        self._active_future = future
        future.add_done_callback(
            lambda completed, expected=snapshot, selected=region_index: self._post(
                lambda: self._finish_refine_generation(
                    completed, expected, selected
                )
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
            self._refine.cancel()
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
        self._refine.cancel()
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
        self._selected_mesh_index = min(
            self._selected_mesh_index, len(project.refine_regions)
        )
        self._displayed_mesh_index = min(
            self._displayed_mesh_index, len(project.refine_regions)
        )
        if (
            self._editing_refine_region_index is not None
            and self._editing_refine_region_index >= len(project.refine_regions)
        ):
            self._editing_refine_region_index = None
            self._refine_cube_visible = False
        self.view.set_selected_mesh(self._selected_mesh_index)
        displayed_result = getattr(self, "_displayed_refine_result", None)
        if displayed_result is not None:
            region_index, result_index = displayed_result
            if (
                region_index < len(project.refine_regions)
                and result_index
                < len(project.region_refine_results(region_index))
            ):
                self.view.set_selected_refine_result(
                    region_index, result_index
                )
        cube_visible = (
            self._refine_cube_visible
            and self._displayed_mesh_index == 0
            and project.refine_cube is not None
        )
        self.view.set_refine_cube_visible(cube_visible)
        self._sync_displayed_mesh(project)
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
            self._displayed_mesh_signature = None

    def _sync_refine_cube(self, cube: RefineCube | None) -> None:
        if cube is None:
            self.reconstruction_viewport.clear_refine_cube()
            return
        self.reconstruction_viewport.set_refine_cube(
            _gltf_point_to_viewport(cube.center), cube.side, cube.confirmed
        )

    @staticmethod
    def _region_signature(project: MultiviewProject, index: int) -> tuple:
        region = project.refine_regions[index - 1]
        return (
            project.shape_path,
            region.geometry_fingerprint,
            region.center,
            region.side,
        )

    def _sync_displayed_mesh(self, project: MultiviewProject) -> None:
        displayed_result = getattr(self, "_displayed_refine_result", None)
        if displayed_result is not None:
            region_index, result_index = displayed_result
            if region_index < len(project.refine_regions):
                results = project.region_refine_results(region_index)
                if result_index < len(results):
                    result_path = Path(
                        results[result_index].textured_region_path
                        or results[result_index].refined_region_path
                    ).expanduser().resolve()
                    if result_path.is_file():
                        self._refine_mask_visible = False
                        self._refine_mask_painting = False
                        self._load_shape_in_workspace(str(result_path))
                        self.reconstruction_viewport.clear_refine_cube()
                        self._apply_refine_mask_controls()
                        return
            self._displayed_refine_result = None
        mask_available = self._displayed_mesh_index > 0
        if not mask_available:
            self._refine_mask_visible = False
            self._refine_mask_painting = False
        apply_mask_controls = getattr(self, "_apply_refine_mask_controls", None)
        if apply_mask_controls is not None:
            apply_mask_controls()
        else:
            set_mask_available = getattr(
                self.view, "set_model_mask_available", None
            )
            if set_mask_available is not None:
                set_mask_available(mask_available)
        if self._displayed_mesh_index == 0:
            self._sync_shape_workspace(project.shape_path)
            if self._refine_cube_visible:
                self._sync_refine_cube(project.refine_cube)
            else:
                self.reconstruction_viewport.clear_refine_cube()
            return
        if not project.shape_path or not Path(project.shape_path).is_file():
            self._displayed_mesh_index = 0
            self._selected_mesh_index = 0
            self._refine_cube_visible = False
            self.view.set_selected_mesh(0)
            self.view.set_refine_cube_visible(False)
            self._sync_shape_workspace(project.shape_path)
            self.reconstruction_viewport.clear_refine_cube()
            return
        signature = self._region_signature(
            project, self._displayed_mesh_index
        )
        refine_mask = project.refine_mask(self._displayed_mesh_index - 1)
        displayed = ("region", signature)
        if displayed == self._displayed_mesh_signature:
            self.reconstruction_viewport.clear_refine_cube()
            self._set_viewport_refine_mask(refine_mask)
            return
        main_signature = (
            "main",
            str(Path(project.shape_path).expanduser().resolve()),
        )
        if self._displayed_mesh_signature != main_signature:
            self._load_shape_in_workspace(project.shape_path, force=True)
        region = project.refine_regions[self._displayed_mesh_index - 1]
        _vertices, triangle_count, _meshes = (
            self.reconstruction_viewport.show_cube_mesh_subset(
                _gltf_point_to_viewport(region.center),
                region.side,
                fit_camera=True,
            )
        )
        if not triangle_count:
            failed_index = self._displayed_mesh_index
            self._displayed_mesh_index = 0
            self._selected_mesh_index = 0
            self._refine_cube_visible = False
            self.view.set_selected_mesh(0)
            self.view.set_refine_cube_visible(False)
            self._sync_shape_workspace(project.shape_path)
            self.reconstruction_viewport.clear_refine_cube()
            self.view.set_status(
                f"Refine region {failed_index} contains no model surface"
            )
            return
        self.reconstruction_viewport.clear_refine_cube()
        self._set_viewport_refine_mask(refine_mask)
        self._displayed_mesh_signature = displayed

    def _load_shape_in_workspace(self, shape_path: str, *, force: bool = False) -> None:
        resolved = str(Path(shape_path).expanduser().resolve())
        displayed = ("main", resolved)
        if (
            not force
            and resolved == self._loaded_shape_path
            and self._displayed_mesh_signature == displayed
        ):
            return
        self.reconstruction_viewport.load_glb(resolved, fit_camera=True)
        self._loaded_shape_path = resolved
        self._displayed_mesh_signature = displayed

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

        adopted_groups = []
        results_changed = False
        for group in self.controller.project.refine_shape_results:
            adopted_group = []
            for result in group:
                source_run = Path(result.manifest_path).expanduser().resolve().parent
                if not source_run.is_relative_to(session_root):
                    adopted_group.append(result)
                    continue
                destination_run = copied_runs.get(source_run)
                if destination_run is None:
                    destination_root = project_root / "refine-runs"
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

                def rebased(value: str) -> str:
                    if not value:
                        return ""
                    value_path = Path(value).expanduser().resolve()
                    try:
                        relative = value_path.relative_to(source_run)
                    except ValueError:
                        relative = Path(value).name
                    return str(destination_run / relative)

                adopted_group.append(RefineShapeResult(
                    geometry_fingerprint=result.geometry_fingerprint,
                    request_key=result.request_key,
                    source_region_path=rebased(result.source_region_path),
                    refined_region_path=rebased(result.refined_region_path),
                    source_slat_path=rebased(result.source_slat_path),
                    refined_slat_path=rebased(result.refined_slat_path),
                    manifest_path=rebased(result.manifest_path),
                    textured_region_path=rebased(result.textured_region_path),
                    texture_key=result.texture_key,
                    texture_shape_slat_path=rebased(
                        result.texture_shape_slat_path
                    ),
                    texture_slat_path=rebased(result.texture_slat_path),
                    texture_manifest_path=rebased(
                        result.texture_manifest_path
                    ),
                ))
                results_changed = True
            adopted_groups.append(tuple(adopted_group))
        if results_changed:
            self.controller.replace_refine_shape_results(tuple(adopted_groups))

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

    def _finish_refine_texture_generation(
        self,
        future: Future,
        expected_project: MultiviewProject,
        region_index: int,
        result_index: int,
        source_key: str,
    ) -> None:
        self._active_future = None
        self._set_busy(False)
        try:
            result = future.result()
        except Exception as error:
            message = str(error)
            if "cancel" in message.lower():
                self.view.set_status("Refined region texturing cancelled")
            else:
                self._show_error("TRELLIS.2 region texturing failed", message)
            self.view.apply_project(
                self.controller.project,
                self.controller.project_path,
                self.controller.dirty,
            )
            return
        if self.controller.project != expected_project:
            self.view.set_status("Ignored region texture for a different project")
            return
        current = self.controller.project.region_refine_results(region_index)
        if (
            not 0 <= result_index < len(current)
            or current[result_index].request_key != source_key
        ):
            self.view.set_status("Ignored texture for a different refine result")
            return
        self.controller.replace_refine_shape_result(
            region_index, result_index, result
        )
        if self.controller.project_path is not None:
            self._adopt_unsaved_artifacts(self.controller.project_path)
            self.controller.save()
        self.select_refine_result(region_index, result_index)
        self.view.set_status(
            f"Textured refined region {region_index + 1}.{result_index + 1} ready · "
            "assembly not run"
        )

    def _finish_refine_generation(
        self,
        future: Future,
        expected_project: MultiviewProject,
        region_index: int,
    ) -> None:
        self._active_future = None
        self._set_busy(False)
        try:
            result = future.result()
        except Exception as error:
            message = str(error)
            if "cancel" in message.lower():
                self.view.set_status("Standalone region refine cancelled")
            else:
                self._show_error("TRELLIS.2 region refine failed", message)
            self.view.apply_project(
                self.controller.project,
                self.controller.project_path,
                self.controller.dirty,
            )
            return
        if self.controller.project != expected_project:
            self.view.set_status("Ignored refine result for a different project")
            return
        self.controller.add_refine_shape_result(region_index, result)
        if self.controller.project_path is not None:
            self._adopt_unsaved_artifacts(self.controller.project_path)
            self.controller.save()
        results = self.controller.project.region_refine_results(region_index)
        result_index = next(
            index
            for index, item in enumerate(results)
            if item.request_key == result.request_key
        )
        self.select_refine_result(region_index, result_index)
        self.view.set_status(
            f"Standalone refined region {region_index + 1} ready · "
            "assembly not run"
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
        log.error(f"{title}: {message}")
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
