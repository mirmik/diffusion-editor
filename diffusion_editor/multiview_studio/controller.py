"""Toolkit-neutral state transitions for the multiview studio."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable

from .model import (
    MAX_REFINE_REGIONS,
    MultiviewProject,
    RefineCube,
    RefineFaceMask,
    RefineShapeResult,
    RefineShapeSettings,
    RefineViewPatch,
    ViewKey,
    all_view_keys,
    geometry_fingerprint,
)


ProjectListener = Callable[[MultiviewProject, Path | None, bool], None]


class MultiviewStudioController:
    def __init__(self, project: MultiviewProject | None = None) -> None:
        self.project = project or MultiviewProject()
        self.project_path: Path | None = None
        self.dirty = False
        self._listeners: list[ProjectListener] = []

    def connect(self, listener: ProjectListener) -> None:
        self._listeners.append(listener)
        listener(self.project, self.project_path, self.dirty)

    def new_project(self) -> None:
        self.project = MultiviewProject()
        self.project_path = None
        self.dirty = False
        self._publish()

    def open_project(self, path: str | Path) -> None:
        resolved = Path(path).expanduser().resolve()
        self.project = MultiviewProject.load(resolved)
        self.project_path = resolved
        self.dirty = False
        self._publish()

    def save(self, path: str | Path | None = None) -> Path:
        target = (
            Path(path).expanduser().resolve()
            if path is not None
            else self.project_path
        )
        if target is None:
            raise ValueError("project has no manifest path")
        self.project_path = self.project.save(target)
        self.dirty = False
        self._publish()
        return self.project_path

    def set_source(self, source: str, path: str | Path) -> None:
        self._replace(self.project.with_source(source, str(Path(path).resolve())))

    def set_slot_image(self, key: ViewKey, path: str | Path) -> None:
        self._replace(
            self.project.with_slot(key, image_path=str(Path(path).resolve()))
        )

    def set_slot_images(self, images: dict[ViewKey, str | Path]) -> None:
        project = self.project
        for key, path in images.items():
            project = project.with_slot(
                key, image_path=str(Path(path).resolve())
            )
        self._replace(project)

    def clear_slot(self, key: ViewKey) -> None:
        project = self.project.with_slot(key, image_path="")
        if key == ViewKey("eye", 0):
            project = replace(project, front_path="")
        elif key == ViewKey("eye", 180):
            project = replace(project, back_path="")
        project = replace(
            project,
            refine_view_patches=tuple(
                tuple(patch for patch in patches if patch.key != key)
                for patches in project.refine_view_patches
            ),
        )
        self._replace(project)

    def set_qwen_seed(self, seed: int) -> None:
        self._replace(replace(self.project, qwen_seed=int(seed)))

    def set_trellis_setting(self, field: str, value: int) -> None:
        if field not in {
            "seed",
            "total_steps",
            "warmup_steps",
            "resolution",
            "decimation_target",
        }:
            raise ValueError(f"unknown TRELLIS.2 setting: {field}")
        settings = replace(self.project.trellis, **{field: int(value)})
        self._replace(replace(self.project, trellis=settings))

    def set_mesh_postprocess(self, field: str, value: bool | float) -> None:
        boolean_fields = {
            "fill_holes",
            "remesh",
            "simplify",
            "cleanup",
            "final_repair",
            "remove_isolated_double_faces",
            "remove_degenerate_faces",
        }
        if field in boolean_fields:
            normalized: bool | float = bool(value)
        elif field == "fill_hole_perimeter":
            normalized = float(value)
        else:
            raise ValueError(f"unknown mesh postprocess setting: {field}")
        postprocess = replace(
            self.project.trellis.postprocess,
            **{field: normalized},
        )
        settings = replace(self.project.trellis, postprocess=postprocess)
        self._replace(replace(self.project, trellis=settings))

    def set_texture_setting(self, field: str, value: int) -> None:
        if field not in {
            "seed",
            "total_steps",
            "warmup_steps",
            "resolution",
            "texture_size",
        }:
            raise ValueError(f"unknown texture setting: {field}")
        settings = replace(self.project.texture, **{field: int(value)})
        self._replace(replace(self.project, texture=settings))

    def set_geometry_path(self, path: str | Path) -> None:
        resolved = str(Path(path).resolve())
        self.set_model_paths(resolved, resolved)

    def set_textured_shape_path(self, path: str | Path) -> None:
        self._replace(
            replace(self.project, shape_path=str(Path(path).resolve()))
        )

    def set_model_paths(
        self,
        geometry_path: str | Path,
        shape_path: str | Path,
    ) -> None:
        resolved_geometry = str(Path(geometry_path).resolve())
        cube = self.project.refine_cube
        regions = self.project.refine_regions
        masks = self.project.refine_masks
        view_patches = self.project.refine_view_patches
        refine_settings = self.project.refine_shape_settings
        refine_results = self.project.refine_shape_results
        if cube is not None:
            source = Path(resolved_geometry)
            if (
                not source.is_file()
                or geometry_fingerprint(source) != cube.geometry_fingerprint
            ):
                cube = None
        source = Path(resolved_geometry)
        if source.is_file():
            fingerprint = geometry_fingerprint(source)
            retained = [
                (
                    region,
                    masks[index] if index < len(masks) else None,
                    view_patches[index] if index < len(view_patches) else (),
                    refine_settings[index]
                    if index < len(refine_settings)
                    else RefineShapeSettings(),
                    refine_results[index] if index < len(refine_results) else (),
                )
                for index, region in enumerate(regions)
                if region.geometry_fingerprint == fingerprint
            ]
            regions = tuple(
                region
                for region, _mask, _patches, _settings, _results in retained
            )
            masks = tuple(
                mask
                for _region, mask, _patches, _settings, _results in retained
                if mask is not None
            )
            view_patches = tuple(
                patches
                for _region, _mask, patches, _settings, _results in retained
            )
            refine_settings = tuple(
                settings
                for _region, _mask, _patches, settings, _results in retained
            )
            refine_results = tuple(
                results
                for _region, _mask, _patches, _settings, results in retained
            )
        else:
            regions = ()
            masks = ()
            view_patches = ()
            refine_settings = ()
            refine_results = ()
        self._replace(replace(
            self.project,
            geometry_path=resolved_geometry,
            shape_path=str(Path(shape_path).resolve()),
            refine_cube=cube,
            refine_regions=regions,
            refine_masks=masks,
            refine_view_patches=view_patches,
            refine_shape_settings=refine_settings,
            refine_shape_results=refine_results,
        ))

    def set_shape_path(self, path: str | Path) -> None:
        """Compatibility alias for callers that produce new geometry."""
        self.set_geometry_path(path)

    def set_refine_cube(
        self,
        center: tuple[float, float, float],
        side: float,
        *,
        confirmed: bool = False,
    ) -> None:
        geometry = Path(self.project.geometry_path)
        if not geometry.is_file():
            raise ValueError("a geometry model is required before selecting a cube")
        cube = RefineCube(
            center=center,
            side=side,
            geometry_fingerprint=geometry_fingerprint(geometry),
            confirmed=confirmed,
        )
        self._replace(replace(self.project, refine_cube=cube))

    def update_refine_cube(
        self,
        center: tuple[float, float, float],
        side: float,
    ) -> None:
        """Apply an interactive edit without re-hashing unchanged geometry."""
        cube = self.project.refine_cube
        if cube is None:
            raise ValueError("no refine cube is selected")
        self._replace(replace(
            self.project,
            refine_cube=replace(
                cube,
                center=tuple(map(float, center)),
                side=float(side),
                confirmed=False,
            ),
        ))

    def set_refine_cube_value(self, field: str, value: float) -> None:
        cube = self.project.refine_cube
        if cube is None:
            raise ValueError("no refine cube is selected")
        if field == "side":
            updated = replace(cube, side=float(value), confirmed=False)
        elif field in {"x", "y", "z"}:
            center = list(cube.center)
            center[{"x": 0, "y": 1, "z": 2}[field]] = float(value)
            updated = replace(cube, center=tuple(center), confirmed=False)
        else:
            raise ValueError(f"unknown refine cube field: {field}")
        self._replace(replace(self.project, refine_cube=updated))

    def select_refine_region(self, index: int) -> None:
        if not 0 <= index < len(self.project.refine_regions):
            raise ValueError("refine region index is out of range")
        selected = self.project.refine_regions[index]
        if selected == self.project.refine_cube:
            return
        # The active editor selection is UI state: publish it so numeric
        # controls update, but do not dirty an otherwise unchanged project.
        self.project = replace(self.project, refine_cube=selected)
        self._publish()

    def confirm_refine_cube(self, region_index: int | None = None) -> None:
        cube = self.project.refine_cube
        if cube is None:
            raise ValueError("no refine cube is selected")
        if region_index is not None and not (
            0 <= region_index < len(self.project.refine_regions)
        ):
            raise ValueError("refine region index is out of range")
        if (
            region_index is None
            and len(self.project.refine_regions) >= MAX_REFINE_REGIONS
        ):
            raise ValueError(
                f"a project supports at most {MAX_REFINE_REGIONS} refine regions"
            )
        confirmed = replace(cube, confirmed=True)
        regions = list(self.project.refine_regions)
        masks = list(self.project.refine_masks)
        view_patches = list(self.project.refine_view_patches)
        refine_settings = list(self.project.refine_shape_settings)
        refine_results = list(self.project.refine_shape_results)
        if region_index is None:
            regions.append(confirmed)
            while len(masks) < len(regions) - 1:
                masks.append(RefineFaceMask(regions[len(masks)].geometry_fingerprint))
            masks.append(RefineFaceMask(confirmed.geometry_fingerprint))
            while len(view_patches) < len(regions):
                view_patches.append(())
            while len(refine_settings) < len(regions):
                refine_settings.append(RefineShapeSettings())
            while len(refine_results) < len(regions):
                refine_results.append(())
        else:
            regions[region_index] = confirmed
            while len(refine_results) < len(regions):
                refine_results.append(())
            refine_results[region_index] = ()
        self._replace(replace(
            self.project,
            refine_cube=confirmed,
            refine_regions=tuple(regions),
            refine_masks=tuple(masks),
            refine_view_patches=tuple(view_patches),
            refine_shape_settings=tuple(refine_settings),
            refine_shape_results=tuple(refine_results),
        ))

    def set_refine_shape_setting(
        self, region_index: int, field: str, value: int | float
    ) -> None:
        if not 0 <= region_index < len(self.project.refine_regions):
            raise ValueError("refine region index is out of range")
        if field not in {
            "seed",
            "steps",
            "strength",
            "cfg",
            "resolution",
            "preview_face_target",
        }:
            raise ValueError(f"unknown refine shape setting: {field}")
        settings = list(self.project.refine_shape_settings)
        while len(settings) < len(self.project.refine_regions):
            settings.append(RefineShapeSettings())
        normalized = (
            float(value) if field in {"strength", "cfg"} else int(value)
        )
        settings[region_index] = replace(
            settings[region_index], **{field: normalized}
        )
        self._replace(replace(
            self.project,
            refine_shape_settings=tuple(settings),
        ))

    def add_refine_shape_result(
        self, region_index: int, result: RefineShapeResult
    ) -> None:
        if not 0 <= region_index < len(self.project.refine_regions):
            raise ValueError("refine region index is out of range")
        region = self.project.refine_regions[region_index]
        if result.geometry_fingerprint != region.geometry_fingerprint:
            raise ValueError("refine result geometry does not match its region")
        groups = list(self.project.refine_shape_results)
        while len(groups) < len(self.project.refine_regions):
            groups.append(())
        if any(item.request_key == result.request_key for item in groups[region_index]):
            return
        groups[region_index] = (*groups[region_index], result)
        self._replace(replace(
            self.project,
            refine_shape_results=tuple(groups),
        ))

    def replace_refine_shape_result(
        self,
        region_index: int,
        result_index: int,
        result: RefineShapeResult,
    ) -> None:
        if not 0 <= region_index < len(self.project.refine_regions):
            raise ValueError("refine region index is out of range")
        groups = list(self.project.refine_shape_results)
        if not 0 <= result_index < len(groups[region_index]):
            raise ValueError("refine result index is out of range")
        current = groups[region_index][result_index]
        if result.request_key != current.request_key:
            raise ValueError("replacement refine result identity changed")
        updated = list(groups[region_index])
        updated[result_index] = result
        groups[region_index] = tuple(updated)
        self._replace(replace(
            self.project,
            refine_shape_results=tuple(groups),
        ))

    def replace_refine_shape_results(
        self,
        groups: tuple[tuple[RefineShapeResult, ...], ...],
    ) -> None:
        if len(groups) > len(self.project.refine_regions):
            raise ValueError("refine results must correspond to stored regions")
        self._replace(replace(
            self.project,
            refine_shape_results=groups,
        ))

    def set_refine_view_patch(
        self,
        region_index: int,
        key: ViewKey,
        bounds: tuple[float, float, float, float] | None,
    ) -> None:
        if not 0 <= region_index < len(self.project.refine_regions):
            raise ValueError("refine region index is out of range")
        if bounds is not None and not self.project.slot(key).populated:
            raise ValueError("a populated view is required for a refine patch")
        groups = list(self.project.refine_view_patches)
        while len(groups) < len(self.project.refine_regions):
            groups.append(())
        by_key = {patch.key: patch for patch in groups[region_index]}
        if bounds is None:
            by_key.pop(key, None)
        else:
            by_key[key] = RefineViewPatch(key, bounds)
        groups[region_index] = tuple(
            by_key[view_key]
            for view_key in all_view_keys()
            if view_key in by_key
        )
        self._replace(replace(
            self.project,
            refine_view_patches=tuple(groups),
        ))

    def set_refine_mask(
        self,
        region_index: int,
        mesh_faces: tuple[tuple[int, ...], ...],
    ) -> None:
        if not 0 <= region_index < len(self.project.refine_regions):
            raise ValueError("refine region index is out of range")
        masks = list(self.project.refine_masks)
        while len(masks) < len(self.project.refine_regions):
            region = self.project.refine_regions[len(masks)]
            masks.append(RefineFaceMask(region.geometry_fingerprint))
        region = self.project.refine_regions[region_index]
        masks[region_index] = RefineFaceMask(
            region.geometry_fingerprint,
            mesh_faces,
        )
        self._replace(replace(self.project, refine_masks=tuple(masks)))

    def set_refine_mask_weights(
        self,
        region_index: int,
        mesh_vertex_weights: tuple[tuple[tuple[int, float], ...], ...],
    ) -> None:
        if not 0 <= region_index < len(self.project.refine_regions):
            raise ValueError("refine region index is out of range")
        masks = list(self.project.refine_masks)
        while len(masks) < len(self.project.refine_regions):
            region = self.project.refine_regions[len(masks)]
            masks.append(RefineFaceMask(region.geometry_fingerprint))
        region = self.project.refine_regions[region_index]
        masks[region_index] = RefineFaceMask(
            geometry_fingerprint=region.geometry_fingerprint,
            mesh_vertex_weights=mesh_vertex_weights,
        )
        self._replace(replace(self.project, refine_masks=tuple(masks)))

    def replace_refine_cube(self, cube: RefineCube | None) -> None:
        self._replace(replace(self.project, refine_cube=cube))

    def clear_refine_cube(self) -> None:
        self.replace_refine_cube(None)

    def _replace(self, project: MultiviewProject) -> None:
        if project == self.project:
            return
        self.project = project
        self.dirty = True
        self._publish()

    def _publish(self) -> None:
        for listener in tuple(self._listeners):
            listener(self.project, self.project_path, self.dirty)
