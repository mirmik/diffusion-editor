"""Persistent project model for the native multiview studio."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


PROJECT_FORMAT = "diffusion-editor-multiview"
PROJECT_VERSION = 1
ELEVATIONS = ("low", "eye", "elevated")
ELEVATION_DEGREES = {"low": -30, "eye": 0, "elevated": 30}
AZIMUTHS = tuple(range(0, 360, 45))
REFINE_COORDINATE_SPACE = "geometry-local-gltf-y-up-v1"
MAX_REFINE_REGIONS = 8


@dataclass(frozen=True, order=True)
class ViewKey:
    elevation: str
    azimuth: int

    def __post_init__(self) -> None:
        if self.elevation not in ELEVATIONS:
            raise ValueError(f"unsupported elevation: {self.elevation}")
        if self.azimuth not in AZIMUTHS:
            raise ValueError(f"unsupported azimuth: {self.azimuth}")

    @property
    def stable_id(self) -> str:
        return f"{self.elevation}-{self.azimuth:03d}"

    @property
    def elevation_degrees(self) -> int:
        return ELEVATION_DEGREES[self.elevation]


def all_view_keys() -> tuple[ViewKey, ...]:
    return tuple(
        ViewKey(elevation, azimuth)
        for elevation in ELEVATIONS
        for azimuth in AZIMUTHS
    )


@dataclass(frozen=True)
class ViewSlot:
    key: ViewKey
    image_path: str = ""

    @property
    def populated(self) -> bool:
        return bool(self.image_path)


@dataclass(frozen=True)
class MeshPostprocessSettings:
    fill_holes: bool = True
    fill_hole_perimeter: float = 0.03
    remesh: bool = True
    simplify: bool = True
    cleanup: bool = True
    final_repair: bool = False
    remove_isolated_double_faces: bool = True
    remove_degenerate_faces: bool = True

    def __post_init__(self) -> None:
        if not 0 < self.fill_hole_perimeter <= 1.0:
            raise ValueError("fill hole perimeter must be in (0, 1]")


@dataclass(frozen=True)
class TrellisShapeSettings:
    seed: int = 42
    total_steps: int = 39
    warmup_steps: int = 15
    resolution: int = 1024
    decimation_target: int = 250_000
    postprocess: MeshPostprocessSettings = MeshPostprocessSettings()

    def __post_init__(self) -> None:
        if not 0 <= self.seed <= 2_147_483_647:
            raise ValueError("TRELLIS.2 seed must fit a signed 32-bit integer")
        if self.total_steps <= 0:
            raise ValueError("TRELLIS.2 total steps must be positive")
        if not 0 <= self.warmup_steps <= self.total_steps:
            raise ValueError("warmup must be between zero and total steps")
        if self.resolution < 1024 or self.resolution % 128:
            raise ValueError(
                "TRELLIS.2 resolution must be a multiple of 128 and at least 1024"
            )
        if self.decimation_target <= 0:
            raise ValueError("decimation target must be positive")


@dataclass(frozen=True)
class TrellisTextureSettings:
    seed: int = 43
    total_steps: int = 39
    warmup_steps: int = 15
    resolution: int = 1024
    texture_size: int = 2048

    def __post_init__(self) -> None:
        if not 0 <= self.seed <= 2_147_483_647:
            raise ValueError("texture seed must fit a signed 32-bit integer")
        if self.total_steps <= 0:
            raise ValueError("texture steps must be positive")
        if not 0 <= self.warmup_steps <= self.total_steps:
            raise ValueError("texture warmup must be between zero and total steps")
        if self.resolution not in {512, 1024}:
            raise ValueError("texture resolution must be 512 or 1024")
        if not 1024 <= self.texture_size <= 4096 or self.texture_size % 512:
            raise ValueError(
                "texture size must be a multiple of 512 from 1024 to 4096"
            )


@dataclass(frozen=True)
class RefineCube:
    """One geometry-local cubic region selected for a refine operation."""

    center: tuple[float, float, float]
    side: float
    geometry_fingerprint: str
    coordinate_space: str = REFINE_COORDINATE_SPACE
    confirmed: bool = False

    def __post_init__(self) -> None:
        center = tuple(float(value) for value in self.center)
        if len(center) != 3 or not all(math.isfinite(value) for value in center):
            raise ValueError("refine cube center must contain three finite values")
        if not math.isfinite(self.side) or self.side <= 0.0:
            raise ValueError("refine cube side must be finite and positive")
        if not self.geometry_fingerprint:
            raise ValueError("refine cube must identify its source geometry")
        if self.coordinate_space != REFINE_COORDINATE_SPACE:
            raise ValueError(
                f"unsupported refine coordinate space: {self.coordinate_space}"
            )
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "side", float(self.side))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "axis-aligned-cube",
            "coordinate_space": self.coordinate_space,
            "center": list(self.center),
            "side": self.side,
            "geometry_fingerprint": self.geometry_fingerprint,
            "confirmed": self.confirmed,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RefineCube":
        if payload.get("kind", "axis-aligned-cube") != "axis-aligned-cube":
            raise ValueError(f"unsupported refine region kind: {payload.get('kind')}")
        return cls(
            center=tuple(float(value) for value in payload["center"]),
            side=float(payload["side"]),
            geometry_fingerprint=str(payload["geometry_fingerprint"]),
            coordinate_space=str(
                payload.get("coordinate_space", REFINE_COORDINATE_SPACE)
            ),
            confirmed=bool(payload.get("confirmed", False)),
        )


@dataclass(frozen=True)
class RefineFaceMask:
    """Sparse vertex weights with legacy binary face-mask compatibility."""

    geometry_fingerprint: str
    mesh_faces: tuple[tuple[int, ...], ...] = ()
    mesh_vertex_weights: tuple[tuple[tuple[int, float], ...], ...] = ()

    def __post_init__(self) -> None:
        if not self.geometry_fingerprint:
            raise ValueError("refine mask must identify its source geometry")
        normalized = []
        for faces in self.mesh_faces:
            values = tuple(sorted({int(face) for face in faces}))
            if any(face < 0 for face in values):
                raise ValueError("refine mask face indexes must be non-negative")
            normalized.append(values)
        object.__setattr__(self, "mesh_faces", tuple(normalized))
        normalized_weights = []
        for weights in self.mesh_vertex_weights:
            by_vertex: dict[int, float] = {}
            for vertex, weight in weights:
                vertex = int(vertex)
                weight = float(weight)
                if vertex < 0:
                    raise ValueError(
                        "refine mask vertex indexes must be non-negative"
                    )
                if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
                    raise ValueError("refine mask weights must be from zero to one")
                if weight > 1e-6:
                    by_vertex[vertex] = weight
            normalized_weights.append(tuple(sorted(by_vertex.items())))
        object.__setattr__(
            self, "mesh_vertex_weights", tuple(normalized_weights)
        )

    def to_dict(self) -> dict[str, Any]:
        if self.mesh_vertex_weights or not self.mesh_faces:
            return {
                "kind": "weighted-source-vertices",
                "geometry_fingerprint": self.geometry_fingerprint,
                "mesh_vertex_weights": [
                    [[vertex, weight] for vertex, weight in weights]
                    for weights in self.mesh_vertex_weights
                ],
            }
        return {
            "kind": "binary-source-faces",
            "geometry_fingerprint": self.geometry_fingerprint,
            "mesh_faces": [list(faces) for faces in self.mesh_faces],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RefineFaceMask":
        kind = payload.get("kind", "binary-source-faces")
        if kind == "weighted-source-vertices":
            return cls(
                geometry_fingerprint=str(payload["geometry_fingerprint"]),
                mesh_vertex_weights=tuple(
                    tuple((int(vertex), float(weight)) for vertex, weight in weights)
                    for weights in payload.get("mesh_vertex_weights", ())
                ),
            )
        if kind != "binary-source-faces":
            raise ValueError(f"unsupported refine mask kind: {kind}")
        return cls(
            geometry_fingerprint=str(payload["geometry_fingerprint"]),
            mesh_faces=tuple(
                tuple(int(face) for face in faces)
                for faces in payload.get("mesh_faces", ())
            ),
        )


@dataclass(frozen=True)
class RefineViewPatch:
    """A normalized image rectangle used to condition one refine region."""

    key: ViewKey
    bounds: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        bounds = tuple(float(value) for value in self.bounds)
        if len(bounds) != 4 or not all(
            math.isfinite(value) for value in bounds
        ):
            raise ValueError("refine view patch must contain four finite bounds")
        x0, y0, x1, y1 = bounds
        if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
            raise ValueError(
                "refine view patch bounds must form a rectangle in [0, 1]"
            )
        object.__setattr__(self, "bounds", bounds)

    def to_dict(self) -> dict[str, Any]:
        x0, y0, x1, y1 = self.bounds
        return {
            "elevation": self.key.elevation,
            "azimuth": self.key.azimuth,
            "x": x0,
            "y": y0,
            "width": x1 - x0,
            "height": y1 - y0,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RefineViewPatch":
        x = float(payload["x"])
        y = float(payload["y"])
        right = x + float(payload["width"])
        bottom = y + float(payload["height"])
        if math.isclose(right, 1.0, abs_tol=1e-9):
            right = 1.0
        if math.isclose(bottom, 1.0, abs_tol=1e-9):
            bottom = 1.0
        return cls(
            key=ViewKey(str(payload["elevation"]), int(payload["azimuth"])),
            bounds=(
                x,
                y,
                right,
                bottom,
            ),
        )


@dataclass(frozen=True)
class RefineShapeSettings:
    seed: int = 44
    steps: int = 25
    strength: float = 1.0
    cfg: float = 5.0
    resolution: int = 1024
    preview_face_target: int = 250_000

    def __post_init__(self) -> None:
        if not 0 <= self.seed <= 2_147_483_647:
            raise ValueError("refine seed must fit a signed 32-bit integer")
        if not 1 <= self.steps <= 100:
            raise ValueError("refine steps must be from 1 to 100")
        if not math.isfinite(self.strength) or not 0.0 <= self.strength <= 1.0:
            raise ValueError("refine strength must be from zero to one")
        if not math.isfinite(self.cfg) or not 0.0 <= self.cfg <= 20.0:
            raise ValueError("refine CFG must be from zero to 20")
        if not 1024 <= self.resolution <= 1536 or self.resolution % 128:
            raise ValueError(
                "refine resolution must be a multiple of 128 from 1024 to 1536"
            )
        if self.preview_face_target <= 0:
            raise ValueError("refine preview face target must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "steps": self.steps,
            "strength": self.strength,
            "cfg": self.cfg,
            "resolution": self.resolution,
            "preview_face_target": self.preview_face_target,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RefineShapeSettings":
        return cls(
            seed=int(payload.get("seed", 44)),
            steps=int(payload.get("steps", 25)),
            strength=float(payload.get("strength", 1.0)),
            cfg=float(payload.get("cfg", 5.0)),
            resolution=int(payload.get("resolution", 1024)),
            preview_face_target=int(payload.get("preview_face_target", 250_000)),
        )


@dataclass(frozen=True)
class RefineShapeResult:
    geometry_fingerprint: str
    request_key: str
    source_region_path: str
    refined_region_path: str
    source_slat_path: str
    refined_slat_path: str
    manifest_path: str
    textured_region_path: str = ""
    texture_key: str = ""
    texture_shape_slat_path: str = ""
    texture_slat_path: str = ""
    texture_manifest_path: str = ""

    def __post_init__(self) -> None:
        if not self.geometry_fingerprint:
            raise ValueError("refine result must identify its source geometry")
        if not self.request_key:
            raise ValueError("refine result must have a request key")
        if not self.refined_region_path:
            raise ValueError("refine result must have a refined region mesh")

    def to_dict(self, root: Path) -> dict[str, Any]:
        return {
            "geometry_fingerprint": self.geometry_fingerprint,
            "request_key": self.request_key,
            "source_region": _encode_path(self.source_region_path, root),
            "refined_region": _encode_path(self.refined_region_path, root),
            "source_slat": _encode_path(self.source_slat_path, root),
            "refined_slat": _encode_path(self.refined_slat_path, root),
            "manifest": _encode_path(self.manifest_path, root),
            "textured_region": _encode_path(self.textured_region_path, root),
            "texture_key": self.texture_key,
            "texture_shape_slat": _encode_path(
                self.texture_shape_slat_path, root
            ),
            "texture_slat": _encode_path(self.texture_slat_path, root),
            "texture_manifest": _encode_path(
                self.texture_manifest_path, root
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any], root: Path) -> "RefineShapeResult":
        return cls(
            geometry_fingerprint=str(payload["geometry_fingerprint"]),
            request_key=str(payload["request_key"]),
            source_region_path=_decode_path(payload.get("source_region", ""), root),
            refined_region_path=_decode_path(payload["refined_region"], root),
            source_slat_path=_decode_path(payload.get("source_slat", ""), root),
            refined_slat_path=_decode_path(payload.get("refined_slat", ""), root),
            manifest_path=_decode_path(payload.get("manifest", ""), root),
            textured_region_path=_decode_path(
                payload.get("textured_region", ""), root
            ),
            texture_key=str(payload.get("texture_key", "")),
            texture_shape_slat_path=_decode_path(
                payload.get("texture_shape_slat", ""), root
            ),
            texture_slat_path=_decode_path(
                payload.get("texture_slat", ""), root
            ),
            texture_manifest_path=_decode_path(
                payload.get("texture_manifest", ""), root
            ),
        )


@dataclass(frozen=True)
class MultiviewProject:
    front_path: str = ""
    back_path: str = ""
    qwen_seed: int = 20_260_822
    slots: tuple[ViewSlot, ...] = tuple(
        ViewSlot(key) for key in all_view_keys()
    )
    trellis: TrellisShapeSettings = TrellisShapeSettings()
    texture: TrellisTextureSettings = TrellisTextureSettings()
    geometry_path: str = ""
    shape_path: str = ""
    refine_cube: RefineCube | None = None
    refine_regions: tuple[RefineCube, ...] = ()
    refine_masks: tuple[RefineFaceMask, ...] = ()
    refine_view_patches: tuple[tuple[RefineViewPatch, ...], ...] = ()
    refine_shape_settings: tuple[RefineShapeSettings, ...] = ()
    refine_shape_results: tuple[tuple[RefineShapeResult, ...], ...] = ()

    def __post_init__(self) -> None:
        keys = tuple(slot.key for slot in self.slots)
        expected = all_view_keys()
        if keys != expected:
            raise ValueError("multiview project must contain the canonical 3x8 grid")
        if not 0 <= self.qwen_seed <= 2_147_483_647:
            raise ValueError("Qwen seed must fit a signed 32-bit integer")
        if len(self.refine_regions) > MAX_REFINE_REGIONS:
            raise ValueError(
                f"a project supports at most {MAX_REFINE_REGIONS} refine regions"
            )
        if any(not region.confirmed for region in self.refine_regions):
            raise ValueError("stored refine regions must be confirmed")
        if len(self.refine_masks) > len(self.refine_regions):
            raise ValueError("refine masks must correspond to stored regions")
        if len(self.refine_view_patches) > len(self.refine_regions):
            raise ValueError("refine view patches must correspond to stored regions")
        if len(self.refine_shape_settings) > len(self.refine_regions):
            raise ValueError("refine settings must correspond to stored regions")
        if len(self.refine_shape_results) > len(self.refine_regions):
            raise ValueError("refine results must correspond to stored regions")
        for index, mask in enumerate(self.refine_masks):
            if (
                mask.geometry_fingerprint
                != self.refine_regions[index].geometry_fingerprint
            ):
                raise ValueError("refine mask geometry does not match its region")
        for patches in self.refine_view_patches:
            keys = tuple(patch.key for patch in patches)
            if len(keys) != len(set(keys)):
                raise ValueError("a refine region cannot repeat a view patch")
            if keys != tuple(sorted(keys, key=all_view_keys().index)):
                raise ValueError("refine view patches must use canonical view order")
        for index, results in enumerate(self.refine_shape_results):
            fingerprint = self.refine_regions[index].geometry_fingerprint
            if any(
                result.geometry_fingerprint != fingerprint for result in results
            ):
                raise ValueError("refine result geometry does not match its region")

    def refine_mask(self, index: int) -> RefineFaceMask:
        if not 0 <= index < len(self.refine_regions):
            raise ValueError("refine region index is out of range")
        if index < len(self.refine_masks):
            return self.refine_masks[index]
        return RefineFaceMask(self.refine_regions[index].geometry_fingerprint)

    def view_patches(self, index: int) -> tuple[RefineViewPatch, ...]:
        if not 0 <= index < len(self.refine_regions):
            raise ValueError("refine region index is out of range")
        if index < len(self.refine_view_patches):
            return self.refine_view_patches[index]
        return ()

    def view_patch(self, index: int, key: ViewKey) -> RefineViewPatch | None:
        return next(
            (patch for patch in self.view_patches(index) if patch.key == key),
            None,
        )

    def region_refine_settings(self, index: int) -> RefineShapeSettings:
        if not 0 <= index < len(self.refine_regions):
            raise ValueError("refine region index is out of range")
        if index < len(self.refine_shape_settings):
            return self.refine_shape_settings[index]
        return RefineShapeSettings()

    def region_refine_results(self, index: int) -> tuple[RefineShapeResult, ...]:
        if not 0 <= index < len(self.refine_regions):
            raise ValueError("refine region index is out of range")
        if index < len(self.refine_shape_results):
            return self.refine_shape_results[index]
        return ()

    def validate_refine_request(self, index: int) -> tuple[str, ...]:
        if not 0 <= index < len(self.refine_regions):
            return ("Select a refine region",)
        errors = []
        mask = self.refine_mask(index)
        if not any(mask.mesh_vertex_weights) and not any(mask.mesh_faces):
            errors.append("Paint a non-empty refine mask")
        patches = self.view_patches(index)
        if not patches:
            errors.append("Assign at least one conditioning view patch")
        for patch in patches:
            slot = self.slot(patch.key)
            if not slot.populated or not Path(slot.image_path).is_file():
                errors.append(f"Conditioning view is missing: {patch.key.stable_id}")
        if not self.geometry_path or not Path(self.geometry_path).is_file():
            errors.append("Source geometry is missing")
        return tuple(errors)

    def validate_refine_texture_request(
        self, region_index: int, result_index: int
    ) -> tuple[str, ...]:
        if not 0 <= region_index < len(self.refine_regions):
            return ("Select a refine region",)
        results = self.region_refine_results(region_index)
        if not 0 <= result_index < len(results):
            return ("Select a refined region result",)
        errors: list[str] = []
        result = results[result_index]
        if not result.refined_region_path or not Path(
            result.refined_region_path
        ).is_file():
            errors.append("Refined region mesh is missing")
        patches = self.view_patches(region_index)
        if not patches:
            errors.append("Assign at least one conditioning view patch")
        elif self.texture.total_steps < len(patches):
            errors.append("Texture steps must cover every region view patch once")
        for patch in patches:
            slot = self.slot(patch.key)
            if not slot.populated or not Path(slot.image_path).is_file():
                errors.append(
                    f"Conditioning view is missing: {patch.key.stable_id}"
                )
        return tuple(errors)

    def slot(self, key: ViewKey) -> ViewSlot:
        return self.slots[all_view_keys().index(key)]

    def with_slot(
        self,
        key: ViewKey,
        *,
        image_path: str | None = None,
    ) -> "MultiviewProject":
        current = self.slot(key)
        updated = replace(
            current,
            image_path=(
                current.image_path if image_path is None else str(image_path)
            ),
        )
        slots = tuple(updated if slot.key == key else slot for slot in self.slots)
        return replace(self, slots=slots)

    def with_source(self, source: str, image_path: str) -> "MultiviewProject":
        normalized = str(image_path)
        if source == "front":
            project = replace(self, front_path=normalized)
            return project.with_slot(ViewKey("eye", 0), image_path=normalized)
        if source == "back":
            project = replace(self, back_path=normalized)
            return project.with_slot(ViewKey("eye", 180), image_path=normalized)
        raise ValueError(f"unknown source: {source}")

    def populated_slots(self) -> tuple[ViewSlot, ...]:
        """Every installed view participates in 3D reconstruction."""
        return tuple(slot for slot in self.slots if slot.populated)

    def validate_shape_request(self) -> tuple[str, ...]:
        errors: list[str] = []
        if not self.front_path:
            errors.append("Front image is required for TRELLIS.2 warmup")
        populated = self.populated_slots()
        if not populated:
            errors.append("At least one populated view is required")
        elif self.trellis.total_steps - self.trellis.warmup_steps < len(populated):
            errors.append(
                "Post-warmup steps must cover every populated view at least once"
            )
        return tuple(errors)

    def validate_texture_request(self) -> tuple[str, ...]:
        errors: list[str] = []
        if not self.geometry_path:
            errors.append("Build or provide a model before texturing")
        populated = self.populated_slots()
        if not populated:
            errors.append("At least one populated view is required for texturing")
        if self.texture.warmup_steps and not self.front_path:
            errors.append("Front image is required for texture warmup")
        if (
            populated
            and self.texture.total_steps - self.texture.warmup_steps
            < len(populated)
        ):
            errors.append(
                "Texture post-warmup steps must cover every populated view once"
            )
        return tuple(errors)

    def to_dict(self, manifest_path: Path) -> dict[str, Any]:
        root = manifest_path.expanduser().resolve().parent
        return {
            "format": PROJECT_FORMAT,
            "version": PROJECT_VERSION,
            "sources": {
                "front": _encode_path(self.front_path, root),
                "back": _encode_path(self.back_path, root),
            },
            "qwen": {"seed": self.qwen_seed},
            "trellis": {
                "seed": self.trellis.seed,
                "total_steps": self.trellis.total_steps,
                "warmup_steps": self.trellis.warmup_steps,
                "resolution": self.trellis.resolution,
                "decimation_target": self.trellis.decimation_target,
                "postprocess": {
                    "fill_holes": self.trellis.postprocess.fill_holes,
                    "fill_hole_perimeter": (
                        self.trellis.postprocess.fill_hole_perimeter
                    ),
                    "remesh": self.trellis.postprocess.remesh,
                    "simplify": self.trellis.postprocess.simplify,
                    "cleanup": self.trellis.postprocess.cleanup,
                    "final_repair": self.trellis.postprocess.final_repair,
                    "remove_isolated_double_faces": (
                        self.trellis.postprocess.remove_isolated_double_faces
                    ),
                    "remove_degenerate_faces": (
                        self.trellis.postprocess.remove_degenerate_faces
                    ),
                },
            },
            "texture": {
                "seed": self.texture.seed,
                "total_steps": self.texture.total_steps,
                "warmup_steps": self.texture.warmup_steps,
                "resolution": self.texture.resolution,
                "texture_size": self.texture.texture_size,
            },
            "views": [
                {
                    "elevation": slot.key.elevation,
                    "elevation_degrees": slot.key.elevation_degrees,
                    "azimuth": slot.key.azimuth,
                    "image": _encode_path(slot.image_path, root),
                }
                for slot in self.slots
            ],
            "geometry": _encode_path(self.geometry_path, root),
            "shape": _encode_path(self.shape_path, root),
            "refine": {
                "cube": (
                    self.refine_cube.to_dict()
                    if self.refine_cube is not None
                    else None
                ),
                "regions": [region.to_dict() for region in self.refine_regions],
                "masks": [mask.to_dict() for mask in self.refine_masks],
                "view_patches": [
                    [patch.to_dict() for patch in patches]
                    for patches in self.refine_view_patches
                ],
                "shape_settings": [
                    settings.to_dict() for settings in self.refine_shape_settings
                ],
                "shape_results": [
                    [result.to_dict(root) for result in results]
                    for results in self.refine_shape_results
                ],
            },
        }

    def save(self, manifest_path: str | Path) -> Path:
        path = Path(manifest_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(path), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, manifest_path: str | Path) -> "MultiviewProject":
        path = Path(manifest_path).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != PROJECT_FORMAT:
            raise ValueError(f"not a {PROJECT_FORMAT} project")
        if int(payload.get("version", 0)) != PROJECT_VERSION:
            raise ValueError(
                f"unsupported multiview project version: {payload.get('version')}"
            )
        root = path.parent
        source = payload.get("sources", {})
        qwen = payload.get("qwen", {})
        trellis = payload.get("trellis", {})
        texture = payload.get("texture", {})
        postprocess = trellis.get("postprocess", {})
        slot_payloads = {
            ViewKey(str(item["elevation"]), int(item["azimuth"])): item
            for item in payload.get("views", ())
        }
        slots = tuple(
            ViewSlot(
                key,
                _decode_path(slot_payloads.get(key, {}).get("image", ""), root),
            )
            for key in all_view_keys()
        )
        settings = TrellisShapeSettings(
            seed=int(trellis.get("seed", 42)),
            total_steps=int(trellis.get("total_steps", 39)),
            warmup_steps=int(trellis.get("warmup_steps", 15)),
            resolution=int(trellis.get("resolution", 1024)),
            decimation_target=int(trellis.get("decimation_target", 250_000)),
            postprocess=MeshPostprocessSettings(
                fill_holes=bool(postprocess.get("fill_holes", True)),
                fill_hole_perimeter=float(
                    postprocess.get("fill_hole_perimeter", 0.03)
                ),
                remesh=bool(postprocess.get("remesh", True)),
                simplify=bool(postprocess.get("simplify", True)),
                cleanup=bool(postprocess.get("cleanup", True)),
                final_repair=(
                    bool(postprocess.get("final_repair", False))
                    if "remove_degenerate_faces" in postprocess
                    else False
                ),
                remove_isolated_double_faces=bool(
                    postprocess.get("remove_isolated_double_faces", True)
                ),
                remove_degenerate_faces=bool(
                    postprocess.get("remove_degenerate_faces", True)
                ),
            ),
        )
        texture_settings = TrellisTextureSettings(
            seed=int(texture.get("seed", 43)),
            total_steps=int(texture.get("total_steps", 39)),
            warmup_steps=int(texture.get("warmup_steps", 15)),
            resolution=int(texture.get("resolution", 1024)),
            texture_size=int(texture.get("texture_size", 2048)),
        )
        shape_path = _decode_path(payload.get("shape", ""), root)
        geometry_path = _decode_path(
            payload.get("geometry", payload.get("shape", "")), root
        )
        refine_payload = payload.get("refine", {})
        cube_payload = (
            refine_payload.get("cube")
            if isinstance(refine_payload, dict)
            else None
        )
        refine_cube = RefineCube.from_dict(cube_payload) if cube_payload else None
        region_payloads = (
            refine_payload.get("regions", ())
            if isinstance(refine_payload, dict)
            else ()
        )
        refine_regions = tuple(
            RefineCube.from_dict(item) for item in region_payloads
        )
        mask_payloads = (
            refine_payload.get("masks", ())
            if isinstance(refine_payload, dict)
            else ()
        )
        refine_masks = tuple(
            RefineFaceMask.from_dict(item) for item in mask_payloads
        )
        patch_payloads = (
            refine_payload.get("view_patches", ())
            if isinstance(refine_payload, dict)
            else ()
        )
        refine_view_patches = tuple(
            tuple(RefineViewPatch.from_dict(item) for item in patches)
            for patches in patch_payloads
        )
        settings_payloads = (
            refine_payload.get("shape_settings", ())
            if isinstance(refine_payload, dict)
            else ()
        )
        refine_shape_settings = tuple(
            RefineShapeSettings.from_dict(item) for item in settings_payloads
        )
        results_payloads = (
            refine_payload.get("shape_results", ())
            if isinstance(refine_payload, dict)
            else ()
        )
        refine_shape_results = tuple(
            tuple(RefineShapeResult.from_dict(item, root) for item in results)
            for results in results_payloads
        )
        if len(refine_masks) > len(refine_regions):
            raise ValueError("refine masks must correspond to stored regions")
        if len(refine_view_patches) > len(refine_regions):
            raise ValueError("refine view patches must correspond to stored regions")
        if len(refine_shape_settings) > len(refine_regions):
            raise ValueError("refine settings must correspond to stored regions")
        if len(refine_shape_results) > len(refine_regions):
            raise ValueError("refine results must correspond to stored regions")
        if len(refine_regions) > MAX_REFINE_REGIONS:
            raise ValueError(
                f"a project supports at most {MAX_REFINE_REGIONS} refine regions"
            )
        if Path(geometry_path).is_file():
            fingerprint = geometry_fingerprint(geometry_path)
            if (
                refine_cube is not None
                and fingerprint != refine_cube.geometry_fingerprint
            ):
                refine_cube = None
            retained = [
                (
                    region,
                    refine_masks[index] if index < len(refine_masks) else None,
                    refine_view_patches[index]
                    if index < len(refine_view_patches)
                    else (),
                    refine_shape_settings[index]
                    if index < len(refine_shape_settings)
                    else RefineShapeSettings(),
                    refine_shape_results[index]
                    if index < len(refine_shape_results)
                    else (),
                )
                for index, region in enumerate(refine_regions)
                if region.geometry_fingerprint == fingerprint
            ]
            refine_regions = tuple(
                region
                for region, _mask, _patches, _settings, _results in retained
            )
            refine_masks = tuple(
                mask
                for region, mask, _patches, _settings, _results in retained
                if mask is not None
                and mask.geometry_fingerprint == region.geometry_fingerprint
            )
            refine_view_patches = tuple(
                patches
                for _region, _mask, patches, _settings, _results in retained
            )
            refine_shape_settings = tuple(
                settings
                for _region, _mask, _patches, settings, _results in retained
            )
            refine_shape_results = tuple(
                results
                for _region, _mask, _patches, _settings, results in retained
            )
        return cls(
            front_path=_decode_path(source.get("front", ""), root),
            back_path=_decode_path(source.get("back", ""), root),
            qwen_seed=int(qwen.get("seed", 20_260_822)),
            slots=slots,
            trellis=settings,
            texture=texture_settings,
            geometry_path=geometry_path,
            shape_path=shape_path,
            refine_cube=refine_cube,
            refine_regions=refine_regions,
            refine_masks=refine_masks,
            refine_view_patches=refine_view_patches,
            refine_shape_settings=refine_shape_settings,
            refine_shape_results=refine_shape_results,
        )


def geometry_fingerprint(path: str | Path) -> str:
    """Return a stable content identity for geometry-bound project state."""
    resolved = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _encode_path(value: str, root: Path) -> str:
    if not value:
        return ""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _decode_path(value: Any, root: Path) -> str:
    if not value:
        return ""
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = root / path
    return str(path.resolve())


def view_schedule(
    slots: Iterable[ViewSlot], total_steps: int, warmup_steps: int
) -> tuple[ViewKey, ...]:
    """Return the exact stable view order used by shape denoising."""
    populated = tuple(slot.key for slot in slots if slot.populated)
    if not populated:
        raise ValueError("cannot build a view schedule without populated views")
    front = ViewKey("eye", 0)
    return tuple(
        front
        if index < warmup_steps
        else populated[(index - warmup_steps) % len(populated)]
        for index in range(total_steps)
    )
