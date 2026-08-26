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

    def __post_init__(self) -> None:
        keys = tuple(slot.key for slot in self.slots)
        expected = all_view_keys()
        if keys != expected:
            raise ValueError("multiview project must contain the canonical 3x8 grid")
        if not 0 <= self.qwen_seed <= 2_147_483_647:
            raise ValueError("Qwen seed must fit a signed 32-bit integer")

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
                )
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
        if refine_cube is not None and Path(geometry_path).is_file():
            if geometry_fingerprint(geometry_path) != refine_cube.geometry_fingerprint:
                refine_cube = None
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
