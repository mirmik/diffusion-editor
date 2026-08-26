"""Persistent project model for the native multiview studio."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Iterable


PROJECT_FORMAT = "diffusion-editor-multiview"
PROJECT_VERSION = 1
ELEVATIONS = ("low", "eye", "elevated")
ELEVATION_DEGREES = {"low": -30, "eye": 0, "elevated": 30}
AZIMUTHS = tuple(range(0, 360, 45))


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
    include_in_trellis: bool = True

    @property
    def populated(self) -> bool:
        return bool(self.image_path)


@dataclass(frozen=True)
class TrellisShapeSettings:
    seed: int = 42
    total_steps: int = 39
    warmup_steps: int = 15
    resolution: int = 1024
    decimation_target: int = 250_000

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
class MultiviewProject:
    front_path: str = ""
    back_path: str = ""
    qwen_seed: int = 20_260_822
    slots: tuple[ViewSlot, ...] = tuple(
        ViewSlot(key) for key in all_view_keys()
    )
    trellis: TrellisShapeSettings = TrellisShapeSettings()
    shape_path: str = ""

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
        include_in_trellis: bool | None = None,
    ) -> "MultiviewProject":
        current = self.slot(key)
        updated = replace(
            current,
            image_path=(
                current.image_path if image_path is None else str(image_path)
            ),
            include_in_trellis=(
                current.include_in_trellis
                if include_in_trellis is None
                else bool(include_in_trellis)
            ),
        )
        slots = tuple(updated if slot.key == key else slot for slot in self.slots)
        return replace(self, slots=slots)

    def with_source(self, source: str, image_path: str) -> "MultiviewProject":
        normalized = str(image_path)
        if source == "front":
            project = replace(self, front_path=normalized)
            return project.with_slot(
                ViewKey("eye", 0), image_path=normalized, include_in_trellis=True
            )
        if source == "back":
            project = replace(self, back_path=normalized)
            return project.with_slot(
                ViewKey("eye", 180), image_path=normalized, include_in_trellis=True
            )
        raise ValueError(f"unknown source: {source}")

    def included_slots(self) -> tuple[ViewSlot, ...]:
        return tuple(
            slot
            for slot in self.slots
            if slot.populated and slot.include_in_trellis
        )

    def validate_shape_request(self) -> tuple[str, ...]:
        errors: list[str] = []
        if not self.front_path:
            errors.append("Front image is required for TRELLIS.2 warmup")
        included = self.included_slots()
        if not included:
            errors.append("At least one populated view must be included")
        elif self.trellis.total_steps - self.trellis.warmup_steps < len(included):
            errors.append(
                "Post-warmup steps must cover every included view at least once"
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
            },
            "views": [
                {
                    "elevation": slot.key.elevation,
                    "elevation_degrees": slot.key.elevation_degrees,
                    "azimuth": slot.key.azimuth,
                    "image": _encode_path(slot.image_path, root),
                    "include_in_trellis": slot.include_in_trellis,
                }
                for slot in self.slots
            ],
            "shape": _encode_path(self.shape_path, root),
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
        slot_payloads = {
            ViewKey(str(item["elevation"]), int(item["azimuth"])): item
            for item in payload.get("views", ())
        }
        slots = tuple(
            ViewSlot(
                key,
                _decode_path(slot_payloads.get(key, {}).get("image", ""), root),
                bool(
                    slot_payloads.get(key, {}).get(
                        "include_in_trellis", True
                    )
                ),
            )
            for key in all_view_keys()
        )
        settings = TrellisShapeSettings(
            seed=int(trellis.get("seed", 42)),
            total_steps=int(trellis.get("total_steps", 39)),
            warmup_steps=int(trellis.get("warmup_steps", 15)),
            resolution=int(trellis.get("resolution", 1024)),
            decimation_target=int(trellis.get("decimation_target", 250_000)),
        )
        return cls(
            front_path=_decode_path(source.get("front", ""), root),
            back_path=_decode_path(source.get("back", ""), root),
            qwen_seed=int(qwen.get("seed", 20_260_822)),
            slots=slots,
            trellis=settings,
            shape_path=_decode_path(payload.get("shape", ""), root),
        )


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
    included = tuple(
        slot.key
        for slot in slots
        if slot.populated and slot.include_in_trellis
    )
    if not included:
        raise ValueError("cannot build a view schedule without included views")
    front = ViewKey("eye", 0)
    return tuple(
        front
        if index < warmup_steps
        else included[(index - warmup_steps) % len(included)]
        for index in range(total_steps)
    )
