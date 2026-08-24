"""Pure selection rules for character meshes in Blender dataset scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CharacterMeshCandidate:
    name: str
    render_hidden: bool = False
    viewport_visible: bool = True


def select_character_mesh_names(
    candidates: Iterable[CharacterMeshCandidate],
    *,
    mode: str,
    include_meshes: Iterable[str] = (),
    only_meshes: Iterable[str] = (),
) -> list[str]:
    """Return selected names while preserving the authored scene order.

    ``include_meshes`` keeps the historical additive behavior. ``only_meshes``
    is an exact allow-list used for Blend files that contain several character
    variants in one visible scene.
    """

    values = list(candidates)
    include = set(include_meshes)
    only = set(only_meshes)
    if include and only:
        raise ValueError("include_meshes and only_meshes are mutually exclusive")
    available = {candidate.name for candidate in values}
    if only:
        missing = sorted(only - available)
        if missing:
            raise ValueError(
                "requested character meshes are missing: " + ", ".join(missing)
            )
        return [candidate.name for candidate in values if candidate.name in only]

    selected = []
    for candidate in values:
        if candidate.name in include:
            selected.append(candidate.name)
            continue
        if candidate.render_hidden:
            continue
        if mode == "rain":
            if candidate.name.startswith("GEO-rain_"):
                selected.append(candidate.name)
        elif mode == "opened-blend":
            upper = candidate.name.upper()
            if not upper.startswith(("WGT-", "HLP-")) and candidate.viewport_visible:
                selected.append(candidate.name)
        else:
            selected.append(candidate.name)
    return selected
