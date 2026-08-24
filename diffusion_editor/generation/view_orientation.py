"""Horizontal view-orientation contract for generated character images."""

from __future__ import annotations

import json
import re
from typing import Any


ORIENTATIONS = ("left", "center", "right", "uncertain")

ORIENTATION_PROMPT = """In which horizontal direction is the rendered
character facing? Use image coordinates:

- left: the nose and chest point toward the LEFT edge of the image.
- right: the nose and chest point toward the RIGHT edge of the image.
- center: the character is shown symmetrically from the front or from the back.
- uncertain: the direction cannot be determined reliably.

Ignore eye gaze and any slight head tilt. Return exactly one lowercase word:
left, center, right, or uncertain."""


def expected_orientation(azimuth_degrees: float) -> str:
    """Map the Multiple Angles camera azimuth to image-space facing.

    Positive camera azimuths see the character's right side, so the character
    points toward the left edge of the resulting image.  The inverse applies
    to negative (315 degree) views.  Exact front and back views are symmetric.
    """

    azimuth = float(azimuth_degrees) % 360.0
    if min(abs(azimuth), abs(azimuth - 180.0), abs(azimuth - 360.0)) < 1e-6:
        return "center"
    if 0.0 < azimuth < 180.0:
        return "left"
    return "right"


def parse_orientation_response(response: str) -> str:
    """Extract one unambiguous orientation label from a VL response."""

    text = response.strip().lower()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced is not None:
        text = fenced.group(1).strip()
    try:
        value: Any = json.loads(text)
    except json.JSONDecodeError:
        value = None
    aliases = {"red": "left", "blue": "right"}
    if isinstance(value, str):
        value = aliases.get(value, value)
        if value in ORIENTATIONS:
            return value
    if isinstance(value, dict):
        direction = value.get("direction")
        if isinstance(direction, str):
            direction = aliases.get(direction.lower(), direction.lower())
            if direction in ORIENTATIONS:
                return direction

    labels = {
        aliases.get(match.group(1), match.group(1))
        for match in re.finditer(
            r"\b(left|center|right|uncertain|red|blue)\b",
            text,
            re.IGNORECASE,
        )
    }
    if len(labels) == 1:
        return labels.pop().lower()
    return "uncertain"


def orientation_verification(
    *,
    azimuth_degrees: float,
    response: str,
) -> dict[str, Any]:
    expected = expected_orientation(azimuth_degrees)
    observed = parse_orientation_response(response)
    return {
        "expected": expected,
        "observed": observed,
        "accepted": observed == expected,
        "raw_response": response.strip(),
    }
