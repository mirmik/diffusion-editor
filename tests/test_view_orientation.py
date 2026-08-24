from __future__ import annotations

import pytest

from diffusion_editor.generation.view_orientation import (
    expected_orientation,
    orientation_verification,
    parse_orientation_response,
)


@pytest.mark.parametrize(
    ("azimuth", "expected"),
    (
        (0, "center"),
        (45, "left"),
        (90, "left"),
        (135, "left"),
        (180, "center"),
        (225, "right"),
        (270, "right"),
        (315, "right"),
        (-45, "right"),
        (360, "center"),
    ),
)
def test_expected_orientation_uses_image_coordinates(
    azimuth: float, expected: str
) -> None:
    assert expected_orientation(azimuth) == expected


@pytest.mark.parametrize(
    ("response", "expected"),
    (
        ("left", "left"),
        ("RIGHT", "right"),
        ('{"direction": "center"}', "center"),
        ("```json\n{\"direction\": \"right\"}\n```", "right"),
        ("red", "left"),
        ("blue", "right"),
        ('{"direction": "blue"}', "right"),
        ("The character points left.", "left"),
        ("It could be left or right.", "uncertain"),
        ("I cannot tell.", "uncertain"),
    ),
)
def test_parse_orientation_response(response: str, expected: str) -> None:
    assert parse_orientation_response(response) == expected


def test_verification_rejects_wrong_or_uncertain_direction() -> None:
    assert orientation_verification(
        azimuth_degrees=315, response="right"
    )["accepted"]
    assert not orientation_verification(
        azimuth_degrees=315, response="left"
    )["accepted"]
    assert not orientation_verification(
        azimuth_degrees=315, response="uncertain"
    )["accepted"]
