from __future__ import annotations

import importlib.util
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "extract-qwen-canonical-features.py"
SPEC = importlib.util.spec_from_file_location("canonical_feature_extractor", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_prompt_azimuth_offset_wraps_without_changing_actual_angle() -> None:
    actual = 315.0

    assert MODULE._prompt_azimuth(actual, 90.0) == 45.0
    assert actual == 315.0


def test_offset_angle_selects_supported_semantic_descriptor() -> None:
    prompt_angle = MODULE._prompt_azimuth(0.0, 45.0)

    assert MODULE._prompt(prompt_angle, 0.0) == (
        "<sks> front-right quarter view eye-level shot medium shot"
    )


def test_view_prompt_azimuth_prefers_explicit_nominal_angle() -> None:
    view = {
        "azimuth_degrees": 12.5,
        "prompt_azimuth_degrees": 0.0,
    }
    assert MODULE._view_prompt_azimuth(view) == 0.0
    assert MODULE._signed_angle_delta(12.5, 0.0) == 12.5
