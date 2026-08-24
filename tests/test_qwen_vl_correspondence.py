from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _script(name: str):
    inserted = []
    for dependency in ("cv2", "torch"):
        try:
            __import__(dependency)
        except ImportError:
            sys.modules[dependency] = types.ModuleType(dependency)
            inserted.append(dependency)
    path = REPOSITORY_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    try:
        spec.loader.exec_module(module)
    finally:
        for dependency in inserted:
            sys.modules.pop(dependency, None)
    return module


def test_canonical_correspondence_uses_feature_nearest_neighbour() -> None:
    module = _script("analyze-qwen-vl-canonical-correspondence.py")
    target_features = np.eye(3, dtype=np.float32)
    source_features = target_features[[2, 0, 1]]
    target_xyz = np.array(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    source_xyz = target_xyz[[2, 0, 1]]

    errors, similarities = module._nearest_errors(
        target_features, target_xyz, source_features, source_xyz
    )

    np.testing.assert_allclose(errors, 0.0)
    np.testing.assert_allclose(similarities, 1.0)


def test_pair_analyzer_prefers_real_alpha_over_background_heuristic() -> None:
    module = _script("analyze-qwen-vl-vision-features.py")
    pixels = np.zeros((3, 4, 4), dtype=np.uint8)
    pixels[1, 2] = (100, 120, 140, 255)
    image = Image.fromarray(pixels, "RGBA")

    mask, description = module._subject_mask(
        image, lambda _image: np.ones((3, 4), dtype=bool)
    )

    assert description == "image alpha"
    assert mask.sum() == 1
    assert mask[1, 2]
