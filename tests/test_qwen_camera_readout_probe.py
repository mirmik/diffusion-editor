from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "probe-qwen-camera-readout.py"
SPEC = importlib.util.spec_from_file_location("probe_qwen_camera_readout", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_circular_delta_wraps_at_zero() -> None:
    delta = MODULE.circular_delta_degrees(np.array([359.0, 1.0]), np.array([1.0, 359.0]))
    np.testing.assert_allclose(delta, [-2.0, 2.0])


def test_dual_ridge_recovers_heldout_angles() -> None:
    train_angles = np.arange(0.0, 360.0, 45.0)
    train_targets = MODULE.angle_targets(train_angles)
    train_features = np.column_stack(
        (train_targets, train_targets[:, 0] * 0.5 + train_targets[:, 1] * 0.25)
    )
    model = MODULE.fit_dual_ridge(train_features, train_targets, ridge=1e-6)
    predicted = model.predict_angles(train_features)
    errors = np.abs(MODULE.circular_delta_degrees(predicted, train_angles))
    assert errors.max() < 1e-3


def test_residual_target_separates_prompt_from_actual_camera() -> None:
    data = {
        "actual_angles": np.array([355.0, 57.0]),
        "prompt_angles": np.array([0.0, 45.0]),
    }
    np.testing.assert_allclose(MODULE._target_angles(data, "residual"), [-5.0, 12.0])


def test_block_lookup_handles_different_cache_block_order() -> None:
    assert MODULE._block_index({"blocks": np.array([15, 30, 45, 59])}, 45) == 2
