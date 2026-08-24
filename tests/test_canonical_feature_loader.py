from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "train-canonical-pointmap-head.py"
SPEC = importlib.util.spec_from_file_location("canonical_head_training", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _samples(tmp_path: Path) -> list[dict]:
    dataset_root = tmp_path / "dataset"
    feature_roots = [tmp_path / "features-a", tmp_path / "features-b"]
    dataset_root.mkdir()
    for feature_root in feature_roots:
        feature_root.mkdir()
    geometry_path = dataset_root / "geometry.npz"
    np.savez_compressed(
        geometry_path,
        canonical_xyz=np.zeros((4, 4, 3), dtype=np.float32),
        mask=np.ones((4, 4), dtype=np.uint8),
    )
    camera_path = dataset_root / "camera.json"
    camera_path.write_text(
        json.dumps(
            {
                "camera_from_canonical": np.eye(4).tolist(),
                "intrinsics": [[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]],
                "image_size": [4, 4],
            }
        ),
        encoding="utf-8",
    )
    samples = []
    for index in range(3):
        feature_root = feature_roots[0 if index < 2 else 1]
        feature_name = f"sample-{index}.npy"
        np.save(
            feature_root / feature_name,
            np.full((2, 2, 2, 3), index, dtype=np.float16),
        )
        samples.append(
            {
                "feature_root": feature_root,
                "dataset_root": dataset_root,
                "dataset_view": {
                    "geometry": geometry_path.name,
                    "camera": camera_path.name,
                },
                "features": feature_name,
                "identity_id": "a" if index < 2 else "b",
                "pose_id": "rest",
                "view_id": f"v{index}",
                "sample_id": f"sample-{index}",
                "normalized_timestep": index / 2,
            }
        )
    return samples


def test_bounded_loader_evicts_without_changing_payload(tmp_path: Path) -> None:
    samples = _samples(tmp_path)
    assert len({sample["feature_root"] for sample in samples}) == 2
    bounded = MODULE.CanonicalFeatureDataset(samples, host_cache_samples=1)
    eager = MODULE.CanonicalFeatureDataset(samples, host_cache_samples=-1)

    first = bounded[0]
    assert np.array_equal(first["features"], eager[0]["features"])
    assert bounded[0] is first
    bounded[1]
    bounded[2]

    info = bounded.cache_info()
    assert info["configured_samples"] == 1
    assert info["resident_samples"] == 1
    assert info["peak_resident_samples"] == 1
    assert info["requests"] == 4
    assert info["hits"] == 1
    assert info["misses"] == 3


def test_identity_balanced_weights_equalize_identity_mass() -> None:
    samples = [
        {"identity_id": "a"},
        {"identity_id": "a"},
        {"identity_id": "b"},
    ]
    weights = MODULE._identity_balanced_weights(samples)
    assert weights == [0.5, 0.5, 1.0]
    assert sum(weights[:2]) == weights[2]


def test_bottleneck_decoder_scope_is_symmetric_except_for_vl_branch() -> None:
    selected = MODULE._parameter_in_trainable_scope

    assert selected("bottleneck.blocks.0.conv1.weight", "bottleneck-decoder")
    assert selected("decoder64.blocks.0.conv1.weight", "bottleneck-decoder")
    assert selected("output.xyz.0.weight", "bottleneck-decoder")
    assert selected("vl_context.attention.in_proj_weight", "bottleneck-decoder")
    assert not selected("projectors.0.0.weight", "bottleneck-decoder")
    assert not selected("encoder64.blocks.0.conv1.weight", "bottleneck-decoder")
    assert selected("vl_context.gate", "vl-only")
    assert not selected("output.xyz.0.weight", "vl-only")
