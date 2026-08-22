import numpy as np
import pytest

from diffusion_editor.generation.types import (
    DEFAULT_DEPTH_MODEL_PROFILE_ID,
    DEPTH_MODEL_PROFILES,
    DepthBackend,
    DepthEstimationResult,
    DepthValueKind,
    depth_model_profile,
)


def test_depth_profiles_cover_high_quality_and_baseline_models():
    profiles = {profile.stable_id: profile for profile in DEPTH_MODEL_PROFILES}

    assert DEFAULT_DEPTH_MODEL_PROFILE_ID == "da3-nested-giant-large-1.1"
    assert set(profiles) == {
        "da3-nested-giant-large-1.1",
        "da3-mono-large",
        "depth-pro",
        "v2-large",
        "v2-small",
    }
    best = profiles["da3-nested-giant-large-1.1"]
    assert best.value_kind is DepthValueKind.DIRECT_METRIC
    assert best.metric is True
    assert best.predicts_intrinsics is True
    assert best.use_ray_pose is True
    assert profiles["da3-mono-large"].backend is DepthBackend.DA3
    assert profiles["da3-mono-large"].process_resolution == 1008
    assert profiles["depth-pro"].metric is True
    assert profiles["depth-pro"].value_kind is DepthValueKind.DIRECT_METRIC
    assert (
        profiles["da3-mono-large"].value_kind
        is DepthValueKind.DIRECT_SCALE_AMBIGUOUS
    )
    assert profiles["v2-large"].value_kind is DepthValueKind.INVERSE_RELATIVE
    assert profiles["v2-small"].value_kind is DepthValueKind.INVERSE_RELATIVE
    assert depth_model_profile("v2-large").model_id.endswith("V2-Large-hf")
    assert len({profile.layer_name for profile in profiles.values()}) == 5


def test_canonical_depth_result_rejects_quantized_input_and_is_immutable():
    with pytest.raises(ValueError, match="float32"):
        DepthEstimationResult(np.ones((2, 3), dtype=np.uint8))

    source = np.array([[1.25, 2.5]], dtype=np.float32)
    result = DepthEstimationResult(source)
    source[0, 0] = 99.0

    np.testing.assert_array_equal(result.depth_map, [[1.25, 2.5]])
    assert result.depth_map.flags.writeable is False
