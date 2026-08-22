import numpy as np

from diffusion_editor.document.layer import Layer
from diffusion_editor.generation.depth_controller import (
    DepthGenerationController,
)
from diffusion_editor.generation.types import (
    DEFAULT_DEPTH_MODEL_PROFILE_ID,
    DEPTH_ANYTHING_V2_SMALL_MODEL_ID,
    DepthEstimationResult,
    DepthValueKind,
    EnginePollEvent,
)


def _rgba(width, height, color):
    array = np.zeros((height, width, 4), dtype=np.uint8)
    array[:] = color
    return array


class _Engine:
    def __init__(self):
        self.is_busy = False
        self.calls = []
        self.poll_result = None

    def submit_request(self, request):
        self.calls.append(request)
        return True

    def poll_event(self):
        result = self.poll_result
        self.poll_result = None
        return result


def test_start_captures_current_composite_for_best_calibrated_da3_profile():
    layer = Layer("Layer", 8, 6, _rgba(8, 6, (0, 0, 0, 0)))
    composite = _rgba(8, 6, (10, 20, 30, 255))
    engine = _Engine()
    controller = DepthGenerationController(
        engine=engine,
        composite=lambda: composite,
    )

    event = controller.start(layer)

    assert event.status == "DA3 Nested Giant Large 1.1: starting..."
    assert controller.pending_context.layer_id == layer.id
    assert len(engine.calls) == 1
    assert engine.calls[0].profile_id == DEFAULT_DEPTH_MODEL_PROFILE_ID
    assert engine.calls[0].model_id == (
        "depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    np.testing.assert_array_equal(engine.calls[0].image, composite)
    assert engine.calls[0].image is not composite


def test_start_accepts_depth_anything_v2_small_profile():
    layer = Layer("Layer", 8, 6, _rgba(8, 6, (0, 0, 0, 0)))
    engine = _Engine()
    controller = DepthGenerationController(
        engine=engine,
        composite=lambda: _rgba(8, 6, (10, 20, 30, 255)),
    )

    event = controller.start(layer, "v2-small")

    assert event.status == "Depth Anything V2 Small: starting..."
    assert engine.calls[0].profile_id == "v2-small"
    assert engine.calls[0].model_id == DEPTH_ANYTHING_V2_SMALL_MODEL_ID


def test_poll_returns_canonical_float_depth_and_camera_for_pending_context():
    layer = Layer("Layer", 8, 6, _rgba(8, 6, (0, 0, 0, 0)))
    engine = _Engine()
    controller = DepthGenerationController(
        engine=engine,
        composite=lambda: _rgba(8, 6, (10, 20, 30, 255)),
    )
    controller.start(layer)
    depth_map = np.arange(48, dtype=np.float32).reshape((6, 8)) + 1.0
    intrinsics = np.array([
        [10.0, 0.0, 4.0],
        [0.0, 10.0, 3.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    engine.poll_result = EnginePollEvent(
        task_type="depth",
        result=DepthEstimationResult(
            depth_map,
            value_kind=DepthValueKind.DIRECT_METRIC,
            intrinsics=intrinsics,
        ),
    )

    event = controller.poll()

    context, result, output_mask = event.depth_result
    assert context.layer_id == layer.id
    np.testing.assert_array_equal(result.depth_map, depth_map)
    np.testing.assert_array_equal(result.intrinsics, intrinsics)
    assert result.depth_map.dtype == np.float32
    assert output_mask is None
    assert controller.pending_context is None


def test_start_freezes_and_normalizes_output_selection():
    layer = Layer("Layer", 8, 6, _rgba(8, 6, (0, 0, 0, 0)))
    composite = _rgba(8, 6, (10, 20, 30, 255))
    composite[:, :2, :3] = (201, 151, 101)
    selection = np.zeros((6, 8), dtype=np.float32)
    selection[:, 2:6] = 0.5
    engine = _Engine()
    controller = DepthGenerationController(
        engine=engine,
        composite=lambda: composite,
        selection=lambda: selection,
    )

    controller.start(layer)
    selection[:] = 0.0
    composite[:] = 0

    # The inference contract contains the untouched full frame, including
    # pixels outside the selection, and cannot expose the post-output mask.
    assert not hasattr(engine.calls[0], "output_mask")
    assert np.all(engine.calls[0].image[:, :2, :3] == (201, 151, 101))
    parameters = (
        controller.pending_context.request_provenance.to_dict()["parameters"]
    )
    assert parameters["inference_scope"] == "full_frame"
    assert parameters["output_mask_stage"] == "post_inference"

    engine.poll_result = EnginePollEvent(
        task_type="depth",
        result=DepthEstimationResult(
            np.ones((6, 8), dtype=np.float32),
            value_kind=DepthValueKind.DIRECT_METRIC,
        ),
    )
    event = controller.poll()
    _context, _result, output_mask = event.depth_result

    assert np.all(output_mask[:, 2:6] == 128)
    assert np.all(output_mask[:, :2] == 0)


def test_progress_event_keeps_depth_job_pending():
    layer = Layer("Layer", 8, 6, _rgba(8, 6, (0, 0, 0, 0)))
    engine = _Engine()
    controller = DepthGenerationController(
        engine=engine,
        composite=lambda: _rgba(8, 6, (10, 20, 30, 255)),
    )
    controller.start(layer)
    engine.poll_result = EnginePollEvent(
        task_type="depth",
        meta={"progress": "Loading depth model..."},
    )

    event = controller.poll()

    assert event.status == "Loading depth model..."
    assert controller.pending_context is not None
    assert controller.is_busy
