import numpy as np

from diffusion_editor.document.layer import Layer
from diffusion_editor.generation.depth_controller import (
    DepthGenerationController,
)
from diffusion_editor.generation.types import (
    DEPTH_ANYTHING_V2_SMALL_MODEL_ID,
    DepthEstimationResult,
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


def test_start_captures_current_composite_for_depth_anything_small():
    layer = Layer("Layer", 8, 6, _rgba(8, 6, (0, 0, 0, 0)))
    composite = _rgba(8, 6, (10, 20, 30, 255))
    engine = _Engine()
    controller = DepthGenerationController(
        engine=engine,
        composite=lambda: composite,
    )

    event = controller.start(layer)

    assert event.status == "Depth Anything V2 Small: starting..."
    assert controller.pending_context.layer_id == layer.id
    assert len(engine.calls) == 1
    assert engine.calls[0].model_id == DEPTH_ANYTHING_V2_SMALL_MODEL_ID
    np.testing.assert_array_equal(engine.calls[0].image, composite)
    assert engine.calls[0].image is not composite


def test_poll_returns_grayscale_depth_map_for_pending_context():
    layer = Layer("Layer", 8, 6, _rgba(8, 6, (0, 0, 0, 0)))
    engine = _Engine()
    controller = DepthGenerationController(
        engine=engine,
        composite=lambda: _rgba(8, 6, (10, 20, 30, 255)),
    )
    controller.start(layer)
    depth_map = np.arange(48, dtype=np.uint8).reshape((6, 8))
    engine.poll_result = EnginePollEvent(
        task_type="depth",
        result=DepthEstimationResult(depth_map),
    )

    event = controller.poll()

    context, result = event.depth_result
    assert context.layer_id == layer.id
    np.testing.assert_array_equal(result, depth_map)
    assert controller.pending_context is None


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
