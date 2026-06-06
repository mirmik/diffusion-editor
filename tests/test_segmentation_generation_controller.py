import numpy as np

from diffusion_editor.generation.types import (
    EnginePollEvent,
    SegmentationResult,
)
from diffusion_editor.document.layer import Layer
from diffusion_editor.generation.segmentation_controller import (
    SegmentationGenerationController,
)


def _rgba(width, height, color):
    arr = np.zeros((height, width, 4), dtype=np.uint8)
    arr[:] = color
    return arr


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


def test_start_select_background_submits_segmentation_request():
    layer = Layer("Layer", 8, 8, _rgba(8, 8, (0, 0, 0, 0)))
    composite = _rgba(8, 8, (10, 20, 30, 255))
    engine = _Engine()
    controller = SegmentationGenerationController(
        engine=engine,
        composite_below=lambda _layer: composite,
    )

    event = controller.start_select_background(layer)

    assert event.status == "Segmenting background..."
    assert controller.pending_layer is layer
    assert len(engine.calls) == 1
    assert engine.calls[0].image is composite
    assert engine.calls[0].invert is True


def test_poll_returns_pending_layer_and_mask():
    layer = Layer("Layer", 8, 8, _rgba(8, 8, (0, 0, 0, 0)))
    engine = _Engine()
    controller = SegmentationGenerationController(
        engine=engine,
        composite_below=lambda _layer: _rgba(8, 8, (10, 20, 30, 255)),
    )
    controller.start_select_background(layer)
    mask = np.ones((8, 8), dtype=np.uint8) * 255
    engine.poll_result = EnginePollEvent(
        task_type="segmentation",
        result=SegmentationResult(mask=mask),
    )

    event = controller.poll()

    assert event.segmentation_result == (layer, mask)
    assert controller.pending_layer is None
