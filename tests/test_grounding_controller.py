import numpy as np

from diffusion_editor.grounding.controller import GroundingController
from diffusion_editor.grounding.types import (
    GroundingDetection,
    GroundingEngineEvent,
    GroundingParams,
    GroundingResult,
)
from diffusion_editor.document.layer import Layer


def _params():
    return GroundingParams(
        prompt="cup.",
        model_id="dino",
        box_threshold=0.4,
        text_threshold=0.3,
        use_gpu=False,
        sam2_model_id=None,
        sam2_mask_channel=0,
        mask_threshold=0.0,
        max_hole_area=0,
        max_sprinkle_area=0,
        multimask=True,
        non_overlap=False,
    )


class _Engine:
    def __init__(self):
        self.is_busy = False
        self.calls = []
        self.poll_result = None

    def gpu_available(self):
        return False

    def submit_request(self, request):
        self.calls.append(request)
        return True

    def poll_event(self):
        result = self.poll_result
        self.poll_result = None
        return result


def test_start_detection_submits_grounding_request():
    layer = Layer("Layer", 8, 8)
    composite = np.zeros((8, 8, 4), dtype=np.uint8)
    engine = _Engine()
    controller = GroundingController(
        engine=engine,
        composite=lambda: composite,
    )

    event = controller.start_detection(layer, _params())

    assert event.status == "Grounding: detecting..."
    assert controller.pending_layer is layer
    assert len(engine.calls) == 1
    assert engine.calls[0].image is composite
    assert engine.calls[0].params.prompt == "cup."


def test_poll_returns_pending_layer_and_grounding_result():
    layer = Layer("Layer", 8, 8)
    engine = _Engine()
    controller = GroundingController(
        engine=engine,
        composite=lambda: np.zeros((8, 8, 4), dtype=np.uint8),
    )
    controller.start_detection(layer, _params())
    result = GroundingResult(detections=(
        GroundingDetection(
            label="cup",
            x0=1,
            y0=2,
            x1=4,
            y1=6,
            score=0.9,
        ),
    ))
    engine.poll_result = GroundingEngineEvent(
        status="Grounding: 1 hit(s): cup (90%)",
        result=result,
    )

    event = controller.poll()

    assert event.grounding_result == (layer, result)
    assert event.status == "Grounding: 1 hit(s): cup (90%)"
    assert controller.pending_layer is None


def test_poll_error_clears_pending_layer():
    layer = Layer("Layer", 8, 8)
    engine = _Engine()
    controller = GroundingController(
        engine=engine,
        composite=lambda: np.zeros((8, 8, 4), dtype=np.uint8),
    )
    controller.start_detection(layer, _params())
    engine.poll_result = GroundingEngineEvent(error="failed")

    event = controller.poll()

    assert event.grounding_error == "failed"
    assert event.status == "Grounding error: failed"
    assert controller.pending_layer is None
