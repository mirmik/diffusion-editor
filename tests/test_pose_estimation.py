from __future__ import annotations

import numpy as np

from diffusion_editor.document.commands import AddPoseOverlayCommand
from diffusion_editor.document.layer_stack import LayerStack
from diffusion_editor.generation.pose_estimation import (
    PoseConnection,
    PoseEstimationResult,
    PoseInstance,
    PoseKeypoint,
    PoseSegment,
    render_pose_overlay,
)
from diffusion_editor.workers.pose_protocol import (
    PoseProtocolError,
    decode_response,
    encode_message,
)
import pytest


def _result() -> PoseEstimationResult:
    return PoseEstimationResult(
        profile_id="dwpose",
        width=64,
        height=48,
        keypoint_schema="test",
        poses=(PoseInstance((
            PoseKeypoint("left_shoulder", 16.0, 12.0, 0.9),
            PoseKeypoint("left_elbow", 24.0, 24.0, 0.8),
            PoseKeypoint("hidden", 32.0, 12.0, 0.1),
        )),),
        connections=(
            PoseConnection("left_shoulder", "left_elbow", "body"),
            PoseConnection("left_elbow", "hidden", "body"),
        ),
        segments=(PoseSegment(2.0, 2.0, 8.0, 8.0),),
    )


def test_pose_result_json_round_trip_preserves_coordinates_and_confidence():
    result = _result()

    restored = PoseEstimationResult.from_dict(result.to_dict())

    assert restored == result
    assert restored.poses[0].keypoints[0].score == 0.9


def test_pose_overlay_is_transparent_and_hides_low_confidence_connections():
    overlay = render_pose_overlay(_result(), confidence_threshold=0.25)

    assert overlay.shape == (48, 64, 4)
    assert overlay.dtype == np.uint8
    assert overlay.flags.c_contiguous
    assert overlay.flags.writeable
    assert overlay.flags.owndata
    assert overlay[12, 16, 3] > 0
    assert overlay[24, 24, 3] > 0
    assert overlay[12, 32, 3] == 0
    assert overlay[47, 63, 3] == 0


def test_pose_overlay_command_keeps_source_active_and_is_undoable():
    stack = LayerStack()
    source_pixels = np.full((24, 32, 4), 255, dtype=np.uint8)
    stack.init_from_image(source_pixels)
    source = stack.active_layer
    assert source is not None
    overlay = np.zeros_like(source_pixels)
    overlay[4:8, 4:8] = (0, 255, 255, 255)
    command = AddPoseOverlayCommand(source, "Pose", overlay)

    delta = command.apply_with_history(stack)

    assert delta is not None
    assert stack.active_layer is source
    assert len(stack.all_layers()) == 2
    pose_layer = next(layer for layer in stack.all_layers() if layer is not source)
    assert (pose_layer.x, pose_layer.y) == (source.x, source.y)
    assert pose_layer.image.flags.c_contiguous
    assert pose_layer.image.flags.writeable

    delta.undo_fn()
    assert stack.all_layers() == [source]
    delta.redo_fn()
    assert len(stack.all_layers()) == 2
    assert stack.active_layer is source


def test_pose_overlay_uses_canvas_coordinates_for_partial_anchor_layer():
    stack = LayerStack()
    canvas = np.full((24, 32, 4), (20, 30, 40, 255), dtype=np.uint8)
    stack.init_from_image(canvas)
    partial = np.full((8, 10, 4), (200, 80, 40, 255), dtype=np.uint8)
    stack.insert_image_layer("Partial", partial, x=7, y=9)
    source = stack.active_layer
    overlay = np.zeros((24, 32, 4), dtype=np.uint8)
    overlay[2:5, 3:7] = (0, 255, 255, 255)

    delta = AddPoseOverlayCommand(source, "Pose", overlay).apply_with_history(stack)

    assert delta is not None
    pose_layer = next(
        layer for layer in stack.all_layers()
        if layer is not source and layer.name == "Pose"
    )
    assert pose_layer.image.shape == (24, 32, 4)
    assert (pose_layer.x, pose_layer.y) == (0, 0)
    assert stack.active_layer is source


def test_pose_protocol_rejects_non_boolean_gil_flag():
    response = encode_message({
        "protocol": 1,
        "type": "ready",
        "runtime": {
            "python": "/python",
            "version": "3.14",
            "abiflags": "",
            "gil_enabled": "false",
        },
    })

    with pytest.raises(PoseProtocolError, match="invalid runtime"):
        decode_response(response)
