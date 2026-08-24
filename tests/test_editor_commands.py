import numpy as np
import pytest

from diffusion_editor.app.application import EditorApplication, EngineSet
from diffusion_editor.app.editor_commands import EditorCommandCoordinator
from diffusion_editor.generation.types import (
    DepthEstimationResult,
    DepthValueKind,
    EnginePollEvent,
    SegmentationResult,
)
from diffusion_editor.generation.pose_estimation import (
    PoseConnection,
    PoseEstimationResult,
    PoseInstance,
    PoseKeypoint,
)


class _Settings:
    def get(self, _key, default=None):
        return default

    def set(self, _key, _value):
        pass


class _Engine:
    model_info = {}

    def __init__(self):
        self.poll_result = None

    def poll_event(self):
        result = self.poll_result
        self.poll_result = None
        return result

    def shutdown(self):
        pass


def _application() -> EditorApplication:
    engine = _Engine()
    return EditorApplication(
        settings=_Settings(),
        engines=EngineSet(engine, engine, engine, engine, engine),
    )


def test_standard_commands_project_state_and_round_trip_clipboard_history():
    application = _application()
    image = np.zeros((6, 8, 4), dtype=np.uint8)
    image[2:4, 3:6] = (10, 20, 30, 255)
    application.layer_stack.init_from_image(image)
    fits = []
    commands = EditorCommandCoordinator(
        application,
        fit_in_view=lambda: fits.append(True),
    )

    assert application.command_states["edit.undo"] == (False, False)
    assert application.command_states["selection.all"] == (True, False)
    assert application.command_states["selection.clear"] == (False, False)

    commands.handlers["selection.all"]()
    assert np.all(application.layer_stack.selection.data == 1.0)
    assert application.command_states["edit.undo"] == (True, False)
    assert application.command_states["selection.clear"] == (True, False)

    commands.handlers["edit.copy"]()
    np.testing.assert_array_equal(application.clipboard, image)
    assert application.clipboard_pos == (0, 0)
    assert application.command_states["edit.paste"] == (True, False)

    commands.handlers["edit.paste"]()
    assert len(application.layer_stack.all_layers()) == 2
    assert application.layer_stack.active_layer.name == "Floating Selection"
    assert application.status_text == "Pasted 8x6"

    commands.handlers["edit.undo"]()
    assert len(application.layer_stack.all_layers()) == 1
    assert application.command_states["edit.redo"] == (True, False)
    commands.handlers["edit.redo"]()
    assert len(application.layer_stack.all_layers()) == 2

    commands.handlers["view.fit"]()
    assert fits == [True]
    application.close()


def test_layer_and_selection_commands_cancel_active_mutation_first():
    application = _application()
    application.layer_stack.init_from_image(
        np.zeros((4, 4, 4), dtype=np.uint8)
    )
    cancelled = []
    commands = EditorCommandCoordinator(
        application,
        before_mutation=lambda: cancelled.append(True),
    )

    commands.handlers["layer.new"]()
    assert len(application.layer_stack.all_layers()) == 2
    assert application.layer_stack.active_layer.name == "Layer 0"

    commands.handlers["layer.flatten"]()
    assert len(application.layer_stack.all_layers()) == 1
    assert len(cancelled) == 2
    application.close()


def test_select_background_command_submits_full_composite_segmentation():
    application = _application()
    image = np.zeros((4, 5, 4), dtype=np.uint8)
    image[:, :] = (12, 34, 56, 255)
    application.layer_stack.init_from_image(image)
    requests = []
    application.engines.segmentation.submit_request = (
        lambda request, **_kwargs: requests.append(request) or True
    )
    commands = EditorCommandCoordinator(application)

    assert application.command_states["selection.background"] == (True, False)
    commands.handlers["selection.background"]()

    assert application.status_text == "Selecting background..."
    assert len(requests) == 1
    np.testing.assert_array_equal(requests[0].image, image)
    assert requests[0].invert is True
    assert application.command_states["selection.background"] == (False, False)

    mask = np.zeros((4, 5), dtype=np.uint8)
    mask[:, :2] = 255
    application.engines.segmentation.poll_result = EnginePollEvent(
        task_type="segmentation",
        result=SegmentationResult(mask=mask),
    )
    application.poll()

    np.testing.assert_array_equal(
        application.layer_stack.selection.data,
        mask.astype(np.float32) / 255.0,
    )
    assert application.status_text == "Background selected"
    application.document.undo()
    assert application.layer_stack.selection.is_empty
    application.document.redo()
    assert np.all(application.layer_stack.selection.data[:, :2] == 1.0)
    application.close()


def _depth_result(
        height: int,
        width: int,
        profile_id: str = "da3-nested-giant-large-1.1",
        *,
        with_confidence: bool = False,
) -> DepthEstimationResult:
    depth = (
        np.arange(height * width, dtype=np.float32).reshape((height, width))
        + 1.0
    )
    intrinsics = np.array([
        [10.0, 0.0, width * 0.5],
        [0.0, 10.0, height * 0.5],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    return DepthEstimationResult(
        depth,
        profile_id=profile_id,
        value_kind=DepthValueKind.DIRECT_METRIC,
        intrinsics=intrinsics,
        confidence=(
            np.arange(height * width, dtype=np.float32).reshape(
                (height, width)) + 1.0
            if with_confidence else None
        ),
    )


def test_depth_map_command_keeps_float_source_and_adds_one_preview():
    application = _application()
    image = np.full((4, 5, 4), (12, 34, 56, 255), dtype=np.uint8)
    application.layer_stack.init_from_image(image)
    requests = []
    application.depth_engine.submit_request = (
        lambda request, **_kwargs: requests.append(request) or True
    )
    commands = EditorCommandCoordinator(application)

    assert application.command_states["ai.depth_map"] == (True, False)
    commands.handlers["ai.depth_map"]()

    assert len(requests) == 1
    np.testing.assert_array_equal(requests[0].image, image)
    assert application.command_states["ai.depth_map"] == (False, False)
    context = application.depth_controller.pending_context
    # DA3 keeps its native model grid and camera matrix. Only the display
    # derivative is resized back to the 4x5 document.
    result = _depth_result(2, 3, with_confidence=True)
    events = [EnginePollEvent(
        task_type="depth",
        result=result,
        job_id=context.job_id,
    )]
    application.depth_engine.poll_event = (
        lambda: events.pop(0) if events else None
    )

    application.poll()

    assert len(application.layer_stack.all_layers()) == 2
    preview_layer = application.layer_stack.active_layer
    assert preview_layer.name == (
        "Depth DA3 Nested 0 Preview (float32 source)"
    )
    assert preview_layer.visible is True
    assert preview_layer.image.shape == (4, 5, 4)
    assert preview_layer.visible is True
    assert application.latest_depth_result is result
    assert application.latest_depth_result.depth_map.dtype == np.float32
    np.testing.assert_array_equal(
        application.latest_depth_result.depth_map, result.depth_map)
    assert application.latest_depth_point_cloud is not None
    assert application.latest_depth_point_cloud.point_count == 6
    assert application.latest_depth_point_cloud.source_size == (3, 2)
    np.testing.assert_array_equal(
        application.latest_depth_point_cloud.confidence,
        result.confidence.reshape(-1),
    )
    commands.refresh()
    assert application.command_states["ai.depth_point_cloud"] == (
        True,
        False,
    )
    assert application.status_text.startswith("DA3 Nested Giant Large 1.1:")
    application.document.undo()
    assert len(application.layer_stack.all_layers()) == 1
    commands.refresh()
    assert application.command_states["ai.depth_point_cloud"] == (
        False,
        False,
    )
    application.document.redo()
    assert len(application.layer_stack.all_layers()) == 2
    assert application.layer_stack.active_layer.name == preview_layer.name
    application.close()


def test_depth_preview_reports_when_soft_mask_removes_all_cloud_points():
    application = _application()
    image = np.full((4, 5, 4), (12, 34, 56, 255), dtype=np.uint8)
    application.layer_stack.init_from_image(image)
    application.layer_stack.selection.data[:] = 0.25
    requests = []
    application.depth_engine.submit_request = (
        lambda request, **_kwargs: requests.append(request) or True
    )
    commands = EditorCommandCoordinator(application)

    commands.handlers["ai.depth_map.depth_pro"]()
    context = application.depth_controller.pending_context
    assert not hasattr(requests[0], "output_mask")
    np.testing.assert_array_equal(requests[0].image, image)
    events = [EnginePollEvent(
        task_type="depth",
        result=_depth_result(2, 3, "depth-pro"),
        job_id=context.job_id,
    )]
    application.depth_engine.poll_event = (
        lambda: events.pop(0) if events else None
    )

    application.poll()
    commands.refresh()

    assert application.latest_depth_point_cloud is None, (
        application.status_text,
        [layer.name for layer in application.layer_stack.all_layers()],
    )
    assert application.latest_depth_point_cloud_error == (
        "depth point-cloud mask contains no foreground"
    ), (
        application.status_text,
        [layer.name for layer in application.layer_stack.all_layers()],
    )
    assert "point cloud unavailable" in application.status_text
    assert application.command_states["ai.depth_point_cloud"] == (
        True,
        False,
    )
    application.close()


@pytest.mark.parametrize(("command_id", "profile_id"), (
    ("ai.depth_map", "da3-nested-giant-large-1.1"),
    ("ai.depth_map.da3_mono", "da3-mono-large"),
    ("ai.depth_map.depth_pro", "depth-pro"),
    ("ai.depth_map.v2_large", "v2-large"),
    ("ai.depth_map.v2_small", "v2-small"),
))
def test_depth_map_commands_select_explicit_profile(command_id, profile_id):
    application = _application()
    image = np.zeros((4, 5, 4), dtype=np.uint8)
    image[:, :, 3] = 255
    application.layer_stack.init_from_image(image)
    requests = []
    application.depth_engine.submit_request = (
        lambda request, **_kwargs: requests.append(request) or True
    )
    commands = EditorCommandCoordinator(application)

    commands.handlers[command_id]()

    assert requests[0].profile_id == profile_id
    application.close()


@pytest.mark.parametrize(("command_id", "profile_id"), (
    ("ai.pose.dwpose", "dwpose"),
    ("ai.pose.mediapipe", "mediapipe-full"),
    ("ai.pose.silhouette", "silhouette-skeleton"),
))
def test_pose_commands_create_comparable_overlay_layers(
        command_id, profile_id):
    application = _application()
    image = np.full((4, 5, 4), (12, 34, 56, 255), dtype=np.uint8)
    application.layer_stack.init_from_image(image)
    source = application.layer_stack.active_layer
    requests = []
    application.pose_engine.submit_request = (
        lambda request, **_kwargs: requests.append(request) or True
    )
    commands = EditorCommandCoordinator(application)

    commands.handlers[command_id]()

    assert len(requests) == 1
    assert requests[0].profile_id == profile_id
    np.testing.assert_array_equal(requests[0].image, image)
    context = application.pose_controller.pending_context
    assert context is not None
    result = PoseEstimationResult(
        profile_id=profile_id,
        width=5,
        height=4,
        keypoint_schema="test",
        poses=(PoseInstance((
            PoseKeypoint("left_shoulder", 1.0, 1.0, 0.9),
            PoseKeypoint("left_elbow", 3.0, 2.0, 0.8),
        )),),
        connections=(PoseConnection(
            "left_shoulder", "left_elbow", "body"),),
    )
    events = [EnginePollEvent(
        task_type="pose-estimation",
        result=result,
        job_id=context.job_id,
    )]
    application.pose_engine.poll_event = (
        lambda: events.pop(0) if events else None
    )

    application.poll()

    layers = application.layer_stack.all_layers()
    assert len(layers) == 2
    assert application.layer_stack.active_layer is source
    overlay = next(layer for layer in layers if layer is not source)
    assert overlay.name.startswith("Pose")
    assert np.any(overlay.image[:, :, 3] > 0)
    assert application.latest_pose_result is result
    assert "visible keypoints" in application.status_text
    application.document.undo()
    assert application.layer_stack.all_layers() == [source]
    application.document.redo()
    assert len(application.layer_stack.all_layers()) == 2
    assert application.layer_stack.active_layer is source
    application.close()


def test_pose_command_analyzes_visible_canvas_composite_not_active_layer():
    application = _application()
    background = np.full((6, 8, 4), (12, 34, 56, 255), dtype=np.uint8)
    application.layer_stack.init_from_image(background)
    partial = np.full((2, 3, 4), (210, 80, 30, 255), dtype=np.uint8)
    application.layer_stack.insert_image_layer("Partial", partial, x=2, y=3)
    source = application.layer_stack.active_layer
    expected_composite = application.layer_stack.composite()
    requests = []
    application.pose_engine.submit_request = (
        lambda request, **_kwargs: requests.append(request) or True
    )
    commands = EditorCommandCoordinator(application)

    commands.handlers["ai.pose.dwpose"]()

    assert len(requests) == 1
    assert requests[0].image.shape == (6, 8, 4)
    np.testing.assert_array_equal(requests[0].image, expected_composite)
    assert not np.array_equal(requests[0].image[:2, :3], partial)
    context = application.pose_controller.pending_context
    result = PoseEstimationResult(
        profile_id="dwpose",
        width=8,
        height=6,
        keypoint_schema="test",
        poses=(PoseInstance((
            PoseKeypoint("left_shoulder", 2.0, 2.0, 0.9),
            PoseKeypoint("left_elbow", 5.0, 4.0, 0.8),
        )),),
        connections=(PoseConnection(
            "left_shoulder", "left_elbow", "body"),),
    )
    events = [EnginePollEvent(
        task_type="pose-estimation",
        result=result,
        job_id=context.job_id,
    )]
    application.pose_engine.poll_event = (
        lambda: events.pop(0) if events else None
    )

    application.poll()

    pose_layer = next(
        layer for layer in application.layer_stack.all_layers()
        if layer is not source and layer.name.startswith("Pose")
    )
    assert pose_layer.image.shape == (6, 8, 4)
    assert (pose_layer.x, pose_layer.y) == (0, 0)
    assert application.layer_stack.active_layer is source
    application.close()


def test_subject_depth_segments_foreground_and_makes_background_transparent():
    application = _application()
    image = np.full((4, 5, 4), (12, 34, 56, 255), dtype=np.uint8)
    application.layer_stack.init_from_image(image)
    segmentation_requests = []
    depth_requests = []
    application.engines.segmentation.submit_request = (
        lambda request, **_kwargs:
        segmentation_requests.append(request) or True
    )
    application.depth_engine.submit_request = (
        lambda request, **_kwargs: depth_requests.append(request) or True
    )
    commands = EditorCommandCoordinator(application)

    commands.handlers["ai.depth_map.subject"]()

    assert application.status_text == "Isolating subject for depth..."
    assert len(segmentation_requests) == 1
    assert depth_requests == []
    background = np.full((4, 5), 255, dtype=np.uint8)
    background[:, 1:4] = 0
    application.engines.segmentation.poll_result = EnginePollEvent(
        task_type="segmentation",
        result=SegmentationResult(mask=background),
    )

    application.poll()

    assert len(depth_requests) == 1
    foreground = 255 - background
    assert not hasattr(depth_requests[0], "output_mask")
    np.testing.assert_array_equal(depth_requests[0].image, image)
    assert application.layer_stack.selection.is_empty

    depth_result = _depth_result(4, 5)
    depth_context = application.depth_controller.pending_context
    depth_events = [EnginePollEvent(
        task_type="depth",
        result=depth_result,
        job_id=depth_context.job_id,
    )]
    application.depth_engine.poll_event = (
        lambda: depth_events.pop(0) if depth_events else None
    )
    application.poll()

    depth_layers = [
        layer for layer in application.layer_stack.all_layers()
        if layer.name.startswith("Depth DA3 Nested")
    ]
    assert len(depth_layers) == 1
    for layer in depth_layers:
        np.testing.assert_array_equal(layer.image[:, :, 3], foreground)
    assert "full-frame inference; output mask applied afterward" in (
        application.status_text)
    application.close()


def test_subject_depth_can_retry_after_segmentation_error():
    application = _application()
    image = np.zeros((4, 5, 4), dtype=np.uint8)
    image[:, :, 3] = 255
    application.layer_stack.init_from_image(image)
    requests = []
    application.engines.segmentation.submit_request = (
        lambda request, **_kwargs: requests.append(request) or True
    )
    commands = EditorCommandCoordinator(application)
    commands.handlers["ai.depth_map.subject"]()
    application.engines.segmentation.poll_result = EnginePollEvent(
        task_type="segmentation",
        error="segmentation failed",
    )

    application.poll()
    commands.refresh()
    commands.handlers["ai.depth_map.subject"]()

    assert len(requests) == 2
    application.close()


def test_clear_selected_pixels_command_state_and_undo():
    application = _application()
    image = np.full((4, 5, 4), (12, 34, 56, 200), dtype=np.uint8)
    application.layer_stack.init_from_image(image)
    commands = EditorCommandCoordinator(application)

    assert application.command_states["edit.clear_selected_pixels"] == (
        False,
        False,
    )
    commands.handlers["selection.all"]()
    assert application.command_states["edit.clear_selected_pixels"] == (
        True,
        False,
    )

    commands.handlers["edit.clear_selected_pixels"]()

    assert np.all(application.layer_stack.active_layer.image[:, :, 3] == 0)
    assert not application.layer_stack.selection.is_empty
    assert application.status_text == "Selected pixels cleared"
    commands.handlers["edit.undo"]()
    np.testing.assert_array_equal(
        application.layer_stack.active_layer.image,
        image,
    )
    application.close()
