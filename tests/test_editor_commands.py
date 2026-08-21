import numpy as np

from diffusion_editor.app.application import EditorApplication, EngineSet
from diffusion_editor.app.editor_commands import EditorCommandCoordinator
from diffusion_editor.generation.types import (
    DepthEstimationResult,
    EnginePollEvent,
    SegmentationResult,
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


def test_depth_map_command_adds_undoable_grayscale_layer():
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
    depth = np.arange(20, dtype=np.uint8).reshape((4, 5))
    events = [EnginePollEvent(
        task_type="depth",
        result=DepthEstimationResult(depth),
        job_id=context.job_id,
    )]
    application.depth_engine.poll_event = (
        lambda: events.pop(0) if events else None
    )

    application.poll()

    assert len(application.layer_stack.all_layers()) == 2
    layer = application.layer_stack.active_layer
    assert layer.name == "Depth Map 0"
    np.testing.assert_array_equal(layer.image[:, :, 0], depth)
    np.testing.assert_array_equal(layer.image[:, :, 1], depth)
    np.testing.assert_array_equal(layer.image[:, :, 2], depth)
    assert np.all(layer.image[:, :, 3] == 255)
    assert application.status_text.startswith("Depth map created")
    application.document.undo()
    assert len(application.layer_stack.all_layers()) == 1
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
