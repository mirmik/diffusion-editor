import numpy as np
import pytest

from diffusion_editor.app.application import EditorApplication, EngineSet
from diffusion_editor.canvas.edit_transactions import (
    CanvasEditTransactionCoordinator,
)
from diffusion_editor.canvas.brush import BrushToolMode
from diffusion_editor.canvas.editor_canvas_controller import (
    EditorCanvasController,
)
from diffusion_editor.document.commands import AddLayerCommand


class _Settings:
    def get(self, _key, default=None):
        return default

    def set(self, _key, _value):
        pass


class _Engine:
    model_info = {}

    def poll_event(self):
        return None

    def shutdown(self):
        pass


def _editor():
    engine = _Engine()
    application = EditorApplication(
        settings=_Settings(),
        engines=EngineSet(engine, engine, engine, engine, engine),
    )
    application.layer_stack.init_from_image(
        np.zeros((24, 24, 4), dtype=np.uint8)
    )
    controller = EditorCanvasController(
        application.layer_stack,
        gpu_compositing=False,
        set_image=lambda _image: None,
        set_overlay=lambda _overlay: None,
    )
    controller.refresh()
    transactions = CanvasEditTransactionCoordinator(
        application.layer_stack,
        application.document,
        history_replaying=lambda: application.history_replaying,
        on_mutation_begin=application.mark_external_mutation,
    )
    transactions.bind(controller)
    controller.brush.set_size(7)
    controller.brush.set_hardness(1.0)
    controller.brush.set_color(255, 0, 0, 255)
    return application, controller, transactions


def test_paint_gesture_is_one_undoable_transaction_with_edge_clipping():
    application, controller, transactions = _editor()

    controller.pointer_down(0, 0, controller.LEFT_BUTTON)
    controller.pointer_move(8, 0)
    controller.pointer_up(8, 0)
    painted = application.layer_stack.active_layer.image.copy()

    assert np.any(painted[:, :, 3] > 0)
    assert application.history.can_undo
    assert application.document.undo() == "Paint Stroke"
    assert not np.any(application.layer_stack.active_layer.image)
    assert application.document.redo() == "Paint Stroke"
    np.testing.assert_array_equal(
        application.layer_stack.active_layer.image,
        painted,
    )
    transactions.close()
    application.close()


def test_active_layer_switch_during_stroke_keeps_original_target():
    application, controller, transactions = _editor()
    original = application.layer_stack.active_layer
    application.layer_stack.add_layer("Other")
    other = application.layer_stack.active_layer
    application.layer_stack.active_layer = original

    controller.pointer_down(5, 5, controller.LEFT_BUTTON)
    application.layer_stack.active_layer = other
    controller.pointer_move(12, 5)
    controller.pointer_up(12, 5)

    assert np.any(original.image[:, :, 3] > 0)
    assert not np.any(other.image)
    assert application.document.undo() == "Paint Stroke"
    original_after_undo = application.layer_stack.find_layer_by_id(original.id)
    assert original_after_undo is not None
    assert not np.any(original_after_undo.image)
    transactions.close()
    application.close()


def test_cancel_rolls_back_pixels_and_does_not_add_history():
    application, controller, transactions = _editor()

    controller.pointer_down(6, 6, controller.LEFT_BUTTON)
    assert np.any(application.layer_stack.active_layer.image)
    controller.pointer_cancel()

    assert not np.any(application.layer_stack.active_layer.image)
    assert not application.history.can_undo
    assert not transactions.active
    transactions.close()
    application.close()


def test_tool_begin_failure_rolls_back_and_clears_interaction():
    application, controller, transactions = _editor()
    tool = controller._active_stroke_tool
    original_begin = tool.begin

    def fail_after_begin(context, layer, x, y):
        original_begin(context, layer, x, y)
        raise RuntimeError("tool begin failed")

    tool.begin = fail_after_begin

    with pytest.raises(RuntimeError, match="tool begin failed"):
        controller.pointer_down(6, 6, controller.LEFT_BUTTON)

    assert not controller.pointer_interaction_active
    assert not transactions.active
    assert not np.any(application.layer_stack.active_layer.image)
    assert not application.history.can_undo
    transactions.close()
    application.close()


def test_tool_end_failure_rolls_back_and_clears_interaction():
    application, controller, transactions = _editor()
    tool = controller._active_stroke_tool
    original_end = tool.end

    def fail_after_end(context, layer):
        original_end(context, layer)
        raise RuntimeError("tool end failed")

    tool.end = fail_after_end
    controller.pointer_down(6, 6, controller.LEFT_BUTTON)
    controller.pointer_move(12, 6)

    with pytest.raises(RuntimeError, match="tool end failed"):
        controller.pointer_up(12, 6)

    assert not controller.pointer_interaction_active
    assert not transactions.active
    assert not np.any(application.layer_stack.active_layer.image)
    assert not application.history.can_undo
    transactions.close()
    application.close()


def test_overlay_failure_during_cancel_cannot_suppress_pixel_rollback():
    application, controller, transactions = _editor()
    controller.pointer_down(6, 6, controller.LEFT_BUTTON)
    assert np.any(application.layer_stack.active_layer.image)
    controller._overlay_bridge.rebuild = lambda: (_ for _ in ()).throw(
        RuntimeError("overlay rebuild failed"))

    with pytest.raises(RuntimeError, match="overlay rebuild failed"):
        controller.pointer_cancel()

    assert not controller.pointer_interaction_active
    assert not transactions.active
    assert not np.any(application.layer_stack.active_layer.image)
    assert not application.history.can_undo
    transactions.close()
    application.close()


def test_history_registration_failure_restores_live_pixels():
    application, controller, transactions = _editor()
    controller.pointer_down(6, 6, controller.LEFT_BUTTON)
    controller.pointer_move(12, 6)
    application.document.push_callbacks = (
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("history push failed")))

    with pytest.raises(RuntimeError, match="history push failed"):
        controller.pointer_up(12, 6)

    assert not controller.pointer_interaction_active
    assert not transactions.active
    assert not np.any(application.layer_stack.active_layer.image)
    assert not application.history.can_undo
    transactions.close()
    application.close()


def test_transform_history_registration_failure_restores_offset():
    application, controller, transactions = _editor()
    layer = application.layer_stack.active_layer
    controller.set_brush_tool(BrushToolMode.MOVE)
    controller.pointer_down(5, 5, controller.LEFT_BUTTON)
    controller.pointer_move(11, 14)
    assert (layer.x, layer.y) == (6, 9)
    application.document.push_callbacks = (
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("history push failed")))

    with pytest.raises(RuntimeError, match="history push failed"):
        controller.pointer_up(11, 14)

    assert (layer.x, layer.y) == (0, 0)
    assert not controller.pointer_interaction_active
    assert not transactions.active
    assert not application.history.can_undo
    transactions.close()
    application.close()


def test_canvas_history_observer_failure_does_not_break_undo_redo():
    application, controller, transactions = _editor()
    controller.pointer_down(6, 6, controller.LEFT_BUTTON)
    controller.pointer_up(6, 6)
    painted = application.layer_stack.active_layer.image.copy()
    application.layer_stack.on_changed = lambda: (_ for _ in ()).throw(
        RuntimeError("observer failed"))

    assert application.document.undo() == "Paint Stroke"
    assert not np.any(application.layer_stack.active_layer.image)
    assert application.document.redo() == "Paint Stroke"
    np.testing.assert_array_equal(
        application.layer_stack.active_layer.image, painted)
    transactions.close()
    application.close()


def test_transform_history_observer_failure_does_not_break_undo_redo():
    application, controller, transactions = _editor()
    layer = application.layer_stack.active_layer
    controller.set_brush_tool(BrushToolMode.MOVE)
    controller.pointer_down(5, 5, controller.LEFT_BUTTON)
    controller.pointer_move(11, 14)
    controller.pointer_up(11, 14)
    application.layer_stack.on_changed = lambda: (_ for _ in ()).throw(
        RuntimeError("observer failed"))

    assert application.document.undo() == "Move Layer"
    assert (layer.x, layer.y) == (0, 0)
    assert application.document.redo() == "Move Layer"
    assert (layer.x, layer.y) == (6, 9)
    transactions.close()
    application.close()


def test_document_command_cancels_in_flight_stroke_before_snapshot():
    application, controller, transactions = _editor()

    controller.pointer_down(6, 6, controller.LEFT_BUTTON)
    assert controller.pointer_interaction_active
    assert np.any(application.layer_stack.active_layer.image)

    application.document.execute(AddLayerCommand(name="Other"))

    assert not controller.pointer_interaction_active
    assert len(application.layer_stack.layers) == 2
    background = application.layer_stack.layers[-1]
    assert not np.any(background.image)
    assert application.document.undo() == "New Layer"
    assert len(application.layer_stack.layers) == 1
    assert not np.any(application.layer_stack.active_layer.image)
    assert application.document.undo() is None
    transactions.close()
    application.close()


def test_undo_cancels_in_flight_stroke_before_replaying_history():
    application, controller, transactions = _editor()
    application.document.execute(AddLayerCommand(name="Other"))
    painted_layer = application.layer_stack.active_layer

    controller.pointer_down(6, 6, controller.LEFT_BUTTON)
    assert np.any(painted_layer.image)

    assert application.document.undo() == "New Layer"
    assert not controller.pointer_interaction_active
    assert len(application.layer_stack.layers) == 1
    assert not np.any(application.layer_stack.active_layer.image)
    transactions.close()
    application.close()


def test_detached_layer_cancels_instead_of_painting_new_active_layer():
    application, controller, transactions = _editor()
    target = application.layer_stack.active_layer
    application.layer_stack.add_layer("Survivor")
    application.layer_stack.active_layer = target

    controller.pointer_down(6, 6, controller.LEFT_BUTTON)
    application.layer_stack.remove_layer(target)
    survivor = application.layer_stack.active_layer
    controller.pointer_move(12, 6)

    assert not controller.pointer_interaction_active
    assert not np.any(survivor.image)
    assert not application.history.can_undo
    transactions.close()
    application.close()


def test_mask_gesture_is_one_undoable_transaction():
    application, controller, transactions = _editor()
    layer = application.layer_stack.active_layer
    before = layer.mask.data.copy()
    controller.set_brush_tool(BrushToolMode.MASK)

    controller.pointer_down(5, 5, controller.LEFT_BUTTON)
    controller.pointer_move(13, 5)
    controller.pointer_up(13, 5)
    after = layer.mask.data.copy()

    assert not np.array_equal(after, before)
    assert application.document.undo() == "Mask Stroke"
    np.testing.assert_array_equal(layer.mask.data, before)
    assert application.document.undo() is None
    assert application.document.redo() == "Mask Stroke"
    np.testing.assert_array_equal(layer.mask.data, after)
    transactions.close()
    application.close()


def test_selection_gesture_is_one_undoable_transaction():
    application, controller, transactions = _editor()
    before = application.layer_stack.selection.data.copy()
    controller.set_selection_mode(True)

    controller.pointer_down(5, 5, controller.LEFT_BUTTON)
    controller.pointer_move(13, 5)
    controller.pointer_up(13, 5)
    after = application.layer_stack.selection.data.copy()

    assert not np.array_equal(after, before)
    assert application.document.undo() == "Selection Stroke"
    np.testing.assert_array_equal(
        application.layer_stack.selection.data, before)
    assert application.document.undo() is None
    assert application.document.redo() == "Selection Stroke"
    np.testing.assert_array_equal(
        application.layer_stack.selection.data, after)
    transactions.close()
    application.close()


def test_move_gesture_is_one_undoable_transaction():
    application, controller, transactions = _editor()
    layer = application.layer_stack.active_layer
    before = (layer.x, layer.y)
    controller.set_brush_tool(BrushToolMode.MOVE)

    controller.pointer_down(5, 5, controller.LEFT_BUTTON)
    controller.pointer_move(11, 14)
    controller.pointer_up(11, 14)
    after = (layer.x, layer.y)

    assert after == (6, 9)
    assert application.document.undo() == "Move Layer"
    restored = application.layer_stack.find_layer_by_id(layer.id)
    assert restored is not None
    assert (restored.x, restored.y) == before
    assert application.document.undo() is None
    assert application.document.redo() == "Move Layer"
    restored = application.layer_stack.find_layer_by_id(layer.id)
    assert restored is not None
    assert (restored.x, restored.y) == after
    transactions.close()
    application.close()


def test_erase_gesture_is_one_undoable_transaction():
    application, controller, transactions = _editor()
    layer = application.layer_stack.active_layer
    layer.image[:] = (30, 60, 90, 255)
    application.layer_stack.mark_layer_dirty(layer)
    controller.refresh()
    before = layer.image.copy()
    controller.set_brush_tool(BrushToolMode.ERASER)

    controller.pointer_down(5, 5, controller.LEFT_BUTTON)
    controller.pointer_move(13, 5)
    controller.pointer_up(13, 5)
    after = layer.image.copy()

    assert not np.array_equal(after, before)
    assert application.document.undo() == "Erase Stroke"
    np.testing.assert_array_equal(layer.image, before)
    assert application.document.undo() is None
    assert application.document.redo() == "Erase Stroke"
    np.testing.assert_array_equal(layer.image, after)
    transactions.close()
    application.close()


def test_smudge_gesture_is_one_undoable_transaction():
    application, controller, transactions = _editor()
    layer = application.layer_stack.active_layer
    layer.image[:, :12] = (220, 20, 20, 255)
    layer.image[:, 12:] = (20, 20, 220, 255)
    application.layer_stack.mark_layer_dirty(layer)
    controller.refresh()
    before = layer.image.copy()
    controller.set_brush_tool(BrushToolMode.SMUDGE)

    controller.pointer_down(8, 12, controller.LEFT_BUTTON)
    controller.pointer_move(16, 12)
    controller.pointer_up(16, 12)
    after = layer.image.copy()

    assert not np.array_equal(after, before)
    assert application.document.undo() == "Smudge Stroke"
    np.testing.assert_array_equal(layer.image, before)
    assert application.document.undo() is None
    assert application.document.redo() == "Smudge Stroke"
    np.testing.assert_array_equal(layer.image, after)
    transactions.close()
    application.close()


def test_mask_erase_gesture_is_one_undoable_transaction():
    application, controller, transactions = _editor()
    layer = application.layer_stack.active_layer
    layer.mask.data[:] = 1.0
    layer.image[:] = (30, 60, 90, 255)
    application.layer_stack.mark_layer_dirty(layer)
    controller.refresh()
    before_mask = layer.mask.data.copy()
    before_image = layer.image.copy()
    controller.set_brush_tool(BrushToolMode.MASK_ERASER)

    controller.pointer_down(5, 5, controller.LEFT_BUTTON)
    controller.pointer_move(13, 5)
    controller.pointer_up(13, 5)
    after_mask = layer.mask.data.copy()
    after_image = layer.image.copy()

    assert not np.array_equal(after_mask, before_mask)
    assert not np.array_equal(after_image, before_image)
    assert application.document.undo() == "Mask Erase Stroke"
    np.testing.assert_array_equal(layer.mask.data, before_mask)
    np.testing.assert_array_equal(layer.image, before_image)
    assert application.document.undo() is None
    assert application.document.redo() == "Mask Erase Stroke"
    np.testing.assert_array_equal(layer.mask.data, after_mask)
    np.testing.assert_array_equal(layer.image, after_image)
    transactions.close()
    application.close()


def test_cancel_mask_erase_restores_mask_and_image_exactly():
    application, controller, transactions = _editor()
    layer = application.layer_stack.active_layer
    layer.mask.data[:] = 1.0
    layer.image[:] = (30, 60, 90, 255)
    application.layer_stack.mark_layer_dirty(layer)
    controller.refresh()
    before_mask = layer.mask.data.copy()
    before_image = layer.image.copy()
    controller.set_brush_tool(BrushToolMode.MASK_ERASER)

    controller.pointer_down(5, 5, controller.LEFT_BUTTON)
    controller.pointer_move(13, 5)
    controller.pointer_cancel()

    np.testing.assert_array_equal(layer.mask.data, before_mask)
    np.testing.assert_array_equal(layer.image, before_image)
    assert not application.history.can_undo
    transactions.close()
    application.close()
