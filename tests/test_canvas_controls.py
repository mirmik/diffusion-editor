import numpy as np

from diffusion_editor.app.canvas_controls import (
    BrushControlAction,
    BrushControlsIntent,
    CanvasControlsCoordinator,
    SelectionControlAction,
    SelectionControlsIntent,
)
from diffusion_editor.canvas.brush import BrushToolMode
from diffusion_editor.canvas.editor_canvas_controller import EditorCanvasController
from diffusion_editor.document.commands import (
    ClearLayerPatchRectCommand,
    SetLayerPatchRectCommand,
    SetLayerSelectionCommand,
)
from diffusion_editor.document.layer_stack import LayerStack


class _Document:
    def __init__(self, stack):
        self.stack = stack
        self.commands = []

    def execute(self, command):
        self.commands.append(command)
        command.apply(self.stack)


class _View:
    def __init__(self):
        self.brush_states = []
        self.selection_states = []

    def apply_brush_state(self, state):
        self.brush_states.append(state)

    def apply_selection_state(self, state):
        self.selection_states.append(state)


def _coordinator():
    image = np.zeros((24, 32, 4), dtype=np.uint8)
    stack = LayerStack(tile_size=8)
    stack.init_from_image(image)
    cursors = []
    canvas = EditorCanvasController(
        stack,
        gpu_compositing=False,
        set_image=lambda _image: None,
        set_overlay=lambda _overlay: None,
        set_cursor=cursors.append,
    )
    canvas.refresh()
    document = _Document(stack)
    coordinator = CanvasControlsCoordinator(stack, document, canvas)
    view = _View()
    coordinator.bind_view(view)
    return stack, canvas, document, coordinator, view, cursors


def test_brush_intents_update_canvas_and_mutually_exclusive_modes():
    _stack, canvas, _document, coordinator, view, cursors = _coordinator()
    coordinator.handle_selection_intent(SelectionControlsIntent(
        SelectionControlAction.EDIT_MODE,
        True,
    ))
    assert coordinator.selection_state.edit_mode
    assert cursors[-1] == "crosshair"

    coordinator.handle_brush_intent(BrushControlsIntent(
        BrushControlAction.TOOL,
        BrushToolMode.SMUDGE,
    ))
    coordinator.handle_brush_intent(BrushControlsIntent(
        BrushControlAction.SIZE,
        73,
    ))
    coordinator.handle_brush_intent(BrushControlsIntent(
        BrushControlAction.HARDNESS,
        0.75,
    ))
    coordinator.handle_brush_intent(BrushControlsIntent(
        BrushControlAction.FLOW,
        0.25,
    ))

    assert canvas.brush_tool_mode == BrushToolMode.SMUDGE
    assert canvas.brush.size == 73
    assert canvas.brush.hardness == 0.75
    assert canvas.brush.flow == 0.25
    assert not coordinator.selection_state.edit_mode
    assert cursors[-1] == "default"
    assert view.brush_states[-1] == coordinator.brush_state


def test_patch_and_selection_rect_results_use_document_commands():
    stack, canvas, document, coordinator, _view, _cursors = _coordinator()

    coordinator.handle_brush_intent(BrushControlsIntent(
        BrushControlAction.DRAW_PATCH,
        True,
    ))
    canvas.pointer_down(3, 4, canvas.LEFT_BUTTON)
    canvas.pointer_move(18, 16)
    canvas.pointer_up(18, 16)

    assert isinstance(document.commands[-1], SetLayerPatchRectCommand)
    assert stack.active_layer.patch_rect == (3, 4, 18, 16)
    assert not coordinator.brush_state.draw_patch

    coordinator.handle_brush_intent(BrushControlsIntent(
        BrushControlAction.CLEAR_PATCH,
    ))
    assert isinstance(document.commands[-1], ClearLayerPatchRectCommand)
    assert stack.active_layer.patch_rect is None

    coordinator.handle_selection_intent(SelectionControlsIntent(
        SelectionControlAction.RECT_MODE,
        True,
    ))
    canvas.pointer_down(2, 3, canvas.LEFT_BUTTON)
    canvas.pointer_move(9, 11)
    canvas.pointer_up(9, 11)

    assert isinstance(document.commands[-1], SetLayerSelectionCommand)
    assert np.all(stack.selection.data[3:12, 2:10] == 1.0)
    assert not coordinator.selection_state.rect_mode


def test_eyedropper_updates_state_and_close_restores_callbacks():
    stack, canvas, _document, coordinator, view, _cursors = _coordinator()
    stack.active_layer.image[5, 7] = (11, 22, 33, 255)
    stack.mark_layer_dirty(stack.active_layer, (7, 5, 8, 6))
    old_color_callback = coordinator._previous_color_picked
    old_patch_callback = coordinator._previous_patch_rect_drawn
    old_selection_callback = coordinator._previous_selection_rect_drawn
    old_size_callback = coordinator._previous_brush_size_changed

    canvas.pointer_down(
        7,
        5,
        canvas.LEFT_BUTTON,
        canvas.CTRL_MODIFIER,
    )

    assert canvas.brush.color == (11, 22, 33, 255)
    assert coordinator.brush_state.color == (11, 22, 33, 255)
    assert view.brush_states[-1].color == (11, 22, 33, 255)
    canvas.adjust_brush_size(5)
    assert coordinator.brush_state.size == 25
    assert view.brush_states[-1].size == 25

    coordinator.close()
    assert canvas.on_color_picked is old_color_callback
    assert canvas.on_patch_rect_drawn is old_patch_callback
    assert canvas.on_selection_rect_drawn is old_selection_callback
    assert canvas.on_brush_size_changed is old_size_callback
