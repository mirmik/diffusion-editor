from termin.gui_native import (
    Color,
    EventResult,
    KeyCode,
    KeyEvent,
    KeyEventType,
    PointerEvent,
    PointerEventType,
    Rect,
    tc_ui_document_create,
    tc_ui_document_destroy,
)

from diffusion_editor.app.canvas_controls import (
    BrushControlAction,
    BrushControlsState,
    SelectionControlAction,
    SelectionControlsState,
)
from diffusion_editor.app.native_canvas_controls import NativeCanvasControls
from diffusion_editor.canvas.brush import BrushToolMode


def test_native_controls_programmatic_sync_suppresses_feedback():
    document = tc_ui_document_create()
    brush_intents = []
    selection_intents = []
    controls = NativeCanvasControls(
        document,
        BrushControlsState(
            tool=BrushToolMode.PAINT,
            size=20,
            hardness=0.4,
            flow=1.0,
            color=(255, 255, 255, 255),
        ),
        SelectionControlsState(),
        brush_intents.append,
        selection_intents.append,
        viewport_rect=lambda: Rect(0.0, 0.0, 640.0, 480.0),
    )
    assert document.add_root(controls.widget.handle)

    controls.apply_brush_state(BrushControlsState(
        tool=BrushToolMode.MOVE,
        size=41,
        hardness=0.8,
        flow=0.3,
        color=(10, 20, 30, 255),
        draw_patch=True,
        show_patch=False,
    ))
    controls.apply_selection_state(SelectionControlsState(
        edit_mode=True,
        eraser=True,
        size=61,
        hardness=0.7,
        flow=0.5,
        show=False,
    ))

    assert brush_intents == []
    assert selection_intents == []
    model, command_id = controls.brush.tool_commands[BrushToolMode.MOVE]
    assert model.command(command_id).data.checked
    assert not controls.brush.size.widget.visible
    assert controls.selection.edit_mode.checked
    assert controls.selection.size.value == 61

    model, command_id = controls.brush.tool_commands[BrushToolMode.PAINT]
    controls.brush._on_tool_activated(
        0, command_id, model.command(command_id).data)
    controls.selection.rect_mode.checked = True

    assert brush_intents[-1].action == BrushControlAction.TOOL
    assert brush_intents[-1].value == BrushToolMode.PAINT
    assert selection_intents[-1].action == SelectionControlAction.RECT_MODE
    assert selection_intents[-1].value is True

    controls.close()
    tc_ui_document_destroy(document)


def test_native_brush_mode_toolbar_routes_pointer_and_keyboard():
    document = tc_ui_document_create()
    brush_intents = []
    controls = NativeCanvasControls(
        document,
        BrushControlsState(
            tool=BrushToolMode.PAINT,
            size=20,
            hardness=0.4,
            flow=1.0,
            color=(255, 255, 255, 255),
        ),
        SelectionControlsState(),
        brush_intents.append,
        lambda _intent: None,
        viewport_rect=lambda: Rect(0.0, 0.0, 640.0, 480.0),
    )
    assert document.add_root(controls.widget.handle)
    document.layout_roots(Rect(0.0, 0.0, 320.0, 700.0))
    toolbar = controls.brush.toolbars[0]

    pointer = PointerEvent()
    pointer.type = PointerEventType.Down
    pointer.button = 0
    pointer.x = toolbar.item_rects[1].x + 3.0
    pointer.y = toolbar.item_rects[1].y + 3.0
    assert document.dispatch_pointer_event(pointer) == EventResult.Handled
    pointer.type = PointerEventType.Up
    assert document.dispatch_pointer_event(pointer) == EventResult.Handled
    assert brush_intents[-1].value == BrushToolMode.ERASER

    key = KeyEvent()
    key.type = KeyEventType.Down
    key.key = KeyCode.Left
    assert document.dispatch_key_event(key) == EventResult.Handled
    key.key = KeyCode.Space
    assert document.dispatch_key_event(key) == EventResult.Handled
    assert brush_intents[-1].value == BrushToolMode.PAINT

    controls.close()
    tc_ui_document_destroy(document)


def test_native_brush_color_dialog_accept_cancel_and_reopen():
    document = tc_ui_document_create()
    brush_intents = []
    controls = NativeCanvasControls(
        document,
        BrushControlsState(
            tool=BrushToolMode.PAINT,
            size=20,
            hardness=0.4,
            flow=1.0,
            color=(255, 255, 255, 255),
        ),
        SelectionControlsState(),
        brush_intents.append,
        lambda _intent: None,
        viewport_rect=lambda: Rect(0.0, 0.0, 640.0, 480.0),
    )
    assert document.add_root(controls.widget.handle)
    dialog = controls.brush.color_dialog

    assert dialog.show(Rect(0.0, 0.0, 640.0, 480.0))
    dialog.color = Color(0.2, 0.4, 0.6, 0.8)
    assert dialog.activate("ok")
    assert brush_intents[-1].action == BrushControlAction.COLOR
    assert brush_intents[-1].value == (51, 102, 153, 204)

    count = len(brush_intents)
    assert dialog.show(Rect(0.0, 0.0, 640.0, 480.0))
    assert dialog.activate("cancel")
    assert len(brush_intents) == count

    controls.close()
    tc_ui_document_destroy(document)
