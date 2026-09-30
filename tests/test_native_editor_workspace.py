"""The workspace contract, exercised through native controls and input."""

import numpy as np
import pytest
from termin.gui_native import KeyCode, PointerEvent, PointerEventType, Rect

from diffusion_editor.app.native_root import NativeEditorRoot
from test_native_root import _application


def _click(root, rect):
    event = PointerEvent()
    event.button = 0
    event.x, event.y = rect.x + rect.width / 2, rect.y + rect.height / 2
    for kind in (PointerEventType.Down, PointerEventType.Up):
        event.type = kind
        root.composition.document.dispatch_pointer_event(event)
    root.tick()


def _selected(controls):
    return [tool for tool, (model, command_id) in controls.tool_commands.items()
            if model.command(command_id).data.checked]


def _assert_tool_glyphs_rendered(root):
    frame = root.composition.read_frame_rgba_float()
    for tool, toolbar in root.canvas_controls.toolbars.items():
        rect = toolbar.item_rects[0]
        x, y = round(rect.x), round(rect.y)
        crop = frame[y + 6:y + round(rect.height) - 6,
                     x + 6:x + round(rect.width) - 6, :3]
        # Actual bright glyph pixels, excluding the button border/highlight.
        assert np.count_nonzero(np.all(crop > .6, axis=2)) > 5, tool


@pytest.mark.parametrize("width,height", [(1280, 800), (1024, 700)])
def test_workspace_tool_flow_and_inspector_state(width, height):
    app = _application()
    app.layer_stack.init_from_image(np.full((100, 160, 4), 255, dtype=np.uint8))
    with NativeEditorRoot.create_headless(app, width=width, height=height) as root:
        root.tick()
        controls, coordinator, view = root.canvas_controls, root.canvas_controls_coordinator, root.view
        assert _selected(controls) == ["paint"]
        assert not view.ai_panel_visible
        _assert_tool_glyphs_rendered(root)

        palette, options, canvas, inspector = (controls.widget.bounds, controls.options_widget.bounds,
                                              root.canvas.widget.bounds, view.inspector.bounds)
        assert palette.x == 0
        assert palette.width == 52
        assert canvas.x == 272
        assert palette.x + palette.width <= canvas.x
        buttons = [toolbar.widget.bounds for toolbar in controls.toolbars.values()]
        assert len({rect.x for rect in buttons}) == 1
        assert len({rect.y for rect in buttons}) == 10
        assert all(rect.width == 44 and rect.height == 44 for rect in buttons)
        assert options.x == palette.x + palette.width
        assert options.x + options.width <= canvas.x
        assert canvas.y == view.workspace_row.bounds.y
        assert canvas.height == view.workspace_row.bounds.height
        assert view.tool_options_scroll.content_size.width <= 220
        assert canvas.x + canvas.width <= inspector.x
        assert canvas.width >= 400
        assert controls.toolbars["transform"].widget.bounds.y < height - 30

        _click(root, controls.toolbars["select_rect"].item_rects[0])
        assert _selected(controls) == ["select_rect"]
        root.canvas.controller.pointer_down(10, 10, 0)
        root.canvas.controller.pointer_up(40, 40)
        assert coordinator.active_tool == "select_rect"
        _click(root, controls.toolbars["transform"].item_rects[0])
        assert root.transform_controller.session.target == "selection"
        assert _selected(controls) == ["transform"]
        assert not controls.options_widget.visible
        root.transform_controller.set_dimension("width", 50)
        _click(root, root.transform_controls.apply_button.widget.bounds)
        assert coordinator.active_tool == "select_rect"
        assert _selected(controls) == ["select_rect"]
        assert app.history.can_undo
        # Target switching and Escape must preserve the pre-transform tool.
        controls.activate_tool("transform")
        root.transform_controls.target.selected_index = 1
        assert root.transform_controller.session.target == "layer"
        root.composition.document.clear_focus(root.composition.document.focused_widget)
        root.composition.push_key(KeyCode.Escape.value)
        root.tick()
        assert coordinator.active_tool == "select_rect"

        # Native tab clicks must switch panels without resizing the canvas.
        tabs = view.inspector.bounds
        _click(root, Rect(tabs.x + 80, tabs.y + 4, 55, 20))
        assert view.ai_panel_visible
        assert view.inspector_tabs.selected_index == 1
        ai = view.ai_scroll.widget.bounds
        canvas = root.canvas.widget.bounds
        assert canvas.x + canvas.width <= ai.x
        assert ai.width >= 260
        _click(root, controls.toolbars["mask"].item_rects[0])
        assert _selected(controls) == ["mask"]
        assert not coordinator.selection_state.rect_mode
        controls.brush.size.value = 73
        root.layer_panel.ai_tool_combo.selected_index = 3  # LaMa, no model loading.
        assert app.layer_stack.active_layer.tool.tool_type == "lama"
        root.tick()
        open_canvas_width = root.canvas.widget.bounds.width
        tabs = view.inspector.bounds
        _click(root, Rect(tabs.x + 4, tabs.y + 4, 60, 20))
        assert not view.ai_panel_visible
        _assert_tool_glyphs_rendered(root)
        assert root.canvas.widget.bounds.width == open_canvas_width
        assert controls.widget.bounds.width == 52
        assert view.inspector_tabs.selected_index == 0
        _click(root, controls.toolbars["select_rect"].item_rects[0])
        assert _selected(controls) == ["select_rect"]
        _click(root, controls.toolbars["mask"].item_rects[0])
        assert _selected(controls) == ["mask"]
        view.set_agent_panel_visible(True)
        root.tick()
        view.set_agent_panel_visible(False)
        root.tick()
        tabs = view.inspector.bounds
        _click(root, Rect(tabs.x + 80, tabs.y + 4, 55, 20))
        assert view.ai_panel_visible
        assert controls.brush.size.value == 73
        assert root.layer_panel.ai_tool_combo.selected_index == 3
        assert view.ai_scroll.content_size.width <= view.ai_scroll.widget.bounds.width
        assert _selected(controls) == ["mask"]
        assert root.layer_panel.ai_remove_tool.widget.enabled
        root.composition.document.clear_focus(root.composition.document.focused_widget)
        root.composition.push_key(KeyCode.B.value)
        root.tick()
        assert _selected(controls) == ["paint"]


@pytest.mark.parametrize("input_kind", ["text_input", "text_area", "spin_box"])
def test_tool_shortcuts_do_not_steal_text_input(input_kind):
    app = _application()
    app.layer_stack.init_from_image(np.full((32, 32, 4), 255, dtype=np.uint8))
    with NativeEditorRoot.create_headless(app, width=1024, height=700) as root:
        document = root.composition.document
        text = getattr(document, f"create_{input_kind}")()
        root.view.canvas_host.add_preferred_child(text.widget)
        root.tick()
        assert document.set_focus(text.handle)
        root.composition.push_key(KeyCode.M.value)
        root.tick()
        assert root.canvas_controls_coordinator.active_tool == "paint"
        document.clear_focus(document.focused_widget)
        root.composition.push_key(KeyCode.M.value)
        root.tick()
        assert root.canvas_controls_coordinator.active_tool == "select_rect"


def test_transform_restores_each_tool_and_brackets_edit_current_brush():
    app = _application()
    app.layer_stack.init_from_image(np.full((32, 32, 4), 255, dtype=np.uint8))
    with NativeEditorRoot.create_headless(app, width=1024, height=700) as root:
        controls, coordinator = root.canvas_controls, root.canvas_controls_coordinator
        for tool in ("paint", "eraser", "smudge", "move", "select_rect", "select_brush", "mask", "mask_eraser", "patch"):
            controls.activate_tool(tool)
            controls.activate_tool("transform")
            assert coordinator.active_tool == "transform"
            root.transform_controller.cancel()
            assert coordinator.active_tool == tool
            assert _selected(controls) == [tool]
        controls.activate_tool("select_brush")
        root.tick()
        root.tick()
        assert coordinator.active_tool == "select_brush"
        size = controls.selection.size.value
        paint_size = controls.brush.size.value
        assert root.view.dispatch_shortcut(ord("]"), 0)
        assert controls.selection.size.value == size + 5
        assert controls.brush.size.value == paint_size
        controls.activate_tool("paint")
        assert root.view.dispatch_shortcut(ord("]"), 0)
        assert controls.brush.size.value == paint_size + 5


def test_tooltip_names_icon_and_shortcut_without_blocking_input():
    app = _application()
    app.layer_stack.init_from_image(np.full((32, 32, 4), 255, dtype=np.uint8))
    with NativeEditorRoot.create_headless(app, width=1024, height=700) as root:
        root.tick()
        controls = root.canvas_controls
        rect = controls.toolbars["select_rect"].item_rects[0]
        event = PointerEvent()
        event.type = PointerEventType.Move
        event.x, event.y = rect.x + 10, rect.y + 10
        root.composition.document.dispatch_pointer_event(event)
        root.tick()
        assert controls.tooltip_text.text == "Rectangle (M)"
        assert root.composition.document.overlay_count == 1
        _click(root, rect)
        assert root.canvas_controls_coordinator.active_tool == "select_rect"
        event.x, event.y = 700, 600
        root.composition.document.dispatch_pointer_event(event)
        root.tick()
        assert root.composition.document.overlay_count == 0
