import numpy as np
from termin.gui_native import (
    KeyCode, Point, PointerEvent, PointerEventType,
)

from diffusion_editor.app.native_root import NativeEditorRoot
from test_native_root import _application


def test_native_transform_controls_input_history_and_layout(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMIN_SDK_SHADER_CACHE_ROOT", str(tmp_path / "shader-cache"))
    application = _application()
    image = np.zeros((120, 160, 4), dtype=np.uint8)
    image[20:60, 30:80] = (245, 90, 25, 255)
    application.layer_stack.init_from_image(image)
    stack = application.layer_stack
    stack.selection.data[20:60, 30:80] = 1
    with NativeEditorRoot.create_headless(application, width=1280, height=800) as root:
        root.tick()
        assert root.canvas_controls_coordinator.begin_transform()
        transform = root.transform_controller
        controls = root.transform_controls
        root.tick()
        assert transform.session.target == "selection"
        assert not root.canvas_controls.brush.widget.enabled
        assert controls.target.selected_index == 0
        assert controls.widget.bounds.x + controls.widget.bounds.width <= root.canvas.widget.bounds.x
        assert root.canvas.widget.bounds.y == root.view.workspace_row.bounds.y
        assert root.view.tool_options_scroll.content_size.width <= 220
        assert controls.apply_button.widget.bounds.width >= 40
        controls.dimensions["width"].value = 75
        assert transform.session.rect == (30, 20, 105, 80)
        assert controls.dimensions["height"].value == 60
        controls.aspect.checked = False
        controls.dimensions["height"].value = 50
        assert transform.session.rect == (30, 20, 105, 70)
        root.tick()

        start = root.canvas.canvas.image_to_widget(Point(60, 40))
        end = root.canvas.canvas.image_to_widget(Point(70, 45))
        event = PointerEvent()
        event.button = 0
        event.type = PointerEventType.Down
        event.x, event.y = start.x, start.y
        root.composition.document.dispatch_pointer_event(event)
        assert root.composition.document.pointer_capture
        event.type = PointerEventType.Up
        event.x, event.y = end.x, end.y
        root.composition.document.dispatch_pointer_event(event)
        assert not root.composition.document.pointer_capture
        assert transform.session.rect == (40, 25, 115, 75)
        np.testing.assert_array_equal(stack.active_layer.image, image)
        assert root.canvas.dispatch_shortcut(KeyCode.Enter.value, 0)
        assert not transform.active and application.history.can_undo
        assert root.canvas_controls.brush.widget.enabled
        application.document.undo()
        np.testing.assert_array_equal(stack.active_layer.image, image)
        assert not application.history.can_undo
        assert root.canvas_controls_coordinator.begin_transform()
        controls.target.selected_index = 1
        assert transform.session.target == "layer"
        assert transform.session.rect == (0, 0, 160, 120)
        controls.dimensions["width"].value = 200
        assert root.canvas.dispatch_shortcut(KeyCode.Right.value, 0)
        assert transform.session.rect[0] == 1
        assert root.canvas.dispatch_shortcut(KeyCode.Escape.value, 0)
        assert not application.history.can_undo
        np.testing.assert_array_equal(stack.active_layer.image, image)
        root.tick()
