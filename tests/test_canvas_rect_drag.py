from diffusion_editor.canvas.canvas_rect_drag import CanvasRectDrag, CanvasRectDragController


def test_rect_drag_ignores_begin_when_disabled():
    drag = CanvasRectDrag(include_end_pixel=True, min_size=1)

    assert drag.begin(1, 2) is False
    assert drag.dragging is False


def test_rect_drag_tracks_raw_preview_rect():
    drag = CanvasRectDrag(include_end_pixel=True, min_size=1)
    drag.set_enabled(True)

    assert drag.begin(5, 6) is True
    assert drag.move(2, 3) is True

    assert drag.preview_rect() == (5, 6, 2, 3)


def test_selection_rect_finish_includes_end_pixel():
    drag = CanvasRectDrag(include_end_pixel=True, min_size=1)
    drag.set_enabled(True)
    drag.begin(5, 6)

    result = drag.finish(2, 3)

    assert result is not None
    assert result.rect == (2, 3, 6, 7)
    assert drag.enabled is False
    assert drag.dragging is False


def test_patch_rect_finish_excludes_end_pixel():
    drag = CanvasRectDrag(include_end_pixel=False, min_size=2)
    drag.set_enabled(True)
    drag.begin(5, 6)

    result = drag.finish(2, 3)

    assert result is not None
    assert result.rect == (2, 3, 5, 6)


def test_rect_drag_rejects_tiny_rect():
    drag = CanvasRectDrag(include_end_pixel=False, min_size=2)
    drag.set_enabled(True)
    drag.begin(5, 5)

    assert drag.finish(6, 6) is None
    assert drag.enabled is False
    assert drag.dragging is False


def test_disabling_rect_drag_cancels_active_drag():
    drag = CanvasRectDrag(include_end_pixel=True, min_size=1)
    drag.set_enabled(True)
    drag.begin(1, 1)

    drag.set_enabled(False)

    assert drag.preview_rect() is None
    assert drag.dragging is False


def test_rect_drag_controller_routes_selection_finish():
    controller = CanvasRectDragController()
    controller.set_selection_rect_mode(True)

    assert controller.begin_selection_rect(1, 2) is True
    assert controller.move(4, 5) is True
    result = controller.finish(4, 5)

    assert result.handled is True
    assert result.target == "selection"
    assert result.rect == (1, 2, 5, 6)


def test_rect_drag_controller_routes_patch_finish():
    controller = CanvasRectDragController()
    controller.set_patch_rect_mode(True)

    assert controller.begin_patch_rect(5, 6) is True
    result = controller.finish(1, 2)

    assert result.handled is True
    assert result.target == "patch"
    assert result.rect == (1, 2, 5, 6)


def test_rect_drag_controller_handles_rejected_rect_without_result_rect():
    controller = CanvasRectDragController()
    controller.set_patch_rect_mode(True)

    assert controller.begin_patch_rect(5, 5) is True
    result = controller.finish(6, 6)

    assert result.handled is True
    assert result.target == "patch"
    assert result.rect is None
