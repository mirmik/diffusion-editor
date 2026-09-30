import numpy as np
import pytest

from diffusion_editor.canvas.canvas_transform import CanvasTransformController
from diffusion_editor.canvas.editor_canvas_controller import EditorCanvasController
from diffusion_editor.document.commands import SetLayerOpacityCommand
from diffusion_editor.document.raster_transform import TransformPreviewRenderer
from test_raster_transform import setup_document


@pytest.fixture
def editor():
    stack, document, history = setup_document()
    images = []
    canvas = EditorCanvasController(stack, gpu_compositing=False,
        set_image=images.append, set_overlay=lambda _: None)
    canvas.refresh()
    transform = CanvasTransformController(stack, document, canvas)
    yield stack, document, history, canvas, transform, images
    transform.close()
    canvas.dispose()


def test_multi_gesture_preview_cancel_and_single_apply(editor):
    stack, document, history, canvas, transform, images = editor
    original = stack.active_layer.image.copy()
    revision = stack.revision
    assert transform.begin()
    transform.hit_radius = .1
    canvas.pointer_down(9, 7, 0)
    canvas.pointer_move(12, 9)
    canvas.pointer_up(12, 9)
    assert transform.session.rect == (8, 6, 18, 14)
    transform.set_dimension("width", 20)
    assert transform.session.rect == (8, 6, 28, 22)
    assert len(canvas.annotations()) == 1
    assert canvas.annotations()[0].kind == "transform"
    assert stack.revision == revision and not history.can_undo
    np.testing.assert_array_equal(stack.active_layer.image, original)
    transform.apply()
    assert not transform.active and history.can_undo
    assert stack.active_layer.bounds == (8, 6, 28, 22)
    document.undo()
    assert not history.can_undo
    np.testing.assert_array_equal(stack.active_layer.image, original)


def test_capture_cancel_rolls_back_gesture_and_escape_can_cancel_session(editor):
    stack, _, history, canvas, transform, images = editor
    assert transform.begin()
    transform.hit_radius = .1
    canvas.pointer_down(9, 7, 0)
    canvas.pointer_move(13, 10)
    canvas.pointer_cancel()
    assert transform.session.rect == (5, 4, 15, 12)
    assert not canvas.pointer_interaction_active
    transform.set_dimension("width", 20)
    transform.cancel()
    assert not transform.active and not history.can_undo
    np.testing.assert_array_equal(images[-1], stack.composite())


def test_mutation_barrier_cancels_preview_before_another_command(editor):
    stack, document, history, _, transform, _ = editor
    assert transform.begin()
    transform.set_dimension("width", 20)
    document.execute(SetLayerOpacityCommand(layer=stack.active_layer, opacity=.5))
    assert not transform.active
    assert stack.active_layer.width == 10
    assert stack.active_layer.opacity == .5
    document.undo()
    assert not history.can_undo


def test_layer_switch_cancels_preview(editor):
    stack, _, history, _, transform, _ = editor
    assert transform.begin()
    transform.set_dimension("width", 20)
    stack.active_layer = stack.layers[-1]
    assert not transform.active and not history.can_undo


def test_corner_keeps_opposite_corner_and_can_resize_freely(editor):
    _, _, _, canvas, transform, _ = editor
    assert transform.begin()
    transform.hit_radius = .1
    canvas.pointer_down(5, 4, 0)
    canvas.pointer_up(0, 0)
    assert transform.session.rect == (0, 0, 15, 12)
    transform.keep_aspect = False
    canvas.pointer_down(15, 12, 0)
    canvas.pointer_up(18, 20)
    assert transform.session.rect == (0, 0, 18, 20)


def test_target_switch_restarts_from_original_without_committing_preview(editor):
    stack, _, history, _, transform, _ = editor
    stack.selection.data[5:7, 6:8] = 1
    assert transform.begin()
    assert transform.session.target == "selection"
    transform.set_dimension("width", 8)
    assert transform.begin("layer")
    assert transform.session.rect == (5, 4, 15, 12)
    assert not history.can_undo


def test_incremental_preview_matches_full_render_across_moves_resize_and_return(editor):
    stack, _, _, _, transform, images = editor
    stack.selection.data[5:7, 6:8] = .5
    assert transform.begin()
    for rect in ((15, 8, 20, 12), (-2, -1, 2, 3), (6, 5, 8, 7), (30, 22, 34, 26)):
        transform.set_rect(rect)
        result, _ = transform.session.result()
        expected = TransformPreviewRenderer(stack, stack.active_layer,
            result.preview_layer(stack.active_layer)).composite_full_straight_rgba()
        np.testing.assert_array_equal(images[-1], expected)


def test_corner_can_resize_using_vertical_movement_only(editor):
    _, _, _, canvas, transform, _ = editor
    assert transform.begin()
    transform.hit_radius = .1
    canvas.pointer_down(15, 12, 0)
    canvas.pointer_up(15, 20)
    assert transform.session.rect == (5, 4, 25, 20)


def test_drag_never_builds_full_result_or_resamples_unchanged_size(editor, monkeypatch):
    from diffusion_editor.document import raster_transform
    stack, _, _, canvas, transform, _ = editor
    stack.selection.data[5:7, 6:8] = .5
    assert transform.begin()
    session = transform.session
    def forbidden():
        pytest.fail("Pointer movement must not build a full raster/masks")
    monkeypatch.setattr(session, "result", forbidden)
    transform.set_rect((16, 10, 20, 14))
    # Once sized, moving must reuse its premultiplied samples.
    monkeypatch.setattr(raster_transform, "_resize", lambda *_: forbidden())
    transform.set_rect((17, 11, 21, 15))
    assert canvas.composite_bridge.preview_active


def test_dirty_upload_excludes_source_and_path_for_distant_fragment(editor):
    stack, _, _, canvas, transform, _ = editor
    stack.selection.data[5:7, 6:8] = 1
    uploads = []
    canvas.composite_bridge._update_image_region = lambda x, y, image: uploads.append((x, y, image.shape[:2]))
    assert transform.begin()
    transform.set_rect((20, 15, 22, 17))
    assert uploads == [(6, 5, (2, 2)), (20, 15, (2, 2))]
    uploads.clear()
    transform.set_rect((21, 15, 23, 17))
    assert uploads == [(20, 15, (2, 3))]
    uploads.clear()
    transform.set_rect((21, 15, 23, 17))
    assert uploads == []
