import numpy as np
import pytest

from diffusion_editor.document.document_service import DocumentService
from diffusion_editor.document.commands import ClearSelectedPixelsCommand, ClearSelectionCommand
from diffusion_editor.document.history import HistoryManager
from diffusion_editor.document.layer import Layer
from diffusion_editor.document.layer_stack import LayerStack
from diffusion_editor.document.raster_transform import (
    ApplyRasterTransformCommand, RasterTransform, TransformPreviewRenderer,
)


def setup_document():
    stack = LayerStack()
    stack.init_from_image(np.zeros((24, 32, 4), dtype=np.uint8))
    image = np.full((8, 10, 4), (220, 40, 10, 255), dtype=np.uint8)
    stack.insert_image_layer("Subject", image, x=5, y=4)
    history = HistoryManager(stack.load_state)
    document = DocumentService(stack, history, stack.load_state)
    return stack, document, history


def test_move_selection_on_offset_layer_expands_bounds_and_undoes_once():
    stack, document, history = setup_document()
    layer = stack.active_layer
    before = layer.image.copy()
    stack.selection.data[5:8, 6:10] = 1
    selection_before = stack.selection.data.copy()
    session = RasterTransform(stack)
    assert session.target == "selection"
    assert session.rect == (6, 5, 10, 8)
    session.set_rect((-2, 6, 2, 9))
    np.testing.assert_array_equal(layer.image, before)
    document.execute(ApplyRasterTransformCommand(session))
    assert layer.x == -2 and layer.y == 4
    np.testing.assert_array_equal(layer.image[2:5, :4], before[1:4, 1:5])
    assert not layer.image[1:4, 8:12, 3].any()
    assert stack.selection.bbox() == (0, 6, 2, 9)
    assert stack.validate_invariants()
    assert document.undo() == "Transform Selection"
    assert (layer.x, layer.y) == (5, 4)
    np.testing.assert_array_equal(layer.image, before)
    np.testing.assert_array_equal(stack.selection.data, selection_before)
    assert not history.can_undo
    assert document.redo() == "Transform Selection"
    assert layer.x == -2
    assert stack.validate_invariants()


def test_soft_selection_identity_is_exact_and_does_not_create_history():
    stack, document, history = setup_document()
    layer = stack.active_layer
    layer.image[:, :, 3] = 123
    stack.selection.data[5:8, 6:10] = .3
    before = layer.image.copy()
    session = RasterTransform(stack)
    session.set_rect((10, 8, 18, 14))
    session.set_rect(session.source_rect)
    document.execute(ApplyRasterTransformCommand(session))
    np.testing.assert_array_equal(layer.image, before)
    assert not history.can_undo


def test_soft_cut_preserves_fractional_alpha_and_unselected_pixels():
    stack, _, _ = setup_document()
    layer = stack.active_layer
    layer.image[:, :, 3] = 200
    stack.selection.data[5:7, 6:8] = .25
    session = RasterTransform(stack)
    session.set_rect((18, 5, 20, 7))
    result, _ = session.result()
    assert np.all(result.image[1:3, 1:3, 3] == 150)
    assert np.all(result.image[1:3, 13:15, 3] == 50)
    np.testing.assert_array_equal(result.image[0, :10], layer.image[0])


def test_resize_is_recomputed_from_original_and_has_no_transparent_rgb_fringe():
    stack, _, _ = setup_document()
    layer = stack.active_layer
    layer.image[:, :5] = (255, 0, 0, 255)
    layer.image[:, 5:] = (0, 255, 0, 0)
    session = RasterTransform(stack)
    session.set_rect((5, 4, 10, 8))
    session.result()
    session.set_rect((5, 4, 25, 20))
    result, _ = session.result()
    fresh = RasterTransform(stack)
    fresh.set_rect(session.rect)
    expected, _ = fresh.result()
    np.testing.assert_array_equal(result.image, expected.image)
    visible = result.image[:, :, 3] > 0
    assert np.all(result.image[:, :, 0][visible] == 255)
    assert not result.image[:, :, 1][visible].any()


def test_layer_resize_resamples_processing_mask_and_patch_and_preserves_metadata():
    stack, document, _ = setup_document()
    layer = stack.active_layer
    layer.mask.data[2:4, 2:6] = 1
    layer.patch_rect = (2, 2, 6, 4)
    layer.opacity = .4
    session = RasterTransform(stack, "layer")
    session.set_rect((2, 3, 22, 19))
    document.execute(ApplyRasterTransformCommand(session))
    assert layer.mask.data.shape == (16, 20)
    assert layer.patch_rect == (4, 4, 12, 8)
    assert layer.name == "Subject" and layer.opacity == .4
    assert stack.validate_invariants()
    document.undo()
    assert layer.patch_rect == (2, 2, 6, 4)
    assert layer.mask.data.shape == (8, 10)


def test_preview_matches_committed_composite_with_occluding_layers_and_opacity():
    stack, document, _ = setup_document()
    source = stack.active_layer
    source.opacity = .5
    stack.insert_image_layer("Foreground", np.full((6, 6, 4), (0, 0, 255, 200), dtype=np.uint8), 9, 7)
    stack.active_layer = source
    session = RasterTransform(stack)
    session.set_rect((8, 6, 23, 18))
    result, _ = session.result()
    preview = TransformPreviewRenderer(stack, source, result.preview_layer(source)).composite_full_straight_rgba()
    assert source.bounds == (5, 4, 15, 12)
    document.execute(ApplyRasterTransformCommand(session))
    np.testing.assert_array_equal(stack.composite(), preview)


def test_selection_in_hole_between_disjoint_regions_does_not_transform_whole_layer():
    stack, _, _ = setup_document()
    stack.selection.data[:2, :2] = 1
    stack.selection.data[20:22, 25:27] = 1
    with pytest.raises(ValueError, match="does not intersect"):
        RasterTransform(stack)


def test_stale_session_rejected_and_oversized_transform_does_not_mutate():
    stack, document, history = setup_document()
    session = RasterTransform(stack)
    with pytest.raises(ValueError, match="too large"):
        session.set_rect((0, 0, 16384, 16384))
    assert session.rect == session.source_rect
    session.set_rect((0, 0, 10, 8))
    stack.set_opacity(stack.active_layer, .5)
    with pytest.raises(ValueError, match="Document changed"):
        document.execute(ApplyRasterTransformCommand(session))
    assert not history.can_undo


def test_expansion_keeps_processing_mask_and_patch_in_canvas_coordinates():
    stack, document, _ = setup_document()
    layer = stack.active_layer
    layer.mask.data[2, 3] = 1
    layer.patch_rect = (2, 2, 6, 4)
    stack.selection.data[5:7, 6:8] = 1
    session = RasterTransform(stack)
    session.set_rect((-2, -3, 0, -1))
    document.execute(ApplyRasterTransformCommand(session))
    assert layer.mask.data[9, 10] == 1
    assert layer.local_rect_to_canvas(layer.patch_rect) == (7, 6, 11, 8)
    assert stack.selection.is_empty
    assert stack.validate_invariants()


def test_transform_undo_preserves_older_pixel_and_selection_history_references():
    stack, document, _ = setup_document()
    layer = stack.active_layer
    original = layer.image.copy()
    stack.selection.data[5:7, 6:8] = 1
    original_selection = stack.selection.data.copy()
    document.execute(ClearSelectedPixelsCommand(layer=layer))
    document.execute(ClearSelectionCommand())
    session = RasterTransform(stack)
    session.set_rect((2, 2, 22, 18))
    document.execute(ApplyRasterTransformCommand(session))
    document.undo()
    document.undo()
    document.undo()
    np.testing.assert_array_equal(layer.image, original)
    np.testing.assert_array_equal(stack.selection.data, original_selection)
    document.redo()
    document.redo()
    document.redo()
    assert layer.bounds == (2, 2, 22, 18)
    assert stack.selection.is_empty


@pytest.mark.parametrize("target", ["layer", "selection"])
@pytest.mark.parametrize("multilayer", [False, True])
def test_tiled_session_preview_matches_full_commit_for_random_alpha(target, multilayer):
    rng = np.random.default_rng(351)
    stack = LayerStack(tile_size=16)
    pixels = rng.integers(0, 256, (48, 64, 4), dtype=np.uint8)
    pixels[5:10, 5:10, 3] = 0  # hidden RGB must follow canonical normalization
    stack.init_from_image(pixels)
    if multilayer:
        stack.insert_image_layer("Subject", pixels[:30, :40].copy(), x=-5, y=6)
        stack.active_layer.opacity = .6
    source = stack.active_layer
    if target == "selection":
        stack.selection.data[5:22, 4:30] = rng.random((17, 26), dtype=np.float32)
    session = RasterTransform(stack, target)
    renderer = TransformPreviewRenderer(stack, source, session)
    for rect in ((20, 20, 50, 42), (-10, -6, 20, 16), (23, 22, 42, 35), session.source_rect):
        session.set_rect(rect)
        result, _ = session.result()
        expected = TransformPreviewRenderer(stack, source, result.preview_layer(source)).composite_full_straight_rgba()
        renderer.set_replacement(session, (0, 0, 64, 48))
        actual = np.zeros_like(expected)
        for y in range(0, 48, 16):
            for x in range(0, 64, 16):
                actual[y:y+16, x:x+16] = renderer.composite_preview_rect((x,y,x+16,y+16))
        np.testing.assert_array_equal(actual, expected)
