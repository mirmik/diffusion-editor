from diffusion_editor.canvas.canvas_geometry import (
    canvas_to_layer_point,
    clip_canvas_rect,
    clip_layer_local_rect,
    union_rect,
    visible_layer_rect,
)
from diffusion_editor.document.layer import Layer


def test_union_rect_handles_empty_inputs():
    assert union_rect(None, (1, 2, 3, 4)) == (1, 2, 3, 4)
    assert union_rect((1, 2, 3, 4), None) == (1, 2, 3, 4)
    assert union_rect(None, None) is None


def test_union_rect_merges_bounds():
    assert union_rect((3, 4, 8, 9), (1, 6, 5, 12)) == (1, 4, 8, 12)


def test_canvas_to_layer_point_accounts_for_layer_offset():
    layer = Layer("Offset", 10, 10, x=5, y=-3)
    assert canvas_to_layer_point(layer, 8, 4) == (3, 7)


def test_clip_canvas_rect_clamps_to_canvas_bounds():
    assert clip_canvas_rect((-2, 3, 12, 20), 10, 15) == (0, 3, 10, 15)


def test_visible_layer_rect_returns_local_visible_region():
    layer = Layer("Offset", 10, 8, x=-3, y=4)
    assert visible_layer_rect(layer, 20, 20) == (3, 0, 10, 8)


def test_visible_layer_rect_returns_empty_for_offscreen_layer():
    layer = Layer("Offscreen", 4, 4, x=10, y=10)
    assert visible_layer_rect(layer, 8, 8) == (0, 0, 0, 0)


def test_clip_layer_local_rect_clips_to_layer_and_canvas():
    layer = Layer("Offset", 10, 10, x=-3, y=2)

    local_rect, canvas_rect = clip_layer_local_rect(
        layer,
        (-5, -5, 9, 9),
        20,
        20,
    )

    assert local_rect == (3, 0, 9, 9)
    assert canvas_rect == (0, 2, 6, 11)


def test_clip_layer_local_rect_returns_empty_when_outside_canvas():
    layer = Layer("Offset", 5, 5, x=20, y=20)
    assert clip_layer_local_rect(layer, (0, 0, 5, 5), 10, 10) == (None, None)
