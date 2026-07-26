"""Tests for LayerStack prefix compositing."""

import numpy as np
import pytest

from diffusion_editor.document.layer import Layer
from diffusion_editor.document.layer_stack import LayerStack
from diffusion_editor.canvas.canvas_tools import MoveTool
from diffusion_editor.canvas.canvas_geometry import union_rect


def _solid_image(w, h, r, g, b, a):
    img = np.zeros((h, w, 4), dtype=np.uint8)
    img[:, :] = [r, g, b, a]
    return img


def _make_stack(n_layers=5, w=64, h=64):
    """Create a stack with a red background and n_layers-1 semi-transparent layers."""
    stack = LayerStack()
    stack.on_changed = lambda: None
    bg = _solid_image(w, h, 255, 0, 0, 255)
    stack.init_from_image(bg)
    for i in range(n_layers - 1):
        img = _solid_image(w, h, 0, (i + 1) * 50, 0, 128)
        stack.add_layer(f"Layer {i}", img)
    return stack


# ---------- basic composite correctness ----------


class TestCompositeCorrectness:
    def test_single_layer(self):
        stack = LayerStack()
        stack.on_changed = lambda: None
        stack.init_from_image(_solid_image(32, 32, 100, 200, 50, 255))
        result = stack.composite()
        assert result.shape == (32, 32, 4)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result[:, :, 0], 100)

    def test_two_opaque_layers(self):
        stack = LayerStack()
        stack.on_changed = lambda: None
        stack.init_from_image(_solid_image(8, 8, 255, 0, 0, 255))
        stack.add_layer("top", _solid_image(8, 8, 0, 255, 0, 255))
        result = stack.composite()
        # Opaque green on top of red -> green
        assert result[0, 0, 0] == 0
        assert result[0, 0, 1] == 255
        assert result[0, 0, 3] == 255

    def test_empty_stack(self):
        stack = LayerStack()
        result = stack.composite()
        assert result.shape == (1, 1, 4)

    def test_composite_idempotent(self):
        stack = _make_stack()
        a = stack.composite()
        b = stack.composite()
        np.testing.assert_array_equal(a, b)

    def test_offset_layer_composites_at_canvas_position(self):
        stack = LayerStack(tile_size=8)
        stack.on_changed = lambda: None
        stack.init_from_image(_solid_image(16, 16, 0, 0, 0, 0))
        patch = _solid_image(4, 3, 255, 0, 0, 255)
        layer = Layer("patch", 4, 3, patch, x=5, y=6)
        stack.insert_layer(layer)

        result = stack.composite()
        assert result[6, 5, 0] == 255
        assert result[8, 8, 0] == 255
        assert result[5, 5, 3] == 0
        assert result[6, 4, 3] == 0

    def test_offset_layer_clips_at_canvas_edge(self):
        stack = LayerStack(tile_size=8)
        stack.on_changed = lambda: None
        stack.init_from_image(_solid_image(8, 8, 0, 0, 0, 0))
        patch = _solid_image(4, 4, 0, 255, 0, 255)
        layer = Layer("patch", 4, 4, patch, x=6, y=6)
        stack.insert_layer(layer)

        result = stack.composite()
        assert result[6, 6, 1] == 255
        assert result[7, 7, 1] == 255
        assert result[:6, :, 3].sum() == 0
        assert result[:, :6, 3].sum() == 0

    def test_public_composite_is_straight_rgba(self):
        stack = LayerStack(tile_size=8)
        image = np.array(
            [[[203, 91, 17, 37], [88, 99, 111, 0]]],
            dtype=np.uint8,
        )
        stack.init_from_image(image)

        result = stack.composite()

        # A straight-alpha boundary preserves the colour of a translucent
        # source.  RGB hidden behind zero alpha is normalized to zero.
        np.testing.assert_array_equal(
            result,
            np.array([[[203, 91, 17, 37], [0, 0, 0, 0]]], dtype=np.uint8),
        )

    def test_group_opacity_matches_porter_duff_reference(self):
        stack = LayerStack(tile_size=8)
        stack._width = 1
        stack._height = 1
        bottom = Layer(
            "bottom", 1, 1, _solid_image(1, 1, 20, 40, 240, 128))
        group = Layer(
            "group", 1, 1, _solid_image(1, 1, 200, 20, 40, 128))
        child = Layer(
            "child", 1, 1, _solid_image(1, 1, 10, 220, 30, 128))
        group.add_child(child)
        group.opacity = 0.4
        stack._layers = [group, bottom]
        stack._active_layer = group
        stack._rebuild_caches()

        def _premultiplied(color):
            rgba = np.asarray(color, dtype=np.float64) / 255.0
            return np.concatenate((rgba[:3] * rgba[3], rgba[3:4]))

        def _over(source, destination):
            return source + destination * (1.0 - source[3])

        bottom_p = _premultiplied((20, 40, 240, 128))
        child_p = _premultiplied((10, 220, 30, 128))
        own_p = _premultiplied((200, 20, 40, 128))
        subtree = _over(own_p, child_p) * group.opacity
        expected_p = _over(subtree, bottom_p)
        expected = np.concatenate((
            expected_p[:3] / expected_p[3],
            expected_p[3:4],
        ))
        expected = np.rint(expected * 255.0).astype(np.uint8)

        np.testing.assert_allclose(
            stack.composite()[0, 0],
            expected,
            atol=1,
        )


# ---------- prefix cache behaviour ----------


class TestPrefixCache:
    def test_cached_composite_returns_equal(self):
        stack = _make_stack()
        first = stack.composite()
        second = stack.composite()
        np.testing.assert_array_equal(first, second)

    def test_mark_dirty_recomputes_correctly(self):
        stack = _make_stack(3)
        before = stack.composite().copy()
        # Modify the top layer image
        stack.layers[0].image[:] = _solid_image(64, 64, 255, 255, 0, 255)
        stack.mark_layer_dirty(stack.layers[0])
        after = stack.composite()
        assert not np.array_equal(before, after)

    def test_mark_bottom_dirty_recomputes(self):
        stack = _make_stack(3)
        before = stack.composite().copy()
        stack.layers[-1].image[:] = _solid_image(64, 64, 0, 0, 255, 255)
        stack.mark_layer_dirty(stack.layers[-1])
        after = stack.composite()
        assert not np.array_equal(before, after)

    def test_structural_change_recomputes(self):
        stack = _make_stack(3)
        before = stack.composite().copy()
        stack.add_layer("new", _solid_image(64, 64, 255, 255, 0, 128))
        after = stack.composite()
        assert not np.array_equal(before, after)


# ---------- visibility ----------


class TestVisibility:
    def test_hidden_layer_not_blended(self):
        stack = LayerStack()
        stack.on_changed = lambda: None
        stack.init_from_image(_solid_image(8, 8, 255, 0, 0, 255))
        stack.add_layer("green", _solid_image(8, 8, 0, 255, 0, 255))
        stack.set_visibility(stack.layers[0], False)
        result = stack.composite()
        # Green hidden -> red background
        assert result[0, 0, 0] == 255
        assert result[0, 0, 1] == 0

    def test_toggle_off_and_on(self):
        stack = _make_stack(3)
        before = stack.composite().copy()
        stack.set_visibility(stack.layers[0], False)
        stack.composite()
        stack.set_visibility(stack.layers[0], True)
        after = stack.composite()
        np.testing.assert_array_equal(before, after)

    def test_visibility_invalidates_composite(self):
        stack = _make_stack(3)
        before = stack.composite().copy()
        stack.set_visibility(stack.layers[0], False)
        after = stack.composite()
        assert not np.array_equal(before, after)


# ---------- opacity ----------


class TestOpacity:
    def test_zero_opacity_equals_hidden(self):
        stack = LayerStack()
        stack.on_changed = lambda: None
        stack.init_from_image(_solid_image(8, 8, 255, 0, 0, 255))
        stack.add_layer("green", _solid_image(8, 8, 0, 255, 0, 255))

        stack.set_opacity(stack.layers[0], 0.0)
        r1 = stack.composite()

        stack.set_opacity(stack.layers[0], 1.0)
        stack.set_visibility(stack.layers[0], False)
        r2 = stack.composite()

        np.testing.assert_array_equal(r1, r2)

    def test_opacity_invalidates_composite(self):
        stack = _make_stack(3)
        before = stack.composite().copy()
        stack.set_opacity(stack.layers[0], 0.5)
        after = stack.composite()
        assert not np.array_equal(before, after)


# ---------- exclude_layer (composite_below) ----------


class TestCompositeExcluding:
    def test_exclude_top_layer(self):
        stack = LayerStack()
        stack.on_changed = lambda: None
        stack.init_from_image(_solid_image(8, 8, 255, 0, 0, 255))
        stack.add_layer("green", _solid_image(8, 8, 0, 255, 0, 255))
        below = stack.composite(exclude_layer=stack.layers[0])
        # Should be just the red background
        assert below[0, 0, 0] == 255
        assert below[0, 0, 1] == 0

    def test_exclude_bottom_layer(self):
        stack = LayerStack()
        stack.on_changed = lambda: None
        stack.init_from_image(_solid_image(8, 8, 255, 0, 0, 255))
        below = stack.composite(exclude_layer=stack.layers[0])
        # Nothing below the only layer -> black/transparent
        assert below[0, 0, 3] == 0

    def test_get_prefix_below(self):
        stack = _make_stack(5)
        stack.composite()
        top = stack.layers[0]
        cache = stack.get_prefix_below(top)
        assert cache is not None
        assert cache.shape == (64, 64, 4)
        assert cache.dtype == np.uint8


# ---------- structural operations ----------


class TestStructuralOps:
    def test_add_layer(self):
        stack = _make_stack(3)
        stack.composite()
        stack.add_layer("extra", _solid_image(64, 64, 255, 0, 255, 128))
        result = stack.composite()
        assert result.shape == (64, 64, 4)

    def test_remove_layer(self):
        stack = _make_stack(4)
        stack.composite()
        stack.remove_layer(stack.layers[0])
        result = stack.composite()
        assert result.shape[0] == 64

    def test_move_layer(self):
        stack = _make_stack(4)
        r_before = stack.composite().copy()
        top = stack.layers[0]
        stack.move_layer(top, None, len(stack.layers))
        # After moving top to bottom, result should differ
        r_after = stack.composite()
        assert not np.array_equal(r_before, r_after)

    def test_flatten(self):
        stack = _make_stack(5)
        expected = stack.composite()
        stack.flatten()
        assert len(stack.layers) == 1
        result = stack.composite()
        np.testing.assert_array_equal(result, expected)

    def test_translucent_flatten_is_idempotent(self):
        stack = LayerStack(tile_size=8)
        image = _solid_image(4, 4, 203, 91, 17, 37)
        stack.init_from_image(image)
        expected = stack.composite().copy()

        stack.flatten()
        once = stack.composite().copy()
        stack.flatten()
        twice = stack.composite()

        np.testing.assert_array_equal(once, expected)
        np.testing.assert_array_equal(twice, expected)

    def test_init_from_image(self):
        stack = _make_stack(5)
        stack.composite()
        new_img = _solid_image(32, 32, 0, 0, 255, 255)
        stack.init_from_image(new_img)
        result = stack.composite()
        assert result.shape == (32, 32, 4)
        assert result[0, 0, 2] == 255

    def test_move_tool_changes_offset_not_pixels(self):
        class FakeToolContext:
            def __init__(self, stack):
                self._layer_stack = stack
                self._composite = None
                self.image_set = False

            @staticmethod
            def union_rect(a, b):
                return union_rect(a, b)

            def set_image(self, image):
                self.image_set = True

            def refresh_layer_transform(self, layer, dirty):
                self._layer_stack.mark_layer_dirty(layer, dirty)
                self._composite = np.ascontiguousarray(
                    self._layer_stack.composite())
                self.set_image(self._composite)

        stack = LayerStack(tile_size=8)
        stack.on_changed = lambda: None
        stack.init_from_image(_solid_image(16, 16, 0, 0, 0, 0))
        image = _solid_image(4, 4, 255, 0, 0, 255)
        layer = Layer("patch", 4, 4, image, x=2, y=3)
        stack.insert_layer(layer)
        before = layer.image.copy()
        context = FakeToolContext(stack)
        tool = MoveTool()

        dirty0 = tool.begin(context, layer, 10, 10)
        dirty1 = tool.move(context, layer, (10, 10), 13, 8)

        assert dirty0 == (2, 3, 6, 7)
        assert dirty1 == (2, 1, 9, 7)
        assert (layer.x, layer.y) == (5, 1)
        np.testing.assert_array_equal(layer.image, before)


# ---------- mark_layer_dirty edge cases ----------


class TestMarkDirtyEdgeCases:
    def test_mark_unknown_layer(self):
        stack = _make_stack(3)
        stack.composite()
        orphan = Layer("orphan", 64, 64)
        # Should not crash
        stack.mark_layer_dirty(orphan)
        result = stack.composite()
        assert result.dtype == np.uint8

    def test_mark_dirty_before_first_composite(self):
        stack = _make_stack(3)
        # All already dirty, should not crash
        stack.mark_layer_dirty(stack.layers[0])
        result = stack.composite()
        assert result.dtype == np.uint8


# ---------- nested layers ----------


class TestNestedLayerCaching:
    """Test prefix caching with nested (child) layers per architecture example."""

    def _make_tree_stack(self):
        r"""Build the architecture example tree:
        A      (B,E,D,C)
        |-C    (B,E,D)
        |.|-D  (E)
        |.\-E  ()
        \-B    ()
        """
        stack = LayerStack()
        stack.on_changed = lambda: None
        # Create canvas
        stack._width = 8
        stack._height = 8
        # B = red background (bottom root)
        B = Layer("B", 8, 8, _solid_image(8, 8, 255, 0, 0, 255))
        # A = blue (top root)
        A = Layer("A", 8, 8, _solid_image(8, 8, 0, 0, 255, 128))
        # C = green (child of A)
        C = Layer("C", 8, 8, _solid_image(8, 8, 0, 255, 0, 128))
        # D = yellow (top child of C)
        D = Layer("D", 8, 8, _solid_image(8, 8, 255, 255, 0, 128))
        # E = cyan (bottom child of C)
        E = Layer("E", 8, 8, _solid_image(8, 8, 0, 255, 255, 128))

        C.add_child(D, 0)  # D = top child (children[0])
        C.add_child(E, 1)  # E = bottom child (children[1])
        A.add_child(C, 0)  # C = child of A

        stack._layers = [A, B]  # [0]=top, [-1]=bottom
        stack._active_layer = A
        stack._rebuild_caches()
        return stack, A, B, C, D, E

    def test_prefix_below_root(self):
        stack, A, B, C, D, E = self._make_tree_stack()
        # prefix(B) = empty (nothing below B)
        prefix_B = stack.get_prefix_below(B)
        assert prefix_B is not None
        assert prefix_B[0, 0, 3] == 0  # transparent

    def test_prefix_below_top_root(self):
        stack, A, B, C, D, E = self._make_tree_stack()
        # prefix(A) should include B, E, D, C
        prefix_A = stack.get_prefix_below(A)
        assert prefix_A is not None
        # Should not be empty (B is below)
        assert prefix_A[0, 0, 3] > 0

    def test_prefix_below_nested_layer(self):
        stack, A, B, C, D, E = self._make_tree_stack()
        # get_prefix_below(E) = external(B) + previous(None) + nested(None)
        # E has no previous sibling and no children, but external context
        # includes B (red) from root level.
        prefix_E = stack.get_prefix_below(E)
        assert prefix_E is not None
        assert prefix_E[0, 0, 0] > 0  # red from B
        assert prefix_E[0, 0, 3] > 0  # not transparent

    def test_prefix_below_second_child(self):
        stack, A, B, C, D, E = self._make_tree_stack()
        # prefix(D) = (E) -- should contain E's contribution
        prefix_D = stack.get_prefix_below(D)
        assert prefix_D is not None
        assert prefix_D[0, 0, 3] > 0  # not empty

    def test_prefix_below_parent_includes_external(self):
        stack, A, B, C, D, E = self._make_tree_stack()
        # prefix(C) = (B, E, D) -- includes B (external) + children
        prefix_C = stack.get_prefix_below(C)
        assert prefix_C is not None
        # Should include red from B
        assert prefix_C[0, 0, 0] > 0  # red channel from B

    def test_composite_exclude_nested(self):
        stack, A, B, C, D, E = self._make_tree_stack()
        # composite(exclude_layer=D) should equal prefix(D)
        excluded = stack.composite(exclude_layer=D)
        prefix_D = stack.get_prefix_below(D)
        np.testing.assert_array_equal(excluded, prefix_D)

    def test_full_composite_includes_all(self):
        stack, A, B, C, D, E = self._make_tree_stack()
        result = stack.composite()
        # Should not be transparent (has visible layers)
        assert result[0, 0, 3] > 0


class TestNestedDirtyPropagation:
    def _make_simple_tree(self):
        stack = LayerStack()
        stack.on_changed = lambda: None
        stack._width = 4
        stack._height = 4
        B = Layer("B", 4, 4, _solid_image(4, 4, 255, 0, 0, 255))
        A = Layer("A", 4, 4, _solid_image(4, 4, 0, 0, 255, 128))
        C = Layer("C", 4, 4, _solid_image(4, 4, 0, 255, 0, 128))
        A.add_child(C, 0)
        stack._layers = [A, B]
        stack._active_layer = A
        stack._rebuild_caches()
        return stack, A, B, C

    def test_dirty_child_recomputes_correctly(self):
        stack, A, B, C = self._make_simple_tree()
        before = stack.composite().copy()
        # Modify C's image
        C.image[:] = _solid_image(4, 4, 255, 255, 0, 255)
        stack.mark_layer_dirty(C)
        after = stack.composite()
        # Result should change
        assert not np.array_equal(before, after)

    def test_dirty_parent_recomputes(self):
        stack, A, B, C = self._make_simple_tree()
        before = stack.composite().copy()
        A.image[:] = _solid_image(4, 4, 255, 0, 255, 255)
        stack.mark_layer_dirty(A)
        after = stack.composite()
        assert not np.array_equal(before, after)

    def test_dirty_bottom_does_not_affect_nothing_below(self):
        stack, A, B, C = self._make_simple_tree()
        before = stack.composite().copy()
        # Modify B and mark dirty
        B.image[:] = _solid_image(4, 4, 0, 0, 255, 255)
        stack.mark_layer_dirty(B)
        after = stack.composite()
        assert not np.array_equal(before, after)


class TestSubtreeOpacity:
    def test_parent_opacity_affects_children(self):
        stack = LayerStack()
        stack.on_changed = lambda: None
        stack._width = 8
        stack._height = 8
        B = Layer("B", 8, 8, _solid_image(8, 8, 0, 0, 0, 0))
        A = Layer("A", 8, 8, _solid_image(8, 8, 0, 0, 0, 0))
        C = Layer("C", 8, 8, _solid_image(8, 8, 0, 255, 0, 255))
        A.add_child(C, 0)
        stack._layers = [A, B]
        stack._active_layer = A
        stack._rebuild_caches()

        # With A.opacity = 1.0
        r1 = stack.composite().copy()

        # With A.opacity = 0.0 -- children should also disappear
        stack.set_opacity(A, 0.0)
        r2 = stack.composite()
        assert r2[0, 0, 3] == 0  # fully transparent


class TestDeepNesting:
    def test_three_levels(self):
        stack = LayerStack()
        stack.on_changed = lambda: None
        stack._width = 4
        stack._height = 4
        root = Layer("root", 4, 4, _solid_image(4, 4, 255, 0, 0, 255))
        mid = Layer("mid", 4, 4, _solid_image(4, 4, 0, 255, 0, 128))
        leaf = Layer("leaf", 4, 4, _solid_image(4, 4, 0, 0, 255, 128))
        mid.add_child(leaf, 0)
        root.add_child(mid, 0)
        stack._layers = [root]
        stack._active_layer = root
        stack._rebuild_caches()

        result = stack.composite()
        assert result[0, 0, 3] > 0

        # prefix(leaf) = external context only (no previous, no children).
        # External context = root's previous = None (root is first root layer).
        # So prefix(leaf) is empty.
        prefix_leaf = stack.get_prefix_below(leaf)
        assert prefix_leaf is not None
        assert prefix_leaf[0, 0, 3] == 0

        # prefix(mid) = external(None) + previous(None) + nested(leaf_composite)
        # mid has leaf as child, so nested is leaf's composite (blue at alpha 128).
        prefix_mid = stack.get_prefix_below(mid)
        assert prefix_mid is not None
        assert prefix_mid[0, 0, 3] > 0  # includes leaf

    def test_translucent_nested_solo_is_not_blended_twice(self):
        stack = LayerStack(tile_size=8)
        stack._width = 2
        stack._height = 2
        ancestor = Layer("ancestor", 2, 2, _solid_image(
            2, 2, 255, 0, 0, 128))
        middle = Layer("middle", 2, 2, _solid_image(
            2, 2, 0, 255, 0, 128))
        solo = Layer("solo", 2, 2, _solid_image(
            2, 2, 40, 100, 220, 96))
        middle.add_child(solo)
        ancestor.add_child(middle)
        stack._layers = [ancestor]
        stack._active_layer = solo
        stack.set_solo_layer(solo)

        reference = LayerStack(tile_size=8)
        reference.init_from_image(solo.image.copy())

        np.testing.assert_array_equal(
            stack.composite(),
            reference.composite(),
        )


# ---------- get_prefix_below_rect ----------


class TestPrefixBelowRect:
    def test_rect_matches_full_prefix(self):
        stack = _make_stack(3, w=64, h=64)
        top = stack.layers[0]
        full = stack.get_prefix_below(top)
        rect = stack.get_prefix_below_rect(top, 10, 10, 50, 50)
        np.testing.assert_array_equal(rect, full[10:50, 10:50])

    def test_rect_empty_region(self):
        stack = _make_stack(3, w=64, h=64)
        top = stack.layers[0]
        rect = stack.get_prefix_below_rect(top, 30, 30, 30, 30)
        assert rect.shape[0] == 0

    def test_rect_clamped_to_bounds(self):
        stack = _make_stack(3, w=64, h=64)
        top = stack.layers[0]
        rect = stack.get_prefix_below_rect(top, -10, -10, 70, 70)
        assert rect.shape == (64, 64, 4)

    def test_composite_rect_matches_canonical_full_composite(self):
        stack = _make_stack(4, w=64, h=64)
        full = stack.composite()

        rect = stack.composite_rect(7, 11, 53, 47)

        np.testing.assert_array_equal(rect, full[11:47, 7:53])

    @pytest.mark.parametrize("group_opacity", (0.0, 0.5, 1.0))
    def test_nested_prefix_applies_ancestor_group_opacity(
            self, group_opacity):
        stack = LayerStack(tile_size=8)
        stack.init_from_image(_solid_image(
            1, 1, 0, 0, 255, 255))
        group = Layer("group", 1, 1)
        stack.insert_layer(group)
        lower = Layer(
            "lower", 1, 1, _solid_image(1, 1, 255, 0, 0, 255))
        stack.insert_layer(lower)
        stack.move_layer(lower, group, 0)
        target = Layer(
            "target", 1, 1, _solid_image(1, 1, 0, 255, 0, 255))
        stack.insert_layer(target)
        stack.move_layer(target, group, 0)
        stack.set_opacity(group, group_opacity)

        prefix = stack.get_prefix_below(target)
        prefix_rect = stack.get_prefix_below_rect(target, 0, 0, 1, 1)
        stack.set_visibility(target, False)
        expected = stack.composite()

        np.testing.assert_array_equal(prefix, expected)
        np.testing.assert_array_equal(prefix_rect, expected)


def test_cache_memory_bytes_reports_renderer_cache_size():
    stack = _make_stack(2, w=16, h=16)
    stack.composite()

    assert isinstance(stack.cache_memory_bytes(), int)
    assert stack.cache_memory_bytes() >= 0


def test_flat_prefix_cache_aliases_previous_composite_storage():
    stack = _make_stack(3, w=64, h=64)

    stack.composite()

    one_float_tile = 64 * 64 * 4 * np.dtype(np.float32).itemsize
    assert stack.cache_memory_bytes() <= 3 * one_float_tile


def test_render_cache_memory_limit_preserves_composite_correctness():
    reference = _make_stack(5, w=64, h=64)
    expected = reference.composite()
    limited = _make_stack(5, w=64, h=64)
    one_float_tile = 64 * 64 * 4 * np.dtype(np.float32).itemsize
    limited.set_cache_memory_limit_bytes(one_float_tile)

    actual = limited.composite()

    np.testing.assert_array_equal(actual, expected)
    assert limited.cache_memory_bytes() <= one_float_tile
    assert limited.cache_memory_limit_bytes == one_float_tile


def test_translucent_stack_without_cache_stays_linear():
    stack = LayerStack()
    stack.on_changed = lambda: None
    stack.init_from_image(_solid_image(16, 16, 255, 0, 0, 255))
    for index in range(14):
        stack.add_layer(
            f"Layer {index}",
            _solid_image(16, 16, 0, 20 + index, 120, 180),
        )
    for layer in stack.layers:
        stack.set_opacity(layer, 0.5)
    expected = stack.composite()
    stack.set_cache_memory_limit_bytes(0)

    renderer = stack._renderer
    original = renderer._layer_canvas_tile
    calls = 0

    def counted(layer, tx, ty):
        nonlocal calls
        calls += 1
        return original(layer, tx, ty)

    renderer._layer_canvas_tile = counted
    actual = stack.composite()

    np.testing.assert_array_equal(actual, expected)
    assert calls == 15
    assert stack.cache_memory_bytes() == 0
