"""Tests for LayerStack prefix compositing."""

import numpy as np
import pytest

from diffusion_editor.layer import Layer, LayerStack


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
        # Opaque green on top of red → green
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


# ---------- prefix cache behaviour ----------


class TestPrefixCache:
    def test_cache_built_on_first_composite(self):
        stack = _make_stack()
        all_layers = stack._all_layers_flat()
        assert all(layer in stack._dirty for layer in all_layers)
        assert all(layer in stack._nested_dirty for layer in all_layers)
        stack.composite()
        assert len(stack._dirty) == 0
        assert len(stack._nested_dirty) == 0

    def test_cached_composite_returns_equal(self):
        stack = _make_stack()
        first = stack.composite()
        second = stack.composite()
        np.testing.assert_array_equal(first, second)

    def test_mark_top_layer_dirty(self):
        stack = _make_stack(5)
        stack.composite()
        stack.mark_layer_dirty(stack.layers[0])
        # Only top layer (layers[0]) should be dirty
        assert stack.layers[0] in stack._dirty
        # Bottom layer should remain clean
        assert stack.layers[-1] not in stack._dirty

    def test_mark_bottom_layer_dirty(self):
        stack = _make_stack(5)
        stack.composite()
        stack.mark_layer_dirty(stack.layers[-1])
        # All layers should be dirty
        for layer in stack.layers:
            assert layer in stack._dirty

    def test_mark_middle_layer_dirty(self):
        stack = _make_stack(5)
        stack.composite()
        stack.mark_layer_dirty(stack.layers[2])
        # layers[2] and all above (layers[0], layers[1]) should be dirty
        for layer in stack.layers[:3]:
            assert layer in stack._dirty
        # layers below (layers[3], layers[4]) should be clean
        for layer in stack.layers[3:]:
            assert layer not in stack._dirty

    def test_structural_change_rebuilds_caches(self):
        stack = _make_stack(3)
        stack.composite()
        assert len(stack._dirty) == 0
        stack.add_layer("new")
        all_layers = stack._all_layers_flat()
        assert all(layer in stack._dirty for layer in all_layers)
        assert len(stack._prefix) == len(all_layers)


# ---------- visibility ----------


class TestVisibility:
    def test_toggle_visibility_invalidates(self):
        stack = _make_stack(3)
        stack.composite()
        stack.set_visibility(stack.layers[0], False)
        assert len(stack._dirty) > 0

    def test_hidden_layer_not_blended(self):
        stack = LayerStack()
        stack.on_changed = lambda: None
        stack.init_from_image(_solid_image(8, 8, 255, 0, 0, 255))
        stack.add_layer("green", _solid_image(8, 8, 0, 255, 0, 255))
        stack.set_visibility(stack.layers[0], False)
        result = stack.composite()
        # Green hidden → red background
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


# ---------- opacity ----------


class TestOpacity:
    def test_set_opacity_invalidates(self):
        stack = _make_stack(3)
        stack.composite()
        stack.set_opacity(stack.layers[0], 0.5)
        assert len(stack._dirty) > 0

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
        # Nothing below the only layer → black/transparent
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
        stack.add_layer("extra")
        all_layers = stack._all_layers_flat()
        assert len(stack._prefix) == len(all_layers)
        assert all(layer in stack._dirty for layer in all_layers)

    def test_remove_layer(self):
        stack = _make_stack(4)
        stack.composite()
        stack.remove_layer(stack.layers[0])
        all_layers = stack._all_layers_flat()
        assert len(stack._prefix) == len(all_layers)
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

    def test_init_from_image(self):
        stack = _make_stack(5)
        stack.composite()
        new_img = _solid_image(32, 32, 0, 0, 255, 255)
        stack.init_from_image(new_img)
        assert len(stack._prefix) == 1
        assert len(stack._dirty) == 1
        result = stack.composite()
        assert result.shape == (32, 32, 4)


# ---------- mark_layer_dirty edge cases ----------


class TestMarkDirtyEdgeCases:
    def test_mark_unknown_layer(self):
        stack = _make_stack(3)
        stack.composite()
        orphan = Layer("orphan", 64, 64)
        # Should not crash, rebuilds all caches
        stack.mark_layer_dirty(orphan)
        all_layers = stack._all_layers_flat()
        assert all(layer in stack._dirty for layer in all_layers)

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
        """Build the architecture example tree:
        A      (B,E,D,C)
        |-C    (B,E,D)
        |.|-D  (E)
        |.\-E  ()
        \\-B    ()
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
        # prefix(D) = (E) — should contain E's contribution
        prefix_D = stack.get_prefix_below(D)
        assert prefix_D is not None
        assert prefix_D[0, 0, 3] > 0  # not empty

    def test_prefix_below_parent_includes_external(self):
        stack, A, B, C, D, E = self._make_tree_stack()
        # prefix(C) = (B, E, D) — includes B (external) + children
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

    def test_nested_cache_is_saved_explicitly(self):
        stack, A, B, C, D, E = self._make_tree_stack()
        stack.composite()

        assert hasattr(stack, "_nested")
        assert stack._nested[C] is not None  # C has children -> nested cache stored
        assert stack._nested[A] is not None  # A has child C -> nested cache stored
        assert stack._nested[B] is None  # no children
        assert stack._nested[D] is None  # no children
        assert stack._nested[E] is None  # no children


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

    def test_dirty_child_propagates_to_parent(self):
        stack, A, B, C = self._make_simple_tree()
        stack.composite()
        assert len(stack._dirty) == 0
        assert len(stack._nested_dirty) == 0
        stack.mark_layer_dirty(C)
        # C, A should be dirty (C is child of A)
        assert C in stack._dirty
        assert A in stack._dirty
        assert C in stack._nested_dirty
        assert A in stack._nested_dirty
        # B should remain clean
        assert B not in stack._dirty
        assert B not in stack._nested_dirty

    def test_dirty_child_recomputes_correctly(self):
        stack, A, B, C = self._make_simple_tree()
        before = stack.composite().copy()
        # Modify C's image
        C.image[:] = _solid_image(4, 4, 255, 255, 0, 255)
        stack.mark_layer_dirty(C)
        after = stack.composite()
        # Result should change
        assert not np.array_equal(before, after)

    def test_dirty_child_clears_parent_nested_cache(self):
        stack, A, B, C = self._make_simple_tree()
        stack.composite()
        assert stack._nested[A] is not None

        stack.mark_layer_dirty(C)

        assert stack._nested[C] is None
        assert stack._nested[A] is None


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

        # With A.opacity = 0.0 — children should also disappear
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
        # External context = root's Прошлый = None (root is first root layer).
        # So prefix(leaf) is empty.
        prefix_leaf = stack.get_prefix_below(leaf)
        assert prefix_leaf is not None
        assert prefix_leaf[0, 0, 3] == 0

        # prefix(mid) = external(None) + previous(None) + nested(leaf_composite)
        # mid has leaf as child, so nested is leaf's composite (blue at alpha 128).
        prefix_mid = stack.get_prefix_below(mid)
        assert prefix_mid is not None
        assert prefix_mid[0, 0, 3] > 0  # includes leaf
