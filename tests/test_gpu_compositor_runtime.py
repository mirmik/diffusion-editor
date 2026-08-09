from types import SimpleNamespace
import struct

import numpy as np
import pytest
from tgfx import TextureEncoding

from diffusion_editor.canvas.gpu_compositor import GPUCompositor
from diffusion_editor.canvas import gpu_compositor as gpu_compositor_module
from diffusion_editor.canvas.gpu_compositor import _linear_rgba_to_srgb8
from diffusion_editor.document.layer_stack import LayerStack


class _FakeContext:
    def __init__(self, *, in_frame=False, fail_binding=False):
        self.in_frame = in_frame
        self.fail_binding = fail_binding
        self.calls = []

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            if name == "begin_frame":
                self.in_frame = True
            elif name == "end_frame":
                self.in_frame = False
            elif name == "bind_texture_by_name" and self.fail_binding:
                raise RuntimeError("injected resource binding failure")

        return call


def _compositor_for_context(ctx):
    compositor = GPUCompositor.__new__(GPUCompositor)
    compositor._stack = SimpleNamespace(width=4, height=4, layers=[])
    compositor._ctx = ctx
    compositor._graphics = SimpleNamespace(context=ctx)
    compositor._main_tex = object()
    compositor._display_tex = object()
    compositor._composite_vs = object()
    compositor._composite_fs = object()
    compositor._composite_shader = object()
    compositor._unpremul_vs = object()
    compositor._unpremul_fs = object()
    compositor._unpremul_shader = object()
    compositor._quad_verts = np.zeros((6, 7), dtype=np.float32)
    compositor._dirty = True
    compositor._ensure_context = lambda: None
    compositor._ensure_attachments = lambda width, height: None
    compositor._sync_dirty_textures = lambda: None
    return compositor


class _FakeGraphics:
    def __init__(self):
        self.calls = []
        self._next_texture = 1

    def create_texture_rgba8(self, width, height, data, encoding):
        assert encoding == TextureEncoding.LINEAR
        texture = f"texture-{self._next_texture}"
        self._next_texture += 1
        self.calls.append((
            "create", texture, width, height, np.asarray(data).copy(), encoding))
        return texture

    def upload_texture(self, texture, data):
        self.calls.append(("upload", texture, np.asarray(data).copy()))

    def upload_texture_region(
            self, texture, x, y, width, height, data):
        self.calls.append((
            "upload-region",
            texture,
            x,
            y,
            width,
            height,
            np.asarray(data).copy(),
        ))

    def destroy_texture(self, texture):
        self.calls.append(("destroy", texture))


def _upload_compositor(width=8, height=6):
    image = np.arange(
        width * height * 4, dtype=np.uint8).reshape(height, width, 4)
    layer = SimpleNamespace(image=image, width=width, height=height)
    stack = SimpleNamespace(_all_layers_flat=lambda: [layer])
    graphics = _FakeGraphics()
    compositor = GPUCompositor(stack, graphics)
    return compositor, layer, graphics


def test_linear_readback_is_encoded_to_srgb_without_changing_alpha():
    linear = np.array([[[0.0, 0.0031308, 0.5, 0.25]]], dtype=np.float32)

    encoded = _linear_rgba_to_srgb8(linear)

    np.testing.assert_array_equal(encoded, [[[0, 10, 188, 64]]])


def test_composite_uses_symbolic_texture_binding():
    ctx = _FakeContext()
    compositor = _compositor_for_context(ctx)

    compositor.composite()

    texture_bindings = [
        args for name, args, _ in ctx.calls if name == "bind_texture_by_name"
    ]
    assert texture_bindings == [("u_texture", compositor._main_tex)]
    assert [
        args for name, args, _ in ctx.calls if name == "set_cull"
    ] == [(0,), (0,)]
    assert ctx.in_frame is False


def test_composite_closes_owned_frame_after_resource_binding_error():
    ctx = _FakeContext(fail_binding=True)
    compositor = _compositor_for_context(ctx)

    with pytest.raises(RuntimeError, match="injected resource binding failure"):
        compositor.composite()

    assert ctx.in_frame is False
    assert [name for name, _, _ in ctx.calls].count("end_frame") == 1
    assert [name for name, _, _ in ctx.calls].count("end_pass") >= 1


def test_composite_does_not_close_borrowed_frame():
    ctx = _FakeContext(in_frame=True)
    compositor = _compositor_for_context(ctx)

    compositor.composite()

    assert ctx.in_frame is True
    assert not any(name == "begin_frame" for name, _, _ in ctx.calls)
    assert not any(name == "end_frame" for name, _, _ in ctx.calls)


def test_dirty_layer_region_uploads_only_the_requested_pixels():
    compositor, layer, graphics = _upload_compositor()
    compositor._sync_dirty_textures()
    graphics.calls.clear()
    layer.image[1:3, 2:5] = 231

    compositor.mark_dirty(layer, (2, 1, 5, 3))
    compositor._sync_dirty_textures()

    assert len(graphics.calls) == 1
    call = graphics.calls[0]
    assert call[:6] == (
        "upload-region", "texture-1", 2, 1, 3, 2,
    )
    np.testing.assert_array_equal(
        call[6],
        np.ascontiguousarray(layer.image[1:3, 2:5]).reshape(-1),
    )


def test_dirty_layer_regions_are_clipped_and_unioned_before_upload():
    compositor, layer, graphics = _upload_compositor()
    compositor._sync_dirty_textures()
    graphics.calls.clear()

    compositor.mark_dirty(layer, (-2, 2, 3, 5))
    compositor.mark_dirty(layer, (2, 1, 12, 4))
    compositor._sync_dirty_textures()

    assert len(graphics.calls) == 1
    call = graphics.calls[0]
    assert call[:6] == (
        "upload-region", "texture-1", 0, 1, 8, 4,
    )
    np.testing.assert_array_equal(
        call[6],
        np.ascontiguousarray(layer.image[1:5, 0:8]).reshape(-1),
    )


def test_full_dirty_mark_dominates_partial_layer_regions():
    compositor, layer, graphics = _upload_compositor()
    compositor._sync_dirty_textures()
    graphics.calls.clear()

    compositor.mark_dirty(layer, (1, 1, 2, 2))
    compositor.mark_dirty(layer)
    compositor.mark_dirty(layer, (3, 3, 4, 4))
    compositor._sync_dirty_textures()

    assert [call[0] for call in graphics.calls] == ["upload"]
    np.testing.assert_array_equal(
        graphics.calls[0][2],
        np.ascontiguousarray(layer.image).reshape(-1),
    )


def test_layer_resize_reallocates_even_after_partial_dirty_mark():
    compositor, layer, graphics = _upload_compositor()
    compositor._sync_dirty_textures()
    graphics.calls.clear()
    layer.image = np.zeros((4, 5, 4), dtype=np.uint8)
    layer.width = 5
    layer.height = 4

    compositor.mark_dirty(layer, (1, 1, 3, 3))
    compositor._sync_dirty_textures()

    assert [call[0] for call in graphics.calls] == ["destroy", "create"]
    assert graphics.calls[1][2:4] == (5, 4)
    assert compositor._dirty_layer_regions == {}


def test_composite_only_dirty_does_not_upload_layer_pixels():
    compositor, _layer, graphics = _upload_compositor()
    compositor._sync_dirty_textures()
    graphics.calls.clear()

    compositor.mark_composite_dirty()
    compositor._sync_dirty_textures()

    assert graphics.calls == []


def test_group_texture_is_declared_premultiplied_for_opacity_draw():
    ctx = _FakeContext()
    compositor = GPUCompositor.__new__(GPUCompositor)
    compositor._ctx = ctx
    compositor._quad_verts = np.zeros((6, 7), dtype=np.float32)

    compositor._draw_texture_quad(
        "group-texture",
        0.5,
        source_premultiplied=True,
    )

    uniform_calls = [
        args for name, args, _ in ctx.calls
        if name == "bind_uniform_by_name"
    ]
    assert len(uniform_calls) == 1
    opacity, premultiplied = struct.unpack(
        gpu_compositor_module._COMPOSITE_PARAMS_FMT,
        uniform_calls[0][1],
    )
    assert opacity == pytest.approx(0.5)
    assert premultiplied == pytest.approx(1.0)


def test_metadata_only_rebuild_does_not_upload_unchanged_layer_pixels():
    stack = LayerStack(tile_size=8)
    stack.init_from_image(np.zeros((6, 8, 4), dtype=np.uint8))
    layer = stack.active_layer
    graphics = _FakeGraphics()
    compositor = GPUCompositor(stack, graphics)
    compositor._sync_dirty_textures()
    graphics.calls.clear()

    stack.set_opacity(layer, 0.5)
    compositor.rebuild()
    compositor._sync_dirty_textures()

    assert graphics.calls == []


def test_pixel_revision_triggers_full_upload_without_an_explicit_region():
    stack = LayerStack(tile_size=8)
    stack.init_from_image(np.zeros((6, 8, 4), dtype=np.uint8))
    layer = stack.active_layer
    graphics = _FakeGraphics()
    compositor = GPUCompositor(stack, graphics)
    compositor._sync_dirty_textures()
    graphics.calls.clear()
    layer.image[2, 3] = (1, 2, 3, 255)

    stack.mark_layer_dirty(layer)
    compositor.rebuild()
    compositor._sync_dirty_textures()

    assert [call[0] for call in graphics.calls] == ["upload"]
