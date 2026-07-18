from types import SimpleNamespace

import numpy as np
import pytest

from diffusion_editor.canvas.gpu_compositor import GPUCompositor


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


def test_composite_uses_symbolic_texture_binding():
    ctx = _FakeContext()
    compositor = _compositor_for_context(ctx)

    compositor.composite()

    texture_bindings = [
        args for name, args, _ in ctx.calls if name == "bind_texture_by_name"
    ]
    assert texture_bindings == [("u_texture", compositor._main_tex)]
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
