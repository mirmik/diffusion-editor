from types import SimpleNamespace

import numpy as np
import pytest
from termin.gui_native import DynamicTextureOwnership

from diffusion_editor.canvas.native_editor_canvas import NativeEditorCanvas


class _Lease:
    def __init__(self, name, trace):
        self.name = name
        self.trace = trace
        self.ownership = DynamicTextureOwnership.EMPTY
        self.width = 0
        self.height = 0

    @property
    def empty(self):
        return self.ownership == DynamicTextureOwnership.EMPTY

    def clear(self):
        self.trace.append((self.name, "clear"))
        self.ownership = DynamicTextureOwnership.EMPTY
        self.width = 0
        self.height = 0

    def borrow(self, owner, texture):
        self.trace.append((self.name, "borrow", owner, texture))
        self.ownership = DynamicTextureOwnership.BORROWED

    def set_rgba8(self, data):
        self.trace.append((self.name, "set", data.copy()))
        self.ownership = DynamicTextureOwnership.OWNED
        self.height, self.width = data.shape[:2]

    def update_region_rgba8(self, x, y, data):
        self.trace.append((self.name, "update", x, y, data.copy()))

    def close(self):
        self.trace.append((self.name, "close"))
        self.ownership = DynamicTextureOwnership.RELEASED


def _native_canvas_shell():
    trace = []
    image_lease = _Lease("image", trace)
    overlay_lease = _Lease("overlay", trace)
    bridge = SimpleNamespace(
        using_gpu=True,
        display_tex="gpu-display",
        composite=np.zeros((8, 10, 4), dtype=np.uint8),
        display_size=lambda: (10, 8),
    )

    class Controller:
        composite_bridge = bridge
        overlay_bridge = SimpleNamespace(overlay=None)

        def dispose(self):
            trace.append(("controller", "dispose"))

    class Canvas:
        def set_paint_callback(self, callback):
            assert callable(callback)
            trace.append(("canvas", "detach-paint"))

    class Subscription:
        closed = False

        def unsubscribe(self):
            self.closed = True

    subscription = Subscription()
    stack = SimpleNamespace(width=10, height=8)
    native = NativeEditorCanvas.__new__(NativeEditorCanvas)
    native._closed = False
    native._request_repaint = lambda: trace.append(("view", "repaint"))
    native._graphics_owner = "graphics-owner"
    native._image_lease = image_lease
    native._overlay_lease = overlay_lease
    native._layer_stack = stack
    native._stack_subscription = subscription
    native.controller = Controller()
    native.canvas = Canvas()
    return native, trace, subscription


def test_native_canvas_switches_borrowed_and_owned_textures_explicitly():
    native, trace, _previous = _native_canvas_shell()

    native._sync_gpu_image()
    assert native.image_lease.ownership == DynamicTextureOwnership.BORROWED

    cpu_image = np.full((8, 10, 4), 17, dtype=np.uint8)
    native._replace_owned(native.image_lease, cpu_image)
    assert native.image_lease.ownership == DynamicTextureOwnership.OWNED
    native._update_image_region(
        3,
        2,
        np.full((2, 4, 4), 23, dtype=np.uint8),
    )
    native._sync_gpu_image()

    operations = [item[1] for item in trace if item[0] == "image"]
    assert operations == ["borrow", "clear", "set", "update", "clear", "borrow"]


def test_native_canvas_reuses_same_size_owned_texture_for_full_update():
    native, trace, _previous = _native_canvas_shell()
    first = np.full((8, 10, 4), 17, dtype=np.uint8)
    replacement = np.full((8, 10, 4), 23, dtype=np.uint8)

    native._replace_owned(native.image_lease, first)
    native._replace_owned(native.image_lease, replacement)

    operations = [item[1] for item in trace if item[0] == "image"]
    assert operations == ["set", "update"]
    assert trace[-1][2:4] == (0, 0)
    assert np.array_equal(trace[-1][4], replacement)


def test_native_canvas_reallocates_owned_texture_after_resize():
    native, trace, _previous = _native_canvas_shell()
    native._replace_owned(
        native.image_lease,
        np.full((8, 10, 4), 17, dtype=np.uint8),
    )

    native._replace_owned(
        native.image_lease,
        np.full((6, 7, 4), 23, dtype=np.uint8),
    )

    operations = [item[1] for item in trace if item[0] == "image"]
    assert operations == ["set", "clear", "set"]
    assert (native.image_lease.width, native.image_lease.height) == (7, 6)


def test_native_canvas_releases_leases_before_gpu_compositor():
    native, trace, subscription = _native_canvas_shell()
    native._sync_gpu_image()

    native.close()
    native.close()

    assert subscription.closed
    assert native.image_lease.ownership == DynamicTextureOwnership.RELEASED
    assert native.overlay_lease.ownership == DynamicTextureOwnership.RELEASED
    lifecycle = [
        item[:2] for item in trace
        if item[0] in {"canvas", "overlay", "image", "controller"}
    ]
    assert lifecycle[-4:] == [
        ("canvas", "detach-paint"),
        ("overlay", "close"),
        ("image", "close"),
        ("controller", "dispose"),
    ]


def test_cancel_releases_document_capture_when_controller_raises():
    native, trace, _previous = _native_canvas_shell()

    class Relay:
        handle = "relay"

    class Document:
        pointer_capture = "relay"

        def release_pointer_capture(self, handle):
            assert handle == "relay"
            trace.append(("document", "release-capture"))
            self.pointer_capture = None

    native._capture_relay = Relay()
    native._document = Document()
    native.controller.pointer_cancel = lambda: (_ for _ in ()).throw(
        RuntimeError("cancel failed"))

    with pytest.raises(RuntimeError, match="cancel failed"):
        native.cancel_pointer_interaction()

    assert native._document.pointer_capture is None
    assert ("document", "release-capture") in trace


def test_close_finishes_all_cleanup_when_cancellation_raises():
    native, trace, subscription = _native_canvas_shell()

    class Relay:
        handle = "relay"

        def set_pointer_handler(self, handler):
            assert handler is None
            trace.append(("relay", "detach-handler"))

    class Document:
        pointer_capture = "relay"

        def release_pointer_capture(self, handle):
            assert handle == "relay"
            trace.append(("document", "release-capture"))
            self.pointer_capture = None

    native._capture_relay = Relay()
    native._document = Document()
    native.controller.pointer_cancel = lambda: (_ for _ in ()).throw(
        RuntimeError("cancel failed"))

    with pytest.raises(RuntimeError, match="cancel failed"):
        native.close()

    assert subscription.closed
    assert native._document.pointer_capture is None
    assert native.overlay_lease.ownership == DynamicTextureOwnership.RELEASED
    assert native.image_lease.ownership == DynamicTextureOwnership.RELEASED
    assert ("relay", "detach-handler") in trace
    assert ("controller", "dispose") in trace
