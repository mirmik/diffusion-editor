from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from termin.gui_native import tc_ui_document_create, tc_ui_document_destroy

from diffusion_editor.multiview_studio.model import MultiviewProject, ViewKey
from diffusion_editor.multiview_studio.native_view import (
    NativeMultiviewStudioView,
)


class _Actions:
    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: None


class _FakeLease:
    def __init__(self):
        self.texture = object()
        self.empty = True
        self.upload = None

    def set_rgba8(self, pixels, encoding):
        assert pixels.dtype == np.uint8
        assert pixels.flags.c_contiguous
        assert pixels.flags.writeable
        self.upload = (pixels.copy(), encoding)
        self.empty = False

    def clear(self):
        self.empty = True

    def close(self):
        self.empty = True


class _FakeImageWidget:
    def __init__(self):
        self.upload = None

    def set_texture(self, texture, size):
        self.upload = (texture, size)

    def clear_texture(self):
        self.upload = None


def test_native_multiview_view_builds_3_by_8_slots_and_applies_state():
    document = tc_ui_document_create()
    view = NativeMultiviewStudioView(
        document,
        _Actions(),
        request_repaint=lambda: None,
        texture_lease_factory=lambda: None,
    )
    try:
        assert view.root.stable_id == "multiview-studio.root"
        assert len(view.slot_widgets) == 24
        project = MultiviewProject().with_slot(
            ViewKey("low", 45), include_in_trellis=False
        )

        view.apply_project(project, None, True)

        assert not view.slot_widgets[ViewKey("low", 45)]["include"].checked
        assert not view.build_shape_button.widget.enabled
        assert "0/24 populated" in view.status.text
    finally:
        view.close()
        tc_ui_document_destroy(document)


def test_preview_upload_copies_pil_pixels_to_writable_c_array(tmp_path: Path):
    image_path = tmp_path / "preview.png"
    Image.new("RGB", (31, 47), "blue").save(image_path)
    document = tc_ui_document_create()
    leases = []

    def lease_factory():
        lease = _FakeLease()
        leases.append(lease)
        return lease

    view = NativeMultiviewStudioView(
        document,
        _Actions(),
        request_repaint=lambda: None,
        texture_lease_factory=lease_factory,
    )
    widget = _FakeImageWidget()
    try:
        view._set_preview("test", widget, str(image_path))

        assert len(leases) == 1
        assert leases[0].upload is not None
        assert widget.upload is not None
    finally:
        view.close()
        tc_ui_document_destroy(document)
