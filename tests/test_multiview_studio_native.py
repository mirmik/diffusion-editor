from __future__ import annotations

from termin.gui_native import tc_ui_document_create, tc_ui_document_destroy

from diffusion_editor.multiview_studio.model import MultiviewProject, ViewKey
from diffusion_editor.multiview_studio.native_view import (
    NativeMultiviewStudioView,
)


class _Actions:
    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: None


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
