"""Contextual transform options in the left tool settings panel."""

from termin.gui_native import EdgeInsets, Size, TextWrapMode


class NativeTransformControls:
    def __init__(self, document, transform, *, begin, on_state_changed=lambda: None):
        self.transform = transform
        self._begin = begin
        self._on_state_changed = on_state_changed
        self._syncing = False
        self._closed = False
        self._connections = []
        self.widget = document.create_vstack("TransformOptions")
        self.widget.stable_id = "diffusion-editor.transform-options"
        self.widget.set_layout_spacing(4)
        self.widget.set_layout_padding(EdgeInsets(6, 4, 6, 6))
        self.widget.add_preferred_child(document.create_label("Transform"))
        self.target = document.create_combo_box()
        self.target.widget.stable_id = "diffusion-editor.transform.target"
        self.target.add_item("Selected pixels")
        self.target.add_item("Entire layer")
        self._connections.append(self.target.connect_changed(self._target_changed))
        self.widget.add_preferred_child(self.target.widget)
        self.apply_button = document.create_button("Apply")
        self.apply_button.widget.stable_id = "diffusion-editor.transform.apply"
        self.cancel_button = document.create_button("Cancel")
        self.cancel_button.widget.stable_id = "diffusion-editor.transform.cancel"
        self._connections.append(self.apply_button.connect_clicked(transform.apply))
        self._connections.append(self.cancel_button.connect_clicked(transform.cancel))
        self.dimensions = {}
        for axis, label in (("width", "Width"), ("height", "Height")):
            fields = document.create_hstack(f"TransformDimension.{axis}")
            fields.set_layout_spacing(4)
            fields.add_fixed_child(document.create_label(label), 68)
            control = document.create_spin_box(1)
            control.widget.stable_id = f"diffusion-editor.transform.{axis}"
            control.set_range(1, 16384)
            control.step, control.decimals = 1, 0
            control.widget.min_size = Size(40, 24)
            self._connections.append(control.connect_changed(
                lambda value, axis=axis: self._dimension_changed(axis, value)))
            fields.add_flex_child(control.widget, 1)
            self.dimensions[axis] = control
            self.widget.add_preferred_child(fields)
        ratio = document.create_hstack("TransformAspect")
        ratio.set_layout_spacing(4)
        self.aspect = document.create_checkbox(True)
        self.aspect.widget.stable_id = "diffusion-editor.transform.keep-aspect"
        self._connections.append(self.aspect.connect_changed(self._aspect_changed))
        ratio.add_fixed_child(self.aspect.widget, 22)
        ratio.add_preferred_child(document.create_label("Keep ratio"))
        self.widget.add_preferred_child(ratio)
        actions = document.create_hstack("TransformActions")
        actions.set_layout_spacing(4)
        actions.add_flex_child(self.apply_button.widget, 1)
        actions.add_flex_child(self.cancel_button.widget, 1)
        self.widget.add_preferred_child(actions)
        self.caption = document.create_label("")
        self.caption.set_wrap_mode(TextWrapMode.Word)
        self.caption.stable_id = "diffusion-editor.transform.caption"
        self.widget.add_preferred_child(self.caption)
        transform.on_changed = self.sync
        self.sync()

    def sync(self):
        if self._closed:
            return
        self._on_state_changed()
        self._syncing = True
        try:
            session = self.transform.session
            self.widget.visible = session is not None or self.transform.error is not None
            self.target.widget.enabled = session is not None
            self.apply_button.widget.enabled = session is not None
            self.cancel_button.widget.enabled = session is not None or self.transform.error is not None
            self.aspect.widget.enabled = session is not None
            self.aspect.checked = self.transform.keep_aspect
            for control in self.dimensions.values():
                control.widget.enabled = session is not None
            if session:
                self.target.selected_index = 0 if session.target == "selection" else 1
                x0, y0, x1, y1 = session.rect
                self.dimensions["width"].value = x1-x0
                self.dimensions["height"].value = y1-y0
                self.caption.text = f"{session.layer.name} · X {x0}, Y {y0} · Enter / Esc"
                if self.transform.error:
                    self.caption.text = self.transform.error
            else:
                self.caption.text = self.transform.message
        finally:
            self._syncing = False

    def _target_changed(self, index, *_args):
        if not self._syncing and index in (0, 1):
            self._begin("selection" if index == 0 else "layer")

    def _dimension_changed(self, axis, value):
        if not self._syncing:
            self.transform.set_dimension(axis, value)

    def _aspect_changed(self, value):
        if not self._syncing:
            self.transform.keep_aspect = bool(value)

    def close(self):
        self._closed = True
        self.transform.on_changed = lambda: None
        self._connections.clear()
