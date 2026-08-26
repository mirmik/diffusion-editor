"""termin-gui-native projection of the multiview studio project."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol

import numpy as np
from PIL import Image
from termin.gui_native import (
    EdgeInsets,
    ImageFit,
    LayoutPolicy,
    Size,
    TcDocument,
)
from tgfx import TextureEncoding

from .model import AZIMUTHS, ELEVATIONS, MultiviewProject, ViewKey


class StudioActions(Protocol):
    def new_project(self) -> None: ...
    def open_project(self) -> None: ...
    def save_project(self) -> None: ...
    def save_project_as(self) -> None: ...
    def pick_source(self, source: str) -> None: ...
    def pick_slot(self, key: ViewKey) -> None: ...
    def clear_slot(self, key: ViewKey) -> None: ...
    def set_slot_included(self, key: ViewKey, included: bool) -> None: ...
    def set_qwen_seed(self, seed: int) -> None: ...
    def set_trellis_setting(self, field: str, value: int) -> None: ...
    def generate_view(self, key: ViewKey) -> None: ...
    def generate_four(self) -> None: ...
    def generate_missing(self) -> None: ...
    def generate_all(self) -> None: ...
    def build_shape(self) -> None: ...
    def open_shape(self) -> None: ...
    def cancel_job(self) -> None: ...


class NativeMultiviewStudioView:
    def __init__(
        self,
        document: TcDocument,
        actions: StudioActions,
        *,
        request_repaint: Callable[[], None],
        texture_lease_factory: Callable[[], object],
    ) -> None:
        self._document = document
        self._actions = actions
        self._request_repaint = request_repaint
        self._texture_lease_factory = texture_lease_factory
        self._connections: list[object] = []
        self._leases: dict[str, object] = {}
        self._preview_signatures: dict[str, tuple[str, int, int]] = {}
        self._syncing = False
        self._closed = False
        self._busy = False
        self.slot_widgets: dict[ViewKey, dict[str, object]] = {}
        self.source_widgets: dict[str, dict[str, object]] = {}
        self.setting_controls: dict[str, object] = {}

        self.root = document.create_vstack("MultiviewStudioRoot")
        self.root.stable_id = "multiview-studio.root"
        self.root.set_layout_spacing(0.0)

        self.toolbar = document.create_hstack("MultiviewStudioToolbar")
        self.toolbar.stable_id = "multiview-studio.toolbar"
        self.toolbar.set_layout_spacing(5.0)
        self._toolbar_button("New", actions.new_project)
        self._toolbar_button("Open...", actions.open_project)
        self._toolbar_button("Save", actions.save_project)
        self._toolbar_button("Save As...", actions.save_project_as)
        self.toolbar.add_fixed_child(document.create_spacer(Size(18, 1)), 18)
        self.generate_four_button = self._toolbar_button(
            "Generate four", actions.generate_four
        )
        self.generate_missing_button = self._toolbar_button(
            "Generate missing", actions.generate_missing
        )
        self.generate_all_button = self._toolbar_button(
            "Generate all", actions.generate_all
        )
        self.build_shape_button = self._toolbar_button(
            "Build TRELLIS.2 shape", actions.build_shape
        )
        self.cancel_button = self._toolbar_button("Cancel", actions.cancel_job)
        self.cancel_button.widget.enabled = False

        self.left_content = document.create_vstack("MultiviewStudioLeft")
        self.left_content.stable_id = "multiview-studio.left-content"
        self.left_content.set_layout_spacing(6.0)
        self.left_content.set_layout_padding(EdgeInsets(6, 6, 6, 6))
        self._build_source("front", "Front")
        self._build_source("back", "Back")
        self._build_settings()
        self._build_shape_output()
        self.left_scroll = document.create_scroll_area()
        self.left_scroll.widget.stable_id = "multiview-studio.left-scroll"
        self.left_scroll.set_scroll_axes(False, True)
        self.left_scroll.set_content(self.left_content)

        self.grid = document.create_grid_layout("MultiviewStudioViewGrid")
        self.grid.stable_id = "multiview-studio.view-grid"
        self.grid.set_padding(EdgeInsets(6, 6, 6, 6))
        self.grid.set_spacing(6.0, 6.0)
        for _azimuth in AZIMUTHS:
            column = self.grid.add_column(LayoutPolicy.Stretch, 1.0)
            self.grid.set_column_extent_limits(column, 142.0, 240.0)
        for _elevation in ELEVATIONS:
            row = self.grid.add_row(LayoutPolicy.Fixed, 224.0)
            self.grid.set_row_extent_limits(row, 224.0, 224.0)
        for row, elevation in enumerate(ELEVATIONS):
            for column, azimuth in enumerate(AZIMUTHS):
                key = ViewKey(elevation, azimuth)
                group = self._build_slot(key)
                self.grid.add_child(group.widget, row, column)

        self.grid_scroll = document.create_scroll_area()
        self.grid_scroll.widget.stable_id = "multiview-studio.grid-scroll"
        self.grid_scroll.set_scroll_axes(True, True)
        self.grid_scroll.set_content(self.grid)

        self.main_splitter = document.create_splitter(
            True, "MultiviewStudioMainSplitter"
        )
        self.main_splitter.widget.stable_id = "multiview-studio.main-splitter"
        self.main_splitter.set_first(self.left_scroll.widget)
        self.main_splitter.set_second(self.grid_scroll.widget)
        self.main_splitter.set_split_fraction(0.19)
        self.main_splitter.set_min_extents(270.0, 900.0)

        self.status = document.create_status_bar("Ready")
        self.status.widget.stable_id = "multiview-studio.status"
        self.root.add_fixed_child(self.toolbar, 38.0)
        self.root.add_flex_child(self.main_splitter.widget, 1.0)
        self.root.add_fixed_child(self.status.widget, 24.0)
        if not document.add_root(self.root.handle):
            raise RuntimeError("failed to add multiview studio root")

    def apply_project(
        self,
        project: MultiviewProject,
        project_path: Path | None,
        dirty: bool,
    ) -> None:
        if self._closed:
            return
        self._syncing = True
        try:
            for source, path in (
                ("front", project.front_path),
                ("back", project.back_path),
            ):
                widgets = self.source_widgets[source]
                widgets["path"].text = self._path_label(path)
                self._set_preview(f"source:{source}", widgets["image"], path)

            for slot in project.slots:
                widgets = self.slot_widgets[slot.key]
                widgets["path"].text = self._path_label(slot.image_path)
                widgets["include"].checked = slot.include_in_trellis
                widgets["include"].widget.enabled = not self._busy
                widgets["choose"].widget.enabled = not self._busy
                widgets["clear"].widget.enabled = slot.populated and not self._busy
                widgets["generate"].widget.enabled = (
                    bool(project.front_path)
                    and slot.key not in (ViewKey("eye", 0), ViewKey("eye", 180))
                    and not self._busy
                )
                self._set_preview(
                    f"slot:{slot.key.stable_id}",
                    widgets["image"],
                    slot.image_path,
                )

            self.setting_controls["qwen_seed"].text = str(project.qwen_seed)
            self.setting_controls["seed"].text = str(project.trellis.seed)
            for field in (
                "total_steps",
                "warmup_steps",
                "resolution",
                "decimation_target",
            ):
                self.setting_controls[field].value = float(
                    getattr(project.trellis, field)
                )

            has_front = bool(project.front_path)
            has_both = has_front and bool(project.back_path)
            saved = project_path is not None
            self.generate_four_button.widget.enabled = has_both and saved and not self._busy
            self.generate_missing_button.widget.enabled = has_both and saved and not self._busy
            self.generate_all_button.widget.enabled = has_both and saved and not self._busy
            errors = project.validate_shape_request()
            self.build_shape_button.widget.enabled = not errors and saved and not self._busy
            self.cancel_button.widget.enabled = self._busy
            self.shape_path_label.text = self._path_label(project.shape_path)
            self.open_shape_button.widget.enabled = (
                bool(project.shape_path)
                and Path(project.shape_path).is_file()
                and not self._busy
            )
            populated = sum(slot.populated for slot in project.slots)
            included = len(project.included_slots())
            name = project_path.name if project_path else "Untitled"
            marker = " *" if dirty else ""
            detail = errors[0] if errors else "Ready to build shape"
            self.status.text = (
                f"{name}{marker} · {populated}/24 populated · "
                f"{included} included · {detail}"
            )
        finally:
            self._syncing = False
        self._request_repaint()

    def set_status(self, text: str) -> None:
        self.status.text = text
        self._request_repaint()

    def set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        for widgets in self.source_widgets.values():
            widgets["choose"].widget.enabled = not self._busy
        for control in self.setting_controls.values():
            control.widget.enabled = not self._busy
        self.cancel_button.widget.enabled = self._busy
        if self._busy:
            self.open_shape_button.widget.enabled = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for lease in self._leases.values():
            lease.close()
        self._leases.clear()
        self._preview_signatures.clear()
        self._connections.clear()

    def _build_source(self, source: str, title: str) -> None:
        group = self._document.create_group_box(title)
        group.widget.stable_id = f"multiview-studio.source.{source}"
        content = self._document.create_vstack(
            f"MultiviewStudio{title}Source"
        )
        content.set_layout_spacing(4.0)
        image = self._document.create_image_widget()
        image.widget.stable_id = f"multiview-studio.source.{source}.preview"
        image.fit = ImageFit.Contain
        content.add_fixed_child(image.widget, 156.0)
        path = self._document.create_label("Empty")
        path.stable_id = f"multiview-studio.source.{source}.path"
        content.add_fixed_child(path, 22.0)
        button = self._document.create_button("Choose image...")
        button.widget.stable_id = f"multiview-studio.source.{source}.choose"
        self._connections.append(
            button.connect_clicked(
                lambda selected=source: self._actions.pick_source(selected)
            )
        )
        content.add_preferred_child(button.widget)
        group.set_content(content)
        self.left_content.add_preferred_child(group.widget)
        self.source_widgets[source] = {
            "image": image,
            "path": path,
            "choose": button,
        }

    def _build_settings(self) -> None:
        group = self._document.create_group_box("Generation settings")
        group.widget.stable_id = "multiview-studio.settings"
        content = self._document.create_vstack("MultiviewStudioSettings")
        content.set_layout_spacing(4.0)
        self.setting_controls["qwen_seed"] = self._seed_input(
            content,
            "Qwen seed",
            "qwen-seed",
            20_260_822,
            lambda value: self._actions.set_qwen_seed(value),
        )
        self.setting_controls["seed"] = self._seed_input(
            content,
            "TRELLIS.2 seed",
            "trellis-seed",
            42,
            lambda value: self._actions.set_trellis_setting("seed", value),
        )
        self.setting_controls["total_steps"] = self._spin(
            content,
            "Steps per stage",
            "total-steps",
            39,
            1,
            200,
            1,
            lambda value: self._actions.set_trellis_setting(
                "total_steps", value
            ),
        )
        self.setting_controls["warmup_steps"] = self._spin(
            content,
            "Front warmup / stage",
            "warmup-steps",
            15,
            0,
            200,
            1,
            lambda value: self._actions.set_trellis_setting(
                "warmup_steps", value
            ),
        )
        self.setting_controls["resolution"] = self._spin(
            content,
            "Shape resolution",
            "resolution",
            1024,
            1024,
            2048,
            128,
            lambda value: self._actions.set_trellis_setting(
                "resolution", value
            ),
        )
        self.setting_controls["decimation_target"] = self._spin(
            content,
            "Target faces",
            "decimation-target",
            250_000,
            10_000,
            4_000_000,
            10_000,
            lambda value: self._actions.set_trellis_setting(
                "decimation_target", value
            ),
        )
        group.set_content(content)
        self.left_content.add_preferred_child(group.widget)

    def _build_shape_output(self) -> None:
        group = self._document.create_group_box("Shape output")
        group.widget.stable_id = "multiview-studio.shape-output"
        content = self._document.create_vstack("MultiviewStudioShapeOutput")
        content.set_layout_spacing(4.0)
        self.shape_path_label = self._document.create_label("Empty")
        self.shape_path_label.stable_id = "multiview-studio.shape-output.path"
        self.open_shape_button = self._document.create_button("Open in Termin")
        self.open_shape_button.widget.stable_id = (
            "multiview-studio.shape-output.open"
        )
        self._connections.append(
            self.open_shape_button.connect_clicked(self._actions.open_shape)
        )
        self.open_shape_button.widget.enabled = False
        content.add_preferred_child(self.shape_path_label)
        content.add_preferred_child(self.open_shape_button.widget)
        group.set_content(content)
        self.left_content.add_preferred_child(group.widget)

    def _seed_input(
        self,
        parent,
        label: str,
        suffix: str,
        value: int,
        callback: Callable[[int], None],
    ):
        row = self._document.create_hstack("MultiviewStudioSeedRow")
        row.set_layout_spacing(4.0)
        caption = self._document.create_label(label)
        field = self._document.create_text_input(str(value))
        field.widget.stable_id = f"multiview-studio.setting.{suffix}"

        def changed(text: str) -> None:
            if self._syncing or not text.strip():
                return
            try:
                seed = int(text)
            except ValueError:
                return
            callback(seed)

        self._connections.append(field.connect_changed(changed))
        row.add_flex_child(caption, 1.0)
        row.add_fixed_child(field.widget, 112.0)
        parent.add_preferred_child(row)
        return field

    def _build_slot(self, key: ViewKey):
        group = self._document.create_group_box(
            f"{key.azimuth:03d}° · {key.elevation}"
        )
        group.widget.stable_id = f"multiview-studio.slot.{key.stable_id}"
        content = self._document.create_vstack(
            f"MultiviewStudioSlot{key.stable_id}"
        )
        content.set_layout_spacing(3.0)
        image = self._document.create_image_widget()
        image.widget.stable_id = (
            f"multiview-studio.slot.{key.stable_id}.preview"
        )
        image.fit = ImageFit.Contain
        content.add_fixed_child(image.widget, 116.0)
        path = self._document.create_label("Empty")
        path.stable_id = f"multiview-studio.slot.{key.stable_id}.path"
        content.add_fixed_child(path, 20.0)

        actions = self._document.create_hstack("MultiviewStudioSlotActions")
        actions.set_layout_spacing(3.0)
        choose = self._document.create_button("Load")
        choose.widget.stable_id = (
            f"multiview-studio.slot.{key.stable_id}.load"
        )
        generate = self._document.create_button("Generate")
        generate.widget.stable_id = (
            f"multiview-studio.slot.{key.stable_id}.generate"
        )
        clear = self._document.create_button("×")
        clear.widget.stable_id = (
            f"multiview-studio.slot.{key.stable_id}.clear"
        )
        self._connections.extend(
            (
                choose.connect_clicked(
                    lambda selected=key: self._actions.pick_slot(selected)
                ),
                generate.connect_clicked(
                    lambda selected=key: self._actions.generate_view(selected)
                ),
                clear.connect_clicked(
                    lambda selected=key: self._actions.clear_slot(selected)
                ),
            )
        )
        actions.add_flex_child(choose.widget, 1.0)
        actions.add_flex_child(generate.widget, 1.0)
        actions.add_fixed_child(clear.widget, 26.0)
        content.add_preferred_child(actions)

        include_row = self._document.create_hstack(
            "MultiviewStudioIncludeRow"
        )
        include_row.set_layout_spacing(3.0)
        include = self._document.create_checkbox(True)
        include.widget.stable_id = (
            f"multiview-studio.slot.{key.stable_id}.include"
        )
        include_label = self._document.create_label("Use in TRELLIS.2")
        self._connections.append(
            include.connect_changed(
                lambda checked, selected=key: None
                if self._syncing
                else self._actions.set_slot_included(selected, checked)
            )
        )
        include_row.add_preferred_child(include.widget)
        include_row.add_flex_child(include_label, 1.0)
        content.add_preferred_child(include_row)
        group.set_content(content)
        self.slot_widgets[key] = {
            "image": image,
            "path": path,
            "include": include,
            "choose": choose,
            "generate": generate,
            "clear": clear,
        }
        return group

    def _spin(
        self,
        parent,
        label: str,
        suffix: str,
        value: int,
        minimum: int,
        maximum: int,
        step: int,
        callback: Callable[[int], None],
    ):
        row = self._document.create_hstack("MultiviewStudioSettingRow")
        row.set_layout_spacing(4.0)
        caption = self._document.create_label(label)
        field = self._document.create_spin_box(float(value))
        field.widget.stable_id = f"multiview-studio.setting.{suffix}"
        field.set_range(float(minimum), float(maximum))
        field.step = float(step)
        field.decimals = 0
        self._connections.append(
            field.connect_changed(
                lambda changed: None
                if self._syncing
                else callback(int(round(changed)))
            )
        )
        row.add_flex_child(caption, 1.0)
        row.add_fixed_child(field.widget, 112.0)
        parent.add_preferred_child(row)
        return field

    def _toolbar_button(self, label: str, callback: Callable[[], None]):
        button = self._document.create_button(label)
        button.widget.stable_id = (
            "multiview-studio.action."
            + label.lower().replace(" ", "-").replace(".", "")
        )
        self._connections.append(button.connect_clicked(callback))
        self.toolbar.add_preferred_child(button.widget)
        return button

    def _set_preview(self, preview_id: str, widget, image_path: str) -> None:
        source_path = Path(image_path) if image_path else None
        if source_path is not None and source_path.is_file():
            stat = source_path.stat()
            signature = (str(source_path), stat.st_mtime_ns, stat.st_size)
        else:
            signature = (image_path, 0, 0)
        if self._preview_signatures.get(preview_id) == signature:
            return
        self._preview_signatures[preview_id] = signature
        lease = self._leases.get(preview_id)
        if source_path is None or not source_path.is_file():
            widget.clear_texture()
            if lease is not None and not lease.empty:
                lease.clear()
            return
        with Image.open(source_path) as source:
            preview = source.convert("RGBA")
            preview.thumbnail((256, 192))
            pixels = np.ascontiguousarray(preview, dtype=np.uint8)
        if lease is None:
            lease = self._texture_lease_factory()
            self._leases[preview_id] = lease
        lease.set_rgba8(pixels, TextureEncoding.SRGB)
        widget.set_texture(lease.texture, Size(preview.width, preview.height))

    @staticmethod
    def _path_label(path: str) -> str:
        if not path:
            return "Empty"
        resolved = Path(path)
        return resolved.name if resolved.is_file() else f"Missing: {resolved.name}"
