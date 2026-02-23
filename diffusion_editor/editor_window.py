import os
import random
import numpy as np
from PIL import Image
from PyQt6.QtWidgets import (
    QMainWindow, QFileDialog, QStatusBar, QToolBar, QMessageBox,
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QDialogButtonBox,
)
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtCore import Qt, QTimer, QSettings

from .layer import LayerStack, DiffusionLayer, LamaLayer, InstructLayer
from .canvas import Canvas
from .layer_panel import LayerPanel
from .brush_panel import BrushPanel
from .diffusion_engine import DiffusionEngine
from .diffusion_panel import DiffusionPanel
from .lama_engine import LamaEngine
from .lama_panel import LamaPanel
from .instruct_engine import InstructEngine
from .instruct_panel import InstructPanel
from .segmentation import SegmentationEngine
from .diffusion_brush import extract_patch, extract_mask_patch, paste_result


class EditorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Diffusion Editor")
        self.resize(1280, 800)

        self._layer_stack = LayerStack(self)
        self._canvas = Canvas(self._layer_stack, self)
        self.setCentralWidget(self._canvas)

        self._layer_panel = LayerPanel(self._layer_stack, self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._layer_panel)

        self._brush_panel = BrushPanel(self._canvas.brush, self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._brush_panel)
        self._brush_panel.setVisible(False)

        self._engine = DiffusionEngine()
        self._seg_engine = SegmentationEngine()
        self._lama_engine = LamaEngine()
        self._instruct_engine = InstructEngine()

        self._diffusion_panel = DiffusionPanel(self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._diffusion_panel)
        self._diffusion_panel.setVisible(False)

        self._lama_panel = LamaPanel(self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._lama_panel)
        self._lama_panel.setVisible(False)

        self._instruct_panel = InstructPanel(self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._instruct_panel)
        self._instruct_panel.setVisible(False)

        self._setup_menu()
        self._setup_toolbar()
        self._setup_statusbar()

        self._canvas.mouse_moved.connect(self._on_mouse_moved)
        self._canvas.color_picked.connect(self._on_color_picked)
        self._brush_panel.eraser_toggled.connect(self._canvas.set_brush_eraser)
        self._diffusion_panel.load_model_requested.connect(self._on_load_model)
        self._diffusion_panel.regenerate_requested.connect(self._on_regenerate)
        self._diffusion_panel.new_seed_requested.connect(self._on_new_seed)
        self._diffusion_panel.clear_mask_requested.connect(self._on_clear_mask)
        self._diffusion_panel.mask_brush_changed.connect(self._on_mask_brush_changed)
        self._diffusion_panel._mask_eraser_btn.toggled.connect(self._canvas.set_mask_eraser)
        self._diffusion_panel._show_mask_btn.toggled.connect(self._canvas.set_show_mask)
        self._diffusion_panel.load_ip_adapter_requested.connect(self._on_load_ip_adapter)
        self._diffusion_panel.draw_rect_toggled.connect(self._canvas.set_ref_rect_mode)
        self._diffusion_panel.show_rect_toggled.connect(self._canvas.set_show_ref_rect)
        self._diffusion_panel.clear_rect_requested.connect(self._on_clear_ref_rect)
        self._diffusion_panel.select_background_requested.connect(self._on_select_background)
        self._canvas.ref_rect_drawn.connect(self._on_ref_rect_drawn)
        self._diffusion_panel.draw_patch_toggled.connect(self._canvas.set_patch_rect_mode)
        self._diffusion_panel.clear_patch_requested.connect(self._on_clear_patch_rect)
        self._canvas.patch_rect_drawn.connect(self._on_patch_rect_drawn)
        self._layer_panel.create_diffusion_requested.connect(self._on_create_diffusion)
        self._layer_panel.create_lama_requested.connect(self._on_create_lama)
        self._layer_panel.create_instruct_requested.connect(self._on_create_instruct)

        # LaMa panel signals
        self._lama_panel.remove_requested.connect(self._on_lama_remove)
        self._lama_panel.clear_mask_requested.connect(self._on_lama_clear_mask)
        self._lama_panel.mask_brush_changed.connect(self._canvas.set_mask_brush)
        self._lama_panel._mask_eraser_btn.toggled.connect(self._canvas.set_mask_eraser)
        self._lama_panel._show_mask_btn.toggled.connect(self._canvas.set_show_mask)
        self._lama_panel.select_background_requested.connect(self._on_lama_select_background)

        # InstructPix2Pix panel signals
        self._instruct_panel.load_model_requested.connect(self._on_instruct_load_model)
        self._instruct_panel.apply_requested.connect(self._on_instruct_apply)
        self._instruct_panel.new_seed_requested.connect(self._on_instruct_new_seed)
        self._instruct_panel.clear_mask_requested.connect(self._on_instruct_clear_mask)
        self._instruct_panel.mask_brush_changed.connect(self._canvas.set_mask_brush)
        self._instruct_panel._mask_eraser_btn.toggled.connect(self._canvas.set_mask_eraser)
        self._instruct_panel._show_mask_btn.toggled.connect(self._canvas.set_show_mask)
        self._instruct_panel.draw_patch_toggled.connect(self._canvas.set_patch_rect_mode)
        self._instruct_panel.clear_patch_requested.connect(self._on_instruct_clear_patch_rect)

        self._layer_stack.changed.connect(self._on_layer_changed)
        self._settings = QSettings("DiffusionEditor", "DiffusionEditor")
        self._project_path = None
        self._last_dir = self._settings.value("last_dir", "")
        self._pending_request = None  # DiffusionLayer for regenerate
        self._pending_lama_layer = None  # LamaLayer for remove
        self._pending_instruct_layer = None  # InstructLayer for apply

        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_engine)
        self._poll_timer.start(50)

    def _setup_menu(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")
        new_action = QAction("&New...", self)
        new_action.setShortcut(QKeySequence("Ctrl+N"))
        new_action.triggered.connect(self.new_project)
        file_menu.addAction(new_action)

        new_from_image_action = QAction("New From &Image...", self)
        new_from_image_action.triggered.connect(self.new_project_from_image)
        file_menu.addAction(new_from_image_action)

        file_menu.addSeparator()

        open_action = QAction("&Open...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

        save_action = QAction("&Save", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_file)
        file_menu.addAction(save_action)

        save_as_action = QAction("Save &As...", self)
        save_as_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        save_as_action.triggered.connect(self.save_file_as)
        file_menu.addAction(save_as_action)

        file_menu.addSeparator()

        import_action = QAction("&Import Image...", self)
        import_action.setShortcut(QKeySequence("Ctrl+I"))
        import_action.triggered.connect(self.import_image)
        file_menu.addAction(import_action)

        export_action = QAction("&Export Image...", self)
        export_action.setShortcut(QKeySequence("Ctrl+E"))
        export_action.triggered.connect(self.export_image)
        file_menu.addAction(export_action)

        file_menu.addSeparator()
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence("Ctrl+Q"))
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        layer_menu = menubar.addMenu("&Layer")
        new_layer_action = QAction("&New Layer", self)
        new_layer_action.setShortcut(QKeySequence("Ctrl+Shift+N"))
        new_layer_action.triggered.connect(self._new_layer)
        layer_menu.addAction(new_layer_action)

        remove_layer_action = QAction("&Remove Layer", self)
        remove_layer_action.triggered.connect(self._remove_layer)
        layer_menu.addAction(remove_layer_action)

        layer_menu.addSeparator()
        flatten_action = QAction("&Flatten Image", self)
        flatten_action.triggered.connect(self._layer_stack.flatten)
        layer_menu.addAction(flatten_action)

    def _setup_toolbar(self):
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        open_action = QAction("Open", self)
        open_action.triggered.connect(self.open_file)
        toolbar.addAction(open_action)

        save_action = QAction("Save", self)
        save_action.triggered.connect(self.save_file)
        toolbar.addAction(save_action)

        toolbar.addSeparator()

        fit_action = QAction("Fit", self)
        fit_action.triggered.connect(self._fit)
        toolbar.addAction(fit_action)

    def _fit(self):
        self._canvas.fit_in_view()
        self._canvas.update()

    def _setup_statusbar(self):
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)
        self._statusbar.showMessage("Ready")

    def _on_mouse_moved(self, x, y):
        size = self._canvas.image_size()
        if size is None:
            return
        w, h = size
        layer = self._layer_stack.active_layer
        layer_name = layer.name if layer else "-"
        brush_size = self._canvas.brush.size
        if 0 <= x < w and 0 <= y < h:
            self._statusbar.showMessage(f"{w}x{h}  |  ({x}, {y})  |  {layer_name}  |  Brush: {brush_size}px")
        else:
            self._statusbar.showMessage(f"{w}x{h}  |  {layer_name}  |  Brush: {brush_size}px")

    def new_project(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("New Project")
        layout = QVBoxLayout(dlg)

        row = QHBoxLayout()
        row.addWidget(QLabel("Width:"))
        w_spin = QSpinBox()
        w_spin.setRange(64, 8192)
        w_spin.setValue(1024)
        row.addWidget(w_spin)
        row.addWidget(QLabel("Height:"))
        h_spin = QSpinBox()
        h_spin.setRange(64, 8192)
        h_spin.setValue(1024)
        row.addWidget(h_spin)
        layout.addLayout(row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        w, h = w_spin.value(), h_spin.value()
        white = np.full((h, w, 4), 255, dtype=np.uint8)
        self._layer_stack.init_from_image(white)
        self._canvas.fit_in_view()
        self._project_path = None
        self.setWindowTitle("Diffusion Editor")

    def new_project_from_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "New Project From Image", self._last_dir,
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp);;All Files (*)")
        if not path:
            return
        self._last_dir = os.path.dirname(path)
        self._settings.setValue("last_dir", self._last_dir)
        img = Image.open(path).convert("RGBA")
        arr = np.array(img, dtype=np.uint8)
        self._layer_stack.init_from_image(arr)
        self._canvas.fit_in_view()
        self._project_path = None
        self.setWindowTitle("Diffusion Editor")

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", self._last_dir,
            "Diffusion Editor Project (*.deproj);;All Files (*)",
        )
        if not path:
            return
        self._last_dir = os.path.dirname(path)
        self._settings.setValue("last_dir", self._last_dir)
        self.open_file_path(path)

    def open_file_path(self, path: str):
        try:
            self._layer_stack.load_project(path)
            self._canvas.fit_in_view()
            self._project_path = path
            self.setWindowTitle(f"Diffusion Editor — {os.path.basename(path)}")
            self._statusbar.showMessage(f"Opened project: {path}", 3000)
        except Exception as e:
            QMessageBox.critical(self, "Open Project Error",
                                 f"Failed to open project:\n{e}")

    def import_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Image", self._last_dir,
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp);;All Files (*)",
        )
        if not path:
            return
        self._last_dir = os.path.dirname(path)
        self._settings.setValue("last_dir", self._last_dir)
        self.import_image_path(path)

    def import_image_path(self, path: str):
        img = Image.open(path).convert("RGBA")
        arr = np.array(img, dtype=np.uint8)
        self._layer_stack.init_from_image(arr)
        self._canvas.fit_in_view()
        self._project_path = None
        self.setWindowTitle("Diffusion Editor")

    def _new_layer(self):
        self._layer_stack.add_layer(self._layer_stack.next_name("Layer"))

    def _remove_layer(self):
        layer = self._layer_stack.active_layer
        if layer is not None:
            self._layer_stack.remove_layer(layer)

    def save_file(self):
        if self._project_path:
            try:
                self._layer_stack.save_project(self._project_path)
                self._statusbar.showMessage(f"Saved: {self._project_path}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Save Error",
                                     f"Failed to save project:\n{e}")
        else:
            self.save_file_as()

    def save_file_as(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project", self._last_dir,
            "Diffusion Editor Project (*.deproj);;All Files (*)",
        )
        if not path:
            return
        if not path.lower().endswith(".deproj"):
            path += ".deproj"
        self._last_dir = os.path.dirname(path)
        self._settings.setValue("last_dir", self._last_dir)
        try:
            self._layer_stack.save_project(path)
            self._project_path = path
            self.setWindowTitle(f"Diffusion Editor — {os.path.basename(path)}")
            self._statusbar.showMessage(f"Saved project: {path}", 3000)
        except Exception as e:
            QMessageBox.critical(self, "Save Project Error",
                                 f"Failed to save project:\n{e}")

    def export_image(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Image", self._last_dir,
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;All Files (*)",
        )
        if not path:
            return
        self._last_dir = os.path.dirname(path)
        self._settings.setValue("last_dir", self._last_dir)
        self._save_to(path)

    def _save_to(self, path):
        arr = self._canvas.get_composite()
        if arr is None:
            return
        img = Image.fromarray(arr, "RGBA")
        if path.lower().endswith((".jpg", ".jpeg")):
            img = img.convert("RGB")
        img.save(path)
        self._statusbar.showMessage(f"Saved: {path}", 3000)

    # --- Diffusion ---

    def _on_load_model(self, path: str, prediction_type: str):
        if self._engine.is_busy:
            return
        pred = prediction_type if prediction_type else None
        self._engine.submit_load(path, pred)

    def _on_layer_changed(self):
        layer = self._layer_stack.active_layer
        # Hide all panels first
        self._brush_panel.setVisible(False)
        self._diffusion_panel.setVisible(False)
        self._lama_panel.setVisible(False)
        self._instruct_panel.setVisible(False)

        if layer is None:
            pass
        elif isinstance(layer, DiffusionLayer):
            self._diffusion_panel.show_diffusion_layer(layer)
            self._diffusion_panel.setVisible(True)
        elif isinstance(layer, LamaLayer):
            self._lama_panel.show_lama_layer(layer)
            self._lama_panel.setVisible(True)
        elif isinstance(layer, InstructLayer):
            self._instruct_panel.show_instruct_layer(layer)
            self._instruct_panel.setVisible(True)
        else:
            self._brush_panel.setVisible(True)

    def _on_create_diffusion(self):
        composite = self._canvas.get_composite()
        if composite is None:
            return

        mode = self._diffusion_panel.mode
        center_x, center_y = self._canvas.view_center_image()

        if mode == "txt2img":
            # No source image needed — patch covers entire canvas
            patch_pil = None
            px, py = 0, 0
            pw = self._layer_stack.width
            ph = self._layer_stack.height
        else:
            patch_pil, px, py, pw, ph = extract_patch(composite, center_x, center_y)

        seed = self._diffusion_panel.seed
        if seed < 0:
            seed = random.randint(0, 2**32 - 1)
            self._diffusion_panel.set_seed(seed)

        dl = DiffusionLayer(
            name=self._layer_stack.next_name("Diffusion"),
            width=self._layer_stack.width,
            height=self._layer_stack.height,
            source_patch=patch_pil,
            patch_x=px, patch_y=py, patch_w=pw, patch_h=ph,
            prompt=self._diffusion_panel.prompt,
            negative_prompt=self._diffusion_panel.negative_prompt,
            strength=self._diffusion_panel.strength,
            guidance_scale=self._diffusion_panel.guidance_scale,
            steps=self._diffusion_panel.steps,
            seed=seed,
            model_path=self._engine.model_path or "",
            prediction_type=self._diffusion_panel.prediction_type,
            mode=mode,
        )
        self._layer_stack.insert_layer(dl)

    def _on_color_picked(self, r, g, b, a):
        self._canvas.brush.set_color(r, g, b, a)
        self._brush_panel.sync_from_brush()

    def _on_clear_mask(self):
        layer = self._layer_stack.active_layer
        if isinstance(layer, DiffusionLayer):
            layer.clear_mask()
            self._layer_stack.changed.emit()

    def _on_mask_brush_changed(self, size: int, hardness: float):
        self._canvas.set_mask_brush(size, hardness)

    def _sync_panel_to_layer(self, layer: DiffusionLayer):
        layer.prompt = self._diffusion_panel.prompt
        layer.negative_prompt = self._diffusion_panel.negative_prompt
        layer.strength = self._diffusion_panel.strength
        layer.guidance_scale = self._diffusion_panel.guidance_scale
        layer.steps = self._diffusion_panel.steps
        layer.seed = self._diffusion_panel.seed
        layer.mode = self._diffusion_panel.mode
        layer.masked_content = self._diffusion_panel.masked_content
        layer.ip_adapter_scale = self._diffusion_panel.ip_adapter_scale
        layer.resize_to_model_resolution = self._diffusion_panel.resize_to_model_resolution
        if self._engine.model_path:
            layer.model_path = self._engine.model_path
        layer.prediction_type = self._diffusion_panel.prediction_type

    def _on_regenerate(self):
        layer = self._layer_stack.active_layer
        if not isinstance(layer, DiffusionLayer):
            return
        if self._engine.is_busy:
            return

        self._sync_panel_to_layer(layer)

        if layer.mode != "txt2img":
            composite = self._canvas.get_composite_below(layer)
            if composite is None:
                return

            if layer.manual_patch_rect is not None:
                # Явно заданная область патча
                x0, y0, x1, y1 = layer.manual_patch_rect
                h, w = composite.shape[:2]
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(w, x1), min(h, y1)
                if x1 - x0 < 1 or y1 - y0 < 1:
                    return
                from PIL import Image
                patch_pil = Image.fromarray(composite[y0:y1, x0:x1]).convert("RGB")
                layer.source_patch = patch_pil
                layer.patch_x = x0
                layer.patch_y = y0
                layer.patch_w = x1 - x0
                layer.patch_h = y1 - y0
            elif layer.has_mask():
                # Автоматический расчёт из маски
                bbox = layer.mask_bbox()
                center = layer.mask_center()
                if bbox is not None and center is not None:
                    bx0, by0, bx1, by1 = bbox
                    mask_w = bx1 - bx0
                    mask_h = by1 - by0
                    patch_size = max(mask_w, mask_h)
                    patch_size = int(patch_size * 1.25)
                    patch_size = max(patch_size, 512)
                    center_x, center_y = center
                    patch_pil, px, py, pw, ph = extract_patch(
                        composite, center_x, center_y, patch_size=patch_size)
                    layer.source_patch = patch_pil
                    layer.patch_x = px
                    layer.patch_y = py
                    layer.patch_w = pw
                    layer.patch_h = ph

            if layer.source_patch is None:
                return

        # Если нужна другая модель — загружаем сначала
        if layer.model_path and layer.model_path != self._engine.model_path:
            self._pending_request = layer  # после загрузки запустим inference
            pred = layer.prediction_type if layer.prediction_type else None
            self._engine.submit_load(layer.model_path, pred)
            self._statusbar.showMessage("Loading model for regeneration...")
            return

        if not self._engine.is_loaded:
            return

        self._submit_regenerate(layer)

    def _submit_regenerate(self, layer: DiffusionLayer):
        self._pending_request = layer
        mask_image = None
        if layer.mode == "inpaint":
            if not layer.has_mask():
                self._statusbar.showMessage("Inpaint requires a mask", 3000)
                return
            mask_image = extract_mask_patch(
                layer.mask, layer.patch_x, layer.patch_y,
                layer.patch_w, layer.patch_h)

        # IP-Adapter: если есть rect, но адаптер не загружен — загружаем сначала
        ip_adapter_image = None
        if layer.ip_adapter_rect is not None:
            if not self._engine.ip_adapter_loaded:
                self._pending_request = layer
                self._engine.submit_load_ip_adapter()
                self._statusbar.showMessage("Loading IP-Adapter...")
                return
            composite = self._canvas.get_composite_below(layer)
            if composite is not None:
                x0, y0, x1, y1 = layer.ip_adapter_rect
                h, w = composite.shape[:2]
                x0 = max(0, min(x0, w))
                x1 = max(0, min(x1, w))
                y0 = max(0, min(y0, h))
                y1 = max(0, min(y1, h))
                if x1 > x0 and y1 > y0:
                    crop = composite[y0:y1, x0:x1, :3]
                    ip_adapter_image = Image.fromarray(crop, "RGB")

        # Rescale patch to model resolution if needed
        submit_image = layer.source_patch
        submit_mask = mask_image
        submit_w, submit_h = layer.patch_w, layer.patch_h
        MODEL_RES = 1024
        if layer.resize_to_model_resolution and submit_image is not None:
            longest = max(submit_w, submit_h)
            if longest != MODEL_RES:
                scale = MODEL_RES / longest
                submit_w = max(8, round(layer.patch_w * scale / 8) * 8)
                submit_h = max(8, round(layer.patch_h * scale / 8) * 8)
                submit_image = submit_image.resize((submit_w, submit_h), Image.LANCZOS)
                if submit_mask is not None:
                    submit_mask = submit_mask.resize((submit_w, submit_h), Image.NEAREST)

        self._engine.submit(
            image=submit_image,
            prompt=layer.prompt,
            negative_prompt=layer.negative_prompt,
            strength=layer.strength,
            steps=layer.steps,
            guidance_scale=layer.guidance_scale,
            seed=layer.seed,
            mode=layer.mode,
            mask_image=submit_mask,
            masked_content=layer.masked_content,
            ip_adapter_image=ip_adapter_image,
            ip_adapter_scale=layer.ip_adapter_scale,
            width=submit_w,
            height=submit_h,
        )
        self._statusbar.showMessage(f"Regenerating ({submit_w}x{submit_h})...")

    def _on_new_seed(self):
        layer = self._layer_stack.active_layer
        if not isinstance(layer, DiffusionLayer):
            return
        new_seed = random.randint(0, 2**32 - 1)
        layer.seed = new_seed
        self._diffusion_panel.set_seed(new_seed)
        self._on_regenerate()

    def _on_load_ip_adapter(self):
        if self._engine.is_busy:
            return
        if not self._engine.is_loaded:
            self._statusbar.showMessage("Load a model first", 3000)
            return
        self._engine.submit_load_ip_adapter()
        self._statusbar.showMessage("Loading IP-Adapter...")

    def _on_ref_rect_drawn(self, x0, y0, x1, y1):
        layer = self._layer_stack.active_layer
        if isinstance(layer, DiffusionLayer):
            layer.ip_adapter_rect = (x0, y0, x1, y1)
            self._diffusion_panel.show_diffusion_layer(layer)
            self._canvas.update()

    def _on_clear_ref_rect(self):
        layer = self._layer_stack.active_layer
        if isinstance(layer, DiffusionLayer):
            layer.ip_adapter_rect = None
            self._diffusion_panel.show_diffusion_layer(layer)
            self._canvas.update()

    def _on_patch_rect_drawn(self, x0, y0, x1, y1):
        layer = self._layer_stack.active_layer
        if isinstance(layer, DiffusionLayer):
            layer.manual_patch_rect = (x0, y0, x1, y1)
            self._diffusion_panel._draw_patch_btn.setChecked(False)
            self._diffusion_panel.show_diffusion_layer(layer)
            self._canvas.update()
        elif isinstance(layer, InstructLayer):
            layer.manual_patch_rect = (x0, y0, x1, y1)
            self._instruct_panel._draw_patch_btn.setChecked(False)
            self._instruct_panel.show_instruct_layer(layer)
            self._canvas.update()

    def _on_clear_patch_rect(self):
        layer = self._layer_stack.active_layer
        if isinstance(layer, DiffusionLayer):
            layer.manual_patch_rect = None
            self._diffusion_panel.show_diffusion_layer(layer)
            self._canvas.update()
        elif isinstance(layer, InstructLayer):
            layer.manual_patch_rect = None
            self._instruct_panel.show_instruct_layer(layer)
            self._canvas.update()

    # --- LaMa ---

    def _on_create_lama(self):
        composite = self._canvas.get_composite()
        if composite is None:
            return
        center_x, center_y = self._canvas.view_center_image()
        patch_pil, px, py, pw, ph = extract_patch(composite, center_x, center_y)
        ll = LamaLayer(
            name=self._layer_stack.next_name("LaMa"),
            width=self._layer_stack.width,
            height=self._layer_stack.height,
            source_patch=patch_pil,
            patch_x=px, patch_y=py, patch_w=pw, patch_h=ph,
        )
        self._layer_stack.insert_layer(ll)

    def _on_lama_remove(self):
        layer = self._layer_stack.active_layer
        if not isinstance(layer, LamaLayer):
            return
        if self._lama_engine.is_busy:
            return
        if not layer.has_mask():
            self._statusbar.showMessage("Draw a mask first", 3000)
            return

        # Recalculate patch from mask bbox
        bbox = layer.mask_bbox()
        center = layer.mask_center()
        if bbox is None or center is None:
            return
        composite = self._canvas.get_composite_below(layer)
        if composite is None:
            return
        bx0, by0, bx1, by1 = bbox
        mask_w = bx1 - bx0
        mask_h = by1 - by0
        patch_size = max(mask_w, mask_h)
        patch_size = int(patch_size * 1.25)
        patch_size = max(patch_size, 512)
        center_x, center_y = center
        patch_pil, px, py, pw, ph = extract_patch(
            composite, center_x, center_y, patch_size=patch_size)
        layer.source_patch = patch_pil
        layer.patch_x = px
        layer.patch_y = py
        layer.patch_w = pw
        layer.patch_h = ph

        mask_pil = extract_mask_patch(layer.mask, px, py, pw, ph)
        self._lama_engine.submit(patch_pil, mask_pil)
        self._pending_lama_layer = layer
        self._statusbar.showMessage("Removing objects (LaMa)...")

    def _on_lama_clear_mask(self):
        layer = self._layer_stack.active_layer
        if isinstance(layer, LamaLayer):
            layer.clear_mask()
            self._layer_stack.changed.emit()

    def _on_lama_select_background(self):
        layer = self._layer_stack.active_layer
        if not isinstance(layer, LamaLayer):
            return
        if self._seg_engine.is_busy:
            return
        composite = self._canvas.get_composite_below(layer)
        if composite is None:
            return
        self._seg_engine.submit(composite, invert=True)
        self._lama_panel._select_bg_btn.setEnabled(False)
        self._statusbar.showMessage("Segmenting background...")

    # --- InstructPix2Pix ---

    def _on_create_instruct(self):
        composite = self._canvas.get_composite()
        if composite is None:
            return
        center_x, center_y = self._canvas.view_center_image()
        patch_pil, px, py, pw, ph = extract_patch(composite, center_x, center_y)
        seed = self._instruct_panel.seed
        if seed < 0:
            seed = random.randint(0, 2**32 - 1)
            self._instruct_panel.set_seed(seed)
        il = InstructLayer(
            name=self._layer_stack.next_name("Instruct"),
            width=self._layer_stack.width,
            height=self._layer_stack.height,
            source_patch=patch_pil,
            patch_x=px, patch_y=py, patch_w=pw, patch_h=ph,
            instruction=self._instruct_panel.instruction,
            image_guidance_scale=self._instruct_panel.image_guidance_scale,
            guidance_scale=self._instruct_panel.guidance_scale,
            steps=self._instruct_panel.steps,
            seed=seed,
        )
        self._layer_stack.insert_layer(il)

    def _on_instruct_clear_mask(self):
        layer = self._layer_stack.active_layer
        if isinstance(layer, InstructLayer):
            layer.clear_mask()
            self._layer_stack.changed.emit()

    def _on_instruct_clear_patch_rect(self):
        layer = self._layer_stack.active_layer
        if isinstance(layer, InstructLayer):
            layer.manual_patch_rect = None
            self._instruct_panel.show_instruct_layer(layer)
            self._canvas.update()

    def _on_instruct_load_model(self):
        if self._instruct_engine.is_busy:
            return
        self._instruct_panel._load_btn.setEnabled(False)
        self._instruct_panel._model_status.setText("Loading...")
        self._instruct_engine.submit_load()
        self._statusbar.showMessage("Loading InstructPix2Pix model...")

    def _on_instruct_apply(self):
        layer = self._layer_stack.active_layer
        if not isinstance(layer, InstructLayer):
            return
        if self._instruct_engine.is_busy:
            return

        # Sync panel to layer
        layer.instruction = self._instruct_panel.instruction
        layer.image_guidance_scale = self._instruct_panel.image_guidance_scale
        layer.guidance_scale = self._instruct_panel.guidance_scale
        layer.steps = self._instruct_panel.steps
        layer.seed = self._instruct_panel.seed

        if not self._instruct_engine.is_loaded:
            # Auto-load model first
            self._pending_instruct_layer = layer
            self._on_instruct_load_model()
            return

        # Re-extract patch from current composite below
        composite = self._canvas.get_composite_below(layer)
        if composite is None:
            return

        if layer.manual_patch_rect is not None:
            x0, y0, x1, y1 = layer.manual_patch_rect
            h, w = composite.shape[:2]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            if x1 - x0 < 1 or y1 - y0 < 1:
                return
            patch_pil = Image.fromarray(composite[y0:y1, x0:x1]).convert("RGB")
            layer.source_patch = patch_pil
            layer.patch_x = x0
            layer.patch_y = y0
            layer.patch_w = x1 - x0
            layer.patch_h = y1 - y0
        elif layer.has_mask():
            bbox = layer.mask_bbox()
            center = layer.mask_center()
            if bbox is not None and center is not None:
                bx0, by0, bx1, by1 = bbox
                mask_w = bx1 - bx0
                mask_h = by1 - by0
                patch_size = max(mask_w, mask_h)
                patch_size = int(patch_size * 1.25)
                patch_size = max(patch_size, 512)
                center_x, center_y = center
                patch_pil, px, py, pw, ph = extract_patch(
                    composite, center_x, center_y, patch_size=patch_size)
                layer.source_patch = patch_pil
                layer.patch_x = px
                layer.patch_y = py
                layer.patch_w = pw
                layer.patch_h = ph
        else:
            center_x = layer.patch_x + layer.patch_w // 2
            center_y = layer.patch_y + layer.patch_h // 2
            patch_pil, px, py, pw, ph = extract_patch(
                composite, center_x, center_y, patch_size=max(layer.patch_w, layer.patch_h))
            layer.source_patch = patch_pil
            layer.patch_x = px
            layer.patch_y = py
            layer.patch_w = pw
            layer.patch_h = ph

        self._instruct_engine.submit(
            image=patch_pil,
            instruction=layer.instruction,
            guidance_scale=layer.guidance_scale,
            image_guidance_scale=layer.image_guidance_scale,
            steps=layer.steps,
            seed=layer.seed,
        )
        self._pending_instruct_layer = layer
        self._statusbar.showMessage("Applying instruction...")

    def _on_instruct_new_seed(self):
        layer = self._layer_stack.active_layer
        if not isinstance(layer, InstructLayer):
            return
        new_seed = random.randint(0, 2**32 - 1)
        layer.seed = new_seed
        self._instruct_panel.set_seed(new_seed)
        self._on_instruct_apply()

    # --- Segmentation ---

    def _on_select_background(self):
        print("[SelectBG] handler called")
        layer = self._layer_stack.active_layer
        if not isinstance(layer, DiffusionLayer):
            print(f"[SelectBG] not a DiffusionLayer: {type(layer)}")
            return
        if self._seg_engine.is_busy:
            print("[SelectBG] seg_engine is busy")
            return
        composite = self._canvas.get_composite_below(layer)
        if composite is None:
            print("[SelectBG] composite is None")
            return
        print(f"[SelectBG] submitting, composite shape={composite.shape}")
        self._seg_engine.submit(composite, invert=True)
        self._diffusion_panel._select_bg_btn.setEnabled(False)
        self._statusbar.showMessage("Segmenting background...")

    def _poll_segmentation(self):
        seg_mask, seg_error = self._seg_engine.poll()
        if seg_mask is not None:
            layer = self._layer_stack.active_layer
            if isinstance(layer, (DiffusionLayer, LamaLayer)):
                layer.mask = seg_mask
                self._layer_stack.changed.emit()
            self._diffusion_panel._select_bg_btn.setEnabled(True)
            self._lama_panel._select_bg_btn.setEnabled(True)
            self._statusbar.showMessage("Background mask applied", 3000)
        elif seg_error is not None:
            self._diffusion_panel._select_bg_btn.setEnabled(True)
            self._lama_panel._select_bg_btn.setEnabled(True)
            self._statusbar.showMessage(f"Segmentation error: {seg_error[:80]}", 5000)

    def _poll_lama(self):
        result_image, lama_error = self._lama_engine.poll()
        if result_image is not None:
            layer = self._pending_lama_layer
            if isinstance(layer, LamaLayer):
                from PIL import ImageFilter
                layer.image[:] = 0
                if layer.has_mask():
                    mask_pil = Image.fromarray(layer.mask, "L")
                    mask_pil = mask_pil.filter(ImageFilter.MaxFilter(7))
                    mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=4))
                    mask_arg = np.array(mask_pil, dtype=np.uint8)
                else:
                    mask_arg = None
                paste_result(layer.image, result_image, layer.patch_x, layer.patch_y,
                             layer.patch_w, layer.patch_h, mask=mask_arg)
                self._layer_stack.changed.emit()
                self._lama_panel.show_lama_layer(layer)
                self._statusbar.showMessage("Objects removed (LaMa)", 3000)
            self._pending_lama_layer = None
        elif lama_error is not None:
            self._statusbar.showMessage(f"LaMa error: {lama_error[:80]}", 5000)
            self._pending_lama_layer = None

    def _poll_instruct(self):
        task_type, result, error, meta = self._instruct_engine.poll()
        if task_type is None:
            return
        if task_type == "load":
            if error:
                self._instruct_panel.on_model_load_error(error)
                self._statusbar.showMessage(f"InstructPix2Pix load error: {error[:80]}", 5000)
                self._pending_instruct_layer = None
            else:
                self._instruct_panel.on_model_loaded()
                self._statusbar.showMessage("InstructPix2Pix model loaded", 3000)
                # If pending apply after load — run it now
                if isinstance(self._pending_instruct_layer, InstructLayer):
                    self._on_instruct_apply()
        elif task_type == "inference":
            if error:
                self._statusbar.showMessage(f"InstructPix2Pix error: {error[:80]}", 5000)
                self._pending_instruct_layer = None
                return
            result_image, used_seed = result
            layer = self._pending_instruct_layer
            if isinstance(layer, InstructLayer):
                layer.image[:] = 0
                if layer.has_mask():
                    from PIL import ImageFilter
                    mask_pil = Image.fromarray(layer.mask, "L")
                    mask_pil = mask_pil.filter(ImageFilter.MaxFilter(7))
                    mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=4))
                    mask_arg = np.array(mask_pil, dtype=np.uint8)
                else:
                    mask_arg = None
                paste_result(layer.image, result_image, layer.patch_x, layer.patch_y,
                             layer.patch_w, layer.patch_h, mask=mask_arg)
                self._layer_stack.changed.emit()
                self._instruct_panel.show_instruct_layer(layer)
                self._statusbar.showMessage(f"Instruction applied (seed={used_seed})", 3000)
            self._pending_instruct_layer = None

    def _poll_engine(self):
        self._poll_segmentation()
        self._poll_lama()
        self._poll_instruct()

        task_type, result, error, meta = self._engine.poll()
        if task_type is None:
            return

        if task_type == "load":
            if error:
                self._diffusion_panel.on_model_load_error(error)
                self._statusbar.showMessage(f"Model load error: {error[:80]}", 5000)
                self._pending_request = None
            else:
                self._diffusion_panel.on_model_loaded(result, self._engine.model_info)
                # Если ждал regenerate после смены модели — запускаем
                if isinstance(self._pending_request, DiffusionLayer):
                    self._submit_regenerate(self._pending_request)
                    return
                self._statusbar.showMessage("Model loaded", 3000)

        elif task_type == "load_ip_adapter":
            if error:
                self._diffusion_panel.on_ip_adapter_load_error(error)
                self._statusbar.showMessage(f"IP-Adapter error: {error[:80]}", 5000)
                self._pending_request = None
            else:
                self._diffusion_panel.on_ip_adapter_loaded()
                if isinstance(self._pending_request, DiffusionLayer):
                    self._submit_regenerate(self._pending_request)
                    return
                self._statusbar.showMessage("IP-Adapter loaded", 3000)

        elif task_type == "inference":
            if error:
                self._statusbar.showMessage(f"Diffusion error: {error[:80]}", 5000)
                self._pending_request = None
                return

            result_image, used_seed = result

            if isinstance(self._pending_request, DiffusionLayer):
                dl = self._pending_request
                dl.image[:] = 0
                if dl.has_mask():
                    # Feather the mask: dilate + blur for smooth edges
                    from PIL import ImageFilter
                    mask_pil = Image.fromarray(dl.mask, "L")
                    mask_pil = mask_pil.filter(ImageFilter.MaxFilter(7))
                    mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=4))
                    mask_arg = np.array(mask_pil, dtype=np.uint8)
                else:
                    mask_arg = None
                paste_result(dl.image, result_image, dl.patch_x, dl.patch_y,
                             dl.patch_w, dl.patch_h, mask=mask_arg)
                self._layer_stack.changed.emit()
                self._diffusion_panel.show_diffusion_layer(dl)
                self._statusbar.showMessage(f"Regenerated (seed={used_seed})", 3000)

            self._pending_request = None
