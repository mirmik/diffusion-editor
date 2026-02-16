import os
import random
import numpy as np
from PIL import Image
from PyQt6.QtWidgets import (
    QMainWindow, QFileDialog, QStatusBar, QToolBar, QMessageBox,
)
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtCore import Qt, QTimer, QSettings

from .layer import LayerStack, DiffusionLayer
from .canvas import Canvas
from .layer_panel import LayerPanel
from .brush_panel import BrushPanel
from .diffusion_engine import DiffusionEngine
from .diffusion_panel import DiffusionPanel
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
        self._diffusion_panel = DiffusionPanel(self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._diffusion_panel)
        self._diffusion_panel.setVisible(False)

        self._setup_menu()
        self._setup_toolbar()
        self._setup_statusbar()

        self._canvas.mouse_moved.connect(self._on_mouse_moved)
        self._canvas.color_picked.connect(self._on_color_picked)
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
        self._canvas.ref_rect_drawn.connect(self._on_ref_rect_drawn)
        self._layer_panel.create_diffusion_requested.connect(self._on_create_diffusion)
        self._layer_stack.changed.connect(self._on_layer_changed)
        self._settings = QSettings("DiffusionEditor", "DiffusionEditor")
        self._project_path = None
        self._last_dir = self._settings.value("last_dir", "")
        self._pending_request = None  # DiffusionLayer for regenerate

        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_engine)
        self._poll_timer.start(50)

    def _setup_menu(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")
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
        if layer is None:
            self._brush_panel.setVisible(False)
            self._diffusion_panel.setVisible(False)
        elif isinstance(layer, DiffusionLayer):
            self._diffusion_panel.show_diffusion_layer(layer)
            self._diffusion_panel.setVisible(True)
            self._brush_panel.setVisible(False)
        else:
            self._diffusion_panel.clear_diffusion_layer()
            self._diffusion_panel.setVisible(False)
            self._brush_panel.setVisible(True)

    def _on_create_diffusion(self):
        composite = self._canvas.get_composite()
        if composite is None:
            return

        center_x, center_y = self._canvas.view_center_image()
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
            mode=self._diffusion_panel.mode,
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
        layer.ip_adapter_scale = self._diffusion_panel.ip_adapter_scale
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

        # Вычисляем позицию и размер патча из маски
        if layer.has_mask():
            bbox = layer.mask_bbox()
            center = layer.mask_center()
            if bbox is not None and center is not None:
                composite = self._canvas.get_composite_below(layer)
                if composite is None:
                    return
                bx0, by0, bx1, by1 = bbox
                mask_w = bx1 - bx0
                mask_h = by1 - by0
                # Патч — квадрат, покрывающий маску + 25% запас, минимум 512
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

        self._engine.submit(
            image=layer.source_patch,
            prompt=layer.prompt,
            negative_prompt=layer.negative_prompt,
            strength=layer.strength,
            steps=layer.steps,
            guidance_scale=layer.guidance_scale,
            seed=layer.seed,
            mode=layer.mode,
            mask_image=mask_image,
            ip_adapter_image=ip_adapter_image,
            ip_adapter_scale=layer.ip_adapter_scale,
        )
        self._statusbar.showMessage("Regenerating...")

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

    def _poll_engine(self):
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
                # In inpaint mode the pipeline handles masking — don't apply
                # mask as alpha again (double-masking creates visible edges).
                # In img2img mode, mask controls where the result is visible.
                mask_arg = dl.mask if (dl.mode != "inpaint" and dl.has_mask()) else None
                paste_result(dl.image, result_image, dl.patch_x, dl.patch_y,
                             dl.patch_w, dl.patch_h, mask=mask_arg)
                self._layer_stack.changed.emit()
                self._diffusion_panel.show_diffusion_layer(dl)
                self._statusbar.showMessage(f"Regenerated (seed={used_seed})", 3000)

            self._pending_request = None
