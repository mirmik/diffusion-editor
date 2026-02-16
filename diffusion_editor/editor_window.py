import numpy as np
from PIL import Image
from PyQt6.QtWidgets import (
    QMainWindow, QFileDialog, QStatusBar, QToolBar, QColorDialog,
)
from PyQt6.QtGui import QAction, QKeySequence, QColor
from PyQt6.QtCore import Qt, QTimer

from .layer import LayerStack
from .canvas import Canvas
from .layer_panel import LayerPanel
from .diffusion_engine import DiffusionEngine
from .diffusion_panel import DiffusionPanel
from .diffusion_brush import extract_patch, paste_result


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

        self._engine = DiffusionEngine()
        self._diffusion_panel = DiffusionPanel(self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._diffusion_panel)

        self._setup_menu()
        self._setup_toolbar()
        self._setup_statusbar()

        self._canvas.mouse_moved.connect(self._on_mouse_moved)
        self._canvas.diffusion_requested.connect(self._on_diffusion_requested)
        self._diffusion_panel.load_model_requested.connect(self._on_load_model)
        self._diffusion_panel.tool_toggled.connect(self._on_tool_toggled)
        self._current_path = None
        self._pending_paste = None

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

        toolbar.addSeparator()

        self._color_action = QAction("Color", self)
        self._color_action.triggered.connect(self._pick_color)
        toolbar.addAction(self._color_action)
        self._update_color_icon()

    def _fit(self):
        self._canvas.fit_in_view()
        self._canvas.update()

    def _pick_color(self):
        r, g, b, a = self._canvas.brush.color
        initial = QColor(r, g, b, a)
        color = QColorDialog.getColor(initial, self, "Brush Color", QColorDialog.ColorDialogOption.ShowAlphaChannel)
        if color.isValid():
            self._canvas.brush.set_color(color.red(), color.green(), color.blue(), color.alpha())
            self._update_color_icon()

    def _update_color_icon(self):
        r, g, b, _ = self._canvas.brush.color
        from PyQt6.QtGui import QPixmap, QIcon
        px = QPixmap(16, 16)
        px.fill(QColor(r, g, b))
        self._color_action.setIcon(QIcon(px))

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
            self, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp);;All Files (*)",
        )
        if not path:
            return
        self._load_image(path)

    def _load_image(self, path):
        img = Image.open(path).convert("RGBA")
        arr = np.array(img, dtype=np.uint8)
        self._layer_stack.init_from_image(arr)
        self._canvas.fit_in_view()
        self._current_path = path
        self.setWindowTitle(f"Diffusion Editor — {path}")

    def _new_layer(self):
        count = len(self._layer_stack.layers)
        self._layer_stack.add_layer(f"Layer {count}")

    def _remove_layer(self):
        self._layer_stack.remove_layer(self._layer_stack.active_index)

    def save_file(self):
        if self._current_path:
            self._save_to(self._current_path)
        else:
            self.save_file_as()

    def save_file_as(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;All Files (*)",
        )
        if not path:
            return
        self._save_to(path)
        self._current_path = path
        self.setWindowTitle(f"Diffusion Editor — {path}")

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

    def _on_tool_toggled(self, active: bool):
        self._canvas.set_tool_mode("diffusion" if active else "brush")

    def _on_diffusion_requested(self, center_x: int, center_y: int):
        composite = self._canvas.get_composite()
        if composite is None:
            self._canvas.diffusion_busy = False
            return
        if not self._engine.is_loaded:
            self._statusbar.showMessage("No diffusion model loaded!", 3000)
            self._canvas.diffusion_busy = False
            return

        patch_pil, px, py, pw, ph = extract_patch(composite, center_x, center_y)
        self._pending_paste = (px, py, pw, ph)

        self._engine.submit(
            image=patch_pil,
            prompt=self._diffusion_panel.prompt,
            negative_prompt=self._diffusion_panel.negative_prompt,
            strength=self._diffusion_panel.strength,
            steps=self._diffusion_panel.steps,
            guidance_scale=self._diffusion_panel.guidance_scale,
        )
        self._statusbar.showMessage("Diffusion processing...")

    def _poll_engine(self):
        task_type, result, error, meta = self._engine.poll()
        if task_type is None:
            return

        if task_type == "load":
            if error:
                self._diffusion_panel.on_model_load_error(error)
                self._statusbar.showMessage(f"Model load error: {error[:80]}", 5000)
            else:
                self._diffusion_panel.on_model_loaded(result, self._engine.model_info)
                self._statusbar.showMessage("Model loaded", 3000)

        elif task_type == "inference":
            if error:
                self._statusbar.showMessage(f"Diffusion error: {error[:80]}", 5000)
            else:
                layer = self._layer_stack.active_layer
                if layer is not None and self._pending_paste is not None:
                    px, py, pw, ph = self._pending_paste
                    paste_result(layer.image, result, px, py, pw, ph)
                    self._layer_stack.changed.emit()
                    self._statusbar.showMessage("Diffusion done", 2000)
                self._pending_paste = None
            self._canvas.diffusion_busy = False
