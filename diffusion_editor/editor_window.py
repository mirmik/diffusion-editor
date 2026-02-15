import numpy as np
from PIL import Image
from PyQt6.QtWidgets import (
    QMainWindow, QFileDialog, QStatusBar, QToolBar,
)
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtCore import Qt

from .canvas import Canvas


class EditorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Diffusion Editor")
        self.resize(1280, 800)

        self._canvas = Canvas(self)
        self.setCentralWidget(self._canvas)

        self._setup_menu()
        self._setup_toolbar()
        self._setup_statusbar()

        self._canvas.mouse_moved.connect(self._on_mouse_moved)
        self._current_path = None

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
        fit_action.triggered.connect(self._canvas.fit_in_view)
        fit_action.triggered.connect(self._canvas.update)
        toolbar.addAction(fit_action)

    def _setup_statusbar(self):
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)
        self._statusbar.showMessage("Ready")

    def _on_mouse_moved(self, x, y):
        size = self._canvas.image_size()
        if size is None:
            return
        w, h = size
        in_bounds = 0 <= x < w and 0 <= y < h
        if in_bounds:
            self._statusbar.showMessage(f"{w}x{h}  |  ({x}, {y})")
        else:
            self._statusbar.showMessage(f"{w}x{h}")

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
        self._canvas.set_image(arr)
        self._current_path = path
        self.setWindowTitle(f"Diffusion Editor — {path}")

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
        arr = self._canvas.get_image()
        if arr is None:
            return
        img = Image.fromarray(arr, "RGBA")
        if path.lower().endswith((".jpg", ".jpeg")):
            img = img.convert("RGB")
        img.save(path)
        self._statusbar.showMessage(f"Saved: {path}", 3000)
