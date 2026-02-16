import sys
from PyQt6.QtWidgets import QApplication
from .editor_window import EditorWindow


def main():
    app = QApplication(sys.argv)
    window = EditorWindow()
    window.show()

    if len(sys.argv) > 1:
        path = sys.argv[1]
        if path.lower().endswith(".deproj"):
            window.open_file_path(path)
        else:
            window.import_image_path(path)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
