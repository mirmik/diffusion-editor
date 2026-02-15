import sys
from PyQt6.QtWidgets import QApplication
from .editor_window import EditorWindow


def main():
    app = QApplication(sys.argv)
    window = EditorWindow()
    window.show()

    if len(sys.argv) > 1:
        window._load_image(sys.argv[1])

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
