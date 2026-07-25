"""Production-settings host for the termin-gui-native comparison mode."""

from __future__ import annotations

import faulthandler
import os
import time

from tcbase import log

from .application import EditorApplication
from .native_root import NativeEditorRoot


def _smoke_frames() -> int:
    frames = int(os.environ.get("DIFFUSION_EDITOR_SMOKE_FRAMES", "0"))
    if frames < 0:
        raise ValueError("DIFFUSION_EDITOR_SMOKE_FRAMES must be non-negative")
    return frames


def _open_path(root: NativeEditorRoot, path: str) -> None:
    if path.lower().endswith(".deproj"):
        root.dialog_coordinator.open_project_path(path)
    else:
        root.dialog_coordinator.import_image_path(path)


def main(path: str | None = None) -> int:
    faulthandler.enable()
    log.set_level(log.Level.INFO)

    application = EditorApplication()
    smoke_frames = _smoke_frames()
    rendered_frames = 0

    with NativeEditorRoot.create_windowed(application) as root:
        log.info("[main] Window created")
        if path is not None:
            _open_path(root, path)

        while application.running:
            if smoke_frames:
                root.composition.request_repaint()
            result = root.tick()
            rendered_frames += int(result.rendered)
            if smoke_frames and rendered_frames >= smoke_frames:
                application.request_stop()
            elif (
                    not result.rendered
                    and result.events == 0
                    and result.dispatched == 0):
                time.sleep(0.01)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
