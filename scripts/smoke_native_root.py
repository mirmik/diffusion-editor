#!/usr/bin/env python3
"""Run the parallel termin-gui-native root for a bounded number of frames."""

from __future__ import annotations

import argparse

import numpy as np
from termin.gui_native import (
    DynamicTextureOwnership,
    Point,
    PointerEvent,
    PointerEventType,
)

from diffusion_editor.app.application import EditorApplication, EngineSet
from diffusion_editor.app.native_root import NativeEditorRoot


class _MemorySettings:
    def get(self, _key, default=None):
        return default

    def set(self, _key, _value):
        pass


class _IdleEngine:
    model_info = {}

    def poll_event(self):
        return None

    def shutdown(self):
        pass


def _application() -> EditorApplication:
    engine = _IdleEngine()
    return EditorApplication(
        settings=_MemorySettings(),
        engines=EngineSet(engine, engine, engine, engine, engine),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--windowed", action="store_true")
    parser.add_argument(
        "--backend",
        choices=("opengl", "vulkan"),
        default="vulkan",
        help="headless composition backend; windowed mode uses TERMIN_BACKEND",
    )
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error("--frames must be positive")

    application = _application()
    factory = (
        NativeEditorRoot.create_windowed
        if args.windowed
        else NativeEditorRoot.create_headless
    )
    options = {} if args.windowed else {"backend": args.backend}
    image = np.zeros((32, 48, 4), dtype=np.uint8)
    image[:, :, 3] = 255
    image[:, :, 0] = np.arange(48, dtype=np.uint8)[None, :] * 5
    image[:, :, 1] = np.arange(32, dtype=np.uint8)[:, None] * 7
    application.layer_stack.init_from_image(image)
    application.layer_stack.active_layer.mask.data[8:16, 12:24] = 1.0
    with factory(application, width=640, height=360, **options) as root:
        if root.canvas_controls is None:
            raise RuntimeError("native Canvas controls were not mounted")
        if root.layer_panel is None:
            raise RuntimeError("native layer panel was not mounted")
        if root.generation_panels is None:
            raise RuntimeError("native generation panels were not mounted")
        root.canvas_controls.brush.size.value = 9.0
        if root.canvas.controller.brush.size != 9:
            raise RuntimeError("native brush control did not update Canvas")
        root.layer_panel.opacity.value = 0.75
        if application.layer_stack.active_layer.opacity != 0.75:
            raise RuntimeError("native layer panel did not update the document")
        expected_ownership = (
            DynamicTextureOwnership.BORROWED
            if args.windowed
            else DynamicTextureOwnership.OWNED
        )
        if root.canvas.image_lease.ownership != expected_ownership:
            raise RuntimeError(
                "native Canvas image lease ownership mismatch: "
                f"expected {expected_ownership}, "
                f"got {root.canvas.image_lease.ownership}"
            )
        if (
                root.canvas.overlay_lease.ownership
                != DynamicTextureOwnership.OWNED):
            raise RuntimeError("native Canvas overlay did not acquire an owned texture")
        root.defer(lambda: application.set_status("Native dispatcher ready"))
        first = root.tick()
        rendered = int(first.rendered)

        center = root.canvas.canvas.image_to_widget(Point(24, 16))
        zoom_before = root.canvas.canvas.zoom
        event = PointerEvent()
        event.x = center.x
        event.y = center.y
        event.type = PointerEventType.Wheel
        event.wheel_y = 1.0
        root.composition.document.dispatch_pointer_event(event)
        if root.canvas.canvas.zoom <= zoom_before:
            raise RuntimeError("native Canvas wheel zoom did not change zoom")

        event.type = PointerEventType.Down
        event.button = root.canvas.controller.LEFT_BUTTON
        root.composition.document.dispatch_pointer_event(event)
        event.type = PointerEventType.Up
        root.composition.document.dispatch_pointer_event(event)

        for _ in range(args.frames - 1):
            result = root.tick()
            rendered += int(result.rendered)
            if not application.running:
                break

    if rendered == 0:
        raise RuntimeError("native root did not render a frame")
    print(
        f"Diffusion Editor native root smoke OK: "
        f"{rendered} frame(s), {'windowed' if args.windowed else args.backend}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
