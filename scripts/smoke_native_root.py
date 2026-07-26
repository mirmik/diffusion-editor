#!/usr/bin/env python3
"""Exercise the native editor root through a bounded user-visible scenario."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import time

import numpy as np
from PIL import Image
from termin.gui_native import (
    DynamicTextureOwnership,
    Point,
    PointerEvent,
    PointerEventType,
)

from diffusion_editor.app.application import EditorApplication, EngineSet
from diffusion_editor.app.canvas_controls import (
    BrushControlAction,
    BrushControlsIntent,
)
from diffusion_editor.app.generation_panels import (
    GenerationAction,
    GenerationIntent,
    GenerationPhase,
)
from diffusion_editor.app.layer_tree import LayerTreeAction, LayerTreeIntent
from diffusion_editor.app.native_root import NativeEditorRoot
from diffusion_editor.canvas.brush import BrushToolMode
from diffusion_editor.canvas.gpu_compositor import GPUCompositor
from diffusion_editor.canvas.render_diagnostics import (
    rgba_signature,
    source_contribution_error,
)
from diffusion_editor.document.layer import Layer
from diffusion_editor.document.layer_stack import LayerStack
from diffusion_editor.generation.types import (
    DiffusionInferenceResult,
    EnginePollEvent,
)


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


class _FakeDiffusionEngine(_IdleEngine):
    """Deterministic engine for automated and interactive native QA."""

    def __init__(self):
        self.is_busy = False
        self.is_loaded = True
        self.model_path = ""
        self.ip_adapter_loaded = False
        self._event = None

    def submit_load(self, path, _prediction_type=None):
        self.is_busy = True
        self.is_loaded = True
        self.model_path = str(path)
        self._event = EnginePollEvent("load", result=self.model_path)
        return True

    def submit_load_ip_adapter(self):
        self.is_busy = True
        self.ip_adapter_loaded = True
        self._event = EnginePollEvent("load_ip_adapter", result=True)
        return True

    def submit_request(self, request):
        self.is_busy = True
        result = np.full(
            (request.height, request.width, 4),
            (38, 156, 214, 255),
            dtype=np.uint8,
        )
        result[:, ::8, :3] = (240, 180, 40)
        self._event = EnginePollEvent(
            "inference",
            result=DiffusionInferenceResult(
                Image.fromarray(result, "RGBA"),
                request.seed if request.seed >= 0 else 4242,
            ),
        )
        return True

    def poll_event(self):
        event, self._event = self._event, None
        if event is not None:
            self.is_busy = False
        return event


def _application() -> EditorApplication:
    engine = _IdleEngine()
    return EditorApplication(
        settings=_MemorySettings(),
        engines=EngineSet(
            _FakeDiffusionEngine(),
            engine,
            engine,
            engine,
            engine,
        ),
    )


def _input_image(path: Path) -> np.ndarray:
    image = np.zeros((32, 48, 4), dtype=np.uint8)
    image[:, :, 3] = 255
    image[:, :, 0] = np.arange(48, dtype=np.uint8)[None, :] * 5
    image[:, :, 1] = np.arange(32, dtype=np.uint8)[:, None] * 7
    Image.fromarray(image, "RGBA").save(path)
    return image


def _dispatch_pointer(root, image_point: Point) -> None:
    widget_point = root.canvas.canvas.image_to_widget(image_point)
    event = PointerEvent()
    event.x = widget_point.x
    event.y = widget_point.y
    event.button = root.canvas.controller.LEFT_BUTTON
    event.type = PointerEventType.Down
    root.composition.document.dispatch_pointer_event(event)
    event.type = PointerEventType.Up
    root.composition.document.dispatch_pointer_event(event)


def _select_brush_mode(root, mode: BrushToolMode) -> None:
    root.canvas_controls_coordinator.handle_brush_intent(
        BrushControlsIntent(BrushControlAction.TOOL, mode))


def _assert_contract_snapshot(root) -> None:
    stable_ids = {
        item["stable_id"]
        for item in root.composition.document.inspect_snapshot()["widgets"]
        if item["stable_id"]
    }
    expected = {
        "diffusion-editor.root",
        "diffusion-editor.menu-bar",
        "diffusion-editor.toolbar",
        "diffusion-editor.main-splitter",
        "diffusion-editor.workspace-splitter",
        "diffusion-editor.canvas",
        "diffusion-editor.canvas-controls",
        "diffusion-editor.generation-panels",
        "diffusion-editor.layer-panel-content",
        "diffusion-editor.layer-tree",
        "diffusion-editor.agent-panel",
        "diffusion-editor.dialog.settings",
        "diffusion-editor.dialog.grounding",
        "diffusion-editor.status",
    }
    missing = sorted(expected - stable_ids)
    if missing:
        raise RuntimeError(
            "native Document contract is missing stable widgets: "
            + ", ".join(missing)
        )


def _assert_translucent_group_gpu_parity(root) -> None:
    bridge = root.canvas.controller.composite_bridge
    if not bridge.using_gpu:
        return
    stack = LayerStack(tile_size=8)
    background = np.full((2, 2, 4), (20, 40, 220, 255), dtype=np.uint8)
    stack.init_from_image(background)
    group = Layer("group", 2, 2)
    child = Layer(
        "child",
        2,
        2,
        np.full((2, 2, 4), (220, 30, 20, 128), dtype=np.uint8),
    )
    group.add_child(child)
    group.opacity = 0.5
    stack.insert_layer(group)
    expected = stack.composite()

    compositor = GPUCompositor(stack, graphics=root.canvas._graphics)
    try:
        compositor.composite()
        actual = compositor.readback()
    finally:
        compositor.dispose()
    difference = np.abs(
        actual.astype(np.int16) - expected.astype(np.int16))
    if int(difference.max(initial=0)) > 1:
        raise RuntimeError(
            "native GPU translucent-group parity failed: "
            f"expected={expected[0, 0].tolist()}, "
            f"actual={actual[0, 0].tolist()}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--windowed", action="store_true")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="keep the window open for the manual checklist until it is closed",
    )
    parser.add_argument(
        "--backend",
        choices=("opengl", "vulkan"),
        default="vulkan",
        help="headless composition backend; windowed mode uses TERMIN_BACKEND",
    )
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error("--frames must be positive")
    if args.interactive and not args.windowed:
        parser.error("--interactive requires --windowed")

    application = _application()
    factory = (
        NativeEditorRoot.create_windowed
        if args.windowed
        else NativeEditorRoot.create_headless
    )
    options = {} if args.windowed else {"backend": args.backend}
    with tempfile.TemporaryDirectory(
            prefix="diffusion-editor-native-smoke-") as directory:
        image_path = Path(directory) / "input.png"
        source_image = _input_image(image_path)
        with factory(application, width=640, height=360, **options) as root:
            if root.canvas_controls is None:
                raise RuntimeError("native Canvas controls were not mounted")
            if root.layer_panel is None:
                raise RuntimeError("native layer panel was not mounted")
            if root.generation_panels is None:
                raise RuntimeError("native generation panels were not mounted")
            if root.agent_chat is None:
                raise RuntimeError("native Agent Chat was not mounted")
            if root.dialogs is None or root.dialog_coordinator is None:
                raise RuntimeError("native application dialogs were not mounted")
            _assert_contract_snapshot(root)
            _assert_translucent_group_gpu_parity(root)

            # Exercise the same application-owned import path as File > Import.
            root.dialog_coordinator.import_image_path(str(image_path))
            if (application.layer_stack.width, application.layer_stack.height) != (
                    48, 32):
                raise RuntimeError("native import did not open the source image")
            if not np.array_equal(
                    application.layer_stack.active_layer.image, source_image):
                raise RuntimeError("native import changed source pixels")
            # The first frame establishes final layout and Canvas transforms.
            rendered = int(root.tick().rendered)
            composite = root.canvas.controller.get_composite()
            contribution_error = source_contribution_error(
                source_image, composite)
            if contribution_error is not None:
                bridge = root.canvas.controller.composite_bridge
                stages = {}
                if bridge.using_gpu:
                    stages = bridge.gpu_compositor.diagnostic_readbacks(
                        application.layer_stack.active_layer)
                stage_details = "; ".join(
                    f"{name}: {rgba_signature(pixels)}"
                    for name, pixels in stages.items()
                )
                suffix = f"; stages: {stage_details}" if stage_details else ""
                raise RuntimeError(
                    "imported source did not reach Canvas composite: "
                    f"{contribution_error}{suffix}")

            root.canvas_controls.brush.size.value = 9.0
            root.canvas_controls.brush.hardness.value = 1.0
            root.canvas_controls.brush.flow.value = 1.0
            if root.canvas.controller.brush.size != 9:
                raise RuntimeError("native brush control did not update Canvas")
            root.layer_panel.opacity.value = 0.75
            if application.layer_stack.active_layer.opacity != 0.75:
                raise RuntimeError(
                    "native layer panel did not update the document")
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
                    root.canvas.image_lease.width,
                    root.canvas.image_lease.height,
            ) != (48, 32):
                raise RuntimeError("native Canvas image texture is blank")

            # Paint the mask and require an actual overlay texture.
            _select_brush_mode(root, BrushToolMode.MASK)
            _dispatch_pointer(root, Point(24, 16))
            mask = application.layer_stack.active_layer.mask.data
            if not np.any(mask > 0.0):
                raise RuntimeError("native Canvas mask stroke was not routed")
            overlay = root.canvas.controller.overlay_bridge.overlay
            if overlay is None or not np.any(overlay[:, :, 3] > 0):
                raise RuntimeError("native Canvas mask overlay is blank")
            if (
                    root.canvas.overlay_lease.ownership
                    != DynamicTextureOwnership.OWNED):
                raise RuntimeError(
                    "native Canvas overlay did not acquire an owned texture")

            # Switch back to image paint and require a real pixel mutation.
            _select_brush_mode(root, BrushToolMode.PAINT)
            root.canvas.controller.brush.set_color(0, 255, 255, 255)
            before_paint = (
                application.layer_stack.active_layer.image.copy())
            _dispatch_pointer(root, Point(8, 8))
            if np.array_equal(
                    application.layer_stack.active_layer.image, before_paint):
                raise RuntimeError("native Canvas paint stroke changed no pixels")

            # Complete one deterministic generation through the real
            # coordinator/poll/result-mapping path.
            layer_id = application.layer_stack.active_layer.id
            root.layer_tree_coordinator.handle_intent(LayerTreeIntent(
                LayerTreeAction.ATTACH_TOOL,
                layer_id=layer_id,
                value="diffusion",
            ))
            root.generation_panels_coordinator.handle_intent(
                GenerationIntent(GenerationAction.SET_SEED, "4242"))
            root.generation_panels_coordinator.handle_intent(
                GenerationIntent(GenerationAction.RUN))
            if (
                    root.generation_panels_coordinator.state.diffusion.phase
                    != GenerationPhase.RUNNING):
                raise RuntimeError("fake native generation did not start")
            rendered += int(root.tick().rendered)
            if (
                    root.generation_panels_coordinator.state.diffusion.phase
                    != GenerationPhase.RESULT):
                raise RuntimeError("fake native generation did not complete")
            if not application.status_text.startswith("Regenerated"):
                raise RuntimeError(
                    "fake native generation result was not mapped")

            root.defer(
                lambda: application.set_status("Native dispatcher ready"))
            first = root.tick()
            rendered += int(first.rendered)

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

            resize = getattr(root.composition, "resize", None)
            if resize is not None:
                resize(800, 480)

            for _ in range(args.frames - 1):
                result = root.tick()
                rendered += int(result.rendered)
                if not application.running:
                    break

            read_frame = getattr(
                root.composition, "read_frame_rgba_float", None)
            if read_frame is not None:
                frame = read_frame()
                if frame.shape != (480, 800, 4):
                    raise RuntimeError(
                        f"native resize produced frame {frame.shape}")
                if not np.all(np.isfinite(frame)) or np.ptp(frame) < 0.05:
                    raise RuntimeError("native composition framebuffer is blank")

            if args.interactive:
                print(
                    "Native manual QA window ready; close the window to finish.",
                    flush=True,
                )
                while application.running:
                    root.tick()
                    time.sleep(0.01)

        if root.canvas.image_lease.ownership != DynamicTextureOwnership.RELEASED:
            raise RuntimeError("native Canvas image texture leaked")
        if root.canvas.overlay_lease.ownership != DynamicTextureOwnership.RELEASED:
            raise RuntimeError("native Canvas overlay texture leaked")
        root.close()
        try:
            root.tick()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("closed native root accepted another frame")

    if rendered == 0:
        raise RuntimeError("native root did not render a frame")
    print(
        f"Diffusion Editor native root smoke OK: "
        f"{rendered} frame(s), {'windowed' if args.windowed else args.backend}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
