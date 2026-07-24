"""Hostless termin-gui-native composition for the editor migration path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from tcbase import log
from termin.dispatch import Dispatcher, DispatchStats
from termin.display import WindowHandle, WindowManager, WindowedGraphicsSession
from termin.gui_native import (
    GuiWindowAdapter,
    OffscreenGuiComposition,
    TcDocument,
    tc_ui_document_create,
    tc_ui_document_destroy,
)
from tgfx import configure_default_shader_runtime

from ..sdk_runtime import resolve_sdk
from .application import EditorApplication
from .native_shell import CommandHandler, NativeEditorView


DEFAULT_NATIVE_WIDTH = 1280
DEFAULT_NATIVE_HEIGHT = 800
DEFAULT_DISPATCH_LIMIT = 256
DEFAULT_FONT_RELATIVE_PATH = Path("share/termin/fonts/DroidSans.ttf")


class NativeComposition(Protocol):
    """Rendering/input owner consumed by :class:`NativeEditorRoot`."""

    document: TcDocument

    @property
    def should_close(self) -> bool: ...

    def pump_events(self) -> int: ...

    def request_repaint(self) -> None: ...

    def render_frame(self) -> bool: ...

    def close(self) -> None: ...


class NativeViewFactory(Protocol):
    def __call__(
            self,
            document: TcDocument,
            request_repaint: Callable[[], None],
            set_window_title: Callable[[str], None],
            command_handlers: dict[str, CommandHandler]) -> NativeEditorView: ...


@dataclass(frozen=True)
class NativeTickResult:
    dispatched: int
    dispatch_failed: int
    dispatch_remaining: int
    events: int
    rendered: bool


def bundled_native_font_path(sdk_root: Path | None = None) -> Path:
    root = resolve_sdk() if sdk_root is None else sdk_root.expanduser().resolve()
    font_path = root / DEFAULT_FONT_RELATIVE_PATH
    if not font_path.is_file():
        raise RuntimeError(f"Termin SDK native UI font is missing: {font_path}")
    return font_path


class WindowedNativeComposition:
    """Application-owned window/session plus the borrowed native GUI adapter."""

    def __init__(
            self,
            *,
            title: str = "Diffusion Editor (native migration)",
            width: int = DEFAULT_NATIVE_WIDTH,
            height: int = DEFAULT_NATIVE_HEIGHT,
            sdk_root: Path | None = None,
            font_size: int = 14) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("native window dimensions must be positive")

        self._closed = False
        self._session: WindowedGraphicsSession | None = None
        self._manager: WindowManager | None = None
        self._handle: WindowHandle | None = None
        self._document: TcDocument | None = None
        self._adapter: GuiWindowAdapter | None = None

        root = resolve_sdk() if sdk_root is None else sdk_root.expanduser().resolve()
        if not configure_default_shader_runtime("diffusion-editor-native"):
            raise RuntimeError("Termin shader runtime is unavailable")

        try:
            self._session = WindowedGraphicsSession.create_native()
            self._manager = WindowManager(self._session)
            self._document = tc_ui_document_create()
            self._handle = self._manager.create_window(title, width, height)
            self._adapter = GuiWindowAdapter(
                self._manager,
                self._handle,
                self._document,
                font_path=str(bundled_native_font_path(root)),
                font_size=font_size,
            )
        except Exception:
            self.close()
            raise

    @property
    def document(self) -> TcDocument:
        if self._document is None or not self._document.valid:
            raise RuntimeError("native window document is closed")
        return self._document

    @property
    def should_close(self) -> bool:
        return self._adapter is None or self._adapter.should_close

    def pump_events(self) -> int:
        if self._manager is None or self._adapter is None or self._handle is None:
            raise RuntimeError("native window composition is closed")
        self._manager.pump_events()
        return self._adapter.consume_pending_events(self._manager, self._handle)

    def request_repaint(self) -> None:
        if self._adapter is None:
            raise RuntimeError("native window composition is closed")
        self._adapter.request_repaint()

    def set_window_title(self, title: str) -> None:
        if self._manager is None or self._handle is None:
            raise RuntimeError("native window composition is closed")
        self._manager.window(self._handle).set_title(title)

    def render_frame(self) -> bool:
        if self._adapter is None:
            raise RuntimeError("native window composition is closed")
        return bool(self._adapter.render_frame())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        adapter, self._adapter = self._adapter, None
        document, self._document = self._document, None
        manager, self._manager = self._manager, None
        handle, self._handle = self._handle, None
        session, self._session = self._session, None

        if adapter is not None:
            self._close_step("native adapter idle", adapter.wait_idle)
            self._close_step("native adapter", adapter.close)
        if document is not None and document.valid:
            self._close_step(
                "native document",
                lambda: tc_ui_document_destroy(document),
            )
        if manager is not None and handle is not None and manager.contains(handle):
            self._close_step(
                "native window",
                lambda: manager.destroy_window(handle),
            )
        if manager is not None:
            self._close_step("native window manager", manager.close)
        if session is not None:
            self._close_step("native graphics session", session.close)

    @staticmethod
    def _close_step(name: str, callback: Callable[[], Any]) -> None:
        try:
            callback()
        except Exception:
            log.exception(f"Failed to close {name}")


class NativeEditorRoot:
    """Own the native view, dispatcher phase and one rendering composition."""

    def __init__(
            self,
            application: EditorApplication,
            composition: NativeComposition,
            *,
            dispatcher: Dispatcher | None = None,
            dispatch_limit: int = DEFAULT_DISPATCH_LIMIT,
            command_handlers: Mapping[str, CommandHandler] | None = None,
            view_factory: NativeViewFactory = NativeEditorView) -> None:
        if dispatch_limit <= 0:
            raise ValueError("dispatch_limit must be positive")
        if application.closed:
            raise RuntimeError("cannot bind a closed editor application")

        self.application = application
        self.composition = composition
        self.dispatcher = dispatcher if dispatcher is not None else Dispatcher()
        self.dispatch_limit = dispatch_limit
        self.closed = False
        self.discarded_on_close = 0
        handlers: dict[str, CommandHandler] = {
            "app.quit": application.request_stop,
        }
        if command_handlers is not None:
            handlers.update(command_handlers)

        try:
            self.view = view_factory(
                composition.document,
                composition.request_repaint,
                self._set_window_title,
                handlers,
            )
            application.bind_view(self.view.ports())
            composition.request_repaint()
        except Exception:
            self.dispatcher.close()
            self.dispatcher.discard_pending()
            composition.close()
            raise

    @classmethod
    def create_headless(
            cls,
            application: EditorApplication,
            *,
            width: int = 320,
            height: int = 200,
            backend: str = "vulkan",
            dispatch_limit: int = DEFAULT_DISPATCH_LIMIT,
            command_handlers: Mapping[str, CommandHandler] | None = None,
            sdk_root: Path | None = None) -> "NativeEditorRoot":
        root = resolve_sdk() if sdk_root is None else sdk_root.expanduser().resolve()
        composition = OffscreenGuiComposition(
            width=width,
            height=height,
            backend=backend,
            font_path=str(bundled_native_font_path(root)),
            continuous_rendering=False,
            sdk_root=str(root),
        )
        return cls(
            application,
            composition,
            dispatch_limit=dispatch_limit,
            command_handlers=command_handlers,
        )

    @classmethod
    def create_windowed(
            cls,
            application: EditorApplication,
            *,
            title: str = "Diffusion Editor (native migration)",
            width: int = DEFAULT_NATIVE_WIDTH,
            height: int = DEFAULT_NATIVE_HEIGHT,
            dispatch_limit: int = DEFAULT_DISPATCH_LIMIT,
            command_handlers: Mapping[str, CommandHandler] | None = None,
            sdk_root: Path | None = None) -> "NativeEditorRoot":
        composition = WindowedNativeComposition(
            title=title,
            width=width,
            height=height,
            sdk_root=sdk_root,
        )
        return cls(
            application,
            composition,
            dispatch_limit=dispatch_limit,
            command_handlers=command_handlers,
        )

    def _set_window_title(self, title: str) -> None:
        setter = getattr(self.composition, "set_window_title", None)
        if setter is not None:
            setter(title)

    def defer(self, callback: Callable[[], None]):
        if self.closed:
            raise RuntimeError("native editor root is closed")
        return self.dispatcher.defer(callback)

    def tick(self) -> NativeTickResult:
        if self.closed:
            raise RuntimeError("native editor root is closed")

        events = int(self.composition.pump_events())
        stats: DispatchStats = self.dispatcher.run_pending(self.dispatch_limit)
        if stats.busy or stats.internal_error:
            raise RuntimeError("native dispatcher failed to drain cleanly")
        if self.composition.should_close:
            self.application.request_stop()
        if self.application.running:
            self.application.poll()
        rendered = bool(self.composition.render_frame())
        return NativeTickResult(
            dispatched=stats.executed,
            dispatch_failed=stats.failed,
            dispatch_remaining=stats.remaining,
            events=events,
            rendered=rendered,
        )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True

        # Application close joins every registered producer before queued UI
        # callbacks are discarded and the document/view lifetime ends.
        self.application.close()
        self.dispatcher.close()
        self.discarded_on_close = self.dispatcher.discard_pending()
        self.view.close()
        self.composition.close()

    def __enter__(self) -> "NativeEditorRoot":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> bool:
        self.close()
        return False
