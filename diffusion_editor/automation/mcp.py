"""Termin MCP adapter for a running Diffusion Editor process."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any
import uuid

from tcbase import log
from termin.mcp import PythonScriptExecutor, TerminMcpServer, create_secure_mcp_config

from ..sdk_runtime import PROJECT_ROOT, resolve_sdk


_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off", ""})


def editor_mcp_enabled(settings: Any | None = None) -> bool:
    """Return the explicit local-automation preference.

    The Termin-compatible environment override wins. Without it, the hidden
    application setting remains disabled by default because this endpoint can
    execute arbitrary Python in the editor process.
    """

    value = os.environ.get("TERMIN_EDITOR_MCP")
    if value is not None:
        normalized = value.strip().lower()
        if normalized in _TRUE_VALUES:
            return True
        if normalized in _FALSE_VALUES:
            return False
        log.warning(
            "[DiffusionEditorMCP] invalid TERMIN_EDITOR_MCP value "
            f"{value!r}; treating it as disabled"
        )
        return False
    if settings is None:
        return False
    return bool(settings.get("mcp_server_enabled", False))


def canonical_sdk_root(sdk_root: str | Path | None = None) -> Path:
    candidate = resolve_sdk() if sdk_root is None else Path(sdk_root)
    return Path(os.path.normcase(str(candidate.expanduser().resolve(strict=False))))


def default_editor_mcp_registry_dir(
    *,
    sdk_root: str | Path | None = None,
    temp_dir: str | Path | None = None,
) -> Path:
    """Match the registry used by Termin's editor MCP CLI."""

    root = canonical_sdk_root(sdk_root)
    identity = hashlib.sha256(os.fsencode(str(root))).hexdigest()[:16]
    temporary = Path(tempfile.gettempdir() if temp_dir is None else temp_dir)
    return temporary / f"termin-editor-mcp-{identity}" / "sessions"


def new_editor_mcp_session_file(
    *,
    sdk_root: str | Path | None = None,
    temp_dir: str | Path | None = None,
) -> Path:
    return default_editor_mcp_registry_dir(
        sdk_root=sdk_root,
        temp_dir=temp_dir,
    ) / f"{uuid.uuid4().hex}.json"


class DiffusionEditorMcpServer(TerminMcpServer):
    def __init__(
        self,
        executor: PythonScriptExecutor,
        *,
        project_path: Path,
        sdk_root: Path,
        config,
    ) -> None:
        super().__init__(
            executor,
            config,
            log_prefix="DiffusionEditorMCP",
            server_name="diffusion-editor",
            server_version="DiffusionEditorMCP/0.1",
            thread_name="diffusion-editor-mcp",
        )
        self._project_path = project_path
        self._sdk_root = sdk_root

    def _session_payload(self) -> dict[str, object]:
        payload = super()._session_payload()
        payload["project_path"] = str(self._project_path)
        payload["sdk_root"] = str(self._sdk_root)
        return payload


class DiffusionEditorAutomation:
    """Own the main-thread executor and its authenticated MCP endpoint."""

    def __init__(
        self,
        root: Any,
        *,
        sdk_root: str | Path | None = None,
        project_path: str | Path = PROJECT_ROOT,
        session_file: str | Path | None = None,
        host: object | None = None,
        port: object | None = None,
        token: object | None = None,
    ) -> None:
        self._root = root
        self._closed = False
        resolved_sdk = canonical_sdk_root(sdk_root)
        resolved_session = Path(
            session_file
            or os.environ.get("TERMIN_EDITOR_MCP_SESSION_FILE")
            or new_editor_mcp_session_file(sdk_root=resolved_sdk)
        )
        config = create_secure_mcp_config(
            host=(
                os.environ.get("TERMIN_EDITOR_MCP_HOST", "127.0.0.1")
                if host is None else host
            ),
            port=(
                os.environ.get("TERMIN_EDITOR_MCP_PORT", "0")
                if port is None else port
            ),
            token=(os.environ.get("TERMIN_EDITOR_MCP_TOKEN") if token is None else token),
            session_file=resolved_session,
            default_host="127.0.0.1",
            default_port=0,
            log_prefix="DiffusionEditorMCP",
        )
        self.executor = PythonScriptExecutor(
            self._script_context,
            log_prefix="DiffusionEditorPythonExecutor",
            compile_filename="<diffusion-editor-mcp>",
        )
        self.server = DiffusionEditorMcpServer(
            self.executor,
            project_path=Path(project_path).expanduser().resolve(strict=False),
            sdk_root=resolved_sdk,
            config=config,
        )
        self.session_file = resolved_session
        self.server.start()

    @property
    def url(self) -> str:
        return self.server.url

    def process_pending(self, *, limit: int = 8) -> int:
        if self._closed:
            return 0
        return self.executor.process_pending(limit=limit)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.executor.close()
        self.server.stop()
        self._root = None

    def _script_context(self) -> dict[str, object | None]:
        root = self._root
        if root is None:
            raise RuntimeError("Diffusion Editor automation is closed")
        application = root.application
        request_repaint = root.composition.request_repaint
        return {
            "application": application,
            "editor": root,
            "root": root,
            "document": application.document,
            "document_service": application.document,
            "layer_stack": application.layer_stack,
            "composition": root.composition,
            "ui_document": root.composition.document,
            "view": root.view,
            "request_render_update": request_repaint,
            "refresh_editor": request_repaint,
            "request_editor_close": application.request_stop,
        }


def start_editor_automation(root: Any) -> DiffusionEditorAutomation | None:
    settings = getattr(root.application, "settings", None)
    if not editor_mcp_enabled(settings):
        return None
    try:
        return DiffusionEditorAutomation(root)
    except OSError as exc:
        log.error(f"[DiffusionEditorMCP] failed to start server: {exc}")
        return None
