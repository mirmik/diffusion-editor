from __future__ import annotations

import json
from pathlib import Path
import threading
import time
import urllib.request

from diffusion_editor.automation.mcp import (
    DiffusionEditorAutomation,
    default_editor_mcp_registry_dir,
    editor_mcp_enabled,
)


class _Application:
    def __init__(self) -> None:
        self.document = "document-service"
        self.layer_stack = "layer-stack"
        self.marker = "initial"
        self.running = True

    def request_stop(self) -> None:
        self.running = False


class _Composition:
    document = "ui-document"

    def __init__(self) -> None:
        self.repaint_requests = 0

    def request_repaint(self) -> None:
        self.repaint_requests += 1


class _Root:
    def __init__(self) -> None:
        self.application = _Application()
        self.composition = _Composition()
        self.view = "native-view"


def _call_endpoint(
    automation: DiffusionEditorAutomation,
    script: str,
) -> dict[str, object]:
    request = urllib.request.Request(
        automation.url,
        data=json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "execute_python_script",
                    "arguments": {"script": script, "timeout": 2.0},
                },
            }
        ).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer test-token",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=3.0) as response:
        return json.loads(response.read().decode("utf-8"))


def _call_while_pumping(
    automation: DiffusionEditorAutomation,
    script: str,
) -> dict[str, object]:
    received = []
    worker = threading.Thread(
        target=lambda: received.append(_call_endpoint(automation, script))
    )
    worker.start()
    deadline = time.monotonic() + 3.0
    while worker.is_alive() and time.monotonic() < deadline:
        automation.process_pending()
        time.sleep(0.001)
    worker.join(timeout=0.1)
    assert not worker.is_alive()
    return received[0]


def test_editor_mcp_environment_override_is_explicit(monkeypatch):
    settings = {"mcp_server_enabled": True}
    monkeypatch.delenv("TERMIN_EDITOR_MCP", raising=False)
    assert editor_mcp_enabled(settings)

    monkeypatch.setenv("TERMIN_EDITOR_MCP", "0")
    assert not editor_mcp_enabled(settings)
    monkeypatch.setenv("TERMIN_EDITOR_MCP", "yes")
    assert editor_mcp_enabled(None)


def test_registry_path_matches_termin_editor_cli_contract(tmp_path: Path):
    sdk = tmp_path / "sdk"
    first = default_editor_mcp_registry_dir(sdk_root=sdk, temp_dir=tmp_path)
    second = default_editor_mcp_registry_dir(sdk_root=sdk / ".", temp_dir=tmp_path)

    assert first == second
    assert first.parent.name.startswith("termin-editor-mcp-")
    assert first.name == "sessions"


def test_running_endpoint_executes_on_owner_thread_and_removes_session(tmp_path: Path):
    root = _Root()
    session_file = tmp_path / "session.json"
    owner_thread = threading.get_ident()
    automation = DiffusionEditorAutomation(
        root,
        sdk_root=tmp_path / "sdk",
        project_path=tmp_path / "project",
        session_file=session_file,
        token="test-token",
    )
    try:
        session = json.loads(session_file.read_text(encoding="utf-8"))
        assert session["project_path"] == str(tmp_path / "project")
        assert session["server"] == "diffusion-editor"
        assert session["tools"] == ["execute_python_script"]

        response = _call_while_pumping(
            automation,
            "import threading\n"
            "print(threading.get_ident())\n"
            "print(document, layer_stack, ui_document, view)\n"
            "application.marker = 'changed'\n"
            "request_render_update()\n",
        )
        result = response["result"]
        assert result["isError"] is False
        assert str(owner_thread) in result["content"][0]["text"]
        assert "document-service layer-stack ui-document native-view" in (
            result["content"][0]["text"]
        )
        assert root.application.marker == "changed"
        assert root.composition.repaint_requests == 1

        response = _call_while_pumping(
            automation,
            "request_editor_close()\n",
        )
        assert response["result"]["isError"] is False
        assert root.application.running is False
    finally:
        automation.close()

    assert not session_file.exists()


def test_script_error_is_returned_without_stopping_endpoint(tmp_path: Path):
    automation = DiffusionEditorAutomation(
        _Root(),
        sdk_root=tmp_path / "sdk",
        session_file=tmp_path / "session.json",
        token="test-token",
    )
    try:
        failed = _call_while_pumping(automation, "raise ValueError('broken')")
        assert failed["result"]["isError"] is True
        assert "ValueError: broken" in failed["result"]["structuredContent"]["error"]

        recovered = _call_while_pumping(automation, "print('still alive')")
        assert recovered["result"]["isError"] is False
        assert recovered["result"]["content"][0]["text"] == "still alive\n"
    finally:
        automation.close()
