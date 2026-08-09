# Live editor MCP automation

Diffusion Editor can expose Termin's authenticated local MCP endpoint and run
Python scripts on the native editor thread. The endpoint is disabled by default
because scripts have the same permissions as the editor process.

Enable **Local editor MCP server** in **Edit → Settings**, then restart
Diffusion Editor. The setting is persistent and remains disabled by default.

For a one-launch override, set `TERMIN_EDITOR_MCP`. It takes precedence over
the saved setting, so `0` explicitly disables the endpoint and `1` enables it:

```bash
TERMIN_EDITOR_MCP=1 ./run.sh [project.deproj-or-image]
```

The editor registers itself in the same SDK-scoped session registry as Termin
Editor. Termin's existing helper discovers it by project path:

```bash
/home/mirmik/project/termin/scripts/termin-editor-mcp \
  --project /home/mirmik/project/diffusion-editor sessions
/home/mirmik/project/termin/scripts/termin-editor-mcp \
  --project /home/mirmik/project/diffusion-editor \
  exec 'print(layer_stack.width, layer_stack.height)'
```

Use `exec-file PATH` for a multiline script. `execute_python_script` is the
supported MCP tool for this host. The namespace is refreshed before every run
and contains:

- `application`, `editor` / `root`, `view`, and `composition`;
- `document` / `document_service`, `layer_stack`, and `ui_document`;
- `request_render_update()` / `refresh_editor()`;
- `request_editor_close()`.

Scripts are received on the endpoint's worker thread, queued, and executed by
`NativeEditorRoot.tick()` on the editor thread. A clean shutdown rejects queued
work, stops the endpoint, and removes only its owned session descriptor.

For a project-scoped Codex MCP connection, configure Termin's stdio broker:

```toml
[mcp_servers.termin_editor]
command = "/home/mirmik/project/termin/scripts/termin-editor-mcp"
args = ["--project", "/home/mirmik/project/diffusion-editor", "serve"]
startup_timeout_sec = 10
tool_timeout_sec = 60
default_tools_approval_mode = "approve"
```

The usual Termin overrides remain available:
`TERMIN_EDITOR_MCP_HOST`, `TERMIN_EDITOR_MCP_PORT`,
`TERMIN_EDITOR_MCP_TOKEN`, and `TERMIN_EDITOR_MCP_SESSION_FILE`.
