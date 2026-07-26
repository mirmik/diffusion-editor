"""Validation contracts published by the editor's agent tools."""

from diffusion_editor.agent import tools


class _RecordingRegistry:
    def __init__(self):
        self.schemas = {}

    def register(self, name, _callback, schema):
        self.schemas[name] = schema


def test_draw_grid_schema_requires_positive_section_counts(monkeypatch):
    monkeypatch.setattr(tools, "ToolRegistry", _RecordingRegistry)

    registry = tools.create_editor_tool_registry()
    properties = registry.schemas["draw_grid"]["function"]["parameters"][
        "properties"]

    assert properties["sections_x"]["minimum"] == 1
    assert properties["sections_y"]["minimum"] == 1


def test_draw_grid_tool_rejects_invalid_sections_before_execute():
    class Document:
        def execute(self, _command):
            raise AssertionError("invalid grid must not reach the document")

    class Stack:
        active_layer = object()

    result = tools._tool_draw_grid(
        {"sections_x": 2, "sections_y": 0},
        {
            "_document_service": Document(),
            "_layer_stack": Stack(),
        },
    )

    assert result == "sections_x and sections_y must both be at least 1."
