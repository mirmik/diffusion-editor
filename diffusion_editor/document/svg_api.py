"""Shared UI / MCP API. All edits go through document history."""
from __future__ import annotations
import logging
from pathlib import Path
import xml.etree.ElementTree as ET

from .svg_commands import AddSvgLayerCommand, UpdateSvgLayerCommand, RasterizeSvgLayerCommand
from .svg_layer import SvgLayer, SVG_NS, MAX_SVG_BYTES, parse_svg, svg_size

logger = logging.getLogger(__name__)


class SvgDocumentApi:
    def __init__(self, document, stack):
        self._document = document
        self._stack = stack

    def layer(self, layer_id: str | None = None) -> SvgLayer:
        layer = self._stack.active_layer if layer_id is None else self._stack.find_layer_by_id(layer_id)
        if not isinstance(layer, SvgLayer):
            raise ValueError("Select an SVG layer or provide its layer id")
        return layer

    def add(self, source: str, name: str = "SVG", *, width: int | None = None,
            height: int | None = None) -> str:
        natural = svg_size(source, (self._stack.width, self._stack.height))
        self._document.execute(AddSvgLayerCommand(name, source,
            natural[0] if width is None else width,
            natural[1] if height is None else height))
        return self._stack.active_layer.id

    def new(self, name: str = "SVG") -> str:
        w, h = self._stack.width, self._stack.height
        return self.add(f'<svg xmlns="{SVG_NS}" width="{w}" height="{h}" viewBox="0 0 {w} {h}"/>', name)

    def import_file(self, path: str, *, name: str | None = None) -> str:
        source = self._read(path)
        return self.add(source, name or Path(path).stem)

    def replace_from_file(self, path: str, layer_id: str | None = None) -> None:
        self.update(self._read(path), layer_id)

    def source(self, layer_id: str | None = None) -> str:
        return self.layer(layer_id).svg_source

    def update(self, source: str, layer_id: str | None = None, *,
               width: int | None = None, height: int | None = None) -> None:
        self._document.execute(UpdateSvgLayerCommand(self.layer(layer_id), source, width, height))

    def elements(self, layer_id: str | None = None) -> list[dict]:
        return [{"id": node.get("id"), "tag": node.tag.rsplit("}", 1)[-1],
                 "attributes": dict(node.attrib), "text": node.text}
                for node in parse_svg(self.source(layer_id)).iter() if node.get("id")]

    def update_element(self, element_id: str, attributes: dict[str, str | None] | None = None,
                       *, text: str | None = None, layer_id: str | None = None) -> None:
        root = parse_svg(self.source(layer_id))
        node = next((n for n in root.iter() if n.get("id") == element_id), None)
        if node is None:
            raise ValueError(f"SVG element not found: {element_id}")
        for key, value in (attributes or {}).items():
            if value is None:
                node.attrib.pop(key, None)
            elif isinstance(value, str):
                node.set(key, value)
            else:
                raise TypeError("SVG attribute values must be strings or None")
        if text is not None:
            node.text = text
        self.update(ET.tostring(root, encoding="unicode"), layer_id)

    def move(self, x: int, y: int, layer_id: str | None = None) -> None:
        """Move the layer viewport in canvas pixels, preserving SVG coordinates."""
        from .svg_commands import MoveSvgLayerCommand
        self._document.execute(MoveSvgLayerCommand(self.layer(layer_id), x, y))

    def export_file(self, path: str, layer_id: str | None = None) -> None:
        try:
            Path(path).write_text(self.source(layer_id), encoding="utf-8")
        except Exception:
            logger.exception("Could not export SVG to %s", path)
            raise

    def rasterize(self, layer_id: str | None = None) -> None:
        self._document.execute(RasterizeSvgLayerCommand(self.layer(layer_id)))

    @staticmethod
    def _read(path: str) -> str:
        try:
            with Path(path).open("rb") as stream:
                content = stream.read(MAX_SVG_BYTES + 1)
            if len(content) > MAX_SVG_BYTES:
                raise ValueError("SVG source exceeds 4 MiB")
            return content.decode("utf-8-sig")
        except Exception:
            logger.exception("Could not read SVG from %s", path)
            raise
