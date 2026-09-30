"""Source-backed SVG layers. Pixels are a disposable, straight RGBA8 cache."""
from __future__ import annotations

from dataclasses import dataclass
import io
import logging
import math
import re
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import resvg_py

from .layer import Layer
from .mask import Mask
from .tiles import DenseTileGrid

logger = logging.getLogger(__name__)
SVG_NS = "http://www.w3.org/2000/svg"
MAX_SVG_BYTES = 4 * 1024 * 1024
MAX_SVG_PIXELS = 64 * 1024 * 1024
MAX_SVG_DIMENSION = 32768
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")


def parse_svg(source: str) -> ET.Element:
    """Accept a self-contained static SVG; never resolve filesystem resources."""
    if not isinstance(source, str) or len(source.encode("utf-8")) > MAX_SVG_BYTES:
        raise ValueError("SVG source must be text no larger than 4 MiB")
    if re.search(r"<!\s*(DOCTYPE|ENTITY)", source, re.I):
        raise ValueError("SVG DTDs and entities are not supported")
    try:
        root = ET.fromstring(source)
    except ET.ParseError as exc:
        raise ValueError(f"Invalid SVG XML: {exc}") from exc
    if root.tag != f"{{{SVG_NS}}}svg":
        raise ValueError("Expected an SVG root with the SVG namespace")
    ids: set[str] = set()
    for node in root.iter():
        tag = node.tag.rsplit("}", 1)[-1]
        if tag in {"script", "foreignObject", "animate", "animateMotion", "animateTransform", "set"}:
            raise ValueError(f"SVG element {tag!r} is not supported in static layers")
        identifier = node.get("id")
        if identifier:
            if identifier in ids:
                raise ValueError(f"Duplicate SVG id: {identifier}")
            ids.add(identifier)
        for key, value in node.attrib.items():
            local = key.rsplit("}", 1)[-1]
            if local.lower().startswith("on") or local == "base":
                raise ValueError(f"SVG attribute {local!r} is not supported")
            if local == "href" and not (
                value.startswith("#") or
                (tag == "image" and re.match(r"data:image/(png|jpeg|webp);base64,", value))
            ):
                raise ValueError("SVG resources must be embedded images or #id references")
            _validate_css(value)
        if tag == "style":
            _validate_css(node.text or "")
    return root


def _validate_css(value: str) -> None:
    # Reject escaped URL spelling, imports and external CSS resources. Local
    # paint servers (gradients, clip paths, filters) remain fully supported.
    if "\\" in value or re.search(r"@import", value, re.I):
        raise ValueError("External or escaped SVG CSS resources are not supported")
    for match in re.finditer(r"url\s*\((.*?)\)", value, re.I | re.S):
        target = match.group(1).strip().strip("\"'")
        if not target.startswith("#"):
            raise ValueError("SVG CSS URLs must reference an internal #id")


def _length(value: str | None) -> float | None:
    if value is None:
        return None
    match = re.fullmatch(r"\s*(\d+(?:\.\d*)?|\.\d+)(px|mm|cm|in|pt|pc)?\s*", value)
    if match is None:
        return None
    factor = {None: 1, "px": 1, "mm": 96 / 25.4, "cm": 96 / 2.54,
              "in": 96, "pt": 96 / 72, "pc": 16}[match.group(2)]
    return float(match.group(1)) * factor


def svg_size(source: str, fallback: tuple[int, int]) -> tuple[int, int]:
    root = parse_svg(source)
    viewbox = root.get("viewBox", "").replace(",", " ").split()
    view_width, view_height = fallback
    if viewbox:
        if len(viewbox) != 4:
            raise ValueError("SVG viewBox must contain four numbers")
        values = [float(v) for v in viewbox]
        if not all(math.isfinite(v) for v in values) or min(values[2:]) <= 0:
            raise ValueError("SVG viewBox must be finite with positive dimensions")
        view_width, view_height = values[2:]
    width = math.ceil(_length(root.get("width")) or view_width)
    height = math.ceil(_length(root.get("height")) or view_height)
    validate_size(width, height)
    return width, height


def validate_size(width: int, height: int) -> None:
    if (type(width) is not int or type(height) is not int or
            min(width, height) < 1 or max(width, height) > MAX_SVG_DIMENSION or
            width * height > MAX_SVG_PIXELS):
        raise ValueError("SVG viewport must be positive integers within 32768 px / 64 megapixels")


@dataclass(frozen=True)
class SvgContent:
    source: str
    pixels: np.ndarray

    @property
    def size_bytes(self) -> int:
        return len(self.source.encode("utf-8")) + self.pixels.nbytes


def render_svg(source: str, width: int, height: int) -> SvgContent:
    try:
        validate_size(width, height)
        root = parse_svg(source)
        natural = svg_size(source, (width, height))
        if "viewBox" not in root.attrib:
            root.set("viewBox", f"0 0 {natural[0]} {natural[1]}")
        root.set("width", str(width))
        root.set("height", str(height))
        png = resvg_py.svg_to_bytes(svg_string=ET.tostring(root, encoding="unicode"))
        with Image.open(io.BytesIO(png)) as image:
            if image.size != (width, height):
                raise ValueError("SVG renderer returned an unexpected viewport")
            pixels = np.array(image.convert("RGBA"), dtype=np.uint8)
        pixels.flags.writeable = False
        return SvgContent(source, pixels)
    except Exception:
        logger.exception("Failed to render SVG layer (%s x %s)", width, height)
        raise


class SvgLayer(Layer):
    node_type = "svg"
    accepts_pixel_edits = False

    def __init__(self, name: str, source: str, width: int, height: int,
                 *, layer_id: str | None = None, x: int = 0, y: int = 0):
        content = render_svg(source, width, height)
        super().__init__(name, width, height, content.pixels,
                         layer_id=layer_id, x=x, y=y)
        self._svg_content = content
        self.image.flags.writeable = False

    @property
    def svg_source(self) -> str:
        return self._svg_content.source

    @property
    def svg_content(self) -> SvgContent:
        return self._svg_content

    def _install_svg_content(self, content: SvgContent) -> None:
        """Only document commands call this, then publish cache invalidation."""
        self.content = DenseTileGrid.from_array(content.pixels, tile_size=self.content.tile_size)
        self.image = self.content.array
        self.image.flags.writeable = False
        if self.mask.data.shape != self.image.shape[:2]:
            self.mask = Mask.zeros(self.height, self.width)
        self._svg_content = content

    def to_dict(self, path: str) -> dict:
        result = super().to_dict(path)
        result["type"] = "svg"
        result["svg_file"] = f"layers/{path.replace('/', '_')}.svg"
        return result

    def save_images_to_zip(self, archive, path: str) -> None:
        super().save_images_to_zip(archive, path)
        archive.writestr(f"layers/{path.replace('/', '_')}.svg", self.svg_source)

    @classmethod
    def from_dict(cls, data: dict, archive, tile_size: int = 256) -> SvgLayer:
        entry = data["svg_file"]
        if archive.getinfo(entry).file_size > MAX_SVG_BYTES:
            raise ValueError("SVG source exceeds 4 MiB")
        source = archive.read(entry).decode("utf-8")
        layer = cls._from_dict_base(data, archive, tile_size=tile_size)
        if layer.tool is not None or not layer.mask.is_empty:
            raise ValueError("SVG layers cannot contain pixel tools or masks")
        # Regenerate from source, never promote a saved preview to authority.
        layer._install_svg_content(render_svg(source, layer.width, layer.height))
        return layer
