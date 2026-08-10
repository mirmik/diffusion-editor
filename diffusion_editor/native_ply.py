"""Small PLY point-cloud reader for native reconstruction previews."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class NativePointCloudData:
    positions: np.ndarray
    colors: np.ndarray


_PLY_DTYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def load_ply_points(path: str | Path) -> NativePointCloudData:
    payload = Path(path).read_bytes()
    marker = b"end_header"
    marker_offset = payload.find(marker)
    if marker_offset < 0:
        raise ValueError("PLY header has no end_header marker")
    line_end = payload.find(b"\n", marker_offset)
    body_offset = len(payload) if line_end < 0 else line_end + 1
    header = payload[:body_offset].decode("ascii")
    lines = [line.strip() for line in header.splitlines()]
    if not lines or lines[0] != "ply":
        raise ValueError("Not a PLY file")

    encoding = ""
    vertex_count = 0
    vertex_properties: list[tuple[str, str]] = []
    current_element = ""
    for line in lines[1:]:
        parts = line.split()
        if not parts or parts[0] == "comment":
            continue
        if parts[0] == "format" and len(parts) >= 2:
            encoding = parts[1]
        elif parts[0] == "element" and len(parts) == 3:
            current_element = parts[1]
            if current_element == "vertex":
                vertex_count = int(parts[2])
        elif parts[0] == "property" and current_element == "vertex":
            if len(parts) != 3 or parts[1] == "list":
                raise ValueError("List properties are not supported on PLY vertices")
            vertex_properties.append((parts[2], parts[1]))

    required = {"x", "y", "z"}
    if vertex_count <= 0 or not required.issubset(name for name, _ in vertex_properties):
        raise ValueError("PLY contains no point vertices")

    if encoding == "binary_big_endian":
        raise ValueError("Big-endian PLY files are not supported")
    if encoding == "binary_little_endian":
        try:
            dtype = np.dtype([
                (name, _PLY_DTYPES[type_name])
                for name, type_name in vertex_properties
            ])
        except KeyError as exc:
            raise ValueError(f"Unsupported PLY property type: {exc.args[0]}") from exc
        values = np.frombuffer(payload, dtype=dtype, count=vertex_count, offset=body_offset)
        if len(values) != vertex_count:
            raise ValueError("Truncated PLY vertex payload")
        columns = {name: values[name] for name, _ in vertex_properties}
    elif encoding == "ascii":
        rows = payload[body_offset:].decode("ascii").splitlines()[:vertex_count]
        if len(rows) != vertex_count:
            raise ValueError("Truncated PLY vertex payload")
        table = np.asarray([[float(value) for value in row.split()] for row in rows])
        if table.shape != (vertex_count, len(vertex_properties)):
            raise ValueError("Invalid ASCII PLY vertex row")
        columns = {
            name: table[:, index]
            for index, (name, _) in enumerate(vertex_properties)
        }
    else:
        raise ValueError(f"Unsupported PLY encoding: {encoding or 'missing'}")

    positions = np.column_stack((columns["x"], columns["y"], columns["z"]))
    color_names = ("red", "green", "blue")
    if all(name in columns for name in color_names):
        colors = np.column_stack(tuple(columns[name] for name in color_names)).astype(
            np.float32
        )
        if colors.size and float(colors.max()) > 1.0:
            colors /= 255.0
    else:
        colors = np.ones((vertex_count, 3), dtype=np.float32)
    return NativePointCloudData(
        positions=np.ascontiguousarray(positions, dtype=np.float32),
        colors=np.ascontiguousarray(np.clip(colors, 0.0, 1.0), dtype=np.float32),
    )
