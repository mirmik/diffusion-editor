#!/usr/bin/env python3
"""Build a labeled contact sheet from inventory preview directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("preview_root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--columns", type=int, default=3)
    args = parser.parse_args()
    if args.columns <= 0:
        parser.error("--columns must be positive")
    return args


def main() -> int:
    args = _arguments()
    records = []
    for manifest_path in sorted(args.preview_root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        views = [Image.open(manifest_path.parent / name).convert("RGBA")
                 for name in manifest["rendered"]]
        records.append((manifest_path.parent.name, manifest, views))
    if not records:
        raise SystemExit("no inventory previews found")

    font = ImageFont.load_default(size=16)
    small = ImageFont.load_default(size=12)
    view_size = 150
    tile_width = view_size * 2
    tile_height = view_size * 2 + 54
    rows = (len(records) + args.columns - 1) // args.columns
    sheet = Image.new(
        "RGB", (args.columns * tile_width, rows * tile_height), (25, 28, 34)
    )
    draw = ImageDraw.Draw(sheet)
    for index, (name, manifest, views) in enumerate(records):
        left = index % args.columns * tile_width
        top = index // args.columns * tile_height
        for view_index, view in enumerate(views):
            thumbnail = Image.new("RGBA", (view_size, view_size), (10, 12, 16, 255))
            fitted = view.copy()
            fitted.thumbnail((view_size, view_size), Image.Resampling.LANCZOS)
            thumbnail.alpha_composite(
                fitted,
                ((view_size - fitted.width) // 2, (view_size - fitted.height) // 2),
            )
            x = left + view_index % 2 * view_size
            y = top + view_index // 2 * view_size
            sheet.paste(thumbnail.convert("RGB"), (x, y))
        draw.text((left + 6, top + 304), name, fill=(240, 244, 250), font=font)
        detail = (
            f"meshes {manifest['mesh_count']} · "
            f"rigs {len(manifest['armatures'])}"
        )
        draw.text((left + 6, top + 330), detail, fill=(145, 165, 190), font=small)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
