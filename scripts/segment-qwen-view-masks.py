#!/usr/bin/env python3
"""Create persistent neural foreground masks for the nine Qwen views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffusion_editor.workers.segmentation_process import (
    SegmentationProcessClient,
)
from qwen_view_names import VIEW_PATTERN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=("rembg", "threshold"), default="rembg",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser.parse_args()


def _thumbnail(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    result = Image.new("RGB", size, (24, 24, 24))
    copy = image.convert("RGB")
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    result.paste(
        copy,
        ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2),
    )
    return result


def _write_contact_sheet(
    rows: list[tuple[str, Image.Image, Image.Image]],
    target: Path,
    model_label: str,
) -> None:
    cell_width = 256
    cell_height = 290
    image_height = 256
    sheet = Image.new(
        "RGB", (cell_width * 3, cell_height * len(rows)), "black",
    )
    draw = ImageDraw.Draw(sheet)
    for row_index, (name, source, mask) in enumerate(rows):
        y = row_index * cell_height
        source_rgb = source.convert("RGB")
        mask_l = mask.convert("L")
        cutout = Image.new("RGB", source_rgb.size, (24, 24, 24))
        cutout.paste(source_rgb, mask=mask_l)
        for column, (label, image) in enumerate((
            (name, source_rgb), (model_label, mask_l), ("cutout", cutout),
        )):
            x = column * cell_width
            draw.text((x + 4, y + 4), label, fill="white")
            sheet.paste(
                _thumbnail(image, (cell_width, image_height)),
                (x, y + cell_height - image_height),
            )
    sheet.save(target)


def main() -> int:
    args = parse_args()
    paths = [
        path for path in sorted(args.image_dir.iterdir())
        if VIEW_PATTERN.match(path.name) is not None
    ]
    if len(paths) != 9:
        raise SystemExit(f"expected 9 Qwen views, found {len(paths)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = SegmentationProcessClient(
        backend=args.backend,
        request_timeout=args.timeout,
    )
    started = time.monotonic()
    items = []
    contact_rows = []
    try:
        for path in paths:
            with Image.open(path) as opened:
                source = opened.convert("RGB")
            image = np.asarray(source, dtype=np.uint8)
            progress = []
            mask_array = client.segment(
                image,
                threading.Event(),
                on_progress=progress.append,
            )
            if mask_array.shape != image.shape[:2]:
                raise RuntimeError(
                    f"mask shape mismatch for {path.name}: "
                    f"{mask_array.shape} != {image.shape[:2]}"
                )
            mask = Image.fromarray(mask_array, "L")
            output = args.output_dir / path.name
            mask.save(output, format="PNG")
            item = {
                "name": path.name,
                "source": str(path.resolve()),
                "mask": str(output.resolve()),
                "width": source.width,
                "height": source.height,
                "foreground_fraction_soft": float(mask_array.mean() / 255.0),
                "foreground_fraction_50": float(np.mean(mask_array >= 128)),
                "progress": progress,
            }
            items.append(item)
            contact_rows.append((path.name, source, mask))
            print(json.dumps(item), flush=True)
    finally:
        client.shutdown()
    contact_sheet = args.output_dir / "contact-sheet.png"
    model_label = "U2-Net foreground" if args.backend == "rembg" else args.backend
    _write_contact_sheet(contact_rows, contact_sheet, model_label)
    manifest = {
        "schema": "diffusion-editor.qwen-neural-foreground-masks",
        "schema_version": 1,
        "backend": args.backend,
        "model": "u2net" if args.backend == "rembg" else None,
        "image_dir": str(args.image_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "contact_sheet": str(contact_sheet.resolve()),
        "elapsed_seconds": time.monotonic() - started,
        "views": items,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "views": len(items),
        "elapsed_seconds": manifest["elapsed_seconds"],
        "contact_sheet": str(contact_sheet),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
