#!/usr/bin/env python3
"""Build contact sheets and silhouette metrics for a TRELLIS.2 FlowEdit run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


VIEWS = (0, 2, 4, 6, 8, 10, 12, 14)


def _edit_title(root: Path) -> str:
    manifest = root / "manifest.json"
    if manifest.exists():
        mode = json.loads(manifest.read_text(encoding="utf-8")).get("edit_mode")
        if mode == "repaint":
            return "Shape RePaint"
    return "Shape FlowEdit"


def _write_upright_preview(source: Path, destination: Path) -> None:
    """Convert the editor preview convention to glTF's upright convention."""
    import trimesh

    scene = trimesh.load(source, force="scene", process=False)
    scene.apply_transform(np.asarray((
        (1, 0, 0, 0),
        (0, 0, -1, 0),
        (0, 1, 0, 0),
        (0, 0, 0, 1),
    ), dtype=np.float64))
    scene.export(destination)


def _write_viewer_ply(source: Path, destination: Path) -> None:
    import trimesh

    scene = trimesh.load(source, force="scene", process=False)
    mesh = trimesh.util.concatenate(tuple(scene.geometry.values()))
    destination.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(destination)


def _rgba(path: Path) -> Image.Image:
    return Image.open(path).convert("RGBA")


def _on_dark(path: Path, size: int) -> Image.Image:
    image = _rgba(path)
    background = Image.new("RGBA", image.size, (24, 24, 24, 255))
    background.alpha_composite(image)
    return ImageOps.contain(
        background.convert("RGB"),
        (size, size),
        method=Image.Resampling.LANCZOS,
    )


def _head(path: Path, size: int) -> Image.Image:
    image = _rgba(path)
    alpha = np.asarray(image)[..., 3]
    ys, xs = np.nonzero(alpha > 8)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    subject_height = y1 - y0 + 1
    bottom = int(y0 + subject_height * 0.34)
    half_width = max(int(subject_height * 0.19), (bottom - y0) // 2)
    center_x = (x0 + x1) // 2
    crop = image.crop((center_x - half_width, y0, center_x + half_width, bottom))
    background = Image.new("RGBA", crop.size, (24, 24, 24, 255))
    background.alpha_composite(crop)
    return ImageOps.fit(
        background.convert("RGB"),
        (size, size),
        method=Image.Resampling.LANCZOS,
    )


def _condition_subject(path: Path, size: int, *, head: bool) -> Image.Image:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image)
    foreground = array.max(axis=2) > 8
    ys, xs = np.nonzero(foreground)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    if head:
        height = y1 - y0 + 1
        y1 = int(y0 + 0.34 * height)
        half_width = max(int(height * 0.19), (y1 - y0) // 2)
        center_x = (x0 + x1) // 2
        x0, x1 = center_x - half_width, center_x + half_width
    crop = image.crop((x0, y0, x1 + 1, y1 + 1))
    return ImageOps.contain(
        crop,
        (size, size),
        method=Image.Resampling.LANCZOS,
    )


def _target_comparison(root: Path, output: Path) -> None:
    cell = 420
    label = 32
    canvas = Image.new("RGB", (3 * cell, 2 * (cell + label)), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    items = (
        (root / "target-preprocessed.png", "Target condition", True),
        (root / "render-source" / "image_ori" / "004.png", "TRELLIS.2 source", False),
        (root / "render-edited" / "image_ori" / "004.png", _edit_title(root), False),
    )
    for column, (path, title, condition) in enumerate(items):
        for row, head in enumerate((False, True)):
            image = (
                _condition_subject(path, cell, head=head)
                if condition else
                (_head(path, cell) if head else _on_dark(path, cell))
            )
            x, y = column * cell, row * (cell + label)
            canvas.paste(image, (x + (cell - image.width) // 2, y + label))
            draw.text((x + 8, y + 8), title + (" · head" if head else ""), fill="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def _sheet(root: Path, output: Path, *, heads: bool) -> None:
    cell = 320
    label = 30
    canvas = Image.new("RGB", (4 * cell, 4 * (cell + label)), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    for slot, view in enumerate(VIEWS):
        column = slot % 4
        pair = slot // 4
        for variant_row, (folder, title) in enumerate((
            ("render-source", "TRELLIS.2 source"),
            ("render-edited", _edit_title(root)),
        )):
            row = pair * 2 + variant_row
            x, y = column * cell, row * (cell + label)
            path = root / folder / "image_ori" / f"{view:03d}.png"
            image = _head(path, cell) if heads else _on_dark(path, cell)
            canvas.paste(image, (x, y + label))
            draw.text((x + 8, y + 8), f"{title} · view {view:03d}", fill="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def _metrics(root: Path) -> dict:
    views = []
    for view in range(16):
        source = np.asarray(_rgba(
            root / "render-source" / "image_ori" / f"{view:03d}.png"
        ))
        edited = np.asarray(_rgba(
            root / "render-edited" / "image_ori" / f"{view:03d}.png"
        ))
        source_mask = source[..., 3] > 8
        edited_mask = edited[..., 3] > 8
        union = source_mask | edited_mask
        ys, _ = np.nonzero(union)
        top, bottom = ys.min(), ys.max()
        head_bottom = int(top + 0.34 * (bottom - top + 1))
        rows = np.indices(source_mask.shape)[0]

        def iou(region):
            intersection = np.count_nonzero(source_mask & edited_mask & region)
            area = np.count_nonzero(union & region)
            return intersection / area if area else 1.0

        views.append({
            "view": view,
            "head_silhouette_iou": iou(rows < head_bottom),
            "body_silhouette_iou": iou(rows >= head_bottom),
        })
    return {
        "schema": "diffusion-editor.trellis2-shape-flowedit-comparison",
        "schema_version": 1,
        "mean_head_silhouette_iou": float(np.mean([
            item["head_silhouette_iou"] for item in views
        ])),
        "mean_body_silhouette_iou": float(np.mean([
            item["body_silhouette_iou"] for item in views
        ])),
        "views": views,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    root = args.experiment.resolve()
    comparison = root / "comparison"
    _write_upright_preview(
        root / "source-shape.glb",
        root / "source-shape-upright.glb",
    )
    _write_upright_preview(
        root / "edited-shape.glb",
        root / "edited-shape-upright.glb",
    )
    _write_viewer_ply(
        root / "source-shape-upright.glb",
        comparison / "viewer" / "01-source-shape.ply",
    )
    _write_viewer_ply(
        root / "edited-shape-upright.glb",
        comparison / "viewer" / "02-edited-shape.ply",
    )
    if args.prepare_only:
        return 0
    _sheet(root, comparison / "source-vs-edit-orbit.png", heads=False)
    _sheet(root, comparison / "source-vs-edit-heads.png", heads=True)
    _target_comparison(root, comparison / "target-vs-source-vs-edit.png")
    metrics = _metrics(root)
    (comparison / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
