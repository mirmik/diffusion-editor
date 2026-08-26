#!/usr/bin/env python3
"""Build visual and numeric comparisons for an Easy3E experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


VIEWS = [4, 2, 0, 14, 12, 10, 8, 6]


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_dir", type=Path)
    return parser.parse_args()


def composite(path: Path, size: int = 320) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", image.size, (22, 22, 22, 255))
    bg.alpha_composite(image)
    result = bg.convert("RGB")
    result.thumbnail((size, size), Image.Resampling.LANCZOS)
    return result


def build_orbit(case_dir: Path, output: Path) -> None:
    cell = 320
    label = 30
    columns = 4
    sheet = Image.new("RGB", (columns * cell, 4 * (cell + label)), (22, 22, 22))
    draw = ImageDraw.Draw(sheet)
    for group, view in enumerate(VIEWS):
        column = group % columns
        block_row = group // columns
        for variant_row, (variant, title) in enumerate([
            ("control-roundtrip", "SLAT roundtrip"),
            ("edited", "Easy3E edit"),
        ]):
            row = block_row * 2 + variant_row
            x, y = column * cell, row * (cell + label)
            image = composite(case_dir / "comparison" / variant / "image" / f"{view:03d}.png", cell)
            sheet.paste(image, (x + (cell - image.width) // 2, y + label))
            draw.text((x + 7, y + 7), f"{title} · view {view:03d} · {view * 22.5:.1f}°", fill="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def head_crop(path: Path, size: int = 360) -> Image.Image:
    rgba = Image.open(path).convert("RGBA")
    alpha = np.asarray(rgba)[..., 3]
    ys, xs = np.nonzero(alpha > 8)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    subject_height = y1 - y0 + 1
    head_bottom = int(y0 + 0.29 * subject_height)
    head_width = max(int(subject_height * 0.29), head_bottom - y0)
    cx = (x0 + x1) // 2
    crop = rgba.crop((cx - head_width // 2, max(0, y0 - 4), cx + head_width // 2, head_bottom))
    bg = Image.new("RGBA", crop.size, (22, 22, 22, 255))
    bg.alpha_composite(crop)
    result = bg.convert("RGB")
    return ImageOps.fit(result, (size, size), method=Image.Resampling.LANCZOS)


def build_heads(case_dir: Path, output: Path, comparison_dir: str = "comparison") -> None:
    views = [4, 2, 0, 14, 12]
    cell = 360
    label = 30
    sheet = Image.new("RGB", (len(views) * cell, 2 * (cell + label)), (22, 22, 22))
    draw = ImageDraw.Draw(sheet)
    for column, view in enumerate(views):
        for row, (variant, title) in enumerate([
            ("control-roundtrip", "SLAT roundtrip"),
            ("edited", "Easy3E edit"),
        ]):
            x, y = column * cell, row * (cell + label)
            image = head_crop(case_dir / comparison_dir / variant / "image" / f"{view:03d}.png", cell)
            sheet.paste(image, (x, y + label))
            draw.text((x + 7, y + 7), f"{title} · {view * 22.5:.1f}°", fill="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def build_front_target_comparison(case_dir: Path, output: Path) -> None:
    items = [
        (case_dir / "cond" / "head.png", "Target image"),
        (case_dir / "comparison" / "control-roundtrip" / "image" / "004.png", "SLAT roundtrip"),
        (case_dir / "comparison" / "edited" / "image" / "004.png", "Easy3E edit"),
    ]
    cell = 480
    label = 34
    sheet = Image.new("RGB", (3 * cell, 2 * (cell + label)), (22, 22, 22))
    draw = ImageDraw.Draw(sheet)
    for column, (path, title) in enumerate(items):
        full = composite(path, cell)
        x = column * cell
        sheet.paste(full, (x + (cell - full.width) // 2, label + (cell - full.height) // 2))
        draw.text((x + 8, 9), title, fill="white")
        crop = head_crop(path, cell)
        y = cell + label
        sheet.paste(crop, (x, y + label))
        draw.text((x + 8, y + 9), f"{title} · head", fill="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def metrics(case_dir: Path) -> dict:
    rows = []
    for view in range(16):
        arrays = []
        for variant in ("control-roundtrip", "edited"):
            arrays.append(np.asarray(Image.open(
                case_dir / "comparison" / variant / "image" / f"{view:03d}.png"
            ).convert("RGBA")))
        control, edited = arrays
        ca, ea = control[..., 3] > 8, edited[..., 3] > 8
        union = ca | ea
        ys, _ = np.nonzero(union)
        y0, y1 = ys.min(), ys.max()
        cutoff = int(y0 + 0.29 * (y1 - y0 + 1))
        head_region = np.indices(ca.shape)[0] < cutoff
        body_region = ~head_region

        def iou(region: np.ndarray) -> float:
            inter = np.count_nonzero(ca & ea & region)
            total = np.count_nonzero((ca | ea) & region)
            return inter / total if total else 1.0

        shared_body = ca & ea & body_region
        body_mae = (np.abs(control[..., :3].astype(np.int16) - edited[..., :3].astype(np.int16))[
            shared_body
        ].mean() / 255.0 if np.any(shared_body) else 0.0)
        rows.append({
            "view": view,
            "azimuth_degrees": view * 22.5,
            "head_silhouette_iou": iou(head_region),
            "body_silhouette_iou": iou(body_region),
            "body_rgb_mae": float(body_mae),
        })
    return {
        "schema": "diffusion-editor.easy3e-comparison",
        "schema_version": 1,
        "comparison": "source SLAT roundtrip vs Easy3E edit",
        "views": rows,
        "mean_head_silhouette_iou": float(np.mean([row["head_silhouette_iou"] for row in rows])),
        "mean_body_silhouette_iou": float(np.mean([row["body_silhouette_iou"] for row in rows])),
        "mean_body_rgb_mae": float(np.mean([row["body_rgb_mae"] for row in rows])),
    }


def main() -> int:
    parsed = args()
    case_dir = parsed.case_dir.resolve()
    build_orbit(case_dir, case_dir / "comparison" / "roundtrip-vs-edit-orbit.png")
    build_heads(case_dir, case_dir / "comparison" / "roundtrip-vs-edit-heads.png")
    build_heads(
        case_dir,
        case_dir / "comparison-clay" / "roundtrip-vs-edit-heads-clay.png",
        comparison_dir="comparison-clay",
    )
    build_front_target_comparison(
        case_dir,
        case_dir / "comparison" / "target-vs-roundtrip-vs-edit-front.png",
    )
    data = metrics(case_dir)
    path = case_dir / "comparison" / "metrics.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = {
        "schema": "diffusion-editor.easy3e-head-edit-result",
        "schema_version": 1,
        "easy3e_commit": "e971ddb3767ab55b98befe16df3e84b0e6baaf52",
        "checkpoint": "microsoft/TRELLIS-image-large",
        "source_reference": "ori/004.png",
        "source_azimuth_degrees": 90.0,
        "incorrect_upstream_default": "ori/012.png (back for this GLB)",
        "seed": 42,
        "cfg_src": 5.0,
        "cfg_tar": 5.0,
        "flowedit_steps": 25,
        "repaint_steps": 25,
        "outputs": {
            "edited_glb": str((case_dir / "output" / "head.glb").resolve()),
            "roundtrip_control_glb": str((case_dir / "control" / "source-slat-roundtrip.glb").resolve()),
            "front_comparison": str((case_dir / "comparison" / "target-vs-roundtrip-vs-edit-front.png").resolve()),
            "orbit_comparison": str((case_dir / "comparison" / "roundtrip-vs-edit-orbit.png").resolve()),
            "head_comparison": str((case_dir / "comparison" / "roundtrip-vs-edit-heads.png").resolve()),
            "clay_head_comparison": str((case_dir / "comparison-clay" / "roundtrip-vs-edit-heads-clay.png").resolve()),
            "metrics": str(path.resolve()),
        },
        "compatibility": {
            "blender": "CPU Cycles; system Blender 4.0.2 CUDA kernel does not support sm_120",
            "dinov2": "XFORMERS_DISABLED=1 for FP32 attention on sm_120",
            "vox2seq": "bundled PyTorch fallback",
            "kaolin": "local shape-check fallback only",
        },
        "summary_metrics": {
            key: data[key]
            for key in ("mean_head_silhouette_iou", "mean_body_silhouette_iou", "mean_body_rgb_mae")
        },
    }
    (case_dir / "result-manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
