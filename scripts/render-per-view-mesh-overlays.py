#!/usr/bin/env python3
"""Render shared or per-view fitted meshes through their frozen cameras."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import trimesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit_report", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--overlay-alpha", type=float, default=0.58)
    parser.add_argument("--thumbnail-width", type=int, default=420)
    return parser.parse_args()


def _load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    if not isinstance(loaded, trimesh.Trimesh):
        raise SystemExit(f"{path} did not contain a triangle mesh")
    if len(loaded.faces) == 0:
        raise SystemExit(f"{path} has no faces")
    return loaded


def _project(
    vertices: np.ndarray,
    camera: dict,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    rotation = np.asarray(camera["world_to_camera_rotation"], dtype=np.float64)
    camera_vertices = vertices @ rotation.T
    normalized = (
        float(camera["scale"]) * camera_vertices[:, :2]
        + np.asarray(camera["translation_normalized"], dtype=np.float64)
    )
    pixels = np.stack((
        normalized[:, 0] * height + width * 0.5,
        height * 0.5 - normalized[:, 1] * height,
    ), axis=-1)
    return pixels, camera_vertices[:, 2]


def _flat_render(
    mesh: trimesh.Trimesh,
    camera: dict,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    pixels, depths = _project(vertices, camera, width, height)
    triangles = pixels[faces]
    face_depths = depths[faces].mean(axis=1)
    rotation = np.asarray(camera["world_to_camera_rotation"], dtype=np.float64)
    camera_normals = np.asarray(mesh.face_normals) @ rotation.T

    # The reflected-forward MakeHuman convention places the front-facing
    # surface at lower camera Z.  Draw high-Z triangles first so the lower-Z
    # (camera-nearest) surface wins.  Reversing this order preserves the exact
    # silhouette while deceptively rendering the back side of the body.
    order = np.argsort(face_depths)[::-1]
    render = np.zeros((height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    base_bgr = np.asarray((238.0, 185.0, 42.0))
    for face_index in order:
        triangle = np.rint(triangles[face_index]).astype(np.int32)
        if not np.isfinite(triangles[face_index]).all():
            continue
        if (
            triangle[:, 0].max() < 0 or triangle[:, 0].min() >= width
            or triangle[:, 1].max() < 0 or triangle[:, 1].min() >= height
        ):
            continue
        facing = float(np.clip(-camera_normals[face_index, 2], 0.0, 1.0))
        brightness = 0.28 + 0.72 * facing
        color = tuple(int(value) for value in base_bgr * brightness)
        cv2.fillConvexPoly(render, triangle, color, lineType=cv2.LINE_AA)
        cv2.fillConvexPoly(mask, triangle, 255, lineType=cv2.LINE_8)

    contour = np.zeros_like(mask)
    contours, _hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour, contours, -1, 255, 3, lineType=cv2.LINE_AA)
    return render, mask, contour


def _labeled(image: Image.Image, text: str) -> Image.Image:
    result = image.convert("RGB")
    draw = ImageDraw.Draw(result)
    draw.rectangle((0, 0, min(result.width, 720), 38), fill=(0, 0, 0))
    draw.text((12, 10), text, fill=(255, 255, 255))
    return result


def _grid(images: list[Image.Image], width: int, target: Path) -> None:
    thumbs = [image.resize(
        (width, round(image.height * width / image.width)),
        Image.Resampling.LANCZOS,
    ) for image in images]
    cell_height = max(image.height for image in thumbs)
    sheet = Image.new("RGB", (width * 3, cell_height * 3), (20, 20, 24))
    for index, image in enumerate(thumbs):
        sheet.paste(image, ((index % 3) * width, (index // 3) * cell_height))
    sheet.save(target)


def main() -> int:
    args = parse_args()
    if not 0.0 < args.overlay_alpha <= 1.0:
        raise SystemExit("overlay alpha must be in (0, 1]")
    if args.thumbnail_width < 64:
        raise SystemExit("thumbnail width must be at least 64")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report = json.loads(args.fit_report.read_text(encoding="utf-8"))
    if "shared_report" in report:
        shared_report_path = Path(report["shared_report"])
        shared_report = json.loads(shared_report_path.read_text(encoding="utf-8"))
        mesh_paths = {
            name: Path(paths["posed_mesh"])
            for name, paths in report["artifacts"]["per_view"].items()
        }
        fit_kind = "per-view-pose"
    else:
        shared_report_path = args.fit_report
        shared_report = report
        shared_mesh = Path(report["artifacts"]["posed_mesh"])
        mesh_paths = None
        fit_kind = "shared-pose"
    camera_report_path = Path(shared_report["inputs"]["camera_report"])
    keypoint_path = Path(shared_report["inputs"]["keypoints"])
    camera_report = json.loads(camera_report_path.read_text(encoding="utf-8"))
    keypoint_report = json.loads(keypoint_path.read_text(encoding="utf-8"))
    cameras = {camera["name"]: camera for camera in camera_report["cameras"]}
    source_views = {view["name"]: view for view in keypoint_report["views"]}

    overlays = []
    contours = []
    renders = []
    artifacts = {}
    for view_name in cameras:
        source_path = Path(source_views[view_name]["path"])
        source = np.asarray(Image.open(source_path).convert("RGB"))
        height, width = source.shape[:2]
        mesh_path = shared_mesh if mesh_paths is None else mesh_paths[view_name]
        mesh = _load_mesh(mesh_path)
        render_bgr, mask, contour = _flat_render(
            mesh, cameras[view_name], width, height)
        render = cv2.cvtColor(render_bgr, cv2.COLOR_BGR2RGB)
        foreground = mask > 0
        overlay = source.copy()
        overlay[foreground] = np.clip(
            (1.0 - args.overlay_alpha) * source[foreground]
            + args.overlay_alpha * render[foreground],
            0, 255,
        ).astype(np.uint8)
        overlay[contour > 0] = (255, 225, 35)

        contour_only = source.copy()
        contour_only[contour > 0] = (255, 40, 215)
        render_panel = np.full_like(source, (38, 38, 44))
        render_panel[foreground] = render[foreground]

        comparison = Image.new("RGB", (width * 3, height), (0, 0, 0))
        comparison.paste(Image.fromarray(source), (0, 0))
        comparison.paste(Image.fromarray(overlay), (width, 0))
        comparison.paste(Image.fromarray(render_panel), (width * 2, 0))
        comparison = _labeled(
            comparison,
            f"{view_name}: source | mesh overlay | frozen-camera mesh render",
        )

        overlay_path = args.output_dir / f"{view_name}-mesh-overlay.png"
        contour_path = args.output_dir / f"{view_name}-mesh-contour.png"
        render_path = args.output_dir / f"{view_name}-mesh-render.png"
        comparison_path = args.output_dir / f"{view_name}-mesh-comparison.png"
        Image.fromarray(overlay).save(overlay_path)
        Image.fromarray(contour_only).save(contour_path)
        Image.fromarray(render_panel).save(render_path)
        comparison.save(comparison_path)
        overlays.append(_labeled(Image.fromarray(overlay), view_name))
        contours.append(_labeled(Image.fromarray(contour_only), view_name))
        renders.append(_labeled(Image.fromarray(render_panel), view_name))
        artifacts[view_name] = {
            "source": str(source_path),
            "mesh": str(mesh_path),
            "overlay": str(overlay_path),
            "contour": str(contour_path),
            "render": str(render_path),
            "comparison": str(comparison_path),
        }

    overlay_grid = args.output_dir / "mesh-overlay-grid.png"
    contour_grid = args.output_dir / "mesh-contour-grid.png"
    render_grid = args.output_dir / "mesh-render-grid.png"
    _grid(overlays, args.thumbnail_width, overlay_grid)
    _grid(contours, args.thumbnail_width, contour_grid)
    _grid(renders, args.thumbnail_width, render_grid)
    result = {
        "schema": "diffusion-editor.per-view-frozen-camera-mesh-overlays",
        "schema_version": 1,
        "fit_report": str(args.fit_report.resolve()),
        "fit_kind": fit_kind,
        "camera_report": str(camera_report_path.resolve()),
        "projection": camera_report.get("projection"),
        "renderer": (
            "orthographic projected triangles; lower camera-Z is near; "
            "flat negative-camera-Z normal shading"
        ),
        "overlay_alpha": args.overlay_alpha,
        "views": artifacts,
        "grids": {
            "overlay": str(overlay_grid),
            "contour": str(contour_grid),
            "render": str(render_grid),
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(report_path),
        "overlay_grid": str(overlay_grid),
        "contour_grid": str(contour_grid),
        "render_grid": str(render_grid),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
