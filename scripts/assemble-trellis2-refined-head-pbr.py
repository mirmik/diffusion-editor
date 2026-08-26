#!/usr/bin/env python3
"""Splice a locally refined PBR head back into a full TRELLIS.2 PBR body.

Both meshes are clipped at one horizontal plane inside the frozen neck band.
UV coordinates and PBR materials are preserved, while topology is deliberately
kept as two objects so this experiment does not rebake the full-body material.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body-pbr", type=Path, required=True)
    parser.add_argument("--head-pbr-body-space", type=Path, required=True)
    parser.add_argument("--shape-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--seam-frozen-fraction",
        type=float,
        default=0.10,
        help="Seam height as a fraction of the local head's world-space height",
    )
    return parser.parse_args()


def _single_world_mesh(path, trimesh):
    scene = trimesh.load(path, force="scene", process=False)
    dumped = scene.dump(concatenate=False)
    if len(dumped) != 1:
        raise ValueError(f"Expected one mesh in {path}, got {len(dumped)}")
    return dumped[0]


def _clip_textured_z(mesh, cutoff, *, keep_above, trimesh, np):
    if not hasattr(mesh.visual, "uv") or mesh.visual.uv is None:
        raise ValueError("Expected a UV-textured mesh")

    source_uv = np.asarray(mesh.visual.uv, dtype=np.float64)
    vertices = []
    uvs = []
    faces = []

    def inside(position):
        if keep_above:
            return position[2] >= cutoff
        return position[2] <= cutoff

    for face in mesh.faces:
        polygon = [
            (
                np.asarray(mesh.vertices[int(index)], dtype=np.float64),
                source_uv[int(index)],
            )
            for index in face
        ]
        clipped = []
        for current, following in zip(polygon, polygon[1:] + polygon[:1]):
            current_inside = inside(current[0])
            following_inside = inside(following[0])
            if current_inside:
                clipped.append(current)
            if current_inside != following_inside:
                denominator = following[0][2] - current[0][2]
                if abs(float(denominator)) <= 1e-12:
                    continue
                amount = (cutoff - current[0][2]) / denominator
                position = current[0] + amount * (following[0] - current[0])
                uv = current[1] + amount * (following[1] - current[1])
                clipped.append((position, uv))
        if len(clipped) < 3:
            continue
        base = len(vertices)
        vertices.extend(item[0] for item in clipped)
        uvs.extend(item[1] for item in clipped)
        for index in range(1, len(clipped) - 1):
            faces.append((base, base + index, base + index + 1))

    visual = trimesh.visual.TextureVisuals(
        uv=np.asarray(uvs, dtype=np.float64),
        material=mesh.visual.material,
    )
    result = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        visual=visual,
        process=False,
    )
    result.remove_unreferenced_vertices()
    return result


def main() -> int:
    args = _arguments()
    for path in (args.body_pbr, args.head_pbr_body_space, args.shape_manifest):
        if not path.exists():
            raise FileNotFoundError(path)
    if not 0 <= args.seam_frozen_fraction <= 1:
        raise ValueError("seam-frozen-fraction must be in [0, 1]")

    import numpy as np
    import trimesh

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    shape = json.loads(args.shape_manifest.read_text(encoding="utf-8"))
    cutoff = float(shape["world_cutoff_z"])
    center_z = float(shape["local_center"][2])
    scale = float(shape["local_scale"])
    local_world_height = 0.9 / scale
    seam_z = cutoff + args.seam_frozen_fraction * local_world_height

    body = _single_world_mesh(args.body_pbr, trimesh)
    head = _single_world_mesh(args.head_pbr_body_space, trimesh)
    body_below = _clip_textured_z(
        body, seam_z, keep_above=False, trimesh=trimesh, np=np
    )
    head_above = _clip_textured_z(
        head, seam_z, keep_above=True, trimesh=trimesh, np=np
    )
    body_below.metadata["name"] = "original_body_below_neck_seam"
    head_above.metadata["name"] = "refined_head_above_neck_seam"

    editor_scene = trimesh.Scene()
    editor_scene.add_geometry(body_below, geom_name="body")
    editor_scene.add_geometry(head_above, geom_name="refined_head")
    editor_path = output / "final-pbr-refined-head-editor-z-up.glb"
    editor_scene.export(editor_path, extension_webp=True)

    y_up_scene = editor_scene.copy()
    y_up_scene.apply_transform(np.asarray((
        (-1, 0, 0, 0),
        (0, 0, -1, 0),
        (0, -1, 0, 0),
        (0, 0, 0, 1),
    ), dtype=np.float64))
    y_up_path = output / "final-pbr-refined-head-gltf-y-up.glb"
    y_up_scene.export(y_up_path, extension_webp=True)

    # Keep isolated parts as diagnostics: renderers occasionally treat a
    # multi-root glTF differently from a single-root asset.
    body_scene = trimesh.Scene(body_below.copy())
    body_scene.apply_transform(np.asarray((
        (-1, 0, 0, 0),
        (0, 0, -1, 0),
        (0, -1, 0, 0),
        (0, 0, 0, 1),
    ), dtype=np.float64))
    body_scene.export(output / "debug-body-below-seam-gltf-y-up.glb", extension_webp=True)
    head_scene = trimesh.Scene(head_above.copy())
    head_scene.apply_transform(np.asarray((
        (-1, 0, 0, 0),
        (0, 0, -1, 0),
        (0, -1, 0, 0),
        (0, 0, 0, 1),
    ), dtype=np.float64))
    head_scene.export(output / "debug-head-above-seam-gltf-y-up.glb", extension_webp=True)

    manifest = {
        "experiment": "PBR splice of local refined head into original body",
        "body_pbr": str(args.body_pbr.resolve()),
        "head_pbr_body_space": str(args.head_pbr_body_space.resolve()),
        "shape_manifest": str(args.shape_manifest.resolve()),
        "shape_center_z": center_z,
        "shape_local_scale": scale,
        "world_cutoff_z": cutoff,
        "seam_frozen_fraction": args.seam_frozen_fraction,
        "seam_z": seam_z,
        "body_faces": int(len(body_below.faces)),
        "head_faces": int(len(head_above.faces)),
        "editor_output": str(editor_path),
        "gltf_y_up_output": str(y_up_path),
        "topology": "two textured meshes sharing one planar seam; not welded",
        "status": "complete",
    }
    (output / "assembly-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
