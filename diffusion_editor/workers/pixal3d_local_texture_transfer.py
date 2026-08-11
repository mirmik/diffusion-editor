"""Transfer and blend base/local textured meshes onto a fused Pixal3D mesh.

This diagnostic runner writes vertex colors.  It proves source projection and
seam blending before committing to a particular UV-atlas packing strategy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


EDITOR_ROOT = Path(__file__).resolve().parents[2]
if str(EDITOR_ROOT) not in sys.path:
    sys.path.insert(0, str(EDITOR_ROOT))

from diffusion_editor.generation.local_detail_texture import (  # noqa: E402
    blend_base_color,
    local_blend_weights,
    sample_texture_bilinear,
)


def _load_textured_mesh(path, trimesh, np):
    mesh = trimesh.load(path, force="mesh", process=False)
    uv = getattr(mesh.visual, "uv", None)
    material = getattr(mesh.visual, "material", None)
    texture = getattr(material, "baseColorTexture", None)
    if uv is None or texture is None:
        raise ValueError(f"{path} has no base-color texture and UV coordinates")
    image = np.asarray(texture.convert("RGBA"), dtype=np.uint8)
    return mesh, np.asarray(uv, dtype=np.float32), image


def _sample_source(
        source_mesh,
        source_uv,
        source_image,
        target_vertices,
        *,
        batch_size,
        torch,
        np,
):
    from cumesh import cuBVH

    vertices = np.asarray(source_mesh.vertices, dtype=np.float32)
    faces = np.asarray(source_mesh.faces, dtype=np.int32)
    bvh = cuBVH(vertices, faces)
    faces_gpu = torch.from_numpy(faces).long().cuda()
    uv_gpu = torch.from_numpy(source_uv).float().cuda()
    colors = np.empty((len(target_vertices), 4), dtype=np.uint8)
    distances = np.empty((len(target_vertices),), dtype=np.float32)
    for begin in range(0, len(target_vertices), batch_size):
        end = min(begin + batch_size, len(target_vertices))
        query = torch.from_numpy(
            np.asarray(target_vertices[begin:end], dtype=np.float32)
        ).cuda()
        distance, face_id, barycentric = bvh.unsigned_distance(
            query, return_uvw=True
        )
        triangle_uv = uv_gpu[faces_gpu[face_id.long()]]
        sampled_uv = (
            triangle_uv * barycentric.unsqueeze(-1)
        ).sum(dim=1).cpu().numpy()
        colors[begin:end] = np.rint(
            np.clip(
                sample_texture_bilinear(source_image, sampled_uv),
                0.0,
                255.0,
            )
        ).astype(np.uint8)
        distances[begin:end] = distance.float().cpu().numpy()
        print(f"[Texture transfer] {end}/{len(target_vertices)}", flush=True)
    return colors, distances


def _simplify_target(mesh, target_faces, *, trimesh, torch, np):
    if target_faces is None or len(mesh.faces) <= target_faces:
        return mesh
    import cumesh

    gpu_mesh = cumesh.CuMesh()
    gpu_mesh.init(
        torch.from_numpy(np.asarray(mesh.vertices)).float().cuda().contiguous(),
        torch.from_numpy(np.asarray(mesh.faces)).int().cuda().contiguous(),
    )
    gpu_mesh.simplify(target_faces, verbose=True)
    vertices, faces = gpu_mesh.read()
    return trimesh.Trimesh(
        vertices=vertices.float().cpu().numpy(),
        faces=faces.long().cpu().numpy(),
        process=False,
    )


def _bake_vertex_colors(mesh, colors, texture_size, *, trimesh, torch, np):
    import cv2
    import cumesh
    import nvdiffrast.torch as dr
    from PIL import Image

    source_vertices = np.asarray(mesh.vertices, dtype=np.float32)
    source_faces = np.asarray(mesh.faces, dtype=np.int32)
    source_bvh = cumesh.cuBVH(source_vertices, source_faces)
    source_faces_gpu = torch.from_numpy(source_faces).long().cuda()
    source_colors_gpu = torch.from_numpy(colors).float().cuda() / 255.0
    gpu_mesh = cumesh.CuMesh()
    gpu_mesh.init(
        torch.from_numpy(source_vertices).float().cuda().contiguous(),
        torch.from_numpy(source_faces).int().cuda().contiguous(),
    )
    vertices, faces, uvs, _vertex_map = gpu_mesh.uv_unwrap(
        return_vmaps=True, verbose=True
    )
    vertices = vertices.cuda()
    faces = faces.cuda()
    uvs = uvs.cuda()
    _distance, source_face_id, barycentric = source_bvh.unsigned_distance(
        vertices, return_uvw=True
    )
    source_triangle_colors = source_colors_gpu[
        source_faces_gpu[source_face_id.long()]
    ]
    vertex_colors = (
        source_triangle_colors * barycentric.unsqueeze(-1)
    ).sum(dim=1)

    context = dr.RasterizeCudaContext()
    clip_uv = torch.cat((
        uvs * 2.0 - 1.0,
        torch.zeros_like(uvs[:, :1]),
        torch.ones_like(uvs[:, :1]),
    ), dim=1).unsqueeze(0)
    raster = torch.zeros(
        (1, texture_size, texture_size, 4),
        dtype=torch.float32,
        device="cuda",
    )
    for begin in range(0, len(faces), 100_000):
        chunk, _ = dr.rasterize(
            context,
            clip_uv,
            faces[begin:begin + 100_000],
            resolution=(texture_size, texture_size),
        )
        covered = chunk[..., 3:4] > 0
        chunk[..., 3:4] += begin
        raster = torch.where(covered, chunk, raster)
    mask = raster[0, ..., 3] > 0
    baked = dr.interpolate(vertex_colors.unsqueeze(0), raster, faces)[0][0]
    rgba = np.clip(baked.cpu().numpy() * 255.0, 0.0, 255.0).astype(np.uint8)
    holes = (~mask.cpu().numpy()).astype(np.uint8)
    rgb = cv2.inpaint(rgba[..., :3], holes, 3, cv2.INPAINT_TELEA)
    alpha = np.full((*rgb.shape[:2], 1), 255, dtype=np.uint8)
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(np.concatenate((rgb, alpha), axis=2)),
        baseColorFactor=np.asarray((255, 255, 255, 255), dtype=np.uint8),
        metallicFactor=0.0,
        roughnessFactor=0.8,
        alphaMode="OPAQUE",
        doubleSided=False,
    )
    exported_uvs = uvs.cpu().numpy()
    exported_uvs[:, 1] = 1.0 - exported_uvs[:, 1]
    return trimesh.Trimesh(
        vertices=vertices.cpu().numpy(),
        faces=faces.long().cpu().numpy(),
        process=False,
        visual=trimesh.visual.TextureVisuals(
            uv=exported_uvs,
            material=material,
        ),
    )


def run(args) -> None:
    import numpy as np
    import torch
    import trimesh

    metadata = json.loads(Path(args.registration_metadata).read_text("utf-8"))
    bounds = np.asarray(metadata["roi_bounds"], dtype=np.float64)
    registration = metadata["registration"]
    scale = float(registration["scale"])
    raw_translation = np.asarray(registration["translation"], dtype=np.float64)
    preview_translation = raw_translation * np.asarray((-1.0, 1.0, -1.0))

    base, base_uv, base_image = _load_textured_mesh(
        args.base_textured, trimesh, np
    )
    local, local_uv, local_image = _load_textured_mesh(
        args.local_textured, trimesh, np
    )
    local.vertices = (
        np.asarray(local.vertices) * scale + preview_translation
    )
    target = trimesh.load(args.fused_mesh, force="mesh", process=False)
    target = _simplify_target(
        target,
        args.target_faces,
        trimesh=trimesh,
        torch=torch,
        np=np,
    )
    target_vertices = np.asarray(target.vertices, dtype=np.float32)

    base_color, base_distance = _sample_source(
        base,
        base_uv,
        base_image,
        target_vertices,
        batch_size=args.batch_size,
        torch=torch,
        np=np,
    )
    local_color, local_distance = _sample_source(
        local,
        local_uv,
        local_image,
        target_vertices,
        batch_size=args.batch_size,
        torch=torch,
        np=np,
    )

    raw_target = target_vertices * np.asarray((-1.0, 1.0, -1.0))
    weights = local_blend_weights(
        raw_target,
        bounds,
        inner_radius=args.inner_radius,
        outer_radius=args.outer_radius,
    )
    colors = blend_base_color(base_color, local_color, weights)
    colors[:, 3] = 255
    target.visual = trimesh.visual.ColorVisuals(
        mesh=target, vertex_colors=colors
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    vertex_path = output
    if args.texture_size:
        vertex_path = output.with_name(f"{output.stem}-vertex{output.suffix}")
    target.export(vertex_path)
    if args.texture_size:
        textured = _bake_vertex_colors(
            target,
            colors,
            args.texture_size,
            trimesh=trimesh,
            torch=torch,
            np=np,
        )
        textured.export(output, extension_webp=True)

    diagnostic = target.copy()
    weight_byte = np.rint(weights * 255.0).astype(np.uint8)
    diagnostic.visual = trimesh.visual.ColorVisuals(
        mesh=diagnostic,
        vertex_colors=np.stack((
            weight_byte,
            np.zeros_like(weight_byte),
            255 - weight_byte,
            np.full_like(weight_byte, 255),
        ), axis=1),
    )
    diagnostic_path = output.with_name(f"{output.stem}-weights.glb")
    diagnostic.export(diagnostic_path)

    result = {
        "protocol": 1,
        "target_vertices": int(len(target.vertices)),
        "target_faces": int(len(target.faces)),
        "requested_target_faces": args.target_faces,
        "texture_size": args.texture_size,
        "registration_scale": scale,
        "registration_translation": raw_translation.tolist(),
        "blend_inner_radius": args.inner_radius,
        "blend_outer_radius": args.outer_radius,
        "local_weight_vertices": int((weights > 0.0).sum()),
        "local_core_vertices": int((weights >= 1.0).sum()),
        "base_distance_quantiles": np.quantile(
            base_distance, (0.5, 0.95, 0.99)
        ).tolist(),
        "local_distance_quantiles": np.quantile(
            local_distance[weights > 0.0], (0.5, 0.95, 0.99)
        ).tolist(),
        "artifacts": {
            "textured": output.name,
            "vertex_color": vertex_path.name,
            "weights": diagnostic_path.name,
        },
    }
    metadata_path = output.with_name(f"{output.stem}-metadata.json")
    metadata_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-textured", required=True)
    parser.add_argument("--local-textured", required=True)
    parser.add_argument("--fused-mesh", required=True)
    parser.add_argument("--registration-metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--inner-radius", type=float, default=0.72)
    parser.add_argument("--outer-radius", type=float, default=1.08)
    parser.add_argument("--batch-size", type=int, default=250_000)
    parser.add_argument("--target-faces", type=int)
    parser.add_argument("--texture-size", type=int)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.target_faces is not None and args.target_faces < 1:
        parser.error("--target-faces must be positive")
    if args.texture_size is not None and args.texture_size < 64:
        parser.error("--texture-size must be at least 64")
    if not 0.0 <= args.inner_radius < args.outer_radius:
        parser.error("blend radii must satisfy 0 <= inner < outer")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
