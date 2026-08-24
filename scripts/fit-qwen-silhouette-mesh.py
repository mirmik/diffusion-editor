#!/usr/bin/env python3
"""Fit a template mesh to Qwen multi-view silhouettes with PyTorch3D."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
import torch
import trimesh

from qwen_view_names import VIEW_PATTERN

from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    SoftSilhouetteShader,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes
from pytorch3d.transforms import axis_angle_to_matrix


ELEVATIONS = {"low": -30.0, "eye": 0.0, "elevated": 30.0}
AZIMUTHS = {"000": 0.0, "045": 45.0, "315": -45.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument(
        "--geometry-name",
        help="Use one named geometry from a scene asset instead of all parts.",
    )
    parser.add_argument(
        "--obj-group", help="Use only this named group from a Wavefront OBJ.",
    )
    parser.add_argument(
        "--source-up-axis", choices=("y", "z"), default="z",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto",
    )
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--template-faces", type=int, default=1000)
    parser.add_argument(
        "--voxel-resolution",
        type=int,
        default=0,
        help="Voxel-remesh the normalized template at this many cells per height.",
    )
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--distance", type=float, default=3.5)
    parser.add_argument("--fov", type=float, default=35.0)
    parser.add_argument("--mask-threshold", type=float, default=18.0)
    parser.add_argument(
        "--mask-dir",
        type=Path,
        help=(
            "Use precomputed foreground masks with the same filenames as the "
            "views instead of the border-colour heuristic."
        ),
    )
    parser.add_argument("--edge-weight", type=float, default=2.0)
    parser.add_argument("--smooth-weight", type=float, default=0.20)
    parser.add_argument("--anchor-weight", type=float, default=0.02)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--evaluation-interval", type=int, default=25)
    parser.add_argument(
        "--hard-forward-silhouette",
        action="store_true",
        help="Use exact face coverage in forward, retaining soft alpha gradients.",
    )
    parser.add_argument(
        "--silhouette-loss",
        choices=("mse", "iou", "dice"),
        default="mse",
        help="Silhouette objective. IoU and Dice avoid background-dominated shrinkage.",
    )
    parser.add_argument("--rig", type=Path)
    parser.add_argument("--skin-weights", type=Path)
    parser.add_argument(
        "--pose-bones",
        default=(
            "clavicle.L,clavicle.R,upperarm01.L,upperarm01.R,"
            "lowerarm01.L,lowerarm01.R"
        ),
    )
    parser.add_argument("--pose-weight", type=float, default=0.002)
    parser.add_argument("--alignment-weight", type=float, default=0.01)
    parser.add_argument(
        "--pose-axis",
        choices=("full", "x", "y", "z"),
        default="full",
        help="Restrict each optimized joint to one rest-space rotation axis.",
    )
    parser.add_argument(
        "--optimize-alignment",
        action="store_true",
        help="Jointly optimize global scale and translation with the pose.",
    )
    parser.add_argument("--seed", type=int, default=20260823)
    return parser.parse_args()


def _subject_mask(image: Image.Image, threshold: float) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    border = np.concatenate((
        rgb[:12].reshape(-1, 3),
        rgb[-12:].reshape(-1, 3),
        rgb[:, :12].reshape(-1, 3),
        rgb[:, -12:].reshape(-1, 3),
    ))
    background = np.median(border, axis=0)
    distance = np.linalg.norm(rgb - background[None, None], axis=2)
    mask = distance >= threshold
    mask = ndimage.binary_opening(mask, iterations=1)
    mask = ndimage.binary_closing(mask, iterations=2)
    labels, count = ndimage.label(mask)
    if count:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        mask = labels == int(np.argmax(sizes))
    mask = ndimage.binary_fill_holes(mask)
    return mask.astype(np.float32)


def _square_resize(mask: np.ndarray, size: int) -> np.ndarray:
    height, width = mask.shape
    side = max(height, width)
    square = np.zeros((side, side), dtype=np.float32)
    y = (side - height) // 2
    x = (side - width) // 2
    square[y:y + height, x:x + width] = mask
    return cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)


def load_views(
    image_dir: Path,
    image_size: int,
    threshold: float,
    mask_dir: Path | None = None,
) -> tuple[list[dict], torch.Tensor]:
    records = []
    masks = []
    for path in sorted(image_dir.iterdir()):
        match = VIEW_PATTERN.match(path.name)
        if match is None:
            continue
        elevation_name, azimuth_name = match.groups()
        if mask_dir is None:
            with Image.open(path) as image:
                mask = _subject_mask(image, threshold)
            mask_source = "border-colour heuristic"
        else:
            mask_path = mask_dir / path.name
            if not mask_path.is_file():
                raise SystemExit(f"missing foreground mask: {mask_path}")
            with Image.open(path) as source, Image.open(mask_path) as image:
                if image.size != source.size:
                    raise SystemExit(
                        f"foreground mask size mismatch for {path.name}: "
                        f"{image.size} != {source.size}"
                    )
                mask = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
            mask_source = str(mask_path.resolve())
        resized = _square_resize(mask, image_size)
        records.append({
            "name": path.name,
            "path": str(path.resolve()),
            "elevation_degrees": ELEVATIONS[elevation_name],
            "azimuth_degrees": AZIMUTHS[azimuth_name],
            "mask_fraction": float(resized.mean()),
            "mask_source": mask_source,
        })
        masks.append(resized)
    if len(records) != 9:
        raise SystemExit(f"expected 9 ON views, found {len(records)}")
    return records, torch.from_numpy(np.stack(masks)).float()


def load_template(
    path: Path,
    target_faces: int,
    geometry_name: str | None,
    obj_group: str | None,
    source_up_axis: str,
    voxel_resolution: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if obj_group is not None:
        if path.suffix.lower() != ".obj":
            raise SystemExit("--obj-group requires a Wavefront OBJ template")
        vertices = []
        faces = []
        active_group = None
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "v" and len(fields) >= 4:
                vertices.append(tuple(map(float, fields[1:4])))
            elif fields[0] == "g":
                active_group = fields[1] if len(fields) > 1 else None
            elif fields[0] == "f" and active_group == obj_group:
                polygon = []
                for field in fields[1:]:
                    index = int(field.split("/", 1)[0])
                    polygon.append(index - 1 if index > 0 else len(vertices) + index)
                for corner in range(1, len(polygon) - 1):
                    faces.append((polygon[0], polygon[corner], polygon[corner + 1]))
        if not faces:
            raise SystemExit(f"OBJ group {obj_group!r} contains no faces")
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    else:
        scene = trimesh.load(path, force="scene")
        if geometry_name is None:
            mesh = scene.to_geometry()
        else:
            matching_nodes = [
                node for node in scene.graph.nodes_geometry
                if scene.graph[node][1] == geometry_name
            ]
            if len(matching_nodes) != 1:
                available = ", ".join(sorted(scene.geometry))
                raise SystemExit(
                    f"expected one node for geometry {geometry_name!r}; "
                    f"available geometries: {available}"
                )
            transform, _ = scene.graph[matching_nodes[0]]
            mesh = scene.geometry[geometry_name].copy()
            mesh.apply_transform(transform)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    # PyTorch3D's look-at helper expects Y-up.
    if source_up_axis == "z":
        vertices = vertices[:, [0, 2, 1]]
    bounds = np.stack((vertices.min(axis=0), vertices.max(axis=0)))
    center = bounds.mean(axis=0)
    vertices -= center
    height = float(vertices[:, 1].max() - vertices[:, 1].min())
    scale = 2.0 / height
    vertices *= scale
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(mesh.faces, dtype=np.int64),
        process=False,
    )
    if voxel_resolution > 0:
        pitch = 2.0 / voxel_resolution
        voxels = mesh.voxelized(pitch).fill()
        mesh = voxels.marching_cubes
        mesh.apply_transform(voxels.transform)
        trimesh.smoothing.filter_taubin(mesh, iterations=3)
    if target_faces > 0 and len(mesh.faces) > target_faces:
        mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    return vertices, faces, {
        "center": center.astype(np.float32),
        "scale": scale,
        "source_up_axis": source_up_axis,
    }


def _obj_vertices(path: Path, source_up_axis: str) -> np.ndarray:
    vertices = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if fields and fields[0] == "v" and len(fields) >= 4:
            vertices.append(tuple(map(float, fields[1:4])))
    result = np.asarray(vertices, dtype=np.float32)
    if source_up_axis == "z":
        result = result[:, [0, 2, 1]]
    return result


def load_articulation(
    template_path: Path,
    rig_path: Path,
    weights_path: Path,
    vertex_count: int,
    normalization: dict,
    pose_bones: list[str],
) -> dict:
    if template_path.suffix.lower() != ".obj":
        raise SystemExit("articulated fitting currently requires an OBJ template")
    raw_vertices = _obj_vertices(
        template_path, normalization["source_up_axis"],
    )
    raw_vertices = (
        raw_vertices - normalization["center"][None, :]
    ) * normalization["scale"]
    rig = json.loads(rig_path.read_text(encoding="utf-8"))
    weight_data = json.loads(weights_path.read_text(encoding="utf-8"))["weights"]
    bones = rig["bones"]

    ordered = []
    visiting = set()

    def append_bone(name: str) -> None:
        if name in ordered:
            return
        if name in visiting:
            raise SystemExit(f"cycle in rig hierarchy at bone {name!r}")
        visiting.add(name)
        parent = bones[name].get("parent")
        if parent is not None:
            if parent not in bones:
                raise SystemExit(f"missing parent bone {parent!r} for {name!r}")
            append_bone(parent)
        visiting.remove(name)
        ordered.append(name)

    for bone_name in bones:
        append_bone(bone_name)
    bone_indices = {name: index for index, name in enumerate(ordered)}
    missing = [name for name in pose_bones if name not in bone_indices]
    if missing:
        raise SystemExit(f"pose bones not found in rig: {', '.join(missing)}")

    heads = []
    head_vertex_indices = []
    parents = []
    for name in ordered:
        joint_name = bones[name]["head"]
        joint_indices = rig["joints"][joint_name]
        heads.append(raw_vertices[joint_indices].mean(axis=0))
        head_vertex_indices.append(list(map(int, joint_indices)))
        parent = bones[name].get("parent")
        parents.append(-1 if parent is None else bone_indices[parent])

    weights = np.zeros((vertex_count, len(ordered)), dtype=np.float32)
    for name, assignments in weight_data.items():
        bone_index = bone_indices.get(name)
        if bone_index is None:
            continue
        for vertex_index, weight in assignments:
            if vertex_index < vertex_count:
                weights[vertex_index, bone_index] += weight
    totals = weights.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise SystemExit("one or more template vertices have no skinning weights")
    weights /= totals
    return {
        "bone_names": ordered,
        "bone_heads": np.asarray(heads, dtype=np.float32),
        "bone_head_vertex_indices": head_vertex_indices,
        "parents": parents,
        "weights": weights,
        "pose_bone_indices": [bone_indices[name] for name in pose_bones],
        "pose_bones": pose_bones,
    }


def articulate_vertices(
    rest_vertices: torch.Tensor,
    bone_heads: torch.Tensor,
    parents: list[int],
    skin_weights: torch.Tensor,
    pose_bone_indices: list[int],
    pose: torch.Tensor,
    log_scale: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    pose_lookup = {
        bone_index: pose_index
        for pose_index, bone_index in enumerate(pose_bone_indices)
    }
    transforms = []
    eye = torch.eye(3, device=rest_vertices.device, dtype=rest_vertices.dtype)
    bottom = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0]],
        device=rest_vertices.device,
        dtype=rest_vertices.dtype,
    )
    for bone_index, parent in enumerate(parents):
        pose_index = pose_lookup.get(bone_index)
        rotation = (
            eye if pose_index is None
            else axis_angle_to_matrix(pose[pose_index])
        )
        pivot = bone_heads[bone_index]
        offset = pivot - rotation @ pivot
        local = torch.cat((
            torch.cat((rotation, offset[:, None]), dim=1),
            bottom,
        ), dim=0)
        transform = local if parent < 0 else transforms[parent] @ local
        transforms.append(transform)
    transforms_tensor = torch.stack(transforms)
    homogeneous = torch.cat((
        rest_vertices,
        torch.ones_like(rest_vertices[:, :1]),
    ), dim=1)
    per_bone = torch.einsum(
        "bij,vj->bvi", transforms_tensor[:, :3], homogeneous,
    )
    skinned = torch.einsum("vb,bvi->vi", skin_weights, per_bone)
    return skinned * torch.exp(log_scale) + translation


def pose_axis_angles(pose: torch.Tensor, axis: str) -> torch.Tensor:
    if axis == "full":
        return pose
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    direction = torch.zeros(3, device=pose.device, dtype=pose.dtype)
    direction[axis_index] = 1.0
    return pose * direction[None]


def silhouette_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    kind: str,
) -> torch.Tensor:
    if kind == "mse":
        return torch.mean((predicted - target) ** 2)
    dimensions = tuple(range(1, predicted.ndim))
    intersection = torch.sum(predicted * target, dim=dimensions)
    predicted_area = torch.sum(predicted, dim=dimensions)
    target_area = torch.sum(target, dim=dimensions)
    epsilon = 1.0e-6
    if kind == "iou":
        union = predicted_area + target_area - intersection
        return torch.mean(1.0 - (intersection + epsilon) / (union + epsilon))
    return torch.mean(
        1.0 - (2.0 * intersection + epsilon)
        / (predicted_area + target_area + epsilon)
    )


def unique_edges(faces: torch.Tensor) -> torch.Tensor:
    edges = torch.cat((faces[:, :2], faces[:, 1:], faces[:, ::2]), dim=0)
    edges = torch.sort(edges, dim=1).values
    return torch.unique(edges, dim=0)


def make_renderer(
    records: list[dict],
    indices: torch.Tensor,
    image_size: int,
    distance: float,
    fov: float,
    device: torch.device,
) -> MeshRenderer:
    elevations = torch.tensor([
        records[int(index)]["elevation_degrees"] for index in indices
    ], device=device)
    azimuths = torch.tensor([
        records[int(index)]["azimuth_degrees"] for index in indices
    ], device=device)
    distances = torch.tensor([
        records[int(index)].get("distance", distance) for index in indices
    ], device=device)
    rotations, translations = look_at_view_transform(
        distances, elevations, azimuths,
    )
    cameras = FoVPerspectiveCameras(
        R=rotations,
        T=translations,
        fov=fov,
        device=device,
    )
    sigma = 1e-4
    raster = RasterizationSettings(
        image_size=image_size,
        blur_radius=math.log(1.0 / 1e-4 - 1.0) * sigma,
        faces_per_pixel=20,
        # The binned CUDA path can silently omit faces for layered character
        # assets even when its per-bin buffer does not report an overflow.
        # At 96-128 px the exact naive path is still fast on a modern GPU.
        bin_size=0,
        max_faces_per_bin=None,
    )
    return MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster),
        shader=SoftSilhouetteShader(),
    )


def render_views(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    records: list[dict],
    indices: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    mesh = Meshes(verts=[vertices], faces=[faces]).extend(len(indices))
    renderer = make_renderer(
        records, indices, args.image_size, args.distance, args.fov,
        vertices.device,
    )
    if not args.hard_forward_silhouette:
        return renderer(mesh)[..., 3]
    fragments = renderer.rasterizer(mesh)
    soft = renderer.shader(fragments, mesh)[..., 3]
    hard = (fragments.pix_to_face[..., 0] >= 0).to(soft.dtype)
    return soft + (hard - soft).detach()


def _comparison_sheet(
    targets: np.ndarray,
    predictions: np.ndarray,
    records: list[dict],
    target: Path,
) -> None:
    scale = 2
    size = targets.shape[-1] * scale
    label_height = 24
    sheet = Image.new("RGB", (size * 2, (size + label_height) * 9), "black")
    draw = ImageDraw.Draw(sheet)
    for row, record in enumerate(records):
        y = row * (size + label_height)
        draw.text((4, y + 4), f"target  {record['name']}", fill="white")
        draw.text((size + 4, y + 4), "render", fill="white")
        for column, array in enumerate((targets[row], predictions[row])):
            image = Image.fromarray(
                np.clip(array * 255.0, 0, 255).astype(np.uint8), "L"
            ).resize((size, size), Image.Resampling.NEAREST).convert("RGB")
            sheet.paste(image, (column * size, y + label_height))
    sheet.save(target)


def _export_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual.vertex_colors = np.tile(
        np.array([[180, 205, 235, 255]], dtype=np.uint8),
        (len(vertices), 1),
    )
    mesh.export(path)


def main() -> int:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, targets = load_views(
        args.image_dir, args.image_size, args.mask_threshold, args.mask_dir,
    )
    vertices_np, faces_np, normalization = load_template(
        args.template,
        args.template_faces,
        args.geometry_name,
        args.obj_group,
        args.source_up_axis,
        args.voxel_resolution,
    )
    targets = targets.to(device)
    vertices = torch.from_numpy(vertices_np).to(device)
    faces = torch.from_numpy(faces_np).to(device)
    if (args.rig is None) != (args.skin_weights is None):
        raise SystemExit("--rig and --skin-weights must be provided together")
    articulation = None
    pose = None
    log_scale = None
    translation = None
    offsets = None
    if args.rig is not None:
        if args.template_faces > 0 or args.voxel_resolution > 0:
            raise SystemExit(
                "articulated fitting requires --template-faces 0 and no voxel remesh"
            )
        pose_bones = [
            name.strip() for name in args.pose_bones.split(",") if name.strip()
        ]
        articulation = load_articulation(
            args.template,
            args.rig,
            args.skin_weights,
            len(vertices_np),
            normalization,
            pose_bones,
        )
        articulation["bone_heads_tensor"] = torch.from_numpy(
            articulation["bone_heads"],
        ).to(device)
        articulation["weights_tensor"] = torch.from_numpy(
            articulation["weights"],
        ).to(device)
        pose_dimensions = 3 if args.pose_axis == "full" else 1
        pose = torch.zeros(
            (len(pose_bones), pose_dimensions),
            device=device,
            requires_grad=True,
        )
        log_scale = torch.zeros(
            (), device=device, requires_grad=args.optimize_alignment,
        )
        translation = torch.zeros(
            3, device=device, requires_grad=args.optimize_alignment,
        )
        trainable = [pose]
        if args.optimize_alignment:
            trainable.extend((log_scale, translation))
    else:
        offsets = torch.zeros_like(vertices, requires_grad=True)
        trainable = [offsets]

    def current_vertices() -> torch.Tensor:
        if articulation is None:
            return vertices + offsets
        return articulate_vertices(
            vertices,
            articulation["bone_heads_tensor"],
            articulation["parents"],
            articulation["weights_tensor"],
            articulation["pose_bone_indices"],
            pose_axis_angles(pose, args.pose_axis),
            log_scale,
            translation,
        )

    edges = unique_edges(faces)
    initial_lengths = torch.linalg.vector_norm(
        vertices[edges[:, 0]] - vertices[edges[:, 1]], dim=1,
    ).detach()
    optimizer = torch.optim.Adam(trainable, lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed)
    all_indices = torch.arange(len(records))
    history = []
    started = time.monotonic()

    with torch.no_grad():
        initial = render_views(
            vertices, faces, records, all_indices, args,
        ).cpu().numpy()
    targets_np = targets.cpu().numpy()
    initial_tensor = torch.from_numpy(initial).to(device)
    initial_full_mse = float(torch.mean((initial_tensor - targets) ** 2))
    initial_full_loss = float(silhouette_loss(
        initial_tensor, targets, args.silhouette_loss,
    ))
    best_full_loss = initial_full_loss
    best_iteration = 0
    best_vertices = vertices.detach().clone()
    best_pose = None
    best_log_scale = None
    best_translation = None
    _comparison_sheet(
        targets_np, initial, records, args.output_dir / "initial.png",
    )
    _export_mesh(args.output_dir / "template.ply", vertices_np, faces_np)

    for iteration in range(args.iterations):
        permutation = torch.randperm(len(records), generator=generator)
        indices = permutation[:args.batch_size]
        current = current_vertices()
        predicted = render_views(current, faces, records, indices, args)
        silhouette = silhouette_loss(
            predicted, targets[indices], args.silhouette_loss,
        )
        edge_vectors = current[edges[:, 0]] - current[edges[:, 1]]
        edge_lengths = torch.linalg.vector_norm(edge_vectors, dim=1)
        edge = torch.mean((edge_lengths - initial_lengths) ** 2)
        displacement = current - vertices
        if articulation is None:
            smooth = torch.mean(torch.sum(
                (offsets[edges[:, 0]] - offsets[edges[:, 1]]) ** 2,
                dim=1,
            ))
            anchor = torch.mean(offsets ** 2)
            pose_penalty = torch.zeros((), device=device)
            alignment_penalty = torch.zeros((), device=device)
        else:
            smooth = torch.zeros((), device=device)
            anchor = torch.zeros((), device=device)
            pose_penalty = torch.mean(pose ** 2)
            alignment_penalty = log_scale ** 2 + torch.mean(translation ** 2)
        loss = (
            silhouette
            + args.edge_weight * edge
            + args.smooth_weight * smooth
            + args.anchor_weight * anchor
            + args.pose_weight * pose_penalty
            + args.alignment_weight * alignment_penalty
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, args.gradient_clip)
        optimizer.step()
        entry = {
            "iteration": iteration + 1,
            "total": float(loss.detach()),
            "silhouette": float(silhouette.detach()),
            "edge": float(edge.detach()),
            "smooth": float(smooth.detach()),
            "anchor": float(anchor.detach()),
            "pose_penalty": float(pose_penalty.detach()),
            "alignment_penalty": float(alignment_penalty.detach()),
            "mean_offset": float(torch.linalg.vector_norm(
                displacement.detach(), dim=1,
            ).mean()),
            "max_offset": float(torch.linalg.vector_norm(
                displacement.detach(), dim=1,
            ).max()),
        }
        history.append(entry)
        should_evaluate = (
            iteration == 0
            or (iteration + 1) % args.evaluation_interval == 0
            or iteration + 1 == args.iterations
        )
        if should_evaluate:
            with torch.no_grad():
                evaluated = render_views(
                    current_vertices(), faces, records, all_indices, args,
                )
                full_mse = float(torch.mean((evaluated - targets) ** 2))
                full_loss = float(silhouette_loss(
                    evaluated, targets, args.silhouette_loss,
                ))
            entry["full_silhouette_mse"] = full_mse
            entry["full_silhouette_loss"] = full_loss
            if full_loss < best_full_loss:
                best_full_loss = full_loss
                best_iteration = iteration + 1
                best_vertices = current_vertices().detach().clone()
                if articulation is not None:
                    best_pose = pose.detach().clone()
                    best_log_scale = log_scale.detach().clone()
                    best_translation = translation.detach().clone()
            if articulation is not None:
                entry["pose_degrees"] = {
                    name: [float(value) for value in row]
                    for name, row in zip(
                        articulation["pose_bones"],
                        torch.rad2deg(pose_axis_angles(
                            pose.detach(), args.pose_axis,
                        )).cpu().tolist(),
                    )
                }
                entry["global_scale"] = float(torch.exp(log_scale.detach()))
                entry["translation"] = [
                    float(value) for value in translation.detach().cpu()
                ]
            print(json.dumps(entry), flush=True)

    fitted = best_vertices
    with torch.no_grad():
        final = render_views(
            fitted, faces, records, all_indices, args,
        ).cpu().numpy()
    _comparison_sheet(
        targets_np, final, records, args.output_dir / "final.png",
    )
    _export_mesh(
        args.output_dir / "fitted.ply", fitted.cpu().numpy(), faces_np,
    )
    report = {
        "schema": "diffusion-editor.qwen-silhouette-mesh-fit",
        "schema_version": 1,
        "algorithm": "PyTorch3D differentiable silhouette mesh fitting",
        "image_dir": str(args.image_dir.resolve()),
        "template": str(args.template.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "parameters": vars(args) | {
            "image_dir": str(args.image_dir),
            "template": str(args.template),
            "output_dir": str(args.output_dir),
            "rig": None if args.rig is None else str(args.rig),
            "skin_weights": (
                None if args.skin_weights is None else str(args.skin_weights)
            ),
            "mask_dir": None if args.mask_dir is None else str(args.mask_dir),
        },
        "views": records,
        "template_vertices": int(len(vertices_np)),
        "template_faces": int(len(faces_np)),
        "initial_full_silhouette_mse": float(np.mean(
            (initial - targets_np) ** 2
        )),
        "initial_full_silhouette_loss": initial_full_loss,
        "final_full_silhouette_mse": float(np.mean(
            (final - targets_np) ** 2
        )),
        "final_full_silhouette_loss": float(silhouette_loss(
            torch.from_numpy(final).to(device), targets, args.silhouette_loss,
        )),
        "device": str(device),
        "best_iteration": best_iteration,
        "articulation": None if articulation is None else {
            "rig": str(args.rig.resolve()),
            "skin_weights": str(args.skin_weights.resolve()),
            "pose_bones": articulation["pose_bones"],
            "best_pose_degrees": None if best_pose is None else {
                name: [float(value) for value in row]
                for name, row in zip(
                    articulation["pose_bones"],
                    torch.rad2deg(pose_axis_angles(
                        best_pose, args.pose_axis,
                    )).cpu().tolist(),
                )
            },
            "best_global_scale": (
                None if best_log_scale is None
                else float(torch.exp(best_log_scale))
            ),
            "best_translation": (
                None if best_translation is None
                else [float(value) for value in best_translation.cpu()]
            ),
        },
        "elapsed_seconds": time.monotonic() - started,
        "history": history,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        key: report[key] for key in (
            "template_vertices", "template_faces",
            "initial_full_silhouette_mse", "final_full_silhouette_mse",
            "initial_full_silhouette_loss", "final_full_silhouette_loss",
            "elapsed_seconds",
        )
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
