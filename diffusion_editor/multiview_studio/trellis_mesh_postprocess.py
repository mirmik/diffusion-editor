"""Fixed, cacheable mesh postprocess pipeline for Multiview Studio."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import time
from typing import Callable


POSTPROCESS_PROTOCOL = 2


def postprocess_key(settings: dict, actual_resolution: int) -> str:
    payload = {
        "protocol": POSTPROCESS_PROTOCOL,
        "actual_resolution": int(actual_resolution),
        "settings": settings,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def postprocess_output_path(
    output_dir: Path,
    settings: dict,
    actual_resolution: int,
) -> Path:
    key = postprocess_key(settings, actual_resolution)
    return Path(output_dir) / f"shape-post-{key}.glb"


def run_mesh_postprocess(
    cache_path: Path,
    output_dir: Path,
    settings: dict,
    actual_resolution: int,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[Path, dict]:
    import cumesh
    import numpy as np
    import torch
    import trimesh

    cache_path = Path(cache_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    key = postprocess_key(settings, actual_resolution)
    output_path = postprocess_output_path(output_dir, settings, actual_resolution)
    report_path = output_dir / f"postprocess-{key}.json"
    if output_path.is_file() and report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if Path(report.get("cache", "")).resolve() == cache_path:
            _progress(progress, f"[postprocess-cache] {output_path.name}")
            return output_path, report

    started = time.perf_counter()
    cached = np.load(cache_path)
    vertices = np.ascontiguousarray(cached["vertices"], dtype=np.float32)
    faces = np.ascontiguousarray(cached["faces"], dtype=np.int32)
    source_counts = _counts(vertices, faces)
    stages: list[dict] = []

    gpu_stages = any(
        bool(settings.get(name, False))
        for name in ("fill_holes", "remesh", "simplify", "cleanup")
    )
    gpu_mesh = None
    gpu_vertices = None
    gpu_faces = None
    if gpu_stages:
        gpu_vertices = torch.from_numpy(vertices).cuda().contiguous()
        gpu_faces = torch.from_numpy(faces).cuda().contiguous()
        gpu_mesh = cumesh.CuMesh()
        gpu_mesh.init(gpu_vertices, gpu_faces)

    if settings.get("fill_holes", False):
        mark = time.perf_counter()
        _progress(progress, "[postprocess] Initial CuMesh fill holes")
        gpu_mesh.fill_holes(
            max_hole_perimeter=float(settings["fill_hole_perimeter"])
        )
        gpu_vertices, gpu_faces = gpu_mesh.read()
        stages.append(
            _stage("fill_holes", gpu_vertices, gpu_faces, mark)
        )

    if settings.get("remesh", False):
        mark = time.perf_counter()
        _progress(progress, "[postprocess] CuMesh narrow-band remesh")
        gpu_vertices, gpu_faces = gpu_mesh.read()
        lower = gpu_vertices.amin(dim=0)
        upper = gpu_vertices.amax(dim=0)
        center = (lower + upper) * 0.5
        extent = float((upper - lower).max().item())
        resolution = int(actual_resolution)
        band = float(settings.get("remesh_band", 1.0))
        bvh = cumesh.cuBVH(gpu_vertices, gpu_faces)
        gpu_vertices, gpu_faces = cumesh.remeshing.remesh_narrow_band_dc(
            gpu_vertices,
            gpu_faces,
            center=center,
            scale=(resolution + 3 * band) / resolution * extent,
            resolution=resolution,
            band=band,
            project_back=float(settings.get("remesh_project", 0.0)),
            bvh=bvh,
            verbose=True,
        )
        gpu_mesh.init(gpu_vertices, gpu_faces)
        stages.append(_stage("remesh", gpu_vertices, gpu_faces, mark))

    if settings.get("simplify", False):
        target = int(settings["decimation_target"])
        gpu_vertices, gpu_faces = gpu_mesh.read()
        if len(gpu_faces) > target:
            mark = time.perf_counter()
            _progress(progress, f"[postprocess] CuMesh simplify to {target:,}")
            gpu_mesh.simplify(target, verbose=True)
            gpu_vertices, gpu_faces = gpu_mesh.read()
            stages.append(_stage("simplify", gpu_vertices, gpu_faces, mark))

    if settings.get("cleanup", False):
        mark = time.perf_counter()
        _progress(progress, "[postprocess] CuMesh topology cleanup")
        gpu_mesh.remove_duplicate_faces()
        gpu_mesh.repair_non_manifold_edges()
        gpu_mesh.remove_small_connected_components(1e-5)
        gpu_mesh.fill_holes(
            max_hole_perimeter=float(settings["fill_hole_perimeter"])
        )
        gpu_vertices, gpu_faces = gpu_mesh.read()
        stages.append(_stage("cleanup", gpu_vertices, gpu_faces, mark))

    if gpu_stages:
        gpu_vertices, gpu_faces = gpu_mesh.read()
        torch.cuda.synchronize()
        vertices = gpu_vertices.detach().cpu().numpy().astype(np.float32, copy=False)
        faces = gpu_faces.detach().cpu().numpy().astype(np.int32, copy=False)

    if settings.get("final_repair", False):
        mark = time.perf_counter()
        _progress(progress, "[postprocess] Final MeshLib repair")
        vertices, faces, repair = _meshlib_repair(
            vertices,
            faces,
            float(settings["fill_hole_perimeter"]),
            output_dir,
        )
        stage = _stage("final_repair", vertices, faces, mark)
        stage.update(repair)
        stages.append(stage)

    if settings.get("remove_isolated_double_faces", False):
        mark = time.perf_counter()
        _progress(progress, "[postprocess] Remove isolated double faces")
        groups = trimesh.grouping.group_rows(
            np.sort(faces, axis=1), require_count=2
        )
        removed = int(len(groups) * 2)
        if removed:
            keep = np.ones(len(faces), dtype=bool)
            keep[np.asarray(groups).reshape(-1)] = False
            faces = faces[keep]
        stage = _stage("remove_isolated_double_faces", vertices, faces, mark)
        stage["removed_faces"] = removed
        stages.append(stage)

    if settings.get("remove_degenerate_faces", False):
        mark = time.perf_counter()
        _progress(progress, "[postprocess] Remove exact zero-area faces")
        degenerate = _exact_degenerate_faces(vertices, faces)
        removed = int(degenerate.sum())
        if removed:
            faces = faces[~degenerate]
        stage = _stage("remove_degenerate_faces", vertices, faces, mark)
        stage["removed_faces"] = removed
        stages.append(stage)

    result = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    result.remove_unreferenced_vertices()
    topology = _topology_counts(
        np.asarray(result.faces, dtype=np.int64)
    )
    degenerate_faces = int(
        _exact_degenerate_faces(
            np.asarray(result.vertices), np.asarray(result.faces)
        ).sum()
    )
    result.apply_transform(_gltf_y_up_transform(np))
    _ = result.vertex_normals
    result.export(output_path)

    report = {
        "protocol": POSTPROCESS_PROTOCOL,
        "key": key,
        "cache": str(cache_path),
        "output": str(output_path),
        "actual_resolution": int(actual_resolution),
        "settings": settings,
        "source": source_counts,
        "stages": stages,
        "output_counts": {
            "vertices": int(len(result.vertices)),
            "faces": int(len(result.faces)),
            "degenerate_faces": degenerate_faces,
            **topology,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    _progress(progress, f"[postprocess-complete] {output_path.name}")
    return output_path, report


def _meshlib_repair(vertices, faces, max_perimeter: float, output_dir: Path):
    import numpy as np
    import trimesh
    from meshlib import mrmeshpy as meshlib

    with tempfile.TemporaryDirectory(
        prefix="mesh-repair-", dir=output_dir
    ) as temporary_dir:
        root = Path(temporary_dir)
        source_path = root / "source.ply"
        repaired_path = root / "repaired.ply"
        trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(
            source_path
        )
        mesh = meshlib.loadMesh(source_path)
        initial_holes = len(mesh.topology.findHoleRepresentiveEdges())
        duplicated = int(meshlib.duplicateMultiHoleVertices(mesh))
        holes = list(mesh.topology.findHoleRepresentiveEdges())
        selected = meshlib.std_vector_Id_EdgeTag()
        for edge in holes:
            perimeter = meshlib.holePerimeter(mesh.topology, mesh.points, edge)
            if perimeter <= max_perimeter:
                selected.append(edge)
        fill_params = meshlib.FillHoleParams()
        fill_params.smoothBd = True
        fill_params.makeDegenerateBand = False
        meshlib.fillHoles(mesh, selected, fill_params)

        degenerates = meshlib.findDegenerateFaces(
            meshlib.MeshPart(mesh), 1e12
        )
        degenerates_before = int(degenerates.count())
        if degenerates_before:
            params = meshlib.FixMeshDegeneraciesParams()
            params.criticalTriAspectRatio = 1e12
            params.tinyEdgeLength = 0.0
            params.maxDeviation = 0.001
            params.region = degenerates
            params.mode = meshlib.FixMeshDegeneraciesParams.Mode.RemeshPatch
            params.mimicPatch = True
            meshlib.fixMeshDegeneracies(mesh, params)
        remaining_holes = len(mesh.topology.findHoleRepresentiveEdges())
        meshlib.saveMesh(mesh, repaired_path)
        loaded = trimesh.load(repaired_path, force="mesh", process=False)
        repaired_vertices = np.asarray(loaded.vertices, dtype=np.float32)
        repaired_faces = np.asarray(loaded.faces, dtype=np.int32)
    return repaired_vertices, repaired_faces, {
        "initial_holes": initial_holes,
        "normalized_holes": len(holes),
        "duplicated_multi_hole_vertices": duplicated,
        "filled_holes": len(selected),
        "remaining_holes": remaining_holes,
        "exact_degenerates_repaired": degenerates_before,
    }


def _stage(name: str, vertices, faces, started: float) -> dict:
    return {
        "name": name,
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _counts(vertices, faces) -> dict:
    return {"vertices": int(len(vertices)), "faces": int(len(faces))}


def _topology_counts(faces) -> dict[str, int]:
    import numpy as np

    if not len(faces):
        return {"boundary_edges": 0, "multi_face_nonmanifold_edges": 0}
    edges = np.concatenate(
        (faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]), axis=0
    )
    _unique, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    return {
        "boundary_edges": int(np.count_nonzero(counts == 1)),
        "multi_face_nonmanifold_edges": int(np.count_nonzero(counts > 2)),
    }


def _exact_degenerate_faces(vertices, faces):
    import numpy as np

    if not len(faces):
        return np.zeros(0, dtype=bool)
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    edge_a = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    edge_b = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    return np.all(np.cross(edge_a, edge_b) == 0, axis=1)


def _gltf_y_up_transform(np):
    return np.asarray(
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def _progress(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(message)
