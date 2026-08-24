#!/usr/bin/env python3
"""Render character RGB images and exact canonical point-map supervision.

Run this script through Blender, for example:

    blender --background --disable-autoexec rain_rig.blend \
      --python scripts/render-rain-canonical-dataset.py -- \
      --output-dir /tmp/rain-canonical --views 24

With no asset-selection option this preserves the original Rain-v2 behavior.
An FBX can instead be imported into an empty Blender scene with ``--asset``, or
the visible character geometry in an already opened blend can be selected with
``--opened-blend-character``.  All modes keep the authored pose, so a
world-space ray hit is the exact canonical surface correspondence for this
identity/pose.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import bpy
from mathutils import Vector
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from diffusion_editor.training.canonical_pointmap import (  # noqa: E402
    CANONICAL_FROM_BLENDER_AXES,
    camera_from_canonical_matrix,
    camera_distance_world,
    camera_intrinsics,
    canonical_from_world,
    pointmap_reprojection_error,
    write_binary_mask_png,
    world_from_canonical_matrix,
)
from diffusion_editor.training.character_mesh_selection import (  # noqa: E402
    CharacterMeshCandidate,
    select_character_mesh_names,
)


SCHEMA_VERSION = 1
RAIN_SOURCE_PAGE = "https://studio.blender.org/characters/rain/v2/"
RAIN_ARCHIVE_SHA256 = (
    "91d9ecf53b62c5ac533d86d711697fbe"
    "177b9ca6968ef46471661e2ffd8586eb"
)


def _script_arguments() -> argparse.Namespace:
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--asset",
        type=Path,
        help="Import this FBX instead of using Rain from the opened .blend file.",
    )
    parser.add_argument(
        "--opened-blend-character",
        action="store_true",
        help=(
            "Use render-visible meshes in the currently opened blend instead "
            "of assuming the Rain v2 naming convention."
        ),
    )
    parser.add_argument(
        "--include-mesh",
        action="append",
        dest="include_meshes",
        help=(
            "Explicit mesh object to include even if authored render-hidden; "
            "repeat for multiple geometric fallbacks."
        ),
    )
    parser.add_argument(
        "--only-mesh",
        action="append",
        dest="only_meshes",
        help=(
            "Use exactly this mesh object from an opened Blend scene; repeat "
            "for a multi-mesh character. All other meshes are hidden."
        ),
    )
    parser.add_argument("--asset-name", help="Human-readable imported asset name.")
    parser.add_argument("--identity-id", help="Stable identity split label.")
    parser.add_argument("--pose-id", default="authored-rest")
    parser.add_argument(
        "--rig-name",
        help="Armature containing --pelvis-bone for an opened blend character.",
    )
    parser.add_argument(
        "--pelvis-bone",
        help=(
            "Pose bone whose head defines the canonical origin. If omitted, "
            "a conservative pelvis/spine-name heuristic is used."
        ),
    )
    parser.add_argument("--creator", help="Asset creator recorded in the manifest.")
    parser.add_argument("--source-page", help="Asset source URL for the manifest.")
    parser.add_argument("--license", dest="asset_license", help="Asset license.")
    parser.add_argument(
        "--subdivision-render-level",
        type=int,
        help=(
            "Override render_levels on subdivision modifiers of selected "
            "character meshes; zero uses the authored base mesh."
        ),
    )
    parser.add_argument(
        "--pelvis-height-fraction",
        type=float,
        default=0.5,
        help=(
            "For unrigged FBX only: pelvis Z as a fraction of evaluated height "
            "above the lowest character point."
        ),
    )
    parser.add_argument("--views", type=int, default=24)
    parser.add_argument(
        "--azimuth",
        action="append",
        type=float,
        help="Render an explicit azimuth; repeat to select multiple views.",
    )
    parser.add_argument(
        "--azimuth-pair",
        action="append",
        dest="azimuth_pairs",
        metavar="PROMPT:CAMERA",
        help=(
            "Render CAMERA azimuth while recording PROMPT as the requested "
            "semantic azimuth; repeat to create camera-jitter supervision."
        ),
    )
    parser.add_argument(
        "--elevation",
        action="append",
        type=float,
        dest="elevations",
        help=(
            "Render an elevation ring; repeat to select multiple rings. "
            "Defaults to eye level."
        ),
    )
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--lens-mm", type=float, default=58.0)
    parser.add_argument("--sensor-width-mm", type=float, default=36.0)
    parser.add_argument("--distance", type=float, default=4.0)
    parser.add_argument(
        "--distance-unit",
        choices=("world", "character-height"),
        default="world",
        help=(
            "Interpret --distance as absolute Blender units (legacy) or as "
            "multiples of evaluated character height."
        ),
    )
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument(
        "--render-engine",
        choices=("eevee", "workbench"),
        default="eevee",
        help="RGB render engine; geometry supervision always uses exact raycasts.",
    )
    parser.add_argument(
        "--minimum-render-alpha",
        type=float,
        default=0.5,
        help="Keep a ray hit only where rendered alpha reaches this value.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(arguments)
    if args.views <= 0:
        parser.error("--views must be positive")
    if args.resolution <= 0:
        parser.error("--resolution must be positive")
    if args.distance <= 0.0:
        parser.error("--distance must be positive")
    if not 0.0 < args.pelvis_height_fraction < 1.0:
        parser.error("--pelvis-height-fraction must be between zero and one")
    if not 0.0 <= args.minimum_render_alpha <= 1.0:
        parser.error("--minimum-render-alpha must be between zero and one")
    if args.asset is not None and args.opened_blend_character:
        parser.error("--asset and --opened-blend-character are mutually exclusive")
    if (args.rig_name or args.pelvis_bone) and not args.opened_blend_character:
        parser.error("--rig-name/--pelvis-bone require --opened-blend-character")
    if args.only_meshes and not args.opened_blend_character:
        parser.error("--only-mesh requires --opened-blend-character")
    if args.include_meshes and args.only_meshes:
        parser.error("--include-mesh and --only-mesh are mutually exclusive")
    if args.azimuth and args.azimuth_pairs:
        parser.error("--azimuth and --azimuth-pair are mutually exclusive")
    if args.subdivision_render_level is not None and args.subdivision_render_level < 0:
        parser.error("--subdivision-render-level must be non-negative")
    return args


def _matrix_array(matrix) -> np.ndarray:
    return np.asarray([[float(value) for value in row] for row in matrix])


def _json_matrix(matrix: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]


def _visible_character_meshes(
    mode: str,
    include_meshes: set[str] | None = None,
    only_meshes: set[str] | None = None,
) -> list[bpy.types.Object]:
    include_meshes = include_meshes or set()
    only_meshes = only_meshes or set()
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    selected_names = set(
        select_character_mesh_names(
            (
                CharacterMeshCandidate(
                    name=obj.name,
                    render_hidden=obj.hide_render,
                    viewport_visible=obj.visible_get(),
                )
                for obj in mesh_objects
            ),
            mode=mode,
            include_meshes=include_meshes,
            only_meshes=only_meshes,
        )
    )
    character = [obj for obj in mesh_objects if obj.name in selected_names]
    if not character:
        expected = "GEO-rain_* meshes" if mode == "rain" else "visible mesh objects"
        raise RuntimeError(f"no visible {expected} found")
    selected = set(character)
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        visible = obj in selected
        obj.hide_render = not visible
        if obj.name in bpy.context.view_layer.objects:
            obj.hide_set(not visible)
        # Blender particle hair has no corresponding scene-ray surface.  It
        # must not leak into RGB alpha when exact point-map supervision is
        # produced from ray-castable meshes only.
        for modifier in obj.modifiers:
            if modifier.type == "PARTICLE_SYSTEM":
                modifier.show_render = False
                modifier.show_viewport = False
    return character


def _pelvis_from_opened_blend(
    args: argparse.Namespace,
) -> tuple[Vector | None, str | None]:
    armatures = [
        obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"
    ]
    if args.rig_name:
        armatures = [obj for obj in armatures if obj.name == args.rig_name]
        if not armatures:
            raise RuntimeError(f"armature {args.rig_name!r} is missing")
    else:
        final_rigs = [obj for obj in armatures if obj.name.upper().startswith("RIG")]
        if final_rigs:
            armatures = final_rigs
        armatures.sort(key=lambda obj: len(obj.data.bones), reverse=True)

    if args.pelvis_bone:
        candidate_names = (args.pelvis_bone,)
    else:
        candidate_names = (
            "DEF-Pelvis",
            "ORG-Spine",
            "DEF-Hips",
            "DEF-Spine",
            "Hip_Center",
            "DEF-Hip_Center",
            "MSTR-Spine_Hips",
        )
    for bone_name in candidate_names:
        for rig in armatures:
            bone = rig.pose.bones.get(bone_name)
            if bone is not None:
                return rig.matrix_world @ bone.head, f"{rig.name}/{bone.name} head"
    if args.pelvis_bone:
        available = ", ".join(obj.name for obj in armatures) or "none"
        raise RuntimeError(
            f"pelvis bone {args.pelvis_bone!r} is missing from armatures: {available}"
        )
    return None, None


def _evaluated_bounds(
    objects: list[bpy.types.Object], depsgraph: bpy.types.Depsgraph
) -> tuple[np.ndarray, np.ndarray]:
    minimum = np.full(3, np.inf, dtype=np.float64)
    maximum = np.full(3, -np.inf, dtype=np.float64)
    for original in objects:
        evaluated = original.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            matrix = evaluated.matrix_world
            for vertex in mesh.vertices:
                world = matrix @ vertex.co
                minimum = np.minimum(minimum, tuple(world))
                maximum = np.maximum(maximum, tuple(world))
        finally:
            evaluated.to_mesh_clear()
    if not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
        raise RuntimeError("failed to calculate evaluated character bounds")
    return minimum, maximum


def _look_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def _add_area_light(
    name: str,
    *,
    location: Vector,
    target: Vector,
    energy: float,
    size: float,
) -> None:
    data = bpy.data.lights.new(name, "AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    light = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(light)
    light.location = location
    _look_at(light, target)


def _configure_scene(
    args: argparse.Namespace,
    *,
    pelvis: Vector,
    character_height: float,
) -> bpy.types.Object:
    scene = bpy.context.scene
    scene.render.engine = (
        "BLENDER_WORKBENCH"
        if args.render_engine == "workbench"
        else "BLENDER_EEVEE"
    )
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.use_nodes = False
    if args.render_engine == "workbench":
        scene.display.shading.light = "STUDIO"
        scene.display.shading.color_type = "MATERIAL"
        scene.display.shading.show_shadows = True
        scene.display.shading.show_cavity = True
        scene.display.shading.cavity_type = "WORLD"
        scene.display.shading.show_specular_highlight = True
    if hasattr(scene, "eevee") and hasattr(scene.eevee, "taa_render_samples"):
        scene.eevee.taa_render_samples = args.samples
    for obj in bpy.data.objects:
        if obj.type == "LIGHT":
            obj.hide_render = True

    camera_data = bpy.data.cameras.new("CanonicalDatasetCamera")
    camera_data.type = "PERSP"
    camera_data.lens = args.lens_mm
    camera_data.sensor_width = args.sensor_width_mm
    camera_data.sensor_fit = "HORIZONTAL"
    camera_data.shift_x = 0.0
    camera_data.shift_y = 0.0
    camera_data.clip_start = 0.01
    camera_data.clip_end = 100.0
    camera = bpy.data.objects.new("CanonicalDatasetCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera

    scale = character_height
    _add_area_light(
        "CanonicalKey",
        location=pelvis + Vector((1.8 * scale, -2.4 * scale, 2.0 * scale)),
        target=pelvis,
        energy=1100.0,
        size=2.4 * scale,
    )
    _add_area_light(
        "CanonicalFill",
        location=pelvis + Vector((-1.8 * scale, -1.0 * scale, 1.0 * scale)),
        target=pelvis,
        energy=700.0,
        size=1.8 * scale,
    )
    _add_area_light(
        "CanonicalRim",
        location=pelvis + Vector((0.0, 2.0 * scale, 1.8 * scale)),
        target=pelvis,
        energy=900.0,
        size=1.8 * scale,
    )
    return camera


def _position_camera(
    camera: bpy.types.Object,
    *,
    pelvis: Vector,
    azimuth_degrees: float,
    elevation_degrees: float,
    distance: float,
) -> None:
    azimuth = math.radians(azimuth_degrees)
    elevation = math.radians(elevation_degrees)
    horizontal = math.cos(elevation) * distance
    camera.location = pelvis + Vector(
        (
            math.sin(azimuth) * horizontal,
            -math.cos(azimuth) * horizontal,
            math.sin(elevation) * distance,
        )
    )
    _look_at(camera, pelvis)
    bpy.context.view_layer.update()


def _raycast_pointmap(
    *,
    camera: bpy.types.Object,
    intrinsics: np.ndarray,
    pelvis_world: np.ndarray,
    character_height: float,
    object_ids: dict[str, int],
    resolution: int,
) -> dict[str, np.ndarray]:
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    camera_world = camera.matrix_world.copy()
    camera_inverse = camera_world.inverted()
    rotation = camera_world.to_quaternion()
    origin = camera_world.translation
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])

    canonical_xyz = np.zeros((resolution, resolution, 3), dtype=np.float32)
    normal = np.zeros_like(canonical_xyz)
    depth = np.zeros((resolution, resolution), dtype=np.float32)
    mask = np.zeros((resolution, resolution), dtype=np.uint8)
    object_id = np.zeros((resolution, resolution), dtype=np.uint16)
    started = time.monotonic()
    for row in range(resolution):
        y_cv = (row + 0.5 - cy) / fy
        for column in range(resolution):
            x_cv = (column + 0.5 - cx) / fx
            local_direction = Vector((x_cv, -y_cv, -1.0)).normalized()
            world_direction = rotation @ local_direction
            hit, location, world_normal, _face, obj, _matrix = scene.ray_cast(
                depsgraph,
                origin,
                world_direction,
                distance=camera.data.clip_end,
            )
            if not hit or obj is None or obj.name not in object_ids:
                continue
            canonical_xyz[row, column] = canonical_from_world(
                np.asarray(tuple(location)),
                pelvis_world=pelvis_world,
                character_height=character_height,
            )
            canonical_normal = CANONICAL_FROM_BLENDER_AXES @ np.asarray(
                tuple(world_normal), dtype=np.float64
            )
            length = np.linalg.norm(canonical_normal)
            if length > 0.0:
                canonical_normal /= length
            normal[row, column] = canonical_normal
            camera_location = camera_inverse @ location
            depth[row, column] = -float(camera_location.z)
            mask[row, column] = 1
            object_id[row, column] = object_ids[obj.name]
        if resolution >= 128 and (row + 1) % 32 == 0:
            print(
                f"  raycast {row + 1}/{resolution} rows "
                f"({time.monotonic() - started:.1f}s)",
                flush=True,
            )
    return {
        "canonical_xyz": canonical_xyz,
        "normal": normal,
        "depth": depth,
        "mask": mask,
        "object_id": object_id,
    }


def _view_id(azimuth: float, elevation: float) -> str:
    azimuth_label = int(round(azimuth)) % 360
    elevation_label = int(round(elevation))
    return f"az{azimuth_label:03d}_el{elevation_label:+03d}"


def _jitter_view_id(
    index: int, prompt_azimuth: float, camera_azimuth: float, elevation: float
) -> str:
    prompt_label = int(round(prompt_azimuth)) % 360
    camera_label = f"{camera_azimuth % 360.0:07.3f}".replace(".", "p")
    elevation_label = int(round(elevation))
    return (
        f"j{index:03d}_nom{prompt_label:03d}_act{camera_label}"
        f"_el{elevation_label:+03d}"
    )


def _signed_angle_delta(actual: float, nominal: float) -> float:
    return (float(actual) - float(nominal) + 180.0) % 360.0 - 180.0


def _render_alpha_mask(
    path: Path, resolution: int, minimum_alpha: float
) -> np.ndarray:
    rendered = bpy.data.images.load(str(path), check_existing=False)
    try:
        render_size = tuple(int(value) for value in rendered.size)
        if render_size != (resolution, resolution):
            raise RuntimeError(
                f"rendered PNG has size {render_size}, expected "
                f"{(resolution, resolution)}"
            )
        rgba = np.asarray(rendered.pixels[:], dtype=np.float32).reshape(
            resolution, resolution, 4
        )
        # Blender exposes loaded image pixels from the lower-left while the
        # dataset uses conventional top-left image rows.
        return np.flipud(rgba[:, :, 3]) >= minimum_alpha
    finally:
        bpy.data.images.remove(rendered)


def _apply_visibility_mask(
    arrays: dict[str, np.ndarray], visible: np.ndarray
) -> None:
    keep = np.logical_and(arrays["mask"] != 0, visible)
    removed = np.logical_not(keep)
    arrays["canonical_xyz"][removed] = 0.0
    arrays["normal"][removed] = 0.0
    arrays["depth"][removed] = 0.0
    arrays["object_id"][removed] = 0
    arrays["mask"] = keep.astype(np.uint8)


def _write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = _script_arguments()
    if bpy.context.preferences.filepaths.use_scripts_auto_execute:
        raise RuntimeError(
            "embedded scripts are enabled; rerun Blender with --disable-autoexec"
        )
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and not args.force:
        raise SystemExit(
            f"output already contains {manifest_path}; pass --force to overwrite views"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    views_dir = args.output_dir / "views"
    views_dir.mkdir(parents=True, exist_ok=True)

    asset_path = args.asset.resolve() if args.asset is not None else None
    if asset_path is not None:
        if not asset_path.is_file():
            raise RuntimeError(f"FBX asset does not exist: {asset_path}")
        bpy.ops.wm.read_factory_settings(use_empty=True)
        result = bpy.ops.import_scene.fbx(filepath=str(asset_path), use_anim=False)
        if "FINISHED" not in result:
            raise RuntimeError(f"failed to import {asset_path}: {result}")
    if args.opened_blend_character:
        if not bpy.data.filepath:
            raise RuntimeError("--opened-blend-character requires an opened blend file")
        asset_mode = "opened-blend"
        opened_blend_path = Path(bpy.data.filepath).resolve()
    elif asset_path is not None:
        asset_mode = "imported"
        opened_blend_path = None
    else:
        asset_mode = "rain"
        opened_blend_path = None
    character = _visible_character_meshes(
        asset_mode,
        set(args.include_meshes or []),
        set(args.only_meshes or []),
    )
    if args.subdivision_render_level is not None:
        for obj in character:
            for modifier in obj.modifiers:
                if modifier.type == "SUBSURF":
                    modifier.render_levels = args.subdivision_render_level
                    modifier.levels = min(
                        modifier.levels, args.subdivision_render_level
                    )
    depsgraph = bpy.context.evaluated_depsgraph_get()
    minimum, maximum = _evaluated_bounds(character, depsgraph)
    character_height = float(maximum[2] - minimum[2])
    if character_height <= 0.0:
        raise RuntimeError("character has a non-positive evaluated height")
    resolved_camera_distance = camera_distance_world(
        args.distance,
        character_height=character_height,
        unit=args.distance_unit,
    )
    if asset_mode == "rain":
        rig = bpy.data.objects.get("RIG-rain")
        if rig is None or rig.type != "ARMATURE":
            raise RuntimeError("Rain armature RIG-rain is missing")
        pelvis_bone = rig.pose.bones.get("DEF-Pelvis")
        if pelvis_bone is None:
            raise RuntimeError("Rain pelvis bone DEF-Pelvis is missing")
        pelvis = rig.matrix_world @ pelvis_bone.head
        pelvis_origin = "RIG-rain/DEF-Pelvis head"
    else:
        pelvis = None
        pelvis_origin = None
        if asset_mode == "opened-blend":
            pelvis, pelvis_origin = _pelvis_from_opened_blend(args)
        if pelvis is None:
            object_origins = np.asarray(
                [tuple(obj.matrix_world.translation) for obj in character],
                dtype=np.float64,
            )
            center_xy = np.median(object_origins[:, :2], axis=0)
            pelvis = Vector(
                (
                    float(center_xy[0]),
                    float(center_xy[1]),
                    float(
                        minimum[2]
                        + args.pelvis_height_fraction * character_height
                    ),
                )
            )
            pelvis_origin = (
                "estimated from median mesh-object XY origins and "
                f"{args.pelvis_height_fraction:.6g} of evaluated height"
            )
    pelvis_array = np.asarray(tuple(pelvis), dtype=np.float64)
    camera = _configure_scene(
        args, pelvis=pelvis, character_height=character_height
    )
    intrinsics = camera_intrinsics(
        width=args.resolution,
        height=args.resolution,
        lens_mm=args.lens_mm,
        sensor_width_mm=args.sensor_width_mm,
    )
    object_ids = {
        name: index + 1 for index, name in enumerate(sorted(obj.name for obj in character))
    }
    if args.azimuth_pairs:
        azimuth_pairs = []
        for value in args.azimuth_pairs:
            try:
                prompt_text, camera_text = value.split(":", 1)
                azimuth_pairs.append((float(prompt_text), float(camera_text)))
            except ValueError as error:
                raise SystemExit(
                    f"invalid --azimuth-pair {value!r}; expected PROMPT:CAMERA"
                ) from error
    else:
        azimuths = args.azimuth or [
            index * 360.0 / args.views for index in range(args.views)
        ]
        azimuth_pairs = [(azimuth, azimuth) for azimuth in azimuths]
    elevations = args.elevations or [0.0]
    view_specs = [
        (prompt_azimuth, camera_azimuth, elevation)
        for elevation in elevations
        for prompt_azimuth, camera_azimuth in azimuth_pairs
    ]
    if asset_mode == "rain":
        asset_record = {
            "name": "Rain v2",
            "identity_id": args.identity_id or "rain-v2",
            "pose_id": args.pose_id,
            "creator": "Blender Studio",
            "source_page": RAIN_SOURCE_PAGE,
            "archive_sha256": RAIN_ARCHIVE_SHA256,
            "license": "CC-BY",
            "blend_file": str(Path(bpy.data.filepath).resolve()),
        }
    elif asset_mode == "imported":
        asset_record = {
            "name": args.asset_name or asset_path.stem,
            "identity_id": args.identity_id or asset_path.stem.lower(),
            "pose_id": args.pose_id,
            "creator": "local MakeHuman export; creator unverified",
            "source_page": None,
            "sha256": hashlib.sha256(asset_path.read_bytes()).hexdigest(),
            "license": "local user asset; license unverified",
            "fbx_file": str(asset_path),
            "rigged_after_import": any(
                obj.type == "ARMATURE" for obj in bpy.data.objects
            ),
        }
    else:
        asset_record = {
            "name": args.asset_name or opened_blend_path.stem,
            "identity_id": args.identity_id or opened_blend_path.stem.lower(),
            "pose_id": args.pose_id,
            "creator": args.creator or "unverified",
            "source_page": args.source_page,
            "sha256": hashlib.sha256(opened_blend_path.read_bytes()).hexdigest(),
            "license": args.asset_license or "unverified",
            "blend_file": str(opened_blend_path),
            "rigged": any(obj.type == "ARMATURE" for obj in bpy.data.objects),
        }
    manifest = {
        "schema": "diffusion-editor.canonical-pointmap-dataset",
        "schema_version": SCHEMA_VERSION,
        "asset": asset_record,
        "generator": {
            "script": str(Path(__file__).resolve()),
            "blender_version": bpy.app.version_string,
            "embedded_scripts_enabled": False,
        },
        "canonical_frame": {
            "origin": pelvis_origin,
            "axes": {"x": "character right", "y": "up", "z": "forward"},
            "handedness": "right",
            "unit": "evaluated character height",
            "character_height_world": character_height,
            "pelvis_world": [float(value) for value in pelvis_array],
            "world_from_canonical": _json_matrix(
                world_from_canonical_matrix(pelvis_array, character_height)
            ),
            "evaluated_bounds_world": {
                "minimum": [float(value) for value in minimum],
                "maximum": [float(value) for value in maximum],
            },
        },
        "render": {
            "engine": bpy.context.scene.render.engine,
            "engine_mode": args.render_engine,
            "resolution": [args.resolution, args.resolution],
            "samples": args.samples,
            "transparent_background": True,
            "lens_mm": args.lens_mm,
            "sensor_width_mm": args.sensor_width_mm,
            "distance": args.distance,
            "distance_unit": args.distance_unit,
            "distance_world": resolved_camera_distance,
            "subdivision_render_level_override": args.subdivision_render_level,
            "color_view_transform": bpy.context.scene.view_settings.view_transform,
        },
        "geometry": {
            "canonical_xyz": "float32, zero outside mask",
            "normal": "float32 canonical-frame unit vector, zero outside mask",
            "depth": "float32 CV camera-forward Z, zero outside mask",
            "mask": "uint8, values 0 or 1",
            "object_id": "uint16, zero is background",
            "object_ids": object_ids,
            "visibility": (
                "scene ray hit intersected with rendered alpha >= "
                f"{args.minimum_render_alpha:g}"
            ),
            "particle_systems_rendered": False,
        },
        "views": [],
    }
    _write_manifest(manifest_path, manifest)

    started = time.monotonic()
    for index, (prompt_azimuth, azimuth, elevation) in enumerate(view_specs, start=1):
        identifier = (
            _jitter_view_id(index, prompt_azimuth, azimuth, elevation)
            if args.azimuth_pairs
            else _view_id(azimuth, elevation)
        )
        view_dir = views_dir / identifier
        view_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[{index}/{len(view_specs)}] {identifier}: positioning camera",
            flush=True,
        )
        _position_camera(
            camera,
            pelvis=pelvis,
            azimuth_degrees=azimuth,
            elevation_degrees=elevation,
            distance=resolved_camera_distance,
        )
        world_from_camera = _matrix_array(camera.matrix_world)
        camera_from_canonical = camera_from_canonical_matrix(
            world_from_camera_blender=world_from_camera,
            pelvis_world=pelvis_array,
            character_height=character_height,
        )

        rgb_path = view_dir / "rgb.png"
        bpy.context.scene.render.filepath = str(rgb_path)
        print(f"[{index}/{len(view_specs)}] {identifier}: RGB", flush=True)
        bpy.ops.render.render(write_still=True)
        visible = _render_alpha_mask(
            rgb_path, args.resolution, args.minimum_render_alpha
        )
        print(f"[{index}/{len(view_specs)}] {identifier}: geometry", flush=True)
        arrays = _raycast_pointmap(
            camera=camera,
            intrinsics=intrinsics,
            pelvis_world=pelvis_array,
            character_height=character_height,
            object_ids=object_ids,
            resolution=args.resolution,
        )
        _apply_visibility_mask(
            arrays,
            visible,
        )
        errors = pointmap_reprojection_error(
            arrays["canonical_xyz"],
            arrays["mask"],
            intrinsics=intrinsics,
            camera_from_canonical=camera_from_canonical,
        )
        maximum_error = float(errors.max(initial=0.0))
        mean_error = float(errors.mean()) if errors.size else 0.0
        if not errors.size:
            raise RuntimeError(f"{identifier} raycast did not hit the character")
        if maximum_error > 1.0e-3:
            raise RuntimeError(
                f"{identifier} reprojection error {maximum_error:.6g}px exceeds tolerance"
            )
        geometry_path = view_dir / "geometry.npz"
        np.savez_compressed(geometry_path, **arrays)
        mask_path = view_dir / "mask.png"
        write_binary_mask_png(mask_path, arrays["mask"])
        camera_path = view_dir / "camera.json"
        camera_record = {
            "projection": "pinhole",
            "image_size": [args.resolution, args.resolution],
            "pixel_coordinates": "origin top-left; samples at (column+0.5,row+0.5)",
            "camera_coordinates": "CV: x right, y down, z forward",
            "azimuth_degrees": float(azimuth % 360.0),
            "prompt_azimuth_degrees": float(prompt_azimuth % 360.0),
            "azimuth_error_degrees": _signed_angle_delta(
                azimuth, prompt_azimuth
            ),
            "elevation_degrees": float(elevation),
            "intrinsics": _json_matrix(intrinsics),
            "world_from_camera_blender": _json_matrix(world_from_camera),
            "camera_from_canonical": _json_matrix(camera_from_canonical),
            "reprojection_error_pixels": {
                "mean": mean_error,
                "maximum": maximum_error,
            },
        }
        camera_path.write_text(
            json.dumps(camera_record, indent=2) + "\n", encoding="utf-8"
        )
        manifest["views"].append(
            {
                "id": identifier,
                "azimuth_degrees": float(azimuth % 360.0),
                "prompt_azimuth_degrees": float(prompt_azimuth % 360.0),
                "azimuth_error_degrees": _signed_angle_delta(
                    azimuth, prompt_azimuth
                ),
                "elevation_degrees": float(elevation),
                "rgb": str(rgb_path.relative_to(args.output_dir)),
                "mask": str(mask_path.relative_to(args.output_dir)),
                "geometry": str(geometry_path.relative_to(args.output_dir)),
                "camera": str(camera_path.relative_to(args.output_dir)),
                "foreground_pixels": int(arrays["mask"].sum()),
                "reprojection_max_pixels": maximum_error,
            }
        )
        manifest["elapsed_seconds"] = time.monotonic() - started
        _write_manifest(manifest_path, manifest)
        print(
            f"[{index}/{len(view_specs)}] {identifier}: "
            f"{int(arrays['mask'].sum())} hits, reprojection max "
            f"{maximum_error:.3g}px",
            flush=True,
        )
    print(
        f"Complete: {len(manifest['views'])} views in "
        f"{time.monotonic() - started:.1f}s at {args.output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
