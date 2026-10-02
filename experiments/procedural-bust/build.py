"""Run: blender -b -t 12 --python-exit-code 1 --python build.py -- --output PATH"""
import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
import sys
import time

import bpy
import bmesh
import numpy as np
import openvdb
from mathutils import Vector

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from anatomy import field, head_to_world


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--params', type=Path, default=HERE / 'neutral.json')
    parser.add_argument('--voxel', type=float, default=.0015)
    parser.add_argument('--resolution', type=int, default=720)
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--views', nargs='+', choices=['front', 'side', 'back', 'three-quarter'], default=['front', 'side', 'back', 'three-quarter'])
    parser.add_argument('--no-render', action='store_true')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    if not .0005 <= args.voxel <= .006:
        parser.error('voxel must be between 0.0005 and 0.006 metres')
    if args.resolution < 64 or args.samples < 1:
        parser.error('resolution >= 64 and samples >= 1 required')
    return args


def material(name, color, roughness=.55, grain=False):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (*color, 1)
    bsdf.inputs['Roughness'].default_value = roughness
    if grain:
        noise = nodes.new('ShaderNodeTexNoise')
        noise.inputs['Scale'].default_value = 190
        noise.inputs['Detail'].default_value = 2
        bump = nodes.new('ShaderNodeBump')
        bump.inputs['Strength'].default_value = .13
        bump.inputs['Distance'].default_value = .0003
        links.new(noise.outputs['Fac'], bump.inputs['Height'])
        links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    return mat


def aim(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat('-Z', 'Y').to_euler()


def components_of(bm):
    bm.verts.ensure_lookup_table()
    visited = set()
    components = []
    for vert in bm.verts:
        if vert.index in visited:
            continue
        stack, component = [vert], []
        visited.add(vert.index)
        while stack:
            cur = stack.pop()
            component.append(cur)
            for edge in cur.link_edges:
                other = edge.other_vert(cur)
                if other.index not in visited:
                    visited.add(other.index)
                    stack.append(other)
        components.append(component)
    return sorted(components, key=len, reverse=True)


def topology(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    result = dict(vertices=len(bm.verts), faces=len(bm.faces),
                  euler_characteristic=len(bm.verts) - len(bm.edges) + len(bm.faces),
                  boundary_edges=sum(e.is_boundary for e in bm.edges),
                  non_manifold_edges=sum(not e.is_manifold for e in bm.edges),
                  degenerate_faces=sum(f.calc_area() < 1e-14 for f in bm.faces),
                  components=[len(c) for c in components_of(bm)],
                  volume_m3=bm.calc_volume(signed=True))
    bm.free()
    return result


def build_surface(params, step):
    # Snap the origin to the voxel lattice so VDB and NumPy use identical coordinates.
    lower = np.floor(np.array([-.33, -.20, .015]) / step).astype(int)
    upper = np.ceil(np.array([.33, .20, .81]) / step).astype(int)
    axes = [np.arange(lower[i], upper[i] + 1, dtype=np.float32) * step for i in range(3)]
    shape = tuple(len(a) for a in axes)
    grid = openvdb.FloatGrid(background=step * 4)
    grid.transform = openvdb.createLinearTransform(voxelSize=step)
    grid.gridClass = openvdb.GridClass.LEVEL_SET
    # Evaluate slabs to bound peak NumPy memory. Clamp outside a narrow band.
    started = time.perf_counter()
    for start in range(0, shape[0], 12):
        p = (axes[0][start:start+12, None, None], axes[1][None, :, None], axes[2][None, None, :])
        values = field(p, params)
        if not np.isfinite(values).all():
            raise ValueError('Non-finite field')
        # A negative boundary would silently truncate the generated surface.
        if (np.any(values[:, (0, -1), :] <= 0) or np.any(values[:, :, (0, -1)] <= 0)
                or (start == 0 and np.any(values[0] <= 0))
                or (start + 12 >= shape[0] and np.any(values[-1] <= 0))):
            raise ValueError('Surface touches sampling bounds')
        np.clip(values, -step * 4, step * 4, out=values)
        grid.copyFromArray(values, ijk=(int(lower[0] + start), int(lower[1]), int(lower[2])))
        if start % 60 == 0:
            print(f'FIELD {start}/{shape[0]} slabs {time.perf_counter()-started:.1f}s', flush=True)
    sampled = time.perf_counter()
    verts, quads = grid.convertToQuads()
    if not len(verts) or not np.isfinite(verts).all():
        raise ValueError('Empty or non-finite surface')
    mesh = bpy.data.meshes.new('SDF surface')
    mesh.from_pydata(verts.tolist(), [], quads.tolist())
    mesh.update()
    obj = bpy.data.objects.new('Anatomy | continuous SDF', mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    # Alternating Laplacian passes reduce voxel faceting with limited shrinkage.
    for index, factor in enumerate((.45, -.47) * 3):
        mod = obj.modifiers.new(f'Smooth {index}', 'SMOOTH')
        mod.factor = factor
        mod.iterations = 1
        bpy.ops.object.modifier_apply(modifier=mod.name)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    removed_fragments = []
    # Subtractive seams can leave voxel-scale islands. Never silently keep
    # only the largest component: any appreciable detached part must fail QA.
    for component in components_of(bm)[1:]:
        positions = np.array([v.co[:] for v in component])
        low, high = positions.min(0), positions.max(0)
        if len(component) <= 64 and np.linalg.norm(high - low) <= 5 * step:
            removed_fragments.append(dict(vertices=len(component), bounds_metres=[low.tolist(), high.tolist()]))
            bmesh.ops.delete(bm, geom=component, context='VERTS')
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(mesh)
    bm.free()
    for poly in mesh.polygons:
        poly.use_smooth = True
    coords = np.empty(len(mesh.vertices) * 3, np.float32)
    mesh.vertices.foreach_get('co', coords)
    report = dict(grid_shape=shape, voxel_metres=step,
                  sampling_seconds=sampled-started,
                  extraction_and_smoothing_seconds=time.perf_counter()-sampled,
                  removed_voxel_fragments=removed_fragments,
                  geometry_sha256=hashlib.sha256(coords.tobytes()).hexdigest(),
                  bounds_metres=[coords.reshape(-1, 3).min(0).tolist(), coords.reshape(-1, 3).max(0).tolist()],
                  topology=topology(obj))
    check = report['topology']
    if check['boundary_edges'] or check['non_manifold_edges'] or check['degenerate_faces'] or len(check['components']) != 1 or check['volume_m3'] <= 0:
        raise ValueError(f'Surface validation failed: {check}')
    print('SURFACE', json.dumps(report), flush=True)
    return obj, report


def stage(obj, params):
    stone = material('Warm limestone', (.32, .27, .20), grain=True)
    obj.data.materials.append(stone)
    for side in (-1, 1):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=.012,
            location=head_to_world((side * .032, -.057, .613), params))
        eye = bpy.context.object
        eye.name = 'Carved eye L' if side < 0 else 'Carved eye R'
        eye.data.materials.append(stone)
        for poly in eye.data.polygons:
            poly.use_smooth = True
    base_mat = material('Basalt plinth', (.030, .043, .051), .34)
    bpy.ops.mesh.primitive_cylinder_add(vertices=128, radius=.121, depth=.042, location=(0, .009, .030))
    base = bpy.context.object
    base.name = 'Turned stone plinth'
    base.data.materials.append(base_mat)
    bevel = base.modifiers.new('Soft machined edge', 'BEVEL')
    bevel.width, bevel.segments = .004, 3
    for poly in base.data.polygons:
        poly.use_smooth = True
    bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, .007))
    bpy.context.object.name = 'Studio ground'
    bpy.context.object.data.materials.append(material('Studio charcoal', (.025, .034, .045), .7))
    for name, pos, power, color, size in [
        ('Large warm key', (-.75, -1.1, 1.35), 65, (1, .87, .71), .75),
        ('Cool fill', (.85, -.40, .8), 22, (.70, .82, 1), .65),
        ('Rim', (.35, .65, 1.1), 80, (1, .93, .80), .55),
    ]:
        data = bpy.data.lights.new(name, 'AREA')
        data.energy, data.color, data.shape, data.size = power, color, 'DISK', size
        light = bpy.data.objects.new(name, data)
        bpy.context.collection.objects.link(light)
        light.location = pos
        aim(light, (0, 0, .4))
    cam = bpy.data.objects.new('Inspection camera', bpy.data.cameras.new('Orthographic'))
    bpy.context.collection.objects.link(cam)
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = .85
    bpy.context.scene.camera = cam
    return cam


def set_view(cam, name):
    azimuth = dict(front=0, side=90, back=180, **{'three-quarter': 35})[name]
    angle = math.radians(azimuth)
    cam.location = (1.7 * math.sin(angle), -1.7 * math.cos(angle), .64)
    aim(cam, (0, 0, .378))


def main():
    args = arguments()
    params = json.loads(args.params.read_text())
    limits = {'shoulder_width': (.75, 1.15), 'neck_extension': (-.015, .045),
              'jaw_width': (.8, 1.15), 'head_yaw_degrees': (-35, 35)}
    if set(params) != set(limits):
        raise ValueError(f'Expected parameters: {list(limits)}')
    for key, (low, high) in limits.items():
        if not low <= params[key] <= high:
            raise ValueError(f'{key} must be between {low} and {high}')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = output / 'source'
    source.mkdir(exist_ok=True)
    for name in ('build.py', 'anatomy.py', 'sdf.py'):
        shutil.copy2(HERE / name, source / name)
    (source / 'neutral.json').write_text(json.dumps(params, indent=2) + '\n')
    started = time.perf_counter()
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    obj, report = build_surface(params, args.voxel)
    cam = stage(obj, params)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.cycles.seed = 0
    scene.render.resolution_x = scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.world.use_nodes = True
    scene.world.node_tree.nodes.get('Background').inputs[0].default_value = (.18, .23, .30, 1)
    scene.world.node_tree.nodes.get('Background').inputs[1].default_value = .22
    scene.view_settings.view_transform = 'AgX'
    report.update(parameters=params, blender_version=bpy.app.version_string,
                  numpy_version=np.__version__, samples=args.samples,
                  resolution=args.resolution, renders={},
                  source_sha256={name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                                 for name in ('build.py', 'anatomy.py', 'sdf.py', 'neutral.json')})
    # Persist a usable artifact and build metrics even if rendering is interrupted.
    set_view(cam, 'three-quarter')
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                space = area.spaces.active
                space.region_3d.view_location = (0, 0, .37)
                space.region_3d.view_distance = 1.15
                space.region_3d.view_rotation = cam.rotation_euler.to_quaternion()
                space.clip_start = .001
                space.shading.color_type = 'MATERIAL'
                space.shading.show_cavity = True
    bpy.ops.wm.save_as_mainfile(filepath=str(output / 'bust.blend'))
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    if not args.no_render:
        for view in args.views:
            set_view(cam, view)
            scene.render.filepath = str(output / f'{view}.png')
            tick = time.perf_counter()
            bpy.ops.render.render(write_still=True)
            report['renders'][view] = {'seconds': time.perf_counter() - tick, 'file': f'{view}.png'}
    report['total_seconds'] = time.perf_counter() - started
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('DONE', output, f"{report['total_seconds']:.1f}s", flush=True)


if __name__ == '__main__':
    main()
