"""Small Blender geometry helpers for the reference-driven character study."""
import math
import time
import sys
from pathlib import Path
import bpy
import bmesh
import numpy as np
import openvdb
from mathutils import Vector

sys.path.append(str(Path(__file__).resolve().parents[1] / 'procedural-bust'))
from sdf import Sculpt, ellipsoid, loft, smooth_min


def mesh(name, vertices, faces, mat=None, smooth=True):
    data = bpy.data.meshes.new(name)
    data.from_pydata(vertices, [], faces)
    data.update()
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    if mat:
        data.materials.append(mat)
    for p in data.polygons:
        p.use_smooth = smooth
    return obj


def material(name, color, rough=.5, metallic=0, noise=0):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    shader = nodes.get('Principled BSDF')
    shader.inputs['Base Color'].default_value = (*color, 1)
    shader.inputs['Roughness'].default_value = rough
    shader.inputs['Metallic'].default_value = metallic
    if noise:
        tex = nodes.new('ShaderNodeTexNoise')
        tex.inputs['Scale'].default_value = 85
        tex.inputs['Detail'].default_value = 2
        ramp = nodes.new('ShaderNodeValToRGB')
        ramp.color_ramp.elements[0].color = (*(c * (1-noise) for c in color), 1)
        ramp.color_ramp.elements[1].color = (*(c * (1+noise) for c in color), 1)
        links.new(tex.outputs['Fac'], ramp.inputs[0])
        links.new(ramp.outputs[0], shader.inputs['Base Color'])
        bump = nodes.new('ShaderNodeBump')
        bump.inputs['Strength'].default_value = .08
        bump.inputs['Distance'].default_value = .0003
        links.new(tex.outputs['Fac'], bump.inputs['Height'])
        links.new(bump.outputs[0], shader.inputs['Normal'])
    return mat


def uv_ellipsoid(name, center, radii, mat, segments=40):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=24, radius=1, location=center)
    obj = bpy.context.object
    obj.name = name
    obj.scale = radii
    obj.data.materials.append(mat)
    for p in obj.data.polygons:
        p.use_smooth = True
    return obj


def box(name, center, size, mat, bevel=.003):
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    obj = bpy.context.object
    obj.name = name
    obj.scale = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(mat)
    if bevel:
        mod = obj.modifiers.new('Rounded sewn edge', 'BEVEL')
        mod.width, mod.segments = bevel, 3
        obj.modifiers.new('Weighted corner normals', 'WEIGHTED_NORMAL')
    return obj


def curve(name, points, radius, mat, cyclic=False):
    data = bpy.data.curves.new(name, 'CURVE')
    data.dimensions = '3D'
    data.resolution_u = 12
    data.bevel_depth, data.bevel_resolution = radius, 3
    spline = data.splines.new('BEZIER')
    spline.bezier_points.add(len(points)-1)
    for p, co in zip(spline.bezier_points, points):
        p.co = co
        p.handle_left_type = p.handle_right_type = 'AUTO'
    spline.use_cyclic_u = cyclic
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    data.materials.append(mat)
    return obj


def bezier(points, count=24):
    p = np.array(points, dtype=float)
    t = np.linspace(0, 1, count)[:, None]
    return (1-t)**3*p[0] + 3*(1-t)**2*t*p[1] + 3*(1-t)*t*t*p[2] + t**3*p[3]


def sweep(name, points, width, depth, mat, normal=(0, -1, 0), taper=False, sides=12):
    """Closed elliptical sweep; width/depth are radii, points form its centreline."""
    points = np.array(points)
    verts, faces = [], []
    for i, p in enumerate(points):
        tangent = points[min(i+1, len(points)-1)]-points[max(i-1, 0)]
        tangent /= np.linalg.norm(tangent)
        n = np.array(normal, dtype=float)
        if abs(np.dot(tangent, n)) > .97:
            n = np.array([1., 0, 0])
        side = np.cross(tangent, n)
        side /= np.linalg.norm(side)
        n = np.cross(side, tangent)
        t = i / (len(points)-1)
        factor = max(.008, (.32 + .95 * math.sin(math.pi*t)**.7) * (1-t)**.4) if taper else 1
        for j in range(sides):
            a = j * 2 * math.pi / sides
            verts.append(tuple(p + factor * (width*math.cos(a)*side + depth*math.sin(a)*n)))
    for i in range(len(points)-1):
        for j in range(sides):
            k = i*sides+j
            nxt = i*sides+(j+1)%sides
            faces.append((k, nxt, nxt+sides, k+sides))
    faces.extend([tuple(reversed(range(sides))), tuple(range((len(points)-1)*sides, len(points)*sides))])
    return mesh(name, verts, faces, mat)


def surface(name, func, nu, nv, mat, thickness=0, skip=None):
    verts = [func(i/nu, j/nv) for i in range(nu+1) for j in range(nv+1)]
    faces = []
    for i in range(nu):
        for j in range(nv):
            if skip and skip((i+.5)/nu, (j+.5)/nv):
                continue
            a = i*(nv+1)+j
            faces.append((a, a+nv+1, a+nv+2, a+1))
    obj = mesh(name, verts, faces, mat)
    if thickness:
        mod = obj.modifiers.new('Fabric thickness', 'SOLIDIFY')
        mod.thickness, mod.offset = thickness, 0
    return obj


def tapered_segment(p, a, b, ra, rb):
    a, b = np.asarray(a), np.asarray(b)
    axis = b-a
    q = [p[i]-a[i] for i in range(3)]
    t = np.clip(sum(q[i]*axis[i] for i in range(3)) / np.dot(axis, axis), 0, 1)
    return np.sqrt(sum((q[i]-t*axis[i])**2 for i in range(3))) - (ra+(rb-ra)*t)


def extract(name, field_fn, lower, upper, step, mat):
    started = time.perf_counter()
    lo = np.floor(np.array(lower)/step).astype(int)
    hi = np.ceil(np.array(upper)/step).astype(int)
    axes = [np.arange(lo[i], hi[i]+1, dtype=np.float32)*step for i in range(3)]
    grid = openvdb.FloatGrid(background=step*4)
    grid.transform = openvdb.createLinearTransform(voxelSize=step)
    grid.gridClass = openvdb.GridClass.LEVEL_SET
    for start in range(0, len(axes[0]), 10):
        p = (axes[0][start:start+10,None,None], axes[1][None,:,None], axes[2][None,None,:])
        values = field_fn(p).astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f'{name}: non-finite field')
        if (np.any(values[:,(0,-1),:] <= 0) or np.any(values[:,:,(0,-1)] <= 0)
                or (start == 0 and np.any(values[0] <= 0))
                or (start+10 >= len(axes[0]) and np.any(values[-1] <= 0))):
            raise ValueError(f'{name}: sampling bounds touched')
        np.clip(values, -step*4, step*4, out=values)
        grid.copyFromArray(values, ijk=(int(lo[0]+start), int(lo[1]), int(lo[2])))
    v, f = grid.convertToQuads()
    obj = mesh(name, v.tolist(), f.tolist(), mat)
    bpy.context.view_layer.objects.active = obj
    for factor in (.4,-.42)*2:
        mod = obj.modifiers.new('Voxel smoothing', 'SMOOTH')
        mod.factor = factor
        bpy.ops.object.modifier_apply(modifier=mod.name)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(obj.data)
    nonmanifold = sum(not e.is_manifold for e in bm.edges)
    euler=len(bm.verts)-len(bm.edges)+len(bm.faces)
    volume=bm.calc_volume(signed=True)
    seen=set(); components=[]
    bm.verts.ensure_lookup_table()
    for vert in bm.verts:
        if vert.index in seen:
            continue
        stack=[vert]; seen.add(vert.index); count=0
        while stack:
            current=stack.pop(); count+=1
            for edge in current.link_edges:
                other=edge.other_vert(current)
                if other.index not in seen:
                    seen.add(other.index); stack.append(other)
        components.append(count)
    bm.free()
    if nonmanifold or volume<=0 or len(components)!=1 or euler!=2:
        raise ValueError(f'{name}: invalid topology: nonmanifold={nonmanifold}, euler={euler}, volume={volume}, components={components}. Thin fingers need a sufficiently fine grid; the validated body setting is 0.0025 m.')
    report = dict(vertices=len(v), quads=len(f), voxel=step, seconds=time.perf_counter()-started,
                  non_manifold_edges=nonmanifold,euler_characteristic=euler,volume_m3=volume,components=components)
    print(name, report, flush=True)
    return obj, report
