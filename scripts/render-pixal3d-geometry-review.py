#!/usr/bin/env python3
"""Blender: render a full-body Pixal3D character with PBR and neutral clay details.

blender --background --python scripts/render-pixal3d-geometry-review.py -- model.glb output-dir
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    args.output.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.gltf(filepath=str(args.model.resolve()))
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    corners = [obj.matrix_world @ Vector(corner) for obj in meshes for corner in obj.bound_box]
    lower = Vector([min(p[a] for p in corners) for a in range(3)])
    upper = Vector([max(p[a] for p in corners) for a in range(3)])
    center = (lower + upper) * 0.5
    height = upper.z - lower.z
    scene = bpy.context.scene
    try:
        scene.render.engine = 'BLENDER_EEVEE_NEXT'
    except TypeError:
        scene.render.engine = 'BLENDER_EEVEE'
    scene.render.image_settings.file_format = 'PNG'
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.view_settings.view_transform = 'AgX'
    scene.view_settings.look = 'AgX - Medium High Contrast'
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get('Background')
    bg.inputs['Color'].default_value = (0.15, 0.15, 0.15, 1)
    bg.inputs['Strength'].default_value = 0.6
    for name, offset, energy in (
        ('Key', (-1.5, 2, 1.5), 90), ('Fill', (1.5, 1, 0.5), 40), ('Rim', (0, -2, 1.5), 100),
    ):
        light_data = bpy.data.lights.new(name, 'AREA')
        light_data.energy = energy * height**2
        light_data.shape = 'DISK'
        light_data.size = height * 1.5
        light = bpy.data.objects.new(name, light_data)
        scene.collection.objects.link(light)
        light.location = center + Vector(offset) * height
        light.rotation_euler = (center - light.location).to_track_quat('-Z', 'Y').to_euler()
    camera_data = bpy.data.cameras.new('Review camera')
    camera_data.type = 'ORTHO'
    camera = bpy.data.objects.new('Review camera', camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    jobs = []

    def render(name, azimuth, target, scale, resolution=768):
        a = math.radians(azimuth)
        direction = Vector((-math.sin(a), math.cos(a), 0))
        camera.location = target + direction * height * 3
        camera.rotation_euler = (target - camera.location).to_track_quat('-Z', 'Y').to_euler()
        camera_data.ortho_scale = scale
        scene.render.resolution_x = resolution
        scene.render.resolution_y = resolution
        destination = args.output / f'{name}.png'
        scene.render.filepath = str(destination.resolve())
        bpy.ops.render.render(write_still=True)
        jobs.append({'image': destination.name, 'azimuth': azimuth, 'target': list(target), 'ortho_scale': scale})

    full_scale = max(height, upper.x - lower.x, upper.y - lower.y) * 1.12
    for angle in (0, 90, 180, 270):
        render(f'pbr-{angle:03d}', angle, center, full_scale)
    clay = bpy.data.materials.new('Neutral clay geometry inspection')
    clay.use_nodes = True
    bsdf = clay.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (0.42, 0.42, 0.42, 1)
    bsdf.inputs['Roughness'].default_value = 0.75
    bsdf.inputs['Metallic'].default_value = 0
    for obj in meshes:
        obj.data.materials.clear()
        obj.data.materials.append(clay)
        for polygon in obj.data.polygons:
            polygon.material_index = 0
    for angle in (0, 90, 180, 270):
        render(f'clay-{angle:03d}', angle, center, full_scale)
    head_center = Vector((center.x, center.y, upper.z - height * 0.10))
    for angle in (0, 45, 90):
        render(f'clay-head-{angle:03d}', angle, head_center, height * 0.24, 1024)
    # A-pose source: extremal X includes the fingers, around mid body height.
    for side, x in (('left', lower.x + height * 0.05), ('right', upper.x - height * 0.05)):
        target = Vector((x, center.y, lower.z + height * 0.52))
        render(f'clay-hand-{side}', 0, target, height * 0.20, 1024)
        render(f'clay-hand-{side}-045', 45, target, height * 0.20, 1024)
    (args.output / 'manifest.json').write_text(json.dumps({
        'model': str(args.model.resolve()), 'blender': bpy.app.version_string,
        'bounds': [list(lower), list(upper)], 'jobs': jobs,
        'note': 'Clay material replaces all texture/normal inputs; original vertex normals retained. Detail targets assume a full-body A-pose.',
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
