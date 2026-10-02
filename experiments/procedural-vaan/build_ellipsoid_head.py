"""Replace the rejected head with an ellipsoid-only volume, inspect without hair.

Start Blender with the revision-3 scene loaded; save only to a fresh directory.
The source body, studio and cameras provide an unchanged comparison context.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import bpy
import numpy as np
from mathutils import Matrix

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from geometry import extract,uv_ellipsoid,material
from build import view
import portrait
import ellipsoid_head as eh


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--height',type=int,default=1000)
    parser.add_argument('--samples',type=int,default=48)
    parser.add_argument('--voxel',type=float,default=.0008)
    parser.add_argument('--views',nargs='+',default=['face','face-side','face-left','face-three-quarter'])
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    assert not (out/'head.blend').exists(),'Use a fresh output directory'
    source_blend=bpy.data.filepath
    assert source_blend and 'Skin | head ears nose' in bpy.data.objects
    started=time.perf_counter()
    scene=bpy.context.scene
    source=out/'source';source.mkdir(exist_ok=True)
    for f in HERE.glob('*.py'):shutil.copy2(f,source/f.name)
    kernel=HERE.parent/'procedural-bust/sdf.py'
    if not kernel.exists():kernel=HERE/'sdf.py'
    shutil.copy2(kernel,source/'sdf.py')
    m={name:bpy.data.materials[name] for name in ['skin','hairshade','brow','eyeline','liplight','lip','mouth']}
    scene.cycles.samples=args.samples
    # The same skin, studio and camera as revision-3. Keep the upper body for
    # the neck/shoulders. Do not use garments/hair to conceal the head shape.
    for obj in scene.objects:
        if obj.type in ('MESH','CURVE') and obj.name not in ('Skin | torso arms hands','Studio floor'):
            obj.hide_render=True
    baseline=[]
    for obj in scene.objects:
        if obj.type in ('MESH','CURVE') and obj.name not in ('Skin | torso arms hands','Studio floor') and obj.get('character_region')!='hair':
            # Explicit facial names avoid accidentally showing the vest/trim.
            if obj.name=='Skin | head ears nose' or obj.name.startswith(('Recessed eyeball','Anatomical eyelid','Lid wet margin','Upper lid fold','Silver eyebrow','Brow filament','Ear helix','Upper lip vermilion','Lower lip vermilion','Closed mouth line')):
                baseline.append(obj)
                obj.hide_render=False
    for name in args.views:
        view(scene.camera,name,args.height)
        scene.render.filepath=str(out/f'baseline-{name}.png')
        bpy.ops.render.render(write_still=True)
    for obj in list(scene.objects):
        if obj.type in ('MESH','CURVE') and obj.name not in ('Skin | torso arms hands','Studio floor'):
            bpy.data.objects.remove(obj,do_unlink=True)
    head,topology=extract('Skin | head ears nose',eh.head_field,(-.11,-.145,1.535),(.11,.145,1.807),args.voxel,m['skin'])
    eh.configure_face(portrait)
    portrait.face(m)
    # Store inspectable construction volumes in a hidden collection.
    construction=bpy.data.collections.new('SOURCE | ellipsoid construction')
    scene.collection.children.link(construction)
    colors=[(.29,.44,.64),(.55,.35,.26),(.45,.61,.39),(.68,.53,.26),(.49,.34,.57)]
    mats=[material(f'Construction group {i}',c,.7) for i,c in enumerate(colors)]
    for i,spec in enumerate(eh.PARTS):
        obj=uv_ellipsoid(spec['name'],spec['center'],spec['radii'],mats[i%len(mats)],32)
        obj.rotation_euler=Matrix(eh.rotation(spec).tolist()).to_euler()
        for collection in list(obj.users_collection):collection.objects.unlink(obj)
        construction.objects.link(obj)
        obj['blend_metres']=spec['blend']
    construction.hide_render=True
    construction.hide_viewport=True
    eye_probes=np.asarray([portrait.eye_point(side,u,v) for side in (-1,1)
                           for u in np.linspace(-.8,.8,17) for v in np.linspace(-.65,.65,9)]).T
    distances=eh.head_field(eye_probes)
    report={'source_blend':source_blend,'construction':'smooth unions of ellipsoid fields, ellipsoid subtractions; no loft or displacement',
            'positive_ellipsoids':eh.PARTS,'negative_ellipsoids':eh.CUTS,'topology':topology,
            'eye_probe_count':eye_probes.shape[1],'eye_min_clearance_m':float(distances.min()),
            'references':{img.name:hashlib.sha256(img.packed_file.data).hexdigest() for img in bpy.data.images if img.packed_file},
            'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in source.glob('*.py')}}
    assert distances.min()>0,report['eye_min_clearance_m']
    view(scene.camera,'face-three-quarter',args.height)
    bpy.ops.object.select_all(action='DESELECT')
    head.select_set(True);bpy.context.view_layer.objects.active=head
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type=='VIEW_3D':
                area.spaces.active.region_3d.view_location=(0,0,1.665)
                area.spaces.active.region_3d.view_distance=.47
                area.spaces.active.region_3d.view_rotation=scene.camera.rotation_euler.to_quaternion()
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'head.blend'))
    for name in args.views:
        view(scene.camera,name,args.height)
        scene.render.filepath=str(out/f'{name}.png')
        bpy.ops.render.render(write_still=True)
    # Raw component view is separate from the fused mesh; default scene stays fused.
    for obj in scene.objects:
        if obj.type in ('MESH','CURVE') and obj.name not in ('Skin | torso arms hands','Studio floor'):
            obj.hide_render=True
    construction.hide_render=False
    construction.hide_viewport=False
    for obj in construction.objects:obj.hide_render=False
    view(scene.camera,'face-three-quarter',args.height)
    scene.render.filepath=str(out/'construction.png');bpy.ops.render.render(write_still=True)
    report['total_seconds']=time.perf_counter()-started
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('DONE',out,report['total_seconds'],flush=True)


if __name__=='__main__':main()
