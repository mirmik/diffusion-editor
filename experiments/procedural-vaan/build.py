"""Procedural Vaan study, built from scratch in Blender's bundled Python."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

import bpy
import bmesh
import numpy as np
from mathutils import Vector

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from geometry import material, extract
from anatomy import body_field, pants_field
from details import clothes, shoes
from portrait import head_field, face, hair


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--body-voxel',type=float,default=.0025)
    p.add_argument('--head-voxel',type=float,default=.0009)
    p.add_argument('--height',type=int,default=1100)
    p.add_argument('--samples',type=int,default=40)
    p.add_argument('--views',nargs='+',default=['front','right','back','left','three-quarter','face'],choices=['front','right','back','left','three-quarter','face','face-side','face-left','face-three-quarter','head-back'])
    p.add_argument('--no-render',action='store_true')
    p.add_argument('--anatomy-references',type=Path,default=HERE/'references/profile-hair-removal',
                   help='Optional user front and hair removal of the existing profiles')
    args=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    if not .0015<=args.body_voxel<=.006 or not .0006<=args.head_voxel<=.003:
        p.error('Body voxel: 1.5–6 mm; head voxel: 0.6–3 mm')
    if args.height<128 or args.samples<1:
        p.error('Invalid image size or sample count')
    return args


def palette():
    spec={
        'skin':((.52,.245,.112),.61,0,0),
        'white':((.86,.87,.83),.3,0,0),
        'iris':((.12,.090,.038),.28,0,0),
        'black':((.009,.012,.012),.3,0,0),
        'eyeline':((.052,.025,.013),.65,0,0),
        'mouth':((.12,.033,.019),.65,0,0),
        'lip':((.37,.128,.075),.63,0,0),
        'liplight':((.49,.221,.123),.59,0,0),
        'hair':((.72,.74,.75),.44,0,0),
        'hairshade':((.62,.64,.66),.48,0,0),
        'hairgroove':((.45,.47,.49),.6,0,0),
        'brow':((.38,.39,.37),.65,0,0),
        'blue':((.003,.034,.25),.70,0,.13),
        'blueedge':((.005,.063,.34),.60,0,0),
        'bluedark':((.004,.036,.15),.7,0,0),
        'pants':((.002,.013,.027),.80,0,.21),
        'pantsedge':((.002,.009,.019),.75,0,0),
        'leather':((.105,.043,.017),.63,0,.20),
        'leatherlight':((.17,.078,.028),.52,0,.12),
        'leatherdark':((.035,.016,.009),.6,0,0),
        'stitch':((.26,.145,.065),.8,0,0),
        'metal':((.36,.39,.38),.31,.65,0),
        'cyan':((.008,.40,.57),.32,.2,0),
        'gem':((.13,.65,.80),.23,.3,0),
        'shoe':((.012,.023,.036),.52,0,.1),
        'sole':((.003,.008,.012),.75,0,0),
        'laces':((.006,.016,.025),.7,0,0),
    }
    result={name:material(name,*args) for name,args in spec.items()}
    shader=result['skin'].node_tree.nodes.get('Principled BSDF')
    shader.inputs['Specular IOR Level'].default_value=.32
    shader.inputs['Subsurface Weight'].default_value=.045
    shader.inputs['Subsurface Scale'].default_value=.002
    shader.inputs['Subsurface Radius'].default_value=(1,.45,.22)
    return result


def aim(obj, target):
    obj.rotation_euler=(Vector(target)-obj.location).to_track_quat('-Z','Y').to_euler()


def studio():
    bpy.ops.mesh.primitive_plane_add(size=200,location=(0,0,.007))
    bpy.context.object.name='Studio floor'
    bpy.context.object.data.materials.append(material('Neutral grey backdrop',(.18,.18,.185),.8))
    for name,pos,power,color,size in [
        ('Key',(-2.4,-3.5,3.8),400,(1,.88,.77),3.0),
        ('Fill',(2.5,-2.0,2.8),180,(.80,.88,1),2.7),
        ('Rim',(.5,2.2,3.0),450,(1,.95,.88),2.0),
    ]:
        data=bpy.data.lights.new(name,'AREA')
        data.energy,data.color,data.shape,data.size=power,color,'DISK',size
        obj=bpy.data.objects.new(name,data)
        bpy.context.collection.objects.link(obj)
        obj.location=pos
        aim(obj,(0,0,1))
    cam=bpy.data.objects.new('Character inspection camera',bpy.data.cameras.new('Orthographic'))
    cam.data.type='ORTHO'
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera=cam
    scene=bpy.context.scene
    scene.world.use_nodes=True
    scene.world.node_tree.nodes['Background'].inputs[0].default_value=(.5,.5,.5,1)
    scene.world.node_tree.nodes['Background'].inputs[1].default_value=.18
    return cam


def view(cam,name,height):
    scene=bpy.context.scene
    scene.render.resolution_x=int(height*.82)
    scene.render.resolution_y=height
    cam.data.ortho_scale=2.04
    azimuth={'front':0,'right':90,'back':180,'left':270,'three-quarter':30,'face':0,'face-side':90,'face-left':270,'face-three-quarter':30,'head-back':180}[name]
    a=math.radians(azimuth)
    target=(0,0,.922)
    cam.location=(4*math.sin(a),-4*math.cos(a),.922)
    if name=='three-quarter':
        cam.location.z=1.15
    if name.startswith('face') or name=='head-back':
        target=(0,-.01,1.671)
        cam.location=(3*math.sin(a),-3*math.cos(a),1.682)
        cam.data.ortho_scale=.37
        scene.render.resolution_x=scene.render.resolution_y=min(height,1100)
    aim(cam,target)


def audit_character():
    bpy.context.view_layer.update()
    report={'meshes':0,'curves':0,'curve_splines':0,'curve_points':0,'vertices':0,'faces':0,'nonfinite_objects':[],'empty_objects':[],'out_of_bounds_objects':[]}
    world_bounds=[]
    for obj in bpy.context.scene.objects:
        if obj.type=='CURVE':
            report['curves']+=1
            for spline in obj.data.splines:
                points=spline.bezier_points if spline.type=='BEZIER' else spline.points
                report['curve_splines']+=1
                report['curve_points']+=len(points)
                coords=np.array([tuple(p.co) for p in points])
                if not np.isfinite(coords).all():
                    report['nonfinite_objects'].append(obj.name)
        if obj.type!='MESH' or obj.name=='Studio floor':
            continue
        bounds=np.array([obj.matrix_world @ Vector(corner) for corner in obj.bound_box])
        world_bounds.append(bounds)
        low,high=bounds.min(0),bounds.max(0)
        if np.any(low<[-.70,-.30,0]) or np.any(high>[.70,.30,1.95]):
            report['out_of_bounds_objects'].append(obj.name)
        # Recalculate procedural patch/sweep winding before saving and rendering.
        bm=bmesh.new(); bm.from_mesh(obj.data)
        bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=1e-7)
        bmesh.ops.dissolve_degenerate(bm,edges=list(bm.edges),dist=1e-8)
        bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces))
        bm.to_mesh(obj.data); bm.free()
        report['meshes']+=1
        report['vertices']+=len(obj.data.vertices)
        report['faces']+=len(obj.data.polygons)
        if not obj.data.polygons:
            report['empty_objects'].append(obj.name)
        coords=np.empty(len(obj.data.vertices)*3,np.float32)
        obj.data.vertices.foreach_get('co',coords)
        if not np.isfinite(coords).all():
            report['nonfinite_objects'].append(obj.name)
    report['world_bounds_metres']=[np.concatenate(world_bounds).min(0).tolist(),np.concatenate(world_bounds).max(0).tolist()]
    if report['nonfinite_objects'] or report['empty_objects'] or report['out_of_bounds_objects']:
        raise ValueError(report)
    return report


def main():
    args=arguments()
    output=args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    source=output/'source'; source.mkdir(exist_ok=True)
    for path in HERE.glob('*.py'):
        shutil.copy2(path,source/path.name)
    kernel=HERE.parent/'procedural-bust/sdf.py'
    if not kernel.exists():
        kernel=HERE/'sdf.py'
    shutil.copy2(kernel,source/'sdf.py')
    refs={'front':Path('/home/mirmik/Vaan/Front.png'),'right':Path('/home/mirmik/Vaan/views/mv-eye-090.png'),'left':Path('/home/mirmik/Vaan/views/mv-eye-270.png'),'back':Path('/home/mirmik/Vaan/Back.png')}
    bare=args.anatomy_references.resolve()
    refs.update({'anatomy_front':bare/'Body.png','anatomy_front_crop':bare/'front.png'})
    for name in ('right','left'):
        refs[f'anatomy_{name}']=bare/f'{name}-hairless.png'
    references={}
    bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
    for name,path in refs.items():
        if path.exists():
            img=bpy.data.images.load(str(path),check_existing=True)
            img.pack()
            img.use_fake_user=True
            references[name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    started=time.perf_counter()
    m=palette()
    reports={}
    _,reports['body']=extract('Skin | torso arms hands',body_field,(-.59,-.15,.86),(.59,.16,1.67),args.body_voxel,m['skin'])
    _,reports['head']=extract('Skin | head ears nose',head_field,(-.10,-.13,1.548),(.10,.16,1.805),args.head_voxel,m['skin'])
    face(m)
    before_hair=set(bpy.context.scene.objects)
    hair(m)
    for obj in set(bpy.context.scene.objects)-before_hair:
        obj['character_region']='hair'
    _,reports['pants']=extract('Navy cargo trousers',pants_field,(-.25,-.15,.175),(.25,.17,1.10),args.body_voxel,m['pants'])
    clothes(m); shoes(m)
    print('DETAILS COMPLETE',flush=True)
    cam=studio()
    scene=bpy.context.scene
    scene.render.engine='CYCLES'
    scene.cycles.device='CPU'; scene.cycles.samples=args.samples
    scene.cycles.use_denoising=True; scene.cycles.seed=47
    scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG'
    scene.view_settings.view_transform='AgX'
    scene.view_settings.look='AgX - Medium High Contrast'
    scene.view_settings.exposure=-.25
    report={'revision':3,'references':references,'sdf':reports,'scene':audit_character(),'blender':bpy.app.version_string,'renders':{},'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')}}
    view(cam,'three-quarter',args.height)
    bpy.ops.object.select_all(action='DESELECT')
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type=='VIEW_3D':
                space=area.spaces.active
                space.region_3d.view_location=(0,0,.95)
                space.region_3d.view_distance=2.8
                space.region_3d.view_rotation=cam.rotation_euler.to_quaternion()
                space.clip_start=.001
                space.shading.color_type='MATERIAL'
    bpy.ops.wm.save_as_mainfile(filepath=str(output/'vaan.blend'))
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    if not args.no_render:
        for name in args.views:
            view(cam,name,args.height)
            scene.render.filepath=str(output/f'{name}.png')
            tick=time.perf_counter(); bpy.ops.render.render(write_still=True)
            report['renders'][name]={'seconds':time.perf_counter()-tick,'file':f'{name}.png'}
    report['total_seconds']=time.perf_counter()-started
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('DONE',output,report['total_seconds'],flush=True)


if __name__=='__main__':
    main()
