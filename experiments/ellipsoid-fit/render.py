"""Blender artifact: target, initial/fitted unions, editable source ellipsoids."""
import argparse
import json
from pathlib import Path
import sys
import bpy
import bmesh
sys.path.insert(0,str(Path(__file__).resolve().parent))
from limits import require_budget
require_budget()
import numpy as np
import openvdb
from mathutils import Vector
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'procedural-vaan'))
from geometry import mesh,material,uv_ellipsoid


def extract_grid(path,name,mat):
    data=np.load(path);values=data['field'];step=float(data['step']);lower=data['lower']
    g=openvdb.FloatGrid(background=step*4);g.transform=openvdb.createLinearTransform(voxelSize=step)
    g.gridClass=openvdb.GridClass.LEVEL_SET
    g.copyFromArray(np.clip(values,-step*4,step*4),ijk=tuple(np.rint(lower/step).astype(int)))
    v,f=g.convertToQuads();obj=mesh(name,v.tolist(),f.tolist(),mat)
    bpy.context.view_layer.objects.active=obj
    for factor in (.4,-.42)*2:
        mod=obj.modifiers.new('Voxel smoothing','SMOOTH');mod.factor=factor
        bpy.ops.object.modifier_apply(modifier=mod.name)
    bm=bmesh.new();bm.from_mesh(obj.data);bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces))
    bm.to_mesh(obj.data);bad=sum(not e.is_manifold for e in bm.edges)
    result=dict(vertices=len(bm.verts),faces=len(bm.faces),non_manifold_edges=bad,
                euler=len(bm.verts)-len(bm.edges)+len(bm.faces),volume=bm.calc_volume(signed=True))
    bm.free();assert bad==0 and result['volume']>0,result
    np.savez_compressed(path.with_name(name+'-mesh.npz'),vertices=np.array([v.co[:] for v in obj.data.vertices]),
                        faces=np.array([p.vertices[:] for p in obj.data.polygons]))
    return obj,result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);parser.add_argument('--target-only',action='store_true')
    parser.add_argument('--height',type=int,default=800);args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    out=args.run.resolve();bpy.ops.wm.read_factory_settings(use_empty=True)
    scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.samples=40
    scene.render.resolution_x=args.height;scene.render.resolution_y=args.height;scene.render.resolution_percentage=100
    scene.world=bpy.data.worlds.new('Studio');scene.world.use_nodes=True
    scene.world.node_tree.nodes['Background'].inputs[0].default_value=(.18,.19,.21,1)
    scene.world.node_tree.nodes['Background'].inputs[1].default_value=.6
    scene.view_settings.view_transform='AgX'
    clay=material('Neutral clay',(.46,.49,.51),.65)
    target=np.load(out/'target.npz');obj=mesh('target',target['vertices'].tolist(),target['faces'].tolist(),clay)
    objects={'target':obj};reports={}
    if not args.target_only:
        for name in ('initial','fitted'):
            objects[name],reports[name]=extract_grid(out/f'{name}-grid.npz',name,clay)
        collection=bpy.data.collections.new('SOURCE | fitted ellipsoids');scene.collection.children.link(collection)
        params=json.loads((out/'fitted.json').read_text())
        stages=params.get('stages',[dict(params,operation='union')])
        scene['union_temperature']=stages[0]['temperature'];scene['csg_stage_count']=len(stages)
        parts=[dict(p,operation=s['operation']) for s in stages for p in s['parts']]
        palette=[(.34,.47,.62),(.63,.44,.32),(.44,.57,.35),(.64,.55,.35),(.52,.41,.59)]
        mats=[material(f'Component colour {i}',c,.7) for i,c in enumerate(palette)]
        cut_mat=material('Subtractive volumes',(.65,.08,.04),.6)
        for i,part in enumerate(parts):
            subtract=part['operation']=='subtract'
            p=uv_ellipsoid(part['name'],part['center'],part['radii'],cut_mat if subtract else mats[i%len(mats)],40)
            p.rotation_mode='QUATERNION';p.rotation_quaternion=part['quaternion_wxyz']
            p['operation']=part['operation']
            if subtract:
                wire=p.modifiers.new('Show cut as wire cage','WIREFRAME');wire.thickness=.0011
            for col in list(p.users_collection):col.objects.unlink(p)
            collection.objects.link(p)
        collection.hide_render=True;collection.hide_viewport=True
    for name,location,power,size in [('Key',(-2,-3,3),350,3),('Fill',(2,-1,1),180,2),('Rim',(0,2,2),250,2)]:
        light=bpy.data.lights.new(name,'AREA');light.energy=power;light.shape='DISK';light.size=size
        o=bpy.data.objects.new(name,light);scene.collection.objects.link(o);o.location=location
        o.rotation_euler=(-o.location).to_track_quat('-Z','Y').to_euler()
    cam=bpy.data.objects.new('Camera',bpy.data.cameras.new('Camera'));scene.collection.objects.link(cam);scene.camera=cam
    cam.data.type='ORTHO';cam.data.ortho_scale=1.20
    views={'front':(0,-3,0),'profile':(3,0,0),'three-quarter':(1.7,-2.94,.07),'back':(0,3,0)}
    def camera(name):
        cam.location=views[name];cam.rotation_euler=(-cam.location).to_track_quat('-Z','Y').to_euler()
    for name,obj in objects.items():
        for other in objects.values():other.hide_render=True
        obj.hide_render=False
        for view in views:
            camera(view);scene.render.filepath=str(out/f'{name}-{view}.png');bpy.ops.render.render(write_still=True)
    if args.target_only:return
    for obj in objects.values():obj.hide_render=True;obj.hide_set(True)
    collection.hide_render=False;collection.hide_viewport=False
    camera('three-quarter');scene.render.filepath=str(out/'ellipsoids.png');bpy.ops.render.render(write_still=True)
    collection.hide_render=True;collection.hide_viewport=True
    objects['fitted'].hide_render=False;objects['fitted'].hide_set(False)
    bpy.ops.object.select_all(action='DESELECT');objects['fitted'].select_set(True);bpy.context.view_layer.objects.active=objects['fitted']
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type=='VIEW_3D':
                area.spaces.active.region_3d.view_location=(0,0,0)
                area.spaces.active.region_3d.view_distance=2
                area.spaces.active.region_3d.view_rotation=cam.rotation_euler.to_quaternion()
    for filename in ('initial.json','fitted.json','target-report.json','fit-report.json'):
        bpy.data.texts.load(str(out/filename))
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'ellipsoid-fit.blend'))
    (out/'mesh-report.json').write_text(json.dumps(reports,indent=2)+'\n')
    print('DONE',json.dumps(reports),flush=True)


if __name__=='__main__':main()
