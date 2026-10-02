"""Reference-guided surface sculpt over the positive-only fitted head.

Run in Blender through run_bounded.py. No CSG fit or target projection is used.
The original mesh and source ellipsoids remain saved as hidden references.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import bpy
import bmesh
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from limits import require_budget
require_budget()
sys.path.insert(0, str(HERE.parent / 'procedural-vaan'))
from geometry import mesh, material, uv_ellipsoid, curve


def smoothstep(a, b, x):
    t = np.clip((x-a)/(b-a), 0, 1)
    return t*t*(3-2*t)


def vertices(obj):
    a = np.empty(len(obj.data.vertices)*3, dtype=np.float32)
    obj.data.vertices.foreach_get('co', a)
    return a.reshape(-1, 3)


def set_vertices(obj, a):
    obj.data.vertices.foreach_set('co', np.asarray(a, dtype=np.float32).ravel())
    obj.data.update()


def gauss(x, z, cx, cz, sx, sz):
    return np.exp(-((x-cx)/sx)**2-((z-cz)/sz)**2)


EX, EZ, HALF, ER, EY, TILT = .142, -.035, .075, .108, -.202, .42
UPPER, LOWER = .019, .013


def eye_front(x, z):
    dx = np.abs(x)-EX
    return EY+TILT*dx-np.sqrt(np.maximum(.00001, ER*ER-dx*dx-(z-EZ)**2))


def eye_opening(x, z):
    u = (np.abs(x)-EX)/HALF
    mid = EZ+.009*u
    h = np.where(z > mid, UPPER, LOWER)
    # Positive inside the almond, negative outside; smooth tapered canthi.
    d1=(1-np.abs(u))*HALF
    d2=h*np.maximum(0, 1-np.minimum(1, np.abs(u))**2)**.72-np.abs(z-mid)
    d=-.003*np.logaddexp(-d1/.003,-d2/.003)
    return d, u, mid


def mouth_line(x):
    return -.299 + .002*np.exp(-((np.abs(x)-.025)/.016)**2)-.006*(x/.093)**2


def face_sculpt(a):
    p = a.copy()
    x, y, z = p.T
    frontal = 1-smoothstep(-.16, .03, y)
    # Compress the overlong nose-to-chin spacing without flattening the cranium.
    lift = .056*np.exp(-((z+.282)/.14)**4)*np.exp(-(x/.245)**6)*frontal
    p[:, 2] += lift
    x, y, z = p.T
    # Broad malar planes, lower-lip hollow, and a quieter chin.
    d = -.008*gauss(x,z,0,-.404,.083,.035)
    for side in (-1, 1):
        d -= .011*gauss(x,z,side*.181,-.117,.07,.049)
        d += .006*gauss(x,z,side*.214,-.244,.057,.09)
        d -= .006*gauss(x,z,side*.033,-.190,.018,.013)
    d += .004*gauss(x,z,0,-.254,.010,.035)  # philtrum
    d -= .008*gauss(x,z,0,-.212,.012,.010)  # columella
    d += .007*gauss(x,z,0,-.346,.065,.017)
    # Model the two lip masses directly in the skin, plus their meeting groove.
    line = mouth_line(x)
    w = np.exp(-(x/.090)**8)
    d += w*(-.009*np.exp(-((z-line-.007)/.009)**2)
             -.011*np.exp(-((z-line+.011)/.010)**2)
             +.002*np.exp(-((z-line)/.0032)**2))
    p[:, 1] += d*frontal
    p[:, 0] *= 1+.08*gauss(x,z,0,-.399,.10,.036)*frontal
    return p


def features_sculpt(a):
    p = a.copy(); x, y, z = p.T
    face = 1-smoothstep(-.17, -.04, y)
    # Continuous orbital deformation: skin recedes inside the opening and
    # follows a slightly raised lid rim outside. Eyeballs remain separate.
    d, u, mid = eye_opening(x,z)
    region = smoothstep(-.026,-.004,d)*face
    eye = eye_front(x,z)
    recess = smoothstep(-.002, .007, d)
    rim = eye-.0035+recess*.028
    p[:,1] = y*(1-region)+rim*region
    # Supratarsal fold and suborbital transition, not an outline cylinder.
    upper = mid+UPPER*np.maximum(0,1-np.minimum(1,np.abs(u))**2)**.72
    folds = np.exp(-(u/.98)**8)*face
    p[:,1] += .0035*np.exp(-((z-upper-.015)/.0045)**2)*folds
    # Nostrils are small sculpted hollows on either side of the columella.
    for side in (-1,1):
        p[:,1] += .009*gauss(x,z,side*.029,-.211,.010,.0045)*face
    # Ear concha and antihelix, applied to the existing ear surface.
    ear_front = 1-smoothstep(.095,.15,y)
    ax = np.abs(x)
    p[:,1] += .028*gauss(ax,z,.360,-.083,.023,.056)*ear_front
    p[:,1] -= .011*gauss(ax,z,.350,-.055,.009,.040)*ear_front
    outward=smoothstep(.305,.36,ax)
    bowl=gauss(y,z,.112,-.067,.038,.062)
    ring=np.sqrt(((y-.112)/.057)**2+((z+.067)/.094)**2)
    p[:,0]+=np.sign(x)*outward*(-.012*bowl+.002*np.exp(-((ring-.87)/.13)**2))
    return p


def build_eyes(skin):
    sclera = material('Sclera | warm ivory',(.62,.58,.49),.28)
    node=sclera.node_tree.nodes.new('ShaderNodeVertexColor');node.layer_name='Eye colour'
    sclera.node_tree.links.new(node.outputs['Color'],sclera.node_tree.nodes.get('Principled BSDF').inputs['Base Color'])
    objects=[]
    for side in (-1,1):
        # Only the exposed spherical cap is needed. Complete spheres protruded
        # through unrelated parts of the cheeks in the first diagnostic pass.
        verts=[];faces=[];nu,nv=192,64
        for i in range(nu+1):
            u=-.999+1.998*i/nu;x=side*(EX+HALF*u)
            h=(1-u*u)**.72;mid=EZ+.009*u
            for j in range(nv+1):
                z=mid+h*(-LOWER+(UPPER+LOWER)*j/nv)
                verts.append((x,float(eye_front(x,z)),z))
        for i in range(nu):
            for j in range(nv):
                a=i*(nv+1)+j;faces.append((a,a+nv+1,a+nv+2,a+1))
        if side<0:faces=[tuple(reversed(f)) for f in faces]
        obj=mesh(f'Eye | exposed spherical cap {side}',verts,faces,sclera)
        p=np.array(verts);dx=p[:,0]-side*EX;dz=p[:,2]-EZ-.003;r=np.hypot(dx,dz)
        theta=np.arctan2(dz,dx);fiber=.5+.5*np.sin(theta*67+2*np.sin(r*850))
        colors=np.ones((len(p),4));colors[:,:3]=(.62,.58,.49)
        iris=r<.027;colors[iris,:3]=np.array([.13,.115,.087])+fiber[iris,None]*np.array([.06,.055,.045])
        colors[(r>.024)&iris,:3]=(.029,.027,.023)
        colors[r<.0105,:3]=(.004,.0045,.004)
        attr=obj.data.color_attributes.new(name='Eye colour',type='FLOAT_COLOR',domain='POINT')
        attr.data.foreach_set('color',colors.astype(np.float32).ravel())
        objects.append(obj)
    return objects


def main():
    parser=argparse.ArgumentParser();parser.add_argument('out',type=Path)
    parser.add_argument('--size',type=int,default=720);parser.add_argument('--samples',type=int,default=32)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();baseline=HERE/'output/head-01'
    bpy.ops.wm.open_mainfile(filepath=str(baseline/'ellipsoid-fit.blend'))
    scene=bpy.context.scene;scene.cycles.samples=args.samples
    scene.render.resolution_x=args.size;scene.render.resolution_y=args.size
    for obj in bpy.data.objects:obj.hide_render=True;obj.hide_set(True)
    for obj in scene.objects:
        if obj.type in ('LIGHT','CAMERA'):obj.hide_render=False;obj.hide_set(False)
    base=bpy.data.objects['fitted'];base.name='BASE | positive ellipsoid union'
    sculpt=base.copy();sculpt.data=base.data.copy();scene.collection.objects.link(sculpt)
    sculpt.name='Vaan | surface sculpt';sculpt.hide_set(False);sculpt.hide_render=False
    bpy.context.view_layer.objects.active=sculpt;sculpt.select_set(True)
    sub=sculpt.modifiers.new('Detail resolution','SUBSURF');sub.levels=1
    bpy.ops.object.modifier_apply(modifier=sub.name)
    original=vertices(sculpt)
    sculpt.shape_key_add(name='Basis | positive ellipsoid mesh')
    broad=face_sculpt(original);final=features_sculpt(broad)
    key=sculpt.shape_key_add(name='Reference sculpt | face and ears')
    key.data.foreach_set('co',final.astype(np.float32).ravel());key.value=1
    set_vertices(sculpt,original)
    clay=bpy.data.materials['Neutral clay']
    skin=material('Skin | warm matte',(.48,.255,.135),.48)
    shader=skin.node_tree.nodes.get('Principled BSDF');shader.inputs['Subsurface Weight'].default_value=.06
    shader.inputs['Subsurface Radius'].default_value=(.12,.055,.03)
    # Paint subtle lip colour on the same skin surface, not a separate shell.
    attr=sculpt.data.color_attributes.new(name='Complexion',type='FLOAT_COLOR',domain='POINT')
    x,y,z=final.T;line=mouth_line(x)
    mask=np.exp(-(x/.090)**8)*np.exp(-((z-line)/.023)**6)*(1-smoothstep(-.20,-.10,y))
    colors=np.ones((len(final),4));colors[:,:3]=np.array([.48,.255,.135])*(1-mask[:,None])+np.array([.34,.125,.075])*mask[:,None]
    attr.data.foreach_set('color',colors.astype(np.float32).ravel())
    node=skin.node_tree.nodes.new('ShaderNodeVertexColor');node.layer_name='Complexion'
    skin.node_tree.links.new(node.outputs['Color'],shader.inputs['Base Color'])
    details=build_eyes(skin)
    # Query the final deformed surface for brows and the mouth seam.
    bm=bmesh.new();bm.from_mesh(sculpt.data)
    for v,co in zip(bm.verts,final):v.co=co
    tree=BVHTree.FromBMesh(bm)
    def front(x,z):
        loc,normal,index,distance=tree.ray_cast(Vector((x,-1,z)),Vector((0,1,0)))
        if loc is None:raise ValueError(('Missing front',x,z))
        return float(loc.y)
    brow=material('Brows | ash brown',(.10,.065,.039),.8)
    for side in (-1,1):
        verts=[];faces=[]
        for i in range(65):
            u=i/64;x=side*(.062+.181*u);mid=.023+.032*u-.010*u*u+.009*math.sin(math.pi*u)
            width=.007*math.sin(math.pi*u)**.35+.0005
            for j in range(5):
                z=mid+width*(j/4-.5)*2
                verts.append((x,front(x,z)-.0012,z))
        for i in range(64):
            for j in range(4):a=i*5+j;faces.append((a,a+5,a+6,a+1))
        if side<0:faces=[tuple(reversed(f)) for f in faces]
        obj=mesh(f'Sculpted brow {side}',verts,faces,brow);details.append(obj)
    seam=material('Lip meeting line',(.105,.027,.017),.6)
    points=[]
    for x in np.linspace(-.087,.087,45):
        z=float(mouth_line(x));points.append((x,front(float(x),z)-.0007,z))
    obj=curve('Mouth | meeting line',points,.0011,seam);details.append(obj)
    # Confirm exposed iris points are in front of the sculpt, away from lids.
    probes=[]
    for side in (-1,1):
        for dx in np.linspace(-.035,.035,9):
            for dz in np.linspace(-.008,.012,5):
                x=side*EX+dx;z=EZ+dz
                if eye_opening(x,z)[0]>.007:probes.append(front(x,z)-float(eye_front(x,z)))
    bm.free()
    original_mats={o.name:list(o.data.materials) for o in details}
    cam=scene.camera;cam.data.ortho_scale=1.2
    views={'front':(0,-3,0),'profile':(3,0,0),'three-quarter':(1.7,-2.94,.07),'left':(-3,0,0)}
    def render(name,view):
        cam.location=views[view];cam.rotation_euler=(-cam.location).to_track_quat('-Z','Y').to_euler()
        scene.render.filepath=str(out/f'{name}-{view}.png');bpy.ops.render.render(write_still=True)
    for mode in ('base','clay','skin'):
        base.hide_render=mode!='base';sculpt.hide_render=mode=='base'
        sculpt.data.materials.clear();sculpt.data.materials.append(skin if mode=='skin' else clay)
        for obj in details:
            obj.hide_render=mode=='base'
            obj.data.materials.clear()
            if mode=='skin':
                for m in original_mats[obj.name]:obj.data.materials.append(m)
            else:
                for _ in original_mats[obj.name]:obj.data.materials.append(clay)
            # Brows and lip colouring are excluded from the clay comparison.
            if mode=='clay' and ('brow' in obj.name or 'meeting line' in obj.name):obj.hide_render=True
        for view in views:render(mode,view)
    base.hide_render=True;base.hide_set(True);sculpt.hide_render=False;sculpt.hide_set(False)
    for obj in details:obj.hide_set(False)
    bpy.ops.object.select_all(action='DESELECT');sculpt.select_set(True);bpy.context.view_layer.objects.active=sculpt
    report=dict(method='Positive ellipsoid union -> subdivided surface sculpt + eyes/brows',
        baseline=str(baseline/'fitted-mesh.npz'),baseline_sha256=hashlib.sha256((baseline/'fitted-mesh.npz').read_bytes()).hexdigest(),
        baseline_primitives=51,csg_baseline_used=False,vertices=len(final),faces=len(sculpt.data.polygons),
        eye_visibility_probes=len(probes),eye_clearance_min=float(min(probes)),
        displacement_mean=float(np.linalg.norm(final-original,axis=1).mean()),
        displacement_max=float(np.linalg.norm(final-original,axis=1).max()),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),seconds=time.perf_counter()-started,
        limits='12 GiB RAM, zero swap; normalized head-height units')
    for filename in ('front.png','right-hairless.png','left-hairless.png'):
        img=bpy.data.images.load(str(HERE.parent/'procedural-vaan/references/profile-hair-removal'/filename));img.use_fake_user=True;img.pack()
    text=bpy.data.texts.new('sculpt-report.json');text.write(json.dumps(report,indent=2))
    bpy.data.texts.load(str(Path(__file__).resolve()))
    np.savez_compressed(out/'sculpt-mesh.npz',vertices=final,faces=np.array([p.vertices[:] for p in sculpt.data.polygons]))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'head.blend'))
    print('DONE',json.dumps(report),flush=True)


if __name__=='__main__':main()
