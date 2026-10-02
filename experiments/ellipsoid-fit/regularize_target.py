"""Make a separate watertight volume copy; preserve the original Pixal3D GLB."""
import json
from pathlib import Path
import sys
import bpy
import bmesh
sys.path.insert(0,str(Path(__file__).resolve().parent))
from limits import require_budget
require_budget()
import numpy as np

out=Path(sys.argv[sys.argv.index('--')+1]).resolve()
report=json.loads((out/'pixal/result.json').read_text())
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(out/'pixal'/report['shape']))
objects=[o for o in bpy.context.scene.objects if o.type=='MESH']
bpy.ops.object.select_all(action='DESELECT')
for obj in objects:obj.select_set(True)
bpy.context.view_layer.objects.active=objects[0];bpy.ops.object.join();obj=bpy.context.object
bpy.ops.object.transform_apply(location=True,rotation=True,scale=True)
original=np.array([v.co[:] for v in obj.data.vertices]);height=np.ptp(original[:,2])
pitch=height*.0025
obj.data.remesh_voxel_size=pitch;obj.data.remesh_voxel_adaptivity=0
bpy.ops.object.voxel_remesh()
for factor in (.4,-.42)*2:
    m=obj.modifiers.new('Voxel smoothing','SMOOTH');m.factor=factor
    bpy.ops.object.modifier_apply(modifier=m.name)
bm=bmesh.new();bm.from_mesh(obj.data);bmesh.ops.triangulate(bm,faces=list(bm.faces))
bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces))
bad=sum(not e.is_manifold for e in bm.edges)
assert bad==0,bad
bm.to_mesh(obj.data)
info=dict(method='Blender voxel remesh + alternating Laplacian smoothing',relative_voxel=.0025,
    source_unit_voxel=float(pitch),non_manifold_edges=bad,vertices=len(bm.verts),faces=len(bm.faces),
    euler=len(bm.verts)-len(bm.edges)+len(bm.faces),volume=bm.calc_volume(signed=True),
    coordinates='Blender glTF import -> Z-up, original source units')
bm.free()
np.savez_compressed(out/'regularized.npz',vertices=np.array([v.co[:] for v in obj.data.vertices],dtype=np.float32),
                    faces=np.array([p.vertices[:] for p in obj.data.polygons],dtype=np.int32))
(out/'regularization.json').write_text(json.dumps(info,indent=2)+'\n')
print('REGULARIZED',json.dumps(info),flush=True)
