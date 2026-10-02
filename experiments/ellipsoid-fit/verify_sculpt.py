"""Reopen a sculpt deliverable and check its actual saved geometry/provenance."""
from pathlib import Path
import hashlib
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from limits import require_budget
require_budget()
import bpy
import bmesh
import numpy as np

out=Path(bpy.data.filepath).parent
report=json.loads((out/'report.json').read_text())
obj=bpy.data.objects['Vaan | surface sculpt']
baseline=bpy.data.objects['BASE | positive ellipsoid union']
assert baseline.hide_render and not obj.hide_render
assert hashlib.sha256(Path(report['baseline']).read_bytes()).hexdigest()==report['baseline_sha256']
base_array=np.array([v.co[:] for v in baseline.data.vertices])
np.testing.assert_allclose(base_array,np.load(report['baseline'])['vertices'],atol=1e-6)
saved=np.load(out/'sculpt-mesh.npz')
depsgraph=bpy.context.evaluated_depsgraph_get()
evaluated=obj.evaluated_get(depsgraph);data=evaluated.to_mesh()
actual=np.empty(len(data.vertices)*3,dtype=np.float32);data.vertices.foreach_get('co',actual)
actual=actual.reshape(-1,3);assert np.isfinite(actual).all()
np.testing.assert_allclose(actual,saved['vertices'],atol=1e-6)
bm=bmesh.new();bm.from_mesh(data)
bad=sum(not e.is_manifold for e in bm.edges)
degenerate=sum(f.calc_area()<1e-12 for f in bm.faces)
seen=set();components=0
for vertex in bm.verts:
    if vertex in seen:continue
    components+=1;queue=[vertex];seen.add(vertex)
    while queue:
        cur=queue.pop()
        for edge in cur.link_edges:
            other=edge.other_vert(cur)
            if other not in seen:seen.add(other);queue.append(other)
volume=bm.calc_volume(signed=True);euler=len(bm.verts)-len(bm.edges)+len(bm.faces)
assert bad==0 and degenerate==0 and components==1 and euler==2 and volume>0
assert obj.data.shape_keys.key_blocks['Reference sculpt | face and ears'].value==1
embedded=bpy.data.texts['sculpt.py'].as_string().encode()
assert hashlib.sha256(embedded).hexdigest()==report['source_sha256']
references=[i for i in bpy.data.images if i.packed_file]
assert len(references)>=3
reference_hashes=json.loads((out/'reference-hashes.json').read_text())
for name,digest in reference_hashes.items():
    assert hashlib.sha256(bpy.data.images[name].packed_file.data).hexdigest()==digest
result=dict(reopened_scene_matches=True,vertices=len(actual),faces=len(bm.faces),
            nonmanifold_edges=bad,degenerate_faces=degenerate,components=components,euler=euler,
            volume=volume,baseline_unchanged=True,sculpt_shape_key=1,
            packed_references=len(references),reference_hashes_match=True,source_hash_matches=True,
            limits='Head surface topology verified; eye/brow patches are separate open surfaces. No global self-intersection proof or likeness claim.')
bm.free();evaluated.to_mesh_clear()
(out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print('VERIFIED',json.dumps(result),flush=True)
