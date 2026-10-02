"""Check a saved ellipsoid-head artifact, using its snapshotted field code."""
import hashlib
import json
from pathlib import Path
import sys

import bpy
import bmesh
import numpy as np


def main():
    out=Path(bpy.data.filepath).parent
    report=json.loads((out/'report.json').read_text())
    sys.path.insert(0,str(out/'source'))
    import ellipsoid_head as eh
    import portrait
    eh.configure_face(portrait)
    obj=bpy.data.objects['Skin | head ears nose']
    assert not obj.hide_render
    vertices=np.asarray([v.co[:] for v in obj.data.vertices])
    assert np.isfinite(vertices).all()
    assert len(vertices)==report['topology']['vertices']
    assert len(obj.data.polygons)==report['topology']['quads']
    bm=bmesh.new();bm.from_mesh(obj.data)
    bad=sum(not e.is_manifold for e in bm.edges)
    euler=len(bm.verts)-len(bm.edges)+len(bm.faces)
    volume=bm.calc_volume(signed=True)
    bm.verts.ensure_lookup_table()
    seen=set();components=[]
    for vertex in bm.verts:
        if vertex.index in seen:continue
        stack=[vertex];seen.add(vertex.index);count=0
        while stack:
            current=stack.pop();count+=1
            for edge in current.link_edges:
                other=edge.other_vert(current)
                if other.index not in seen:
                    seen.add(other.index);stack.append(other)
        components.append(count)
    bm.free()
    assert bad==0 and euler==2 and volume>0 and len(components)==1
    assert components==report['topology']['components']
    construction=bpy.data.collections['SOURCE | ellipsoid construction']
    assert construction.hide_render and construction.hide_viewport
    assert len(construction.objects)==len(eh.PARTS)
    assert eh.PARTS==report['positive_ellipsoids']
    assert eh.CUTS==report['negative_ellipsoids']
    for part in eh.PARTS:
        raw=construction.objects[part['name']]
        np.testing.assert_allclose(raw.location,part['center'],atol=1e-6)
        np.testing.assert_allclose(raw.scale,part['radii'],atol=1e-6)
        np.testing.assert_allclose(raw.rotation_euler.to_matrix(),eh.rotation(part),atol=1e-6)
    probes=np.asarray([portrait.eye_point(side,u,v) for side in (-1,1)
                       for u in np.linspace(-.8,.8,17) for v in np.linspace(-.65,.65,9)]).T
    clearance=float(eh.head_field(probes).min())
    assert clearance>0
    hashes={img.name:hashlib.sha256(img.packed_file.data).hexdigest()
            for img in bpy.data.images if img.packed_file}
    assert hashes==report['references']
    for name,digest in report['source_sha256'].items():
        assert hashlib.sha256((out/'source'/name).read_bytes()).hexdigest()==digest
    result=dict(reopened_scene_matches=True,vertices=len(vertices),
                non_manifold_edges=bad,euler=euler,components=components,volume_m3=volume,
                bounds_m=[vertices.min(0).tolist(),vertices.max(0).tolist()],
                dimensions_mm=(np.ptp(vertices,axis=0)*1000).tolist(),
                positive_ellipsoids=len(eh.PARTS),negative_ellipsoids=len(eh.CUTS),
                construction_hidden_by_default=True,eye_probe_count=probes.shape[1],
                eye_min_clearance_m=clearance,packed_references=len(hashes),
                source_hashes_match=True,
                limits='Geometry checks do not establish likeness or absence of all overlaps/gaps between separate face details.')
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('VERIFIED',json.dumps(result),flush=True)


if __name__=='__main__':main()
