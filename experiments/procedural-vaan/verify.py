"""Reopen-and-check an artifact. Run in Blender after loading vaan.blend."""
import json
import hashlib
from pathlib import Path
import sys
import bpy
import bmesh
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from build import audit_character
from anatomy import body_field
from details import vest_point
from portrait import head_field, eye_point


def main():
    directory=Path(bpy.data.filepath).parent
    report=json.loads((directory/'report.json').read_text())
    actual=audit_character()
    for key in ('meshes','curves','curve_splines','curve_points','vertices','faces'):
        assert actual[key]==report['scene'][key],(key,actual[key],report['scene'][key])
    np.testing.assert_allclose(actual['world_bounds_metres'],report['scene']['world_bounds_metres'],atol=1e-6)
    topology={}
    for name in ['Skin | torso arms hands','Skin | head ears nose','Navy cargo trousers']:
        bm=bmesh.new(); bm.from_mesh(bpy.data.objects[name].data)
        bad=sum(not e.is_manifold for e in bm.edges)
        euler=len(bm.verts)-len(bm.edges)+len(bm.faces)
        volume=bm.calc_volume(signed=True)
        assert bad==0 and euler==2 and volume>0,(name,bad,euler,volume)
        topology[name]={'non_manifold_edges':bad,'euler':euler,'volume_local_m3':volume}
        bm.free()
    # Regression for the original chest protruding through the front of the vest.
    probes=[]
    for side in (-1,1):
        for z in np.linspace(1.28,1.465,24):
            for a in np.linspace(.50,.95,14):
                x,y,z1=vest_point(a,z)
                probes.append((side*x,y-.003,z1))
    p=np.asarray(probes).T
    distances=body_field(p)
    assert np.min(distances)>-.001, float(np.min(distances))
    # Guard against the SDF skin covering the newly recessed eye apertures.
    eye_probes=np.asarray([eye_point(side,u,v) for side in (-1,1)
                           for u in np.linspace(-.80,.80,17)
                           for v in np.linspace(-.65,.65,9)]).T
    eye_distances=head_field(eye_probes)
    assert np.min(eye_distances)>0, float(np.min(eye_distances))
    packed_hashes={hashlib.sha256(img.packed_file.data).hexdigest()
                   for img in bpy.data.images if img.packed_file}
    assert len(report['references'])>=4
    assert all(ref['sha256'] in packed_hashes for ref in report['references'].values())
    result={'reopened_scene_matches':True,'topology':topology,
            'vest_probe_count':len(probes),'minimum_skin_clearance_m':float(np.min(distances)),
            'eye_aperture_probe_count':eye_probes.shape[1],
            'minimum_eye_aperture_clearance_m':float(np.min(eye_distances)),
            'packed_reference_images':sum(bool(img.packed_file) for img in bpy.data.images),
            'all_reference_hashes_match':True}
    (directory/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('VERIFIED',json.dumps(result),flush=True)


if __name__=='__main__':
    main()
