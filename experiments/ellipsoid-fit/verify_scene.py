"""Reopen the Blender deliverable and verify the saved editable parameters."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from limits import require_budget
require_budget()
import bpy
import numpy as np

out=Path(bpy.data.filepath).parent;params=json.loads((out/'fitted.json').read_text())
collection=bpy.data.collections['SOURCE | fitted ellipsoids']
assert collection.hide_render and collection.hide_viewport
parts=[p for s in params.get('stages',[params]) for p in s['parts']]
assert len(collection.objects)==len(parts)
for part in parts:
    obj=collection.objects[part['name']]
    np.testing.assert_allclose(obj.location,part['center'],atol=1e-6)
    np.testing.assert_allclose(obj.scale,part['radii'],atol=1e-6)
    np.testing.assert_allclose(obj.rotation_quaternion,part['quaternion_wxyz'],atol=1e-6)
for name in ('target','initial','fitted'):
    obj=bpy.data.objects[name];path=out/('target.npz' if name=='target' else f'{name}-mesh.npz')
    saved=np.load(path)
    actual=np.array([v.co[:] for v in obj.data.vertices])
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual,saved['vertices'],atol=1e-6)
    assert len(obj.data.polygons)==len(saved['faces'])
    assert obj.hide_render==(name!='fitted')
result=dict(reopened_scene_matches=True,editable_ellipsoids=len(parts),
            geometry_and_parameters_match=True,default_visible='fitted',
            source_collection_hidden=True)
(out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print('VERIFIED',json.dumps(result),flush=True)
