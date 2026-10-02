"""Blender/OpenVDB extraction of the filled external target envelope."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from limits import require_budget
require_budget()
import numpy as np
import bpy
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from render import extract_grid
out=Path(sys.argv[sys.argv.index('--')+1]).resolve()
bpy.ops.wm.read_factory_settings(use_empty=True)
obj,report=extract_grid(out/'envelope-grid.npz','envelope',None)
# extract_grid exports quads; signed-distance queries use explicit triangles.
import bmesh
bm=bmesh.new();bm.from_mesh(obj.data);bmesh.ops.triangulate(bm,faces=list(bm.faces));bm.to_mesh(obj.data);bm.free()
np.savez_compressed(out/'regularized.npz',vertices=np.array([v.co[:] for v in obj.data.vertices],dtype=np.float32),
    faces=np.array([p.vertices[:] for p in obj.data.polygons],dtype=np.int32))
(out/'regularization.json').write_text(json.dumps(dict(report,coordinates='normalized Z-up',
    method='Filled external envelope, OpenVDB extraction, alternating Laplacian smoothing'),indent=2)+'\n')
print('ENVELOPE MESH',report,flush=True)
