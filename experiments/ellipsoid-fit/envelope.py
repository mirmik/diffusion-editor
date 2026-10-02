"""Convert the Pixal3D thin shell into its filled external envelope.

Build a narrow unsigned-distance barrier, fill enclosed empty cells, then
offset the barrier's outer edge back by its thickness. Original GLB is retained.
"""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from limits import require_budget
require_budget()
import numpy as np
import torch
import trimesh
import cumesh
from scipy import ndimage

out=Path(sys.argv[1]).resolve();result=json.loads((out/'pixal/result.json').read_text())
m=trimesh.load(out/'pixal'/result['shape'],force='mesh',process=True)
transform=np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]],float)
m.apply_transform(transform);center=m.bounds.mean(0);scale=m.extents[2]
m.vertices=(m.vertices-center)/scale
np.savez_compressed(out/'original-normalized.npz',vertices=m.vertices.astype(np.float32),faces=m.faces.astype(np.int32))
tree=cumesh.cuBVH(m.vertices.astype(np.float32),m.faces.astype(np.int32))
step=.0064;axis=np.arange(-100,101,dtype=np.float32)*step
y,z=torch.meshgrid(torch.tensor(axis,device='cuda'),torch.tensor(axis,device='cuda'),indexing='ij')
cache=out/'unsigned-grid.npz'
if cache.exists():
    data=np.load(cache);assert np.isclose(float(data['step']),step)
    unsigned=data['field']
else:
    values=[]
    for i,x in enumerate(axis):
        pts=torch.stack([torch.full_like(y,float(x)),y,z],-1).reshape(-1,3)
        d,_,_=tree.unsigned_distance(pts);values.append(d.cpu().numpy().reshape(len(axis),len(axis)))
        if i%80==0:print('distance slab',i,flush=True)
    unsigned=np.stack(values)
    np.savez_compressed(cache,field=unsigned,step=step)
barrier=unsigned<step*1.5
# Source neck ends obliquely: the rear rim is higher than the front rim.
# Declare the ROI cap explicitly, below the chin but above the open neck rim.
caps=np.searchsorted(axis,-.47+.30*(axis+.10))
for j,cap in enumerate(caps):barrier[:,j,cap:cap+2]=True
filled=ndimage.binary_fill_holes(barrier)
for j,cap in enumerate(caps):filled[:,j,:cap+2]=False
volume=float(filled.sum()*step**3)
assert volume>.1,(volume,'Unsigned shell did not enclose the head')
inside=ndimage.distance_transform_edt(filled,sampling=step)
outside=ndimage.distance_transform_edt(~filled,sampling=step)
sdf=(outside-inside+step*1.5).astype(np.float32)
np.savez_compressed(out/'envelope-grid.npz',field=sdf,lower=np.array([-.64]*3),step=step)
info=dict(method='Unsigned-distance barrier + exterior flood fill + thickness compensation',
    grid_step=step,barrier_width=step*1.5,neck_cap_plane='z=-0.47+0.30*(y+0.10), two-cell barrier',filled_volume_before_compensation=volume,
    original_normalized_signed_volume=float(m.volume),original_area=float(m.area),
    center_original_units=center.tolist(),scale_original_units=float(scale),gltf_to_z_up=transform.tolist(),
    limitation='Neck trimmed/capped on an explicit oblique ROI plane. Enclosed internal cavities intentionally filled; sub-barrier openings may close.')
(out/'envelope-report.json').write_text(json.dumps(info,indent=2)+'\n')
print('ENVELOPE',json.dumps(info),flush=True)
