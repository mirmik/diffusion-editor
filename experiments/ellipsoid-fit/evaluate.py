"""Independent bidirectional mesh-distance checks after volume extraction."""
import hashlib
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from limits import require_budget
require_budget()
import numpy as np
import torch
import trimesh
import cumesh


def load_mesh(path):
    d=np.load(path);f=d['faces']
    if f.shape[1]==4:f=np.concatenate([f[:,[0,1,2]],f[:,[0,2,3]]])
    return trimesh.Trimesh(d['vertices'],f,process=False)


out=Path(sys.argv[1]).resolve();target=load_mesh(out/'target.npz')
sample,_=trimesh.sample.sample_surface(target,30000,seed=73029)
target_tree=cumesh.cuBVH(target.vertices.astype(np.float32),target.faces.astype(np.int32))
result={}
for name in ('initial','fitted'):
    mesh=load_mesh(out/f'{name}-mesh.npz')
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume>0
    components=mesh.split(only_watertight=False)
    tree=cumesh.cuBVH(mesh.vertices.astype(np.float32),mesh.faces.astype(np.int32))
    points,_=trimesh.sample.sample_surface(mesh,30000,seed=93029)
    d1,_,_=tree.unsigned_distance(torch.tensor(sample,dtype=torch.float32,device='cuda'))
    d2,_,_=target_tree.unsigned_distance(torch.tensor(points,dtype=torch.float32,device='cuda'))
    a,b=d1.cpu().numpy(),d2.cpu().numpy()
    from fit_csg import region_masks
    local={region:dict(count=int(mask.sum()),mean=float(a[mask.numpy()].mean()),
                      p95=float(np.quantile(a[mask.numpy()],.95)))
           for region,mask in region_masks(torch.tensor(sample)).items() if mask.any()}
    result[name]=dict(target_to_mesh_mean=float(a.mean()),target_to_mesh_p95=float(np.quantile(a,.95)),
        mesh_to_target_mean=float(b.mean()),mesh_to_target_p95=float(np.quantile(b,.95)),
        bidirectional_mean=float((a.mean()+b.mean())/2),
        components=len(components),euler=int(mesh.euler_number),volume=float(mesh.volume),
        watertight=bool(mesh.is_watertight),vertices=len(mesh.vertices),triangles=len(mesh.faces),local=local)
report=dict(metrics=result,units='Fraction of target head + short neck height; not physical metres',
    samples_per_direction=30000,independent_seeds=[73029,93029],
    mean_distance_improvement_factor=result['initial']['bidirectional_mean']/result['fitted']['bidirectional_mean'],
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(out/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
