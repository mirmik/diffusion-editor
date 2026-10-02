"""Audit/normalize Pixal3D geometry and cache independent fitting samples."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'vendor'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from limits import require_budget
require_budget()
import numpy as np
import torch
import trimesh
import cumesh


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);args=parser.parse_args()
    out=args.run.resolve();pixal=out/'pixal'
    report=json.loads((pixal/'result.json').read_text());path=pixal/report['shape']
    mesh=trimesh.load(path,force='mesh',process=True)
    # Direct Studio export is glTF Y-up; convert to the procedural model's Z-up.
    transform=np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]],dtype=float)
    mesh.apply_transform(transform)
    components=mesh.split(only_watertight=False)
    component_sizes=[len(m.faces) for m in components]
    mesh=max(components,key=lambda m:len(m.faces))
    edge_counts=np.bincount(mesh.edges_unique_inverse)
    before=dict(boundary_edges=int((edge_counts==1).sum()),nonmanifold_edges=int((edge_counts>2).sum()),faces=len(mesh.faces))
    original=mesh.copy()
    original.vertices=(original.vertices-original.bounds.mean(0))/original.extents[2]
    regularized=np.load(out/'regularized.npz')
    mesh=trimesh.Trimesh(regularized['vertices'],regularized['faces'],process=False)
    repair=json.loads((out/'regularization.json').read_text())
    repair['original_edge_audit']=before
    assert mesh.is_watertight and mesh.is_winding_consistent,(mesh.is_watertight,mesh.is_winding_consistent)
    if mesh.volume<0:mesh.invert()
    center=mesh.bounds.mean(0);scale=float(mesh.extents[2])
    mesh.vertices=(mesh.vertices-center)/scale
    mesh.export(out/'target.ply')
    np.savez_compressed(out/'target.npz',vertices=mesh.vertices.astype(np.float32),faces=mesh.faces.astype(np.int32))
    original.vertices=(original.vertices-center)/scale
    repair_distances={}
    for label,source,target in [('original_to_volume',original,mesh),('volume_to_original',mesh,original)]:
        tree=cumesh.cuBVH(target.vertices.astype(np.float32),target.faces.astype(np.int32))
        pts,_=trimesh.sample.sample_surface(source,15000,seed=321)
        dd,_,_=tree.unsigned_distance(torch.tensor(pts,dtype=torch.float32,device='cuda'))
        dd=dd.cpu().numpy()
        repair_distances[label]=dict(mean=float(dd.mean()),p95=float(np.quantile(dd,.95)),max=float(dd.max()))
    repair['surface_change_normalized_height']=repair_distances
    bvh=cumesh.cuBVH(mesh.vertices.astype(np.float32),mesh.faces.astype(np.int32))
    def distances(points):
        values=[]
        for p in np.array_split(points,max(1,len(points)//50000)):
            d,_,_=bvh.signed_distance(torch.tensor(p,dtype=torch.float32,device='cuda'))
            values.append(d.cpu().numpy())
        return np.concatenate(values)
    # Independent CPU ray containment guards sign convention / CUDA query use.
    rng=np.random.default_rng(2087)
    check=rng.uniform(mesh.bounds[0]-.1,mesh.bounds[1]+.1,(256,3))
    d=distances(check);inside=np.concatenate([mesh.contains(chunk) for chunk in np.array_split(check,32)])
    sign=1 if np.mean((d<0)==inside)>.5 else -1
    agreement=float(np.mean((d*sign<0)==inside));assert agreement>.995,agreement
    started=time.monotonic()
    arrays={}
    for name,seed,count in [('train',416,180000),('test',9417,60000)]:
        rng=np.random.default_rng(seed)
        surface,fi=trimesh.sample.sample_surface(mesh,count//2,seed=seed)
        normal=mesh.face_normals[fi]
        jitter=rng.normal(0,.022,(len(surface),1))
        near=surface+normal*jitter
        uniform=rng.uniform(mesh.bounds[0]-.08,mesh.bounds[1]+.08,(count//2,3))
        points=np.concatenate([near,uniform]).astype(np.float32)
        sdf=sign*distances(points)
        arrays.update({name+'_points':points,name+'_sdf':sdf,name+'_surface':surface.astype(np.float32),name+'_normals':normal.astype(np.float32)})
    np.savez_compressed(out/'samples.npz',**arrays)
    info=dict(source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        gltf_to_z_up=transform.tolist(),centering=center.tolist(),scale=scale,
        original_component_faces=component_sizes,selected_component='largest by face count',
        local_repair=repair,
        vertices=len(mesh.vertices),faces=len(mesh.faces),watertight=bool(mesh.is_watertight),
        winding_consistent=bool(mesh.is_winding_consistent),euler=int(mesh.euler_number),volume=float(mesh.volume),
        bounds=mesh.bounds.tolist(),distance_sign_multiplier=sign,sign_check_cpu_ray_agreement=agreement,
        sample_counts={k:len(v) for k,v in arrays.items()},seconds=time.monotonic()-started,
        units='Normalized target head + short neck height is one; no physical scale inferred.')
    (out/'target-report.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
