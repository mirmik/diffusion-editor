#!/usr/bin/env python3
"""Apply the recorded testmodeling visualbruno/CuMesh recipe to a Studio GLB.

Run with the TRELLIS.2 CUDA Python. Original mesh and old experiments are read-only.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import time


def audit(mesh):
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    faces = mesh.faces
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    direction = edges[:, 0] < edges[:, 1]
    keys = (np.minimum(edges[:, 0], edges[:, 1]).astype(np.uint64) << np.uint64(32)) | np.maximum(edges[:, 0], edges[:, 1]).astype(np.uint64)
    order = np.argsort(keys); keys = keys[order]; direction = direction[order]
    starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(keys)])
    pairs = starts[counts == 2]
    adj = mesh.face_adjacency
    n, labels = connected_components(coo_matrix((np.ones(len(adj)), (adj[:, 0], adj[:, 1])), shape=(len(faces), len(faces))), directed=False)
    sizes = np.bincount(labels)
    return dict(vertices=len(mesh.vertices), faces=len(faces),
                boundary_edges=int((counts == 1).sum()), nonmanifold_edges=int((counts > 2).sum()),
                winding_inconsistent_edges=int((direction[pairs] == direction[pairs + 1]).sum()),
                zero_area_faces=int((mesh.area_faces == 0).sum()),
                edge_connected_components=int(n), largest_component_faces=int(sizes.max()),
                bounds=mesh.bounds.tolist(), watertight=bool(mesh.is_watertight))


def main():
    import numpy as np
    import torch
    import cumesh
    import trimesh
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--remesher', type=Path, default=Path('/home/mirmik/test/testmodeling/garment_lab/vaan_visualbruno/CuMesh/cumesh/remeshing.py'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = trimesh.load(args.input, force='mesh', process=False)
    # Experiment expects the geometry GLB, not a textured export with UV seams.
    source.visual = trimesh.visual.ColorVisuals(mesh=source)
    spec = importlib.util.spec_from_file_location('cumesh.visualbruno_remeshing', args.remesher)
    remesher = importlib.util.module_from_spec(spec); spec.loader.exec_module(remesher)
    report = dict(source=str(args.input.resolve()), source_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  remesher=str(args.remesher), remesher_sha256=hashlib.sha256(args.remesher.read_bytes()).hexdigest(),
                  cuda_backend=cumesh.__file__, recipe=dict(resolution=1024,band=1,padding=1.1,project_back=0,remove_inner_faces=False),
                  before=audit(source))
    (args.output/'report.json').write_text(json.dumps(report, indent=2))
    print('BEFORE', report['before'], flush=True)
    v=torch.as_tensor(np.asarray(source.vertices),dtype=torch.float32,device='cuda')
    f=torch.as_tensor(np.asarray(source.faces),dtype=torch.int32,device='cuda')
    lo=v.amin(0);hi=v.amax(0);start=time.monotonic()
    v,f=remesher.remesh_narrow_band_dc(v,f,center=(lo+hi)*.5,scale=float((hi-lo).max())*1.1,
        resolution=1024,band=1,project_back=0,remove_inner_faces=False,verbose=True)
    v=v.cpu().numpy();f=f.cpu().numpy()
    np.savez_compressed(args.output/'raw-remesh.npz',vertices=v,faces=f)
    vd=v.astype(np.float64)
    keep=np.any(np.cross(vd[f[:,1]]-vd[f[:,0]],vd[f[:,2]]-vd[f[:,0]]) != 0,axis=1)
    repaired=trimesh.Trimesh(v,f[keep],process=False);repaired.remove_unreferenced_vertices()
    repaired.export(args.output/'repaired.glb')
    source.export(args.output/'before.glb')
    report.update(elapsed_seconds=time.monotonic()-start,zero_area_removed=int((~keep).sum()),after=audit(repaired))
    loaded=trimesh.load(args.output/'repaired.glb',force='mesh',process=False)
    assert len(loaded.faces)==len(repaired.faces)
    assert np.allclose(loaded.bounds,repaired.bounds,atol=1e-7,rtol=0)
    assert hashlib.sha256(args.input.read_bytes()).hexdigest()==report['source_sha256']
    report['glb_roundtrip_verified']=True
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    print('RESULT',json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
