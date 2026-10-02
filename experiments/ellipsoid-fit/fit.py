"""Fit an editable smooth ellipsoid union to a fixed Pixal3D target.

Run with the existing Pixal3D CUDA Python. Test samples are never used in the
optimizer or residual primitive placement. No target vertices are deformed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from limits import require_budget
require_budget()
import numpy as np
import torch
from torch.nn import functional as F
from scipy.spatial.transform import Rotation
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from field import torch_model,numpy_field


def initial_parameters():
    previous=HERE.parent/'procedural-vaan/output/ellipsoid-head-03/report.json'
    raw=json.loads(previous.read_text())['positive_ellipsoids']
    raw=raw+[dict(name='Short neck',center=[0,.022,1.557],radii=[.034,.037,.028],rx=0,rz=0)]
    parts=[]
    for p in raw:
        center=(np.asarray(p['center'])-[0,.013,1.656])/.254
        # Existing construction is Rz @ Rx, i.e. extrinsic x then z.
        r=Rotation.from_euler('xz',[p['rx'],p['rz']],degrees=True)
        q=r.as_quat();q=np.roll(q,1)
        parts.append(dict(name=p['name'],center=center.tolist(),radii=(np.asarray(p['radii'])/.254).tolist(),quaternion_wxyz=q.tolist()))
    return dict(parts=parts,temperature=.012)


@torch.no_grad()
def batched(model,points,batch=8192):
    return torch.cat([model(p) for p in points.split(batch)])


@torch.no_grad()
def score(model,data):
    p=batched(model,data['test_points']);d=data['test_sdf']
    surf=batched(model,data['test_surface']).abs()
    uniform=slice(len(p)//2,None);inside=p[uniform]<0;truth=d[uniform]<0
    return dict(surface_field_mean=float(surf.mean()),surface_field_p95=float(torch.quantile(surf,.95)),
        volume_iou=float((inside&truth).sum()/(inside|truth).sum()),
        uniform_occupancy_accuracy=float((inside==truth).float().mean()),
        narrow_band_field_mae=float((p[d.abs()<.04]-d[d.abs()<.04]).abs().mean()))


def save_params(path,model):path.write_text(json.dumps(model.export(),indent=2)+'\n')


@torch.no_grad()
def grid(model,path,step=.005):
    lower=np.array([-.65,-.65,-.65]);axis=np.arange(-130,131,dtype=np.float32)*step
    # Chunk one x-slab at a time; no huge primitive-by-volume allocation.
    values=[]
    y,z=torch.meshgrid(torch.tensor(axis,device='cuda'),torch.tensor(axis,device='cuda'),indexing='ij')
    for x in axis:
        p=torch.stack([torch.full_like(y,float(x)),y,z],-1).reshape(-1,3)
        values.append(batched(model,p).cpu().numpy().reshape(len(axis),len(axis)))
    volume=np.stack(values).astype(np.float32)
    assert min(volume[0].min(),volume[-1].min(),volume[:,0].min(),volume[:,-1].min(),volume[:,:,0].min(),volume[:,:,-1].min())>0
    np.savez_compressed(path,field=volume,lower=lower,step=step)


def optimize(model,data,steps,seed,log,stage):
    torch.manual_seed(seed)
    initial_c=model.center.detach().clone();initial_r=model.log_radii.detach().clone()
    opt=torch.optim.Adam([{'params':[model.center],'lr':.0015},
                         {'params':[model.log_radii],'lr':.004},
                         {'params':[model.quaternion],'lr':.002}])
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps,eta_min=.00008)
    p,d,s=data['train_points'],data['train_sdf'],data['train_surface']
    for iteration in range(steps):
        idx=torch.randint(len(p),(6144,),device='cuda');si=torch.randint(len(s),(2048,),device='cuda')
        prediction=model(torch.cat([p[idx],s[si]]));v=prediction[:len(idx)];surf=prediction[len(idx):]
        sdf=F.smooth_l1_loss(v.clamp(-.05,.05)/.02,d[idx].clamp(-.05,.05)/.02,beta=.25)
        surface=F.smooth_l1_loss(surf/.01,torch.zeros_like(surf),beta=.25)
        occupancy=F.binary_cross_entropy_with_logits(-v/.008,torch.sigmoid(-d[idx]/.008))
        regularizer=.01*(model.center-initial_c).square().mean()+.0005*(model.log_radii-initial_r).square().mean()
        loss=sdf+.65*surface+.15*occupancy+regularizer
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(),10)
        opt.step();sched.step()
        with torch.no_grad():
            model.log_radii.clamp_(math.log(.006),math.log(.65))
            model.center.copy_(torch.maximum(torch.minimum(model.center,initial_c+.20),initial_c-.20))
        if iteration%250==0 or iteration==steps-1:
            row=dict(stage=stage,iteration=iteration,loss=float(loss),sdf=float(sdf),surface=float(surface),occupancy=float(occupancy))
            log.append(row);print(json.dumps(row),flush=True)


def add_residual_parts(model,data,count=12):
    params=model.export()
    with torch.no_grad():error=batched(model,data['train_surface']).cpu().numpy()
    surface=data['train_surface'].cpu().numpy();normals=data['train_normals'].cpu().numpy()
    candidates=np.argsort(error)[::-1];selected=[]
    for i in candidates:
        if error[i]<.007 or len(selected)>=count:break
        if any(np.linalg.norm(surface[i]-surface[j])<.07 for j in selected):continue
        selected.append(int(i));radius=.025
        params['parts'].append(dict(name=f'Residual volume {len(selected):02d}',
            center=(surface[i]-normals[i]*radius).tolist(),radii=[radius]*3,quaternion_wxyz=[1,0,0,0]))
    return torch_model(params),[dict(surface_point=surface[i].tolist(),field_error=float(error[i])) for i in selected]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path)
    parser.add_argument('--steps',type=int,default=3000);parser.add_argument('--refine-steps',type=int,default=1500)
    args=parser.parse_args();out=args.run.resolve();started=time.monotonic()
    assert not (out/'fitted.json').exists(),'Use a fresh run or explicitly archive the prior fit'
    torch.set_num_threads(8);torch.manual_seed(416)
    source=out/'source';source.mkdir(exist_ok=True)
    for path in HERE.glob('*.py'):shutil.copy2(path,source/path.name)
    raw=np.load(out/'samples.npz');data={k:torch.tensor(raw[k],device='cuda') for k in raw.files}
    model=torch_model(initial_parameters());save_params(out/'initial.json',model)
    probe=np.random.default_rng(919).uniform(-.6,.6,(2000,3)).astype(np.float32)
    actual=model(torch.tensor(probe,device='cuda')).detach().cpu().numpy()
    parity=float(np.max(np.abs(actual-numpy_field(probe,model.export()))));assert parity<1e-6,parity
    initial=score(model,data);print('INITIAL',json.dumps(initial),flush=True)
    grid(model,out/'initial-grid.npz')
    history=[]
    optimize(model,data,args.steps,416,history,'base')
    save_params(out/'base-fitted.json',model)
    model,additions=add_residual_parts(model,data)
    optimize(model,data,args.refine_steps,417,history,'residual')
    save_params(out/'fitted.json',model);final=score(model,data)
    grid(model,out/'fitted-grid.npz')
    report=dict(initial=initial,final=final,initial_primitives=39,final_primitives=len(model.names),
        residual_additions=additions,history=history,seconds=time.monotonic()-started,
        steps=args.steps,refine_steps=args.refine_steps,seed=416,numpy_torch_field_max_error=parity,
        temperature=model.temperature,grid_step=.005,torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
        loss='clamped approximate SDF + target surface + soft occupancy + weak center/radius prior',
        validation='Independent seed 9417 test set; not used in optimization or primitive placement.',
        input_sha256={n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in ['target.npz','samples.npz']},
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')})
    (out/'fit-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('FINAL',json.dumps(final),'seconds',report['seconds'],flush=True)


if __name__=='__main__':main()
