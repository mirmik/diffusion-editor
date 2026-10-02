"""Freeze the fitted bulk, optimize cavities, then add small positive details."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from limits import require_budget
require_budget()
import numpy as np
import torch
from torch.nn import functional as F
from field import torch_model
from csg_field import CSG
from fit import batched,grid,score


def region_masks(points):
    x,y,z=points.T
    regions={}
    for side in (-1,1):
        regions[f'eye {side}']=(side*x>.045)&(side*x<.24)&(y<-.19)&(z>-.03)&(z<.16)
        regions[f'ear {side}']=(side*x>.29)&(z>-.20)&(z<.14)
        regions[f'nostril {side}']=(side*x>.008)&(side*x<.075)&(y<-.28)&(z>-.20)&(z<-.07)
    regions['mouth']=(x.abs()<.15)&(y<-.23)&(z>-.32)&(z<-.16)
    return regions


def weights(points):
    mask=torch.stack(list(region_masks(points).values())).any(0)
    return torch.where(mask,6.,1.)


@torch.no_grad()
def detail_score(model,data):
    p=data['test_surface'];f=batched(model,p).abs()
    return {name:dict(count=int(mask.sum()),mean=float(f[mask].mean()),p95=float(torch.quantile(f[mask],.95)))
            for name,mask in region_masks(p).items() if mask.any()}


def quaternion_normal(normal):
    n=normal/np.linalg.norm(normal)
    q=np.array([1+n[2],-n[1],n[0],0]) if n[2]>-.9999 else np.array([0.,1,0,0])
    return (q/np.linalg.norm(q)).tolist()


@torch.no_grad()
def seed_cuts(base,data):
    surface=data['train_surface'];normals=data['train_normals'];error=batched(base,surface)
    parts=[];records=[]
    for name,mask in region_masks(surface).items():
        idx=torch.where(mask)[0];i=idx[error[idx].argmin()]
        if error[i]>-.0007:continue
        p=surface[i].cpu().numpy();n=normals[i].cpu().numpy()
        radii=[.028,.016,.015] if 'ear' not in name else [.035,.024,.020]
        center=p+n*(radii[2]-.001)
        parts.append(dict(name='Cut '+name,center=center.tolist(),radii=radii,quaternion_wxyz=quaternion_normal(n)))
        records.append(dict(region=name,point=p.tolist(),overfill=float(-error[i])))
    assert parts
    return dict(parts=parts,temperature=.0015,operation='subtract',blend=.0015),records


@torch.no_grad()
def seed_details(model,data):
    surface=data['train_surface'];normals=data['train_normals'];error=batched(model,surface)
    mask=weights(surface)>1
    idx=torch.where(mask)[0];ordered=idx[torch.argsort(error[idx],descending=True)].cpu().tolist()
    selected=[];parts=[]
    for i in ordered:
        if len(parts)>=10 or error[i]<.0009:break
        p=surface[i].cpu().numpy();n=normals[i].cpu().numpy()
        if any(np.linalg.norm(p-q)<.024 for q in selected):continue
        selected.append(p)
        parts.append(dict(name=f'Added detail {len(parts)+1}',center=(p-n*.009).tolist(),
                          radii=[.014,.010,.009],quaternion_wxyz=quaternion_normal(n)))
    assert parts
    return dict(parts=parts,temperature=.0012,operation='union',blend=.0012)


def optimize(model,data,steps,phase,active_groups,history):
    selected=[];priors=[]
    for i,group in enumerate(model.groups):
        for p in group.parameters():p.requires_grad_(i in active_groups)
        if i in active_groups:
            selected.extend([{'params':[group.center],'lr':.0003},{'params':[group.log_radii],'lr':.0015},
                             {'params':[group.quaternion],'lr':.001}])
            priors.append((group,group.center.detach().clone()))
    opt=torch.optim.Adam(selected);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps,eta_min=.000025)
    p,d,s=data['train_points'],data['train_sdf'],data['train_surface']
    feature_ids=torch.where(weights(s)>1)[0]
    for iteration in range(steps):
        idx=torch.randint(len(p),(4096,),device='cuda')
        si=torch.cat([torch.randint(len(s),(1024,),device='cuda'),feature_ids[torch.randint(len(feature_ids),(2048,),device='cuda')]])
        prediction=model(torch.cat([p[idx],s[si]]));v=prediction[:len(idx)];surf=prediction[len(idx):]
        pw=weights(p[idx]);sw=weights(s[si])
        a=F.smooth_l1_loss(v.clamp(-.03,.03)/.008,d[idx].clamp(-.03,.03)/.008,beta=.3,reduction='none')
        b=F.smooth_l1_loss(surf/.006,torch.zeros_like(surf),beta=.3,reduction='none')
        loss=(a*pw).sum()/pw.sum()+.7*(b*sw).sum()/sw.sum()+.08*F.binary_cross_entropy_with_logits(-v/.005,torch.sigmoid(-d[idx]/.005))
        loss+=sum(.01*(group.center-initial).square().mean() for group,initial in priors)
        opt.zero_grad(set_to_none=True);loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],10)
        opt.step();schedule.step()
        with torch.no_grad():
            for group,initial in priors:
                group.center.copy_(torch.maximum(torch.minimum(group.center,initial+.04),initial-.04))
                group.log_radii.clamp_(math.log(.004),math.log(.07))
        if iteration%300==0 or iteration==steps-1:
            row=dict(phase=phase,iteration=iteration,loss=float(loss));history.append(row);print(json.dumps(row),flush=True)


def main():
    import time
    started=time.monotonic();parser=argparse.ArgumentParser();parser.add_argument('run',type=Path)
    parser.add_argument('--baseline',required=True,type=Path);args=parser.parse_args();out=args.run.resolve()
    assert not (out/'fitted.json').exists()
    source=out/'source';source.mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):shutil.copy2(p,source/p.name)
    torch.set_num_threads(4);torch.manual_seed(529)
    raw=np.load(out/'samples.npz');data={k:torch.tensor(raw[k],device='cuda') for k in raw.files}
    base=json.loads(args.baseline.read_text());model=CSG([base])
    (out/'initial.json').write_text(json.dumps(model.export(),indent=2)+'\n')
    initial=score(model,data);local_initial=detail_score(model,data)
    grid(model,out/'initial-grid.npz')
    cuts,seed_records=seed_cuts(model,data);model=CSG([base,cuts]);history=[]
    optimize(model,data,1800,'subtract',set([1]),history)
    (out/'cut-fitted.json').write_text(json.dumps(model.export(),indent=2)+'\n')
    details=seed_details(model,data);model=CSG([*model.export()['stages'],details])
    optimize(model,data,1200,'add_details',set([2]),history)
    optimize(model,data,1400,'refine_cuts_and_details',set([1,2]),history)
    params=model.export();(out/'fitted.json').write_text(json.dumps(params,indent=2)+'\n')
    final=score(model,data);local_final=detail_score(model,data);grid(model,out/'fitted-grid.npz')
    counts=[dict(operation=s['operation'],count=len(s['parts'])) for s in params['stages']]
    report=dict(initial=initial,final=final,local_initial=local_initial,local_final=local_final,
        stages=counts,cut_seeds=seed_records,history=history,seconds=time.monotonic()-started,
        baseline=str(args.baseline.resolve()),baseline_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        coarse_base_frozen=True,seed=529,
        validation='Same held-out geometry samples as head-01; not used to optimize or place primitives.',
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')})
    (out/'fit-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('FINAL',json.dumps(dict(global_metrics=final,local_metrics=local_final,stages=counts,seconds=report['seconds'])),flush=True)


if __name__=='__main__':main()
