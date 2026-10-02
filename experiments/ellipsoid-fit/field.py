"""Matching NumPy/PyTorch ellipsoid fields, negative inside, head height = 1.

IQ ellipsoid distance is approximate. Union is log-sum-exp soft minimum,
with one fixed temperature: optimization and extraction use the same formula.
"""
import numpy as np


def rotation_numpy(q):
    q=np.asarray(q);q=q/np.linalg.norm(q,axis=-1,keepdims=True)
    w,x,y,z=q.T
    return np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
        2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
        2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],axis=-1).reshape(-1,3,3)


def numpy_field(points,params):
    pts=np.asarray(points);parts=params['parts'];k=params['temperature']
    centers=np.asarray([p['center'] for p in parts]);radii=np.asarray([p['radii'] for p in parts])
    matrices=rotation_numpy([p['quaternion_wxyz'] for p in parts])
    q=np.einsum('nki,kij->nkj',pts[:,None,:]-centers[None,:,:],matrices)
    k0=np.sqrt(np.maximum(np.sum((q/radii)**2,axis=-1),1e-16))
    k1=np.sqrt(np.maximum(np.sum((q/radii**2)**2,axis=-1),1e-16))
    d=k0*(k0-1)/k1
    d=np.where(k0<1e-7,-radii.min(axis=1),d)
    a=d.min(axis=1,keepdims=True)
    return a[:,0]-k*np.log(np.exp(-(d-a)/k).sum(axis=1))


def torch_model(params,device='cuda'):
    import torch
    from torch import nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.names=[p['name'] for p in params['parts']]
            self.temperature=params['temperature']
            self.center=nn.Parameter(torch.tensor([p['center'] for p in params['parts']],dtype=torch.float32,device=device))
            self.log_radii=nn.Parameter(torch.tensor([p['radii'] for p in params['parts']],dtype=torch.float32,device=device).log())
            self.quaternion=nn.Parameter(torch.tensor([p['quaternion_wxyz'] for p in params['parts']],dtype=torch.float32,device=device))

        def matrices(self):
            q=nn.functional.normalize(self.quaternion,dim=-1)
            w,x,y,z=q.unbind(-1)
            return torch.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],dim=-1).reshape(-1,3,3)

        def forward(self,p):
            r=self.log_radii.exp()
            q=torch.einsum('nki,kij->nkj',p[:,None,:]-self.center[None,:,:],self.matrices())
            k0=((q/r).square().sum(-1)+1e-14).sqrt()
            k1=((q/r.square()).square().sum(-1)+1e-14).sqrt()
            d=k0*(k0-1)/k1
            return -self.temperature*torch.logsumexp(-d/self.temperature,dim=1)

        def export(self):
            c=self.center.detach().cpu().tolist();r=self.log_radii.detach().exp().cpu().tolist()
            q=nn.functional.normalize(self.quaternion.detach(),dim=-1).cpu().tolist()
            return dict(temperature=self.temperature,coordinates='Z up, front -Y; target bbox height is 1',
                field='Approximate ellipsoid distance, log-sum-exp smooth union',
                parts=[dict(name=n,center=c[i],radii=r[i],quaternion_wxyz=q[i]) for i,n in enumerate(self.names)])
    return Model()
