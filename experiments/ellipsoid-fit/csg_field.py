"""Ordered differentiable CSG: positive base, subtractive cuts, added details."""
import torch
from field import torch_model


class CSG(torch.nn.Module):
    def __init__(self,stages):
        super().__init__()
        self.operations=[s.get('operation','union') for s in stages]
        self.blends=[s.get('blend',.0015) for s in stages]
        self.groups=torch.nn.ModuleList([torch_model(s) for s in stages])
        for p in self.groups[0].parameters():p.requires_grad_(False)

    def forward(self,p):
        result=self.groups[0](p)
        for operation,k,group in zip(self.operations[1:],self.blends[1:],self.groups[1:]):
            d=group(p)
            if operation=='subtract':result=k*torch.logaddexp(result/k,-d/k)
            else:result=-k*torch.logaddexp(-result/k,-d/k)
        return result

    def export(self):
        return dict(field='Ordered smooth CSG with approximate ellipsoid distances',
            coordinates='Z up, front -Y; target bbox height is 1',
            stages=[dict(group.export(),operation=op,blend=k)
                    for group,op,k in zip(self.groups,self.operations,self.blends)])
