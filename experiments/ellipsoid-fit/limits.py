"""Refuse expensive experiments unless the cgroup has the approved RAM cap."""
from pathlib import Path


def require_budget():
    group=Path('/proc/self/cgroup').read_text().strip().split(':')[-1].lstrip('/')
    path=Path('/sys/fs/cgroup')/group
    maximum=(path/'memory.max').read_text().strip()
    swap=(path/'memory.swap.max').read_text().strip()
    if maximum=='max' or int(maximum)>12*1024**3 or swap!='0':
        raise RuntimeError('Use run_bounded.py: this experiment requires MemoryMax<=12 GiB and MemorySwapMax=0')
    print(f'Budget verified: RAM={int(maximum)//1024**2} MiB, swap=0',flush=True)
