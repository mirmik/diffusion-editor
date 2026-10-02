"""Run one experiment command inside a verified, memory-limited user scope."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
lock=root/'output/.compute.lock';lock.parent.mkdir(exist_ok=True)
with lock.open('w') as handle:
    try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('Another bounded ellipsoid job is running')
    env=os.environ.copy();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    command=['systemd-run','--user','--scope','--quiet',f'--unit=ellipsoid-fit-{os.getpid()}',
        '-p','MemoryMax=12G','-p','MemoryHigh=10G','-p','MemorySwapMax=0',
        '-p','CPUQuota=800%',*sys.argv[1:]]
    if not sys.argv[1:]:raise SystemExit('Usage: run_bounded.py COMMAND [ARGS...]')
    raise SystemExit(subprocess.run(command,env=env).returncode)
