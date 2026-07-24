# Shared model-worker environment

LaMa, rembg segmentation, and Diffusers/Transformers execute in three
independent subprocesses, but share one regular CPython 3.11 environment and
one reviewed dependency lock. The CPython 3.14t UI process never imports their
native ML packages.

## Install

Linux:

```sh
./setup-workers.sh
```

Windows:

```powershell
./setup-workers.ps1
```

The default destination is `.venv-workers`. On Linux the installer finds a
regular CPython 3.11 from `PATH` or an installed pyenv 3.11 branch without
changing the selected project interpreter. Override discovery or destination
with `WORKERS_BOOTSTRAP_PYTHON` and `WORKERS_VENV`.

The former feature-specific setup scripts remain compatibility wrappers around
the shared installer.

`requirements-workers.in` contains the reviewed direct CPU requirements.
`requirements-workers.txt` is the complete exact Linux lock. Regenerate it
with:

```sh
UV_CACHE_DIR=/tmp/diffusion-editor-uv-cache \
uv pip compile --python-version 3.11 --python-platform x86_64-manylinux_2_28 \
  --no-build --index-strategy unsafe-best-match \
  requirements-workers.in -o requirements-workers.txt
```

All packages install from wheels. The project owns the small Big-LaMa
TorchScript adapter directly, so the abandoned `simple-lama-inpainting`
package no longer forces obsolete NumPy, Pillow, OpenCV, and PyTorch versions.
`LAMA_MODEL` and `LAMA_MODEL_URL` retain their previous meanings.

At runtime all three clients default to `.venv-workers/bin/python` (or
`.venv-workers/Scripts/python.exe`). Their feature-specific
`DIFFUSION_EDITOR_*_PYTHON` overrides remain available for diagnostics.

## CUDA and ROCm

The accelerator-neutral graph is pinned in `requirements-workers-core.txt`.
Select an official PyTorch index and matching exact versions:

```sh
ML_ACCELERATOR=cuda \
ML_TORCH_INDEX_URL=https://download.pytorch.org/whl/<reviewed-index> \
ML_TORCH_VERSION=<exact-version-with-suffix> \
ML_TORCHVISION_VERSION=<matching-version-with-suffix> \
./setup-workers.sh
```

Use `ML_ACCELERATOR=rocm` for ROCm. The selected PyTorch build is shared by
the ML and LaMa subprocesses.

## Verification

The canonical quality gate exercises deterministic IPC backends without
downloading models:

```sh
./run-quality-gates.sh --skip-resolver
```

Real feature smokes are documented in
[LaMa](lama-worker.md), [segmentation](segmentation-worker.md), and
[Diffusers/Transformers](ml-worker.md).
