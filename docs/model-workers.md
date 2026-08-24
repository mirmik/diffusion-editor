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
`requirements-workers.txt` is the complete exact Linux lock. The official
Depth Anything 3 source and its minimal, exact dependency overlay are pinned
separately by `setup-workers.sh` and `requirements-workers-da3.txt`. Regenerate
the main lock
with:

```sh
UV_CACHE_DIR=/tmp/diffusion-editor-uv-cache \
uv pip compile --python-version 3.11 --python-platform x86_64-manylinux_2_28 \
  --no-build --index-strategy unsafe-best-match \
  requirements-workers.in -o requirements-workers.txt
```

The main lock installs from wheels. DA3's minimal overlay currently includes
the upstream `moviepy` 1.0.3 source distribution. The project owns the small Big-LaMa
TorchScript adapter directly, so the abandoned `simple-lama-inpainting`
package no longer forces obsolete NumPy, Pillow, OpenCV, and PyTorch versions.
`LAMA_MODEL` and `LAMA_MODEL_URL` retain their previous meanings.

At runtime all three clients default to `.venv-workers/bin/python` (or
`.venv-workers/Scripts/python.exe`). Their feature-specific
`DIFFUSION_EDITOR_*_PYTHON` overrides remain available for diagnostics.

## Pose-estimation worker

DWPose, MediaPipe and the silhouette diagnostic use a fourth isolated regular
CPython 3.11 environment. It is deliberately not merged into `.venv-workers`:
MediaPipe and RTMLib own an OpenCV distribution, while the shared Qwen/DA3
environment requires its reviewed headless OpenCV graph.

Install the pose worker on Linux with:

```sh
./setup-pose-worker.sh
```

It defaults to `.venv-pose`. Override the interpreter or destination with
`POSE_BOOTSTRAP_PYTHON`, `POSE_VENV`, or point the editor at an existing
environment with `DIFFUSION_EDITOR_POSE_PYTHON`. Model weights are downloaded
on first use into `~/.cache/diffusion-editor/pose-models`; set
`DIFFUSION_EDITOR_POSE_MODEL_DIR` to relocate that cache.

## Accelerator selection

`./setup-workers.sh` defaults to `ML_ACCELERATOR=auto`: it installs the
reviewed CUDA 12.8 PyTorch pair when the NVIDIA kernel driver is present, and
the CPU lock otherwise. To make the choice explicit:

```sh
ML_ACCELERATOR=cuda ./setup-workers.sh
ML_ACCELERATOR=cpu ./setup-workers.sh
```

The accelerator-neutral graph is pinned in `requirements-workers-core.txt`.
The reviewed CUDA defaults are PyTorch `2.10.0+cu128` and torchvision
`0.25.0+cu128`. `ML_TORCH_INDEX_URL`, `ML_TORCH_VERSION`, and
`ML_TORCHVISION_VERSION` remain available for an explicit reviewed override.
ROCm still requires those three values with `ML_ACCELERATOR=rocm`. The
selected PyTorch build is shared by the ML and LaMa subprocesses.

## Verification

The canonical quality gate exercises deterministic IPC backends without
downloading models:

```sh
./run-quality-gates.sh --skip-resolver
```

Real feature smokes are documented in
[LaMa](lama-worker.md), [segmentation](segmentation-worker.md), and
[Diffusers/Transformers](ml-worker.md).
