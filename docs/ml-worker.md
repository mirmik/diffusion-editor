# Diffusers/Transformers worker

The Termin UI process never imports `torch`, `torchvision`, `diffusers`,
`transformers`, `accelerate`, `safetensors`, or `tokenizers`. Diffusion,
InstructPix2Pix, Grounding DINO, and SAM 2.1 run in a persistent regular
CPython 3.11 subprocess with the GIL enabled.

Requests and responses use versioned, bounded JSON-lines messages. Images and
masks cross the boundary as PNG files in a per-request temporary directory;
detections use one bounded JSON manifest plus optional mask PNGs. Cancelling,
timing out, crashing, or receiving malformed output terminates the suspect
worker. The next request starts a clean process.

## CPU installation

Linux:

```sh
./setup-ml-worker.sh
```

On Linux the installer automatically finds a regular CPython 3.11 from
`PATH` or `pyenv` without changing the selected project interpreter.

Windows:

```powershell
./setup-ml-worker.ps1
```

The default `requirements-ml-worker.txt` is an exact CPython 3.11 CPU lock
resolved with binary wheels only from PyPI and the official PyTorch CPU index.
Both installers run `pip check` and import every worker-only native package.

Override `DIFFUSION_EDITOR_ML_PYTHON` to select an already reviewed worker
interpreter.

## CUDA and ROCm manual gate

PyTorch publishes accelerator wheels in platform/toolkit-specific indexes.
There is deliberately no guessed fallback. Select the official index URL and
matching exact versions from the
[PyTorch installation selector](https://pytorch.org/get-started/locally/),
then run:

```sh
ML_ACCELERATOR=cuda \
ML_TORCH_INDEX_URL=https://download.pytorch.org/whl/<reviewed-cuda-index> \
ML_TORCH_VERSION=<exact-version-with-suffix> \
ML_TORCHVISION_VERSION=<matching-exact-version-with-suffix> \
./setup-ml-worker.sh
```

Use `ML_ACCELERATOR=rocm` with the reviewed ROCm index for AMD. The installer
pins the accelerator-neutral graph from `requirements-ml-worker-core.txt`,
installs only wheels, runs `pip check`, and prints the selected runtime
versions. Record those values with the machine's driver/toolkit versions before
calling that combination supported.

## Verification

The fast IPC and lifecycle smoke uses a deterministic backend and downloads no
models:

```sh
./venv/bin/python scripts/smoke_ml_worker.py
```

Real model smoke requires a local SDXL `.safetensors` checkpoint and may
download the InstructPix2Pix, Grounding DINO, and SAM model weights:

```sh
.venv-ml/bin/python scripts/smoke_ml_worker.py \
  --real --sdxl /path/to/model.safetensors
```
