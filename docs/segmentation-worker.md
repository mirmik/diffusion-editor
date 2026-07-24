# Isolated rembg segmentation worker

Background segmentation runs outside the CPython 3.14t UI process. The editor
starts a regular CPython 3.11 subprocess with `subprocess.Popen`; it never
forks the multithreaded Termin process.

## Install

```bash
./setup-segmentation-worker.sh
```

On Windows use `./setup-segmentation-worker.ps1`. On Linux the installer
automatically finds a regular CPython 3.11 from `PATH` or `pyenv` without
changing the selected project interpreter. The default environment is
`.venv-segmentation`. Override the bootstrap interpreter or destination with
`SEGMENTATION_BOOTSTRAP_PYTHON` and `SEGMENTATION_VENV`.

At runtime, `DIFFUSION_EDITOR_SEGMENTATION_PYTHON` can select another
compatible environment. The worker starts with the GIL enabled and reports its
Python ABI before accepting work.

`requirements-segmentation-worker.in` contains the reviewed direct packages.
`requirements-segmentation-worker.txt` is the complete regular-CPython 3.11
Linux lock. Installation is binary-only: rembg, ONNX Runtime, OpenCV and their
native dependencies must all resolve as wheels.

Regenerate the lock with:

```bash
UV_CACHE_DIR=/tmp/diffusion-editor-uv-cache \
uv pip compile --python-version 3.11 --python-platform x86_64-manylinux_2_28 \
  --no-build requirements-segmentation-worker.in \
  -o requirements-segmentation-worker.txt
```

## Protocol and lifecycle

Protocol version 1 uses bounded newline-delimited JSON over stdin/stdout.
RGB input and grayscale mask output cross the boundary as PNG files in a
per-request temporary directory. Progress and terminal responses carry a
request ID; no ndarray, editor, Termin, SDL, OpenGL, or Vulkan object is
serialized.

The worker lazily creates and retains its rembg ONNX session. Cancellation,
timeout, worker exit, malformed output, and protocol mismatch terminate the
worker and produce a bounded engine error. The next request starts a fresh
process. Shutdown closes stdin, waits briefly, then terminates or kills a
worker that does not exit.

Run the end-to-end smoke (the first real run downloads the U²-Net model):

```bash
./venv/bin/python scripts/smoke_segmentation_worker.py
```
