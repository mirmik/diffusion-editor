# Isolated LaMa worker

LaMa runs outside the CPython 3.14t UI process. The editor starts a regular
CPython 3.11 subprocess with `subprocess.Popen`; it never forks the
multithreaded Termin process.

## Install

Create the separately locked worker environment:

```bash
./setup-lama-worker.sh
```

On Windows use `./setup-lama-worker.ps1`.

The installer automatically finds a regular CPython 3.11 from `PATH` or an
installed `pyenv` 3.11 version. It does not change the project's local or
global `pyenv` selection. The destination is `.venv-lama`. Override discovery
when needed:

```bash
LAMA_BOOTSTRAP_PYTHON=/path/to/python3.11 \
LAMA_VENV=/path/to/lama-venv \
./setup-lama-worker.sh
```

At runtime the editor uses `.venv-lama/bin/python` (or
`.venv-lama/Scripts/python.exe` on Windows). Set
`DIFFUSION_EDITOR_LAMA_PYTHON` to select another compatible environment.

`requirements-lama-worker.in` contains the reviewed direct requirements.
`requirements-lama-worker.txt` is the complete CPython 3.11 Linux lock,
including CPU-only PyTorch. The setup scripts use PyPI plus the official
PyTorch CPU wheel index. Native packages remain binary-only; the sole source
exception is the pure-Python `fire==0.5.0` archive required by the legacy
package. Its build frontend is pinned separately in
`requirements-lama-worker-build.txt` and build isolation is disabled, so this
exception does not resolve unpinned build dependencies. Regenerate the runtime
lock with:

```bash
UV_CACHE_DIR=/tmp/diffusion-editor-uv-cache \
uv pip compile --python-version 3.11 --python-platform x86_64-manylinux_2_28 \
  --index-strategy unsafe-best-match \
  requirements-lama-worker.in -o requirements-lama-worker.txt
```

## Protocol and lifecycle

Protocol version 1 uses newline-delimited JSON over bounded stdin/stdout pipes.
Images cross the boundary only as PNG files in a per-request temporary
directory. No editor, Termin, SDL, OpenGL, or Vulkan object is serialized.

The subprocess is started lazily and retains the model cache between requests.
It reports its Python ABI and GIL state before accepting work. A free-threaded
worker is accepted only when started with its GIL enabled; the selected lock
uses regular CPython 3.11.

Cancellation, timeout, worker exit, a malformed response, or a protocol
mismatch terminates the worker. The current operation receives a bounded error
and the next request starts a fresh process. Editor shutdown closes stdin,
waits briefly, then terminates or kills a worker that does not exit.

Run the end-to-end smoke (the first real run downloads the upstream model):

```bash
./venv/bin/python scripts/smoke_lama_worker.py
```
