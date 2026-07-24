# diffusion-editor

Standalone diffusion image editor built on Termin `tcgui`, `tgfx`, and
`termin-display`.

## Prerequisites

Install or build the Termin SDK first. The default lookup path is:

```text
/opt/termin
```

For development builds, point `TERMIN_SDK` at a local SDK:

```bash
TERMIN_SDK=/path/to/termin/sdk ./install-deps.sh
```

The SDK must contain the schema-v3 runtime manifest, its canonical
free-threaded interpreter, and Termin wheels:

```text
$TERMIN_SDK/python-runtime-manifest.json
$TERMIN_SDK/bin/termin_python
$TERMIN_SDK/wheels/
```

`diffusion-editor` now requires the exact CPython 3.14t ABI advertised by the
SDK (`free_threaded=true`, `py_gil_disabled=true`, and a `cp314t` SOABI).
Regular CPython 3.14 (`cp314`) is intentionally rejected.

## Install

```bash
./install-deps.sh
```

The script creates `./venv` with `$TERMIN_SDK/bin/termin_python`, verifies that
the wheelhouse has one native build ID and only compatible `cp314t` native
wheels, installs the exact Termin dependency closure, and installs this project
in editable mode. After successful import verification it saves the absolute
SDK path in the ignored `.termin-sdk` file.

If `VENV` already exists, its interpreter is checked before anything is
installed. An old `cp310`/`cp314` environment, or a 3.14t process started with
the GIL enabled, fails with an instruction to move the environment aside or
choose another `VENV` path. The installer never silently replaces it.

Third-party dependencies are exact-pinned and installed from wheels only.
Native ML packages without CPython 3.14t wheels are excluded from the UI
process and will be hosted by isolated workers. The current compatibility
matrix and ownership boundary are documented in
[CPython 3.14t dependency contract](docs/cpython-314t-dependencies.md).
LaMa has a separate regular-CPython environment; install and lifecycle details
are in [the isolated LaMa worker guide](docs/lama-worker.md).
Background segmentation uses another isolated environment documented in
[the rembg worker guide](docs/segmentation-worker.md).
Diffusion, InstructPix2Pix, Grounding DINO, and SAM use the regular-CPython
runtime described in [the Diffusers/Transformers worker guide](docs/ml-worker.md).

Termin wheels are force-refreshed from that SDK even when their version string
has not changed. This matters for pure-Python packages such as `tcgui`: an SDK
rebuild can otherwise leave old Python code in the venv beside new native
libraries.

If the SDK runtime has been rebuilt without rebuilding `wheels/`, installation
stops with the expected and available build IDs. Rebuild or download a complete
SDK instead of mixing the artifacts.

## Run

```bash
./run.sh
```

`run.sh` restores `TERMIN_SDK` from `.termin-sdk` and verifies installed Termin
package versions and wheel payload bytes before importing native modules. If
the venv contains stale SDK files, rerun `./install-deps.sh`. An explicit
`TERMIN_SDK` overrides the saved path and is checked by the same gate.

or:

```bash
./venv/bin/diffusion-editor
```

## Tests

```bash
./venv/bin/python -m pytest -q
```

The CI runtime gates can also be run directly:

```bash
python3 -m diffusion_editor.sdk_runtime python-executable
python3 -m diffusion_editor.sdk_runtime verify-python-executable \
  --python ./venv/bin/python
./venv/bin/python -m diffusion_editor.sdk_runtime verify-installed --imports
TERMIN_BACKEND=opengl ./venv/bin/python scripts/smoke_termin_runtime.py
# Optional real-project and long-frame regression run:
TERMIN_BACKEND=vulkan ./venv/bin/python scripts/smoke_termin_runtime.py \
  --frames 1200 --project /path/to/project.deproj
```
