# diffusion-editor

Standalone diffusion image editor built on Termin `tcgui`, `tgfx`, and
`termin-display`.

The editor stores document pixels as straight sRGB RGBA8 and composites RGB
in linear light with linear alpha. The boundary contract is documented in
[Color management](docs/color-management.md).

## Prerequisites

Install or build the Termin SDK first. The default lookup path is:

```text
/opt/termin
```

For development builds, point `TERMIN_SDK` at a local SDK:

```bash
TERMIN_SDK=/path/to/termin/sdk ./install-deps.sh
```

On Windows, use the matching Windows x86-64 SDK and PowerShell:

```powershell
.\install-deps.ps1 -Sdk C:\path\to\termin-sdk
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

```powershell
.\install-deps.ps1
```

The script creates `./venv` with `$TERMIN_SDK/bin/termin_python`, verifies that
the wheelhouse has one native build ID and only compatible `cp314t` native
wheels, installs the exact Termin dependency closure, and installs this project
in editable mode. After successful import verification it saves the absolute
SDK path in the ignored `.termin-sdk` file.

The PowerShell installer applies the same contract: it selects
`bin\termin_python.exe`, prepends only the selected SDK's `bin`/`lib`
directories for deterministic DLL discovery, rejects an existing regular
`cp314` environment, installs binary wheels only, and verifies imports with
the GIL disabled.

If `VENV` already exists, its interpreter is checked before anything is
installed. An old `cp310`/`cp314` environment, or a 3.14t process started with
the GIL enabled, fails with an instruction to move the environment aside or
choose another `VENV` path. The installer never silently replaces it.

Third-party dependencies are exact-pinned and installed from wheels only.
Native ML packages without CPython 3.14t wheels are excluded from the UI
process and will be hosted by isolated workers. The current compatibility
matrix and ownership boundary are documented in
[CPython 3.14t dependency contract](docs/cpython-314t-dependencies.md).
Install the single shared regular-CPython environment for LaMa, segmentation,
Diffusion, InstructPix2Pix, Grounding DINO, and SAM with:

```bash
./setup-workers.sh
```

Installation, accelerator selection, and feature-specific smoke tests are
documented in the [model-worker guide](docs/model-workers.md).

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

```powershell
.\run.ps1
```

`run.sh` restores `TERMIN_SDK` from `.termin-sdk` and verifies installed Termin
package versions and wheel payload bytes before importing native modules. If
the venv contains stale SDK files, rerun `./install-deps.sh`. An explicit
`TERMIN_SDK` overrides the saved path and is checked by the same gate.

During the comparison period the legacy UI remains the default. Select either
implementation explicitly with the same optional project/image argument:

```bash
./run.sh --ui legacy [project.deproj-or-image]
./run.sh --ui native [project.deproj-or-image]

# Equivalent installed entrypoint:
./venv/bin/diffusion-editor --ui native [project.deproj-or-image]
```

Windows keeps the same comparison selector:

```powershell
.\run.ps1 --ui legacy
.\run.ps1 --ui native
```

`--ui native` uses production settings and inference engines; it is distinct
from the deterministic fake-engine harness in `scripts/smoke_native_root.py`.

Dirty documents have crash-recovery snapshots under
`~/.cache/diffusion-editor/recovery`. The first snapshot is written one minute
after a change; further snapshots are written only for a newer document
revision and no more often than once every five minutes. Recovery uses the
same compressed, atomic `.deproj` writer as an explicit project save and has
no fixed project-size ceiling beyond available storage.

## Tests

```bash
./venv/bin/python -m pytest -q
```

```powershell
.\run-quality-gates.ps1 --skip-resolver
.\scripts\smoke_windows_runtime.ps1 -Ui legacy -Backend opengl
```

The CI runtime gates can also be run directly:

```bash
./run-quality-gates.sh
python3 -m diffusion_editor.sdk_runtime python-executable
python3 -m diffusion_editor.sdk_runtime verify-python-executable \
  --python ./venv/bin/python
./venv/bin/python -m diffusion_editor.sdk_runtime verify-installed --imports
TERMIN_BACKEND=opengl ./venv/bin/python scripts/smoke_termin_runtime.py
# Optional real-project and long-frame regression run:
TERMIN_BACKEND=vulkan ./venv/bin/python scripts/smoke_termin_runtime.py \
  --frames 1200 --project /path/to/project.deproj
```

The complete automated/manual graphics and ML matrix is documented in
[CPython 3.14t quality gates](docs/quality-gates.md).
