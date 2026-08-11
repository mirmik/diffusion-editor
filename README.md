# diffusion-editor

Standalone diffusion image editor built on Termin `termin-gui-native`, `tgfx`,
and `termin-display`.

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
has not changed. This prevents an SDK rebuild from leaving old Python payloads
in the venv beside new native libraries.

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

The native UI is the sole application host. An optional project or image can
be passed directly:

```bash
./run.sh [project.deproj-or-image]

# Equivalent installed entrypoint:
./venv/bin/diffusion-editor [project.deproj-or-image]
```

The production host uses real settings and inference engines; it is distinct
from the deterministic fake-engine harness in `scripts/smoke_native_root.py`.

The image-to-3D prototype represents each result as a non-raster
**3D Reconstruction** object in the layer tree. Create one with the toolbar
button or **Layer → New 3D Reconstruction**, then select it and use
**Generate 3D Model** in its contextual left panel, which replaces the usual
brush and generation controls. Selecting a raster layer restores those controls
and the full-width Canvas; selecting a reconstruction restores its left panel
and Canvas/3D workspace. The reconstruction panel can use Pixal3D, TRELLIS.2,
the optional SPAR3D point-aware backend, geometry-only Hi3DGen, or the
geometry-and-PBR Hunyuan3D 2.1 and SAM 3D Objects backends. It reports
staged sampling progress,
lets a ready stage be previewed, and lets a pending stage be selected as the
next generation target. Coordinate stages are
published as lightweight GLB geometry previews; shape and final stages publish
mesh previews. Per-reconstruction generation controls cover seed, sampling
steps, HR resolution, automatic or manual camera FOV, final mesh face budget,
texture size, low-VRAM execution, and an experimental 512/1024 LR conditioning
resolution selector. Only controls consumed by the selected backend are shown;
version selection remains common, while masked-refinement controls appear only
for compatible Pixal3D runs. The 1024 option uses the trained 1024 shape
conditioner
while retaining the LR coordinate grid. Parameter values are saved with the
layer and snapshotted when a job starts. The current vertical slice captures
the full visible composite;
project-owned artifact persistence, checkpoint resume, and the below-object
source boundary remain follow-up work.

The backend also has an experimental, UI-neutral masked-refinement path. Base
runs leave a session-local HR shape checkpoint; a same-sized preprocessed
conditioning image and soft mask can produce a geometry-only child run without
overwriting its parent. See
[Pixal3D masked refinement backend](docs/pixal3d-masked-refinement.md).

A parallel experimental reconstruction workspace is available for Pixal3D.
Its read-through adapter projects current legacy stage progress, completed run
variants, checkpoints and preview artifacts into the operation graph without
writing back to the legacy run. Existing mesh and point artifacts can be
previewed, and connected base operations can invoke the shared legacy worker
through the selected milestone. Pixal3D writes an atomic, pickle-free session
checkpoint after sparse occupancy, LR latent, HR coordinates, HR latent and
texture latent; running a later operation resumes from that state while the
source and upstream generation parameters remain compatible. Changing the
source or an upstream parameter deliberately starts a fresh prefix.
The operation inspector exposes only parameters used by the selected Pixal3D
operation. Sparse, LR shape, HR shape and texture sampling have independent
seed and step overrides; until edited they inherit the Legacy values. A
downstream override may change after resume without invalidating an accepted
upstream checkpoint, while changing a completed phase invalidates that prefix.
The project-owned Pixal3D runner normally stays alive between base and refine
operations. It loads the pipeline once and caches source preprocessing plus
automatic camera estimation by source content. Cancellation, protocol/runtime
errors, a low-VRAM identity change, or editor shutdown terminates the worker;
the next request then starts a clean runtime. Custom runner paths retain the
one-shot subprocess fallback unless persistence is explicitly requested.
Masked LR refine is connected to the experimental workspace: it consumes an
explicitly selected LR session checkpoint, publishes a decoded LR mesh preview
and writes a new `lr_shape_latent` checkpoint that ordinary HR resume can
consume. Base LR and every Refined LR result remain separate session-local
variants. `Generate LR shape` always exposes Base LR; `Refine LR shape` exposes
the refined variants and a separate `Refine source` selector, which defaults to
Base LR. Preview selection does not silently change the next refine source.
Local-detail operations (upscaled conditioning, isolated local geometry,
registration, fusion, local texture and transfer) remain disabled until their
resumable runners and artifact contracts are connected.
The existing reconstruction panel remains the default during migration.
See the accepted
[reconstruction artifact workspace decision](docs/architecture-council/2026-08-11-reconstruction-artifact-workspace.md).
TRELLIS.2 currently supports base staged generation and PBR GLB export only;
see [TRELLIS.2 backend](docs/trellis2-backend.md).
SPAR3D exposes its editable colored point-cloud stage before final PBR export;
see [SPAR3D backend](docs/spar3d-backend.md).
Hi3DGen conditions TRELLIS-style geometry generation on an inferred normal map;
see [Hi3DGen backend](docs/hi3dgen-backend.md).

For the local shape-latent and PBR pipeline, see
[Hunyuan3D 2.1 backend](docs/hunyuan3d21-backend.md).
SAM 3D Objects exposes its MoGe pointmap, sparse occupancy, structured latent,
Gaussian appearance and baked mesh; see
[SAM 3D Objects backend](docs/sam3d-objects-backend.md).

For live development automation, enable the local Editor MCP server in
**Edit → Settings** and restart the editor. Termin's existing MCP/CLI helper
can then execute Python on the running editor thread. `TERMIN_EDITOR_MCP` is
also available as a per-launch override; see
[Live editor MCP automation](docs/editor-mcp.md).

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
.\scripts\smoke_windows_runtime.ps1 -Backend opengl
```

The CI runtime gates can also be run directly:

```bash
./run-quality-gates.sh
python3 -m diffusion_editor.sdk_runtime python-executable
python3 -m diffusion_editor.sdk_runtime verify-python-executable \
  --python ./venv/bin/python
./venv/bin/python -m diffusion_editor.sdk_runtime verify-installed --imports
./venv/bin/python scripts/smoke_native_root.py --backend vulkan --frames 8
TERMIN_BACKEND=opengl SDL_VIDEODRIVER=offscreen \
  ./venv/bin/python scripts/smoke_native_root.py --windowed --frames 8
```

The complete automated/manual graphics and ML matrix is documented in
[CPython 3.14t quality gates](docs/quality-gates.md).
