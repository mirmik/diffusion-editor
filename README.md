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

The SDK must contain its runtime manifest and Termin wheels:

```text
$TERMIN_SDK/python-runtime-manifest.json
$TERMIN_SDK/wheels/
```

## Install

```bash
./install-deps.sh
```

The script creates `./venv`, verifies that the wheelhouse has one native build
ID and that its bindings match the selected SDK payload, installs the exact
Termin dependency closure, and installs this project in editable mode. After
successful import verification it saves the absolute SDK path in the ignored
`.termin-sdk` file.

If the SDK runtime has been rebuilt without rebuilding `wheels/`, installation
stops with the expected and available build IDs. Rebuild or download a complete
SDK instead of mixing the artifacts.

## Run

```bash
./run.sh
```

`run.sh` restores `TERMIN_SDK` from `.termin-sdk` and verifies installed Termin
package versions before importing native modules. An explicit `TERMIN_SDK`
overrides the saved path and is checked by the same gate.

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
./venv/bin/python -m diffusion_editor.sdk_runtime verify-installed --imports
TERMIN_BACKEND=opengl ./venv/bin/python scripts/smoke_termin_runtime.py
```
