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

The SDK must contain Termin wheels:

```text
$TERMIN_SDK/wheels/
```

## Install

```bash
./install-deps.sh
```

The script creates `./venv`, installs Termin packages from
`$TERMIN_SDK/wheels`, installs Python dependencies, and installs this project in
editable mode.

## Run

```bash
./run.sh
```

or:

```bash
./venv/bin/diffusion-editor
```

## Tests

```bash
./venv/bin/python -m pytest -q
```
