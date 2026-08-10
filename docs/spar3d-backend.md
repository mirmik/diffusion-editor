# SPAR3D reconstruction backend

SPAR3D is an optional image-to-3D backend built around an editable colored
point-cloud condition. The editor-owned protocol exposes the actual upstream
pipeline rather than mapping it onto Pixal3D/TRELLIS.2 terminology:

```text
Source image -> Point cloud (.ply, XYZ + RGB) -> Final PBR mesh (.glb)
```

The official implementation is
[`Stability-AI/stable-point-aware-3d`](https://github.com/Stability-AI/stable-point-aware-3d).
Its weights are gated. Accept the model terms and download the files before
selecting SPAR3D in the editor.
The repository uses the Stability AI Community License rather than a
permissive OSS license; commercial use has its registration and revenue terms.

The default isolated runtime layout is:

```text
root:   /home/mirmik/soft/SPAR3D
python: /home/mirmik/soft/SPAR3D/venv/bin/python
model:  /home/mirmik/soft/SPAR3D/models/stable-point-aware-3d
```

It can be overridden per launch with:

```text
DIFFUSION_EDITOR_SPAR3D_ROOT
DIFFUSION_EDITOR_SPAR3D_PYTHON
DIFFUSION_EDITOR_SPAR3D_MODEL
```

The runtime must contain SPAR3D and its compiled `texture_baker` and
`uv_unwrapper` dependencies. Remeshing dependencies are optional because the
editor runner initially requests the upstream `none` remesh mode.

The tested RTX 5090 runtime uses Python 3.10, PyTorch 2.10.0/torchvision 0.25.0
from the `cu130` index, and `flet==0.23.2`. The Flet pin matters:
`transparent-background==1.3.3` declares an open-ended dependency, but newer
Flet releases removed `FilePickerResultEvent` and make SPAR3D fail during
import. When changing PyTorch, remove the `texture_baker/build` and
`uv_unwrapper/build` directories before reinstalling those extensions; stale
objects retain the previous PyTorch C++ ABI.

## Parameters and artifacts

`Seed`, `Point guidance`, `Texture size`, and `Low VRAM` affect SPAR3D.
Pixal3D-specific conditioning/FOV controls are disabled or ignored. The point
stage is exported as a colored PLY with `preview_kind=points`; the final stage
is exported as a textured GLB with `preview_kind=mesh`. `points.ply` includes
the SPAR3D output and editor Z-up transforms so stage switching remains aligned;
`point-conditioning.ply` retains SPAR3D's raw model coordinates for a future
edit-and-rerun path.

Generation can stop after `Point cloud`. Feeding an edited PLY back into the
mesh stage is supported by upstream SPAR3D and is the intended next editor
operation, but editing and feeding the PLY back is not exposed yet.

## Current boundary

The backend boundary, stage selection, subprocess protocol, progress events,
and artifacts are editor-owned and tested without importing SPAR3D into the
editor process. The native viewport reads ASCII or little-endian binary PLY and
uploads its XYZ/RGB data through Termin's persistent point-cloud renderer. A
real GPU smoke test still requires the gated weights and local SPAR3D runtime.
