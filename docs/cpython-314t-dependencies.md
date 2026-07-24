# CPython 3.14t dependency contract

Status: 2026-07-24.

The editor UI process runs only on Termin's CPython 3.14t build and must keep
`sys._is_gil_enabled() == False`. `install-deps.sh` therefore installs the main
environment with `--only-binary=:all:`. A package without a compatible wheel is
not compiled implicitly and is not admitted to the main process.

## Main-process lock

`requirements-runtime.txt` is the installer's exact third-party lock.
`requirements-project.txt` adds the four direct Termin distributions and is
the source of package metadata. A contract test requires its third-party tail
to exactly equal the runtime lock. Termin package versions are resolved
separately from the selected SDK manifest because their local `+sdk...` build
ID changes with the native SDK.

The installer never asks PyPI to resolve Termin dependencies. This prevents
public packages with colliding names, notably `tmesh`, from entering the
environment before the SDK closure is installed.

Verified Linux x86_64 CPython 3.14t wheels:

| Distribution | Version | Result |
| --- | ---: | --- |
| NumPy | 2.5.1 | `cp314t`; imported with the GIL disabled |
| Pillow | 12.3.0 | `cp314t` |
| PyYAML | 6.0.3 | `cp314t`; imported through `tcgui` with the GIL disabled |
| PySDL2 | 0.9.17 | pure Python |

`requirements-test.txt` is the exact build/test layer. `requirements.txt`
includes both files for developer installs.

## ML packages not admitted to the UI process

The PyPI wheel audit found no CPython 3.14t binary wheel for:

- `safetensors==0.8.0` (published Linux wheel is `cp310-abi3`);
- `tokenizers==0.23.1`;
- `opencv-python==5.0.0.93` and `opencv-python-headless==5.0.0.93`.

Consequently the following stacks cannot currently satisfy the main-process
binary-only gate:

- Diffusers, because it requires safetensors;
- Transformers/Grounding DINO, because they require tokenizers and
  safetensors;
- the LaMa TorchScript adapter, because model execution requires PyTorch;
- rembg segmentation, because its runtime closure requires OpenCV.

Their imports remain lazy in the current application, but they are no longer
declared as unconditional main-process dependencies. The corresponding
features must run in isolated workers with a shared reviewed lock. Installing
an sdist or forcing a regular `abi3` wheel into the 3.14t environment is not an
accepted workaround.

Packages that do publish CPython 3.14t wheels (`torch==2.13.0`,
`torchvision==0.28.0`, and `onnxruntime==1.27.0`) still belong to the worker
lock when they participate in one of these stacks. This keeps one ownership
boundary for model execution and avoids a partially duplicated ML runtime.

Relevant package records:

- https://pypi.org/project/safetensors/0.8.0/
- https://pypi.org/project/diffusers/0.39.0/
- https://pypi.org/project/transformers/5.14.1/
- https://pypi.org/project/opencv-python/5.0.0.93/
- https://pypi.org/project/onnxruntime/1.27.0/
