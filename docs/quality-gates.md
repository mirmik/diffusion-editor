# CPython 3.14t quality gates

`run-quality-gates.sh` is the canonical Linux/CI entrypoint. It rejects the
wrong interpreter, a regular `cp314` native ABI, a main-process GIL transition,
worker-only distributions in the UI environment, non-exact locks, unavailable
binary wheels, broken requirements, lifecycle races, and broken worker IPC.

```sh
./run-quality-gates.sh
```

The default run resolves every main/test pin with `--only-binary=:all:`, runs
the full test and stress suite, and exercises deterministic LaMa, segmentation,
and Diffusers/Transformers subprocess backends. `--skip-resolver` is available
for an offline repetition after the networked gate has already passed.

Graphics gates are explicit because CI hosts differ:

```sh
# Hostless termin-gui-native composition:
TERMIN_SDK_SHADER_CACHE_ROOT=/tmp/diffusion-editor-native-cache \
  ./venv/bin/python scripts/smoke_native_root.py --frames 3

# Public WindowManager + GuiWindowAdapter composition:
TERMIN_BACKEND=opengl SDL_VIDEODRIVER=offscreen \
TERMIN_SDK_SHADER_CACHE_ROOT=/tmp/diffusion-editor-native-window-cache \
  ./venv/bin/python scripts/smoke_native_root.py --windowed --frames 3

# Legacy production entrypoint:
xvfb-run -a ./run-quality-gates.sh --skip-resolver \
  --render-backend opengl --frames 16

# Requires a Vulkan presentation-capable display (DRI3 on Linux):
./run-quality-gates.sh --skip-resolver \
  --render-backend vulkan --frames 16
```

The render child checks `Py_GIL_DISABLED`, `cp314t` SOABI, and
`sys._is_gil_enabled()` before native imports, after native imports, and after
rendering. Shader/compiler errors and bounded subprocess timeouts fail the
gate with their transition name.

## Verified matrix (2026-07-24)

| Path | Evidence | Status |
| --- | --- | --- |
| Main UI CPython 3.14t | import/ABI/wheel gate; 300 tests | automated, passing |
| Native headless root | `OffscreenGuiComposition`, native Canvas, owned texture leases, input/resize, bounded Vulkan render | automated, passing |
| Native windowed root | public `WindowManager` + borrowed `GuiWindowAdapter`, borrowed GPU Canvas texture, offscreen SDL/OpenGL smoke | migration gate |
| OpenGL | 16-frame Xvfb render, GPU compositor + tcgui | passing with llvmpipe |
| Vulkan device | `vulkaninfo --summary` | passing with llvmpipe 1.4 |
| Vulkan presentation | 16-frame render command above | manual gate; current Xvfb has no DRI3/present queue |
| LaMa CPU | identity lifecycle gate plus real LaMa smoke | passing |
| rembg CPU | threshold lifecycle gate plus real U2Net smoke | passing |
| Diffusers/Transformers CPU | fake lifecycle gate plus real SDXL, InstructPix2Pix, Grounding DINO and SAM smokes | passing |
| CUDA | auto-selected reviewed CUDA 12.8 wheels, `pip check`, GPU availability probe, then real ML smoke | installer-verified; real model smoke remains hardware-dependent |
| ROCm | `ML_ACCELERATOR=rocm` installer gate, `pip check`, then real ML smoke | manual; not claimed supported without recorded driver/toolkit evidence |

For CUDA/ROCm, follow `docs/ml-worker.md`, record the exact PyTorch wheel
suffix and official index, driver/toolkit versions, `torch.cuda.is_available()`,
and the output of:

```sh
./venv/bin/python scripts/smoke_ml_worker.py \
  --real --sdxl /path/to/model.safetensors
```

## SDK wheelhouse recovery overlay

`DIFFUSION_EDITOR_QA_SITE_PACKAGES` exists only to test a freshly rebuilt SDK
payload while a known wheelhouse/manifest defect prevents refreshing the local
venv. Normal CI and production must not set it: `run-quality-gates.sh` first
runs `sdk_runtime verify-installed`, so stale installed payloads remain a hard
failure. Remove the overlay after rebuilding the Termin SDK and rerunning
`install-deps.sh`.
