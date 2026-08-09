# CPython 3.14t quality gates

`run-quality-gates.sh` is the canonical Linux/CI entrypoint. It rejects the
wrong interpreter, a regular `cp314` native ABI, a main-process GIL transition,
worker-only distributions in the UI environment, non-exact locks, unavailable
binary wheels, broken requirements, lifecycle races, and broken worker IPC.

`run-quality-gates.ps1` applies the same ABI/import/test runner on Windows.
The Windows CI job is enabled when a `windows_sdk_asset` workflow input or
`windows_asset` repository-dispatch payload names a published Windows x86-64
CPython 3.14t SDK archive.

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
  ./venv/bin/python scripts/smoke_native_root.py \
  --backend vulkan --frames 3

# Routine CI gate: public WindowManager + GuiWindowAdapter on OpenGL:
TERMIN_BACKEND=opengl SDL_VIDEODRIVER=offscreen \
TERMIN_SDK_SHADER_CACHE_ROOT=/tmp/diffusion-editor-native-window-cache \
  ./venv/bin/python scripts/smoke_native_root.py --windowed --frames 3

# Canonical routine OpenGL gate (native headless/startup/windowed root):
xvfb-run -a ./run-quality-gates.sh --skip-resolver \
  --render-backend opengl --frames 16

# Requires a Vulkan presentation-capable display (DRI3 on Linux):
./run-quality-gates.sh --skip-resolver \
  --render-backend vulkan --frames 16
```

The render children check `Py_GIL_DISABLED`, `cp314t` SOABI, and
`sys._is_gil_enabled()` before native imports, after native imports, and after
rendering. Shader/compiler errors and bounded subprocess timeouts fail the
gate with their transition name.

Both native scenarios import a deterministic, high-contrast PNG through the
application coordinator, route mask and image strokes, require non-empty
image/overlay textures, render several frames, and verify idempotent shutdown
plus released texture leases. The compositor gate compares the imported
source pixels with the final Canvas texture, so a correctly sized but blank
texture fails. On failure it reports signatures for the source upload, main
composition target and display target to localize the broken pass.

The hostless Vulkan path additionally reads back the resized framebuffer and
rejects a blank or non-finite result. The windowed OpenGL path requires the
Canvas to borrow the live GPU compositor texture and applies the source-pixel
gate on that live device. CI runs the OpenGL scenario on every push and pull
request. See
[`native-ui-checklist.md`](native-ui-checklist.md) for real desktop input and
visual verification, including the deterministic fake-generation path.

## Verified matrix (2026-07-27)

| Path | Evidence | Status |
| --- | --- | --- |
| Main UI CPython 3.14t | import/ABI/wheel gate; 365 tests | automated, passing |
| Windows main UI CPython 3.14t | `install-deps.ps1`, import/ABI/payload gate, `pip check`, full tests | CI definition ready; requires a published Windows SDK asset |
| Native headless root | `OffscreenGuiComposition`, snapshots, import/paint/mask, framebuffer readback, owned leases, resize/shutdown, bounded Vulkan render | automated, passing |
| Native windowed root | public `WindowManager` + borrowed `GuiWindowAdapter`, borrowed GPU Canvas texture, import/paint/mask/shutdown, offscreen SDL/OpenGL smoke | routine CI gate |
| OpenGL | 16-frame Xvfb native render and GPU compositor | passing with llvmpipe |
| Vulkan device | `vulkaninfo --summary` | passing with llvmpipe 1.4 |
| Vulkan presentation | 16-frame render command above | manual gate; current Xvfb has no DRI3/present queue |
| Windows OpenGL presentation | production native startup smoke through `run.ps1` | Windows CI/manual gate |
| Windows Vulkan presentation | production startup smoke on a Vulkan-capable Windows host | manual; backend support is separate from Python ABI support |
| LaMa CPU | identity lifecycle gate plus real LaMa smoke | passing |
| rembg CPU | threshold lifecycle gate plus real U2Net smoke | passing |
| Diffusers/Transformers CPU | fake lifecycle gate plus real SDXL, InstructPix2Pix, Grounding DINO and SAM smokes | passing |
| CUDA | auto-selected reviewed CUDA 12.8 wheels, `pip check`, GPU availability probe, then real ML smoke | installer-verified; real model smoke remains hardware-dependent |
| ROCm | `ML_ACCELERATOR=rocm` installer gate, `pip check`, then real ML smoke | manual; not claimed supported without recorded driver/toolkit evidence |

The current hostless Vulkan smoke can fall back to the CPU compositor when the
runtime device is not the application graphics device. It remains useful for
root lifecycle and framebuffer coverage, but is not evidence for the Vulkan
GPU compositor. Vulkan GPU composition therefore stays in the real-desktop
manual gate until a presentation-capable display is available.

For CUDA/ROCm, follow `docs/ml-worker.md`, record the exact PyTorch wheel
suffix and official index, driver/toolkit versions, `torch.cuda.is_available()`,
and the output of:

```sh
./venv/bin/python scripts/smoke_ml_worker.py \
  --real --sdxl /path/to/model.safetensors
```
