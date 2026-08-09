# Native UI desktop checklist

Use this checklist on a real desktop to verify the production native UI:

```sh
./run.sh [project.deproj-or-image]
```

For deterministic generation without model weights, use the native manual
harness:

```sh
TERMIN_BACKEND=opengl \
TERMIN_SDK_SHADER_CACHE_ROOT=/tmp/diffusion-editor-native-manual \
  ./venv/bin/python scripts/smoke_native_root.py \
  --windowed --interactive
```

Repeat with `TERMIN_BACKEND=vulkan` on a display with a Vulkan present queue.
The harness first imports a generated image, paints an image stroke and mask,
completes one fake Diffusion result, and validates resource ownership. It then
remains open with the Diffusion tool attached until the window is closed.

Record the backend, GPU/driver, desktop session, Termin SDK manifest version,
and any failed item.

## Shell, layout and text

- Open every top-level menu; verify commands highlight, disabled state is
  visible, menus dismiss on Escape and clicks do not leak into the Canvas.
- Drag the main, workspace and right splitters in both directions. Panels must
  remain usable and the Canvas must repaint throughout the drag.
- Focus prompt, negative-prompt, Agent Chat and Settings text fields. Verify
  typing, cursor movement, selection, Backspace/Delete, copy/paste and Tab
  traversal. Editor shortcuts must not consume text-input keystrokes.
- Resize, maximize, restore and minimize the window. The Canvas must remain
  non-blank and fitted after restore.

## Layers and dialogs

- Select a layer, toggle visibility and solo, change opacity, and use the
  context menu to rename it. Cancel one rename with Escape and accept another
  with Enter.
- Add, reorder and remove layers. Confirm drag/drop indicators and selection;
  cancel one delete confirmation before accepting another.
- Open Import, Open, Save As and Export dialogs. Navigate directories, edit a
  file name, cancel and reopen each dialog. Modal input must not activate the
  layer tree or Canvas behind it.
- Open Settings, change text/numeric/checkbox values, cancel, reopen, then
  accept. Open Grounding and verify validation plus cancel/reopen behavior.

## Canvas

- Zoom at the pointer with the wheel, pan with the middle button, then use Fit.
- Paint and erase image pixels with several sizes, hardness and flow values.
  Pick a color and verify the brush cursor/controls stay synchronized.
- Paint and erase the mask, toggle mask visibility, and verify the colored
  overlay updates without hiding the source image.
- Use rectangle and brush selection, erase part of the selection, toggle its
  visibility and clear it.
- Enable patch drawing, drag and clear a patch rectangle. Verify selection and
  patch previews survive zoom/pan and disappear when disabled.

## Fake generation and shutdown

- Detach the preconfigured Diffusion tool, then attach it again from the
  selected layer's context menu.
- Enter prompt text and press Run. The deterministic fake engine should move
  the panel through running to result, report `Regenerated`, and replace the
  generated region with a blue/yellow stripe pattern.
- While a modal dialog is open, verify generation controls behind it do not
  react. Close the dialog and run generation once more.
- Close the application using the menu and window close button in separate
  runs. There must be no crash, hang, stale-handle message, Vulkan validation
  error, leaked-texture warning or double-destroy warning.

The automated companion gates are:

```sh
./venv/bin/pytest -q
./venv/bin/python scripts/smoke_native_root.py --backend vulkan --frames 3
TERMIN_BACKEND=opengl SDL_VIDEODRIVER=offscreen \
  ./venv/bin/python scripts/smoke_native_root.py --windowed --frames 3
```
