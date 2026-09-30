# Raster transform: first working stage

The first stage of the interface proposal in Kanboard #2740 is implemented by
#2742. The sketch is `output/editor-transform-sketch.html`; the full tool-panel
redesign, rotation, perspective, transforming the selection boundary and keeping
a fragment as a separate layer remain follow-up work.

## Using the tool

1. Select a visible raster layer. Optionally draw a selection.
2. Click **Transform selection / layer** at the top of the left controls.
   With a selection the target is **Selected pixels**; otherwise it is
   **Entire layer**. A selection that does not intersect the active layer reports
   an error instead of silently moving the whole layer.
3. Drag inside the frame to move, or use its eight handles to resize. The options
   above the canvas expose width/height in pixels and **Keep ratio**. With canvas
   keyboard focus, arrow keys nudge by one image pixel.
4. **Apply** / Enter writes the result into the source layer. **Cancel** / Esc
   discards the entire session. Mouse release ends only the current gesture;
   several drags and numeric adjustments still produce one history entry.

Changing the target in the options restarts the preview from the original data.
During preview the brush and selection controls are disabled. Document mutations
(including undo, layer switching and generation results) cancel the preview.
Losing pointer capture rolls back only the current drag. No preview pixels are
written into saved documents, exports, recovery snapshots or history.

The source layer expands if the fragment is placed outside its previous bounds;
the document canvas retains its size. Transformed selection coverage is clipped
to the document canvas, but raster pixels outside it are retained. The layer's
name, opacity, identity, child layers and attached processing tool are preserved.
An entire-layer transform affects that layer's own pixels, not descendants.
Whole-layer scaling also scales its processing mask and patch rectangle. A
fragment transform leaves the processing area at the original canvas location.

## Pixel and history contract

- Every preview samples the session's original pixels. Resizing down and then
  back up within one session does not compound resampling losses.
- Resampling and source-over blending use premultiplied linear RGB; document
  buffers remain straight sRGB RGBA8. A soft cut splits source alpha by selection
  coverage. Unchanged geometry is an exact no-op, including soft edges.
- Preview uses a separate canonical CPU renderer with a replacement raster tile
  source. Layer order, parent compositing and opacity remain canonical; only
  affected preview rectangles are recomposited and uploaded during dragging.
  It samples the session directly, without constructing full layer/selection/mask
  results on mouse movement. Resampled pixels are cached by size; translations
  update the previous and new fragment locations, not the path back to the cut.
  Opaque top-layer regions use direct pixel copies. The full raster is assembled
  only on Apply.
- Apply switches between before/after raster storage in a single history delta.
  Undo restores the original array objects so earlier pixel, selection and mask
  deltas continue to address the correct storage.
- Working raster dimensions and expanded bounds are limited to 16384 pixels per
  axis and 32 MP. Oversized edits report an error and retain the last valid frame.

## Verification

Run with the local environment:

```sh
./venv/bin/python -m pytest tests/test_raster_transform.py tests/test_canvas_transform.py tests/test_native_transform_controls.py tests/test_native_editor_canvas.py -q
```

Native tests exercise actual controls, captured pointer input, numeric dimensions,
target switching, Enter/Esc, undo, and layout above the canvas. Manually repeat
the selection → move → resize → apply → undo scenario in `./run.sh` on a desktop.
Also check a shifted layer, a soft selection and an unselected layer. Transparent
holes currently appear against the native canvas background; adding a checkerboard
is tracked separately in #2744.

## Transform performance regression (#2746)

`./venv/bin/python scripts/benchmark_transform.py` measures CPU preview work,
excluding GUI frame rendering and texture upload. On the development machine,
eight-step median times for a translated opaque selection were:

| Canvas | Fragment | Before | After |
| --- | --- | --- | --- |
| 2048 × 2048 | 256 × 256 | 33.71 ms | 0.08 ms |
| 4096 × 4096 | 512 × 512 | 132.67 ms | 0.28 ms |
| 4096 × 4096 | 1024 × 1024 | 272.55 ms | 1.57 ms |

Use `--multilayer` for an opaque top layer above a background, or `--soft` for
uniform 50% selection coverage. The latter requires per-pixel linear blending:
the 1024² case still measured about 62 ms and remains a separate optimization
target. Regression tests assert that dragging never calls the full result builder,
translations reuse resampling, unchanged coordinates skip rendering, and distant
movement uploads only the affected rectangles. Random-alpha tests compare tiled
preview against the committed raster, including layers, offsets and clipping.
