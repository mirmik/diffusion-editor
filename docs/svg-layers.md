# SVG layers

SVG layers keep editable SVG source alongside ordinary raster layers. Import a
floor plan, annotate it with a separate raster layer, and update the vector
geometry through the editor's local MCP without flattening the plan.

## Interface

Create a document (or open an image), then use the **Layer** menu:

- **New SVG Layer** creates an empty vector viewport matching the canvas.
- **Import SVG Layer…** inserts a new layer; it does not replace the document.
- **Replace SVG Source…** updates the selected SVG layer, preserving its ID,
  viewport size, canvas offset, opacity, children and position in the tree.
- **Export SVG Source…** writes the vector source, not the composited image.
- **Rasterize SVG Layer** makes the selected layer paintable. Undo restores the
  SVG source and its children.

Visibility, solo, opacity, renaming and layer-tree ordering work as usual.
Choose **Move** in the left panel to drag the SVG layer as a whole. Paint,
erase, smudge and pixel masks are disabled until rasterization; draw annotations
on another raster layer. Selection tools remain available for the document.
Importing one SVG produces one layer; SVG groups remain within its source.
This release does not include direct mouse editing of individual vector nodes.

## MCP / Python API

Enable the editor's existing local MCP endpoint (see [editor-mcp.md](editor-mcp.md)).
`execute_python_script` already exposes `document`; use `document.svg`. Operations
execute on the editor thread and participate in the same Undo/Redo history as
UI edits. Passing a layer ID is recommended; omission uses the selected layer.

```python
layer_id = document.svg.import_file('/path/to/plan.svg')
print(document.svg.elements(layer_id))  # objects with IDs and their attributes
print(document.svg.source(layer_id))   # complete original SVG text

document.svg.update_element(
    'room-1', {'transform': 'translate(20, 30)', 'fill': '#567d91'},
    layer_id=layer_id,
)
document.svg.update_element('label-1', text='Control room', layer_id=layer_id)
# Attribute value None removes an attribute. SVG style attributes/stylesheets
# still follow ordinary CSS precedence over presentation attributes.

document.svg.update(new_svg_text, layer_id)  # atomically replace the source
# Explicit viewport resampling keeps vectors and source coordinates:
document.svg.update(document.svg.source(layer_id), layer_id, width=1600, height=1100)
document.svg.move(120, 40, layer_id)  # integer canvas pixels, undoable

document.svg.export_file('/path/to/edited-plan.svg', layer_id)
document.undo()
document.redo()
```

`document.svg.add(source, name, width=..., height=...)` creates a layer from text.
`document.svg.new(name)` creates an empty one. `replace_from_file(path, layer_id)`
and `rasterize(layer_id)` correspond to menu commands. SVG element IDs must be
unique within one layer. Coordinates inside the SVG remain in its `viewBox`;
layer offsets are separate canvas pixels and are saved in `.deproj`, not in an
export of the original SVG source.

## Document and rendering contract

- `SvgLayer` is a composited, non-paintable document node. Its SVG source is
  authoritative; its straight sRGB RGBA8 image is an immutable render cache.
- `resvg_py==0.5.0` renders paths, text, groups, transforms, gradients, clip paths
  and other supported static SVG features. It is pinned in both main-process
  dependency lists; the Linux x86_64 `cp314t` wheel runs with the GIL disabled.
- Rendering occurs on import/update/load, not every frame. The existing CPU and
  GPU compositors consume the cache; native GPU uploads adapt immutable buffers
  to the current writable-buffer binding contract.
- Source validation and rendering finish before a document mutation is
  published. Failed replacement leaves source, pixels and history unchanged.
  Invalid input is reported by the UI/MCP; renderer failures are logged.
- Format version **9** embeds source as `layers/<path>.svg` plus the ordinary
  cached preview. Loading regenerates the preview from source. Version 8 and
  older supported documents still load; old editor builds reject version 9.
- Source updates, viewport changes, insertion, movement and rasterization are
  undoable. SVG history resolves live layer IDs after snapshot restoration.
- Static, self-contained SVG only: no scripts, animation, foreignObject, DTDs,
  external files/URLs or CSS imports. Local `#id` resources and embedded
  PNG/JPEG/WebP images are supported. CSS escapes are rejected. Embed resources
  before import. Text rendering uses locally installed fonts.
- Limits: 4 MiB SVG source, 32768 pixels per side and 64 megapixels per viewport.
  The viewport cache uses document pixel resolution; zooming in does not
  automatically rerasterize it. Resize the viewport explicitly when needed.

## Verification

```bash
./venv/bin/python -m pytest -q tests/test_svg_layers.py tests/test_svg_ui.py tests/test_editor_mcp.py tests/test_gpu_compositor_runtime.py
./venv/bin/python scripts/probe_main_process_dependencies.py
./venv/bin/python scripts/smoke_svg_layers.py --gpu --output /tmp/svg-smoke
# Optionally pass --svg /path/to/plan.svg
```

The tests cover mixed composition, visibility/opacity/offsets, source editing,
invalid input atomicity, archive round trips, snapshot-aware history,
rasterization with children, native menus/Move/paint guards, and live MCP edits.
A separate Vulkan application-domain smoke checked the actual ColdRelay SVG
(79 identified objects), including an update and Undo: CPU/GPU pixel difference
was zero, with the GIL disabled.

The installed SDK is missing the editor's two custom compositor shader artifacts.
For the Vulkan smoke they were compiled into a temporary cache using the
OffscreenGuiComposition `shader_compiler_path`, `slang_compiler_path`,
`shader_cache_root`, `shader_artifact_root` and `enable_shader_dev_compile=True`
parameters. Shipping those artifacts is tracked separately in Kanboard **#2694**;
this is independent of SVG source rendering. The default headless UI composition
uses the existing CPU composition path, so passing that test alone is not proof
of GPU compositor coverage.
