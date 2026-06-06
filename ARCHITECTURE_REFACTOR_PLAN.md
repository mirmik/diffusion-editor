# Diffusion Editor Architecture Refactor Plan

## Goal

Bring the diffusion editor to a healthier long-term architecture where UI,
document state, canvas interaction, generation workflows, engines, and
serialization have explicit boundaries.

The goal is not cosmetic file reshuffling. Module moves should happen only
after responsibilities are separated enough that the new layout reflects real
architecture.

## Target Shape

- `EditorWindow` is a composition root and UI shell:
  - creates panels, engines, controllers, and services;
  - wires callbacks;
  - shows dialogs/message boxes;
  - updates status text;
  - delegates application workflows to controllers/services.
- Document mutations go through commands and `DocumentService`, so undo/redo and
  dirty-state behavior remain consistent.
- Canvas owns viewport, input dispatch, rendering, and interactive tool state,
  but not generation business logic.
- Generation workflows use typed request/result/event objects.
- Engines are backend adapters with narrow request APIs, not workflow
  coordinators.
- Dialogs and panels do not reach into private `EditorWindow` state.
- Pixel-level image operations live in pure helpers covered by focused tests.
- Legacy migrations are isolated and testable.

## Current Main Problems

1. `editor_window.py` is still too large and owns several workflows.

   Diffusion workflow has been extracted, but LaMa, InstructPix2Pix,
   segmentation, and Grounding still leave pending state and submit/poll logic in
   the window.

2. `editor_canvas.py` mixes too many responsibilities.

   It currently combines input handling, tool behavior, viewport coordinate
   mapping, repaint/invalidation, and GPU compositor coordination.

3. `grounding_dialog.py` has the weakest boundary.

   It mixes UI construction, model cache, model loading, inference, threading,
   status updates, and direct writes into editor private state.

4. Some UI paths still mutate document state directly.

   Layer rename currently assigns `layer.name` directly from `LayerPanel`,
   bypassing commands/history.

5. Engine APIs are inconsistent.

   `DiffusionEngine.submit(...)` has a long argument list. Other engines have
   their own ad hoc request/poll shapes.

6. Tool runtime state is cleaner, but serialization still belongs to the
   document boundary.

   Tool dict/zip IO has been moved out of `tool.py` into
   `document/tool_serialization.py`, next to archive serialization helpers.

7. The package is flat.

   Document modules now live under `diffusion_editor/document/`, and canvas
   interaction/rendering modules live under `diffusion_editor/canvas/`.
   Generation, engine, app, and UI modules still remain mostly flat.

## Phase 1: Close Document Model Violations

Add commands for remaining direct document mutations.

Tasks:

- Add `SetLayerNameCommand`.
- Move layer rename in `LayerPanel` to `DocumentService`/command execution.
- Audit direct UI/canvas writes to `Layer` fields.
- Convert state changes that should participate in undo/redo into commands.
- Add undo/redo tests for layer rename and any newly commandified actions.

Success criteria:

- User-visible document mutations do not bypass the command/history path.
- Rename can be undone/redone.
- Direct mutation remains only for transient UI/runtime state.

## Phase 2: Complete Generation Workflow Boundaries

Bring LaMa and InstructPix2Pix to the same workflow shape as Diffusion.

Tasks:

- Add `InstructGenerationController`.
- Add typed Instruct request/result/event objects.
- Move Instruct load/apply/poll/pending-layer state out of `EditorWindow`.
- Add `LamaGenerationController`.
- Add typed LaMa request/result/event objects.
- Move LaMa submit/poll/pending-layer state out of `EditorWindow`.
- Keep `EditorWindow` responsible only for panel sync, status display, and
  command execution.

Success criteria:

- `EditorWindow` no longer owns `_pending_lama_layer` or
  `_pending_instruct_layer`.
- Instruct/LaMa workflows can be unit-tested with fake engines.
- Result application still goes through document commands.

## Phase 3: Unify Engine APIs

Make engines backend adapters with typed requests.

Tasks:

- Add `DiffusionEngine.submit_request(DiffusionRequest)`.
- Replace the long `DiffusionEngine.submit(...)` call path after
  `submit_request` is stable. Done; the legacy public method was removed.
- Add request/result dataclasses for Instruct, LaMa, and segmentation.
- Standardize poll output into typed events instead of ad hoc tuples.
- Keep logging inside engines for failures and important lifecycle events.

Success criteria:

- Controllers call engines with a single request object.
- Engine poll result shapes are explicit and testable.
- Argument-order bugs become unlikely.

## Phase 4: Redesign Grounding

Split Grounding into UI, engine, controller, and DTOs.

Target modules:

- `grounding/types.py`: request params, box/mask result DTOs, events.
- `grounding_engine.py`: model cache, loading, DINO/SAM inference.
- `grounding/controller.py`: async lifecycle, pending result, event mapping.
- `grounding_dialog.py`: UI only.

Tasks:

- Remove direct writes to `editor._pending_grounding_result`.
- Move model loading/cache out of the dialog.
- Route results through controller events.
- Apply accepted selection/masks through document commands.
- Add tests for controller lifecycle and result mapping.

Success criteria:

- Dialog does not know editor internals.
- Grounding can be tested without constructing UI.
- Errors are logged and surfaced through typed events/status handling.

## Phase 5: Decompose Canvas Responsibilities

Turn `EditorCanvas` into a smaller canvas shell plus tool/controller helpers.

Possible boundaries:

- viewport and coordinate mapping;
- input dispatch;
- brush/paint operations;
- mask/selection operations;
- patch rectangle interaction;
- move tool behavior;
- GPU invalidation bridge.

Tasks:

- Identify cohesive method clusters in `editor_canvas.py`.
- Extract pure coordinate/viewport helpers first.
- Extract tool state machines second.
- Keep rendering and invalidation behavior stable during each step.
- Add focused tests around each extracted behavior.

Success criteria:

- Canvas remains responsible for canvas interaction, not editor workflows.
- Tool behavior is easier to test without full UI setup.
- GPU compositor invalidation paths remain explicit.

## Phase 6: Split Tool Runtime Model From Serialization

Prepare tool data for ControlNet and multi-reference inputs.

Current status:

- Runtime tool classes in `tool.py` no longer implement dict/zip
  serialization.
- Tool serialization/deserialization now lives in
  `document/tool_serialization.py`.
- Shared array/PIL zip helpers live in `document/archive_serialization.py`, so
  layer, selection, and tool serialization do not depend on each other for
  archive IO.
- `Layer` and `LayerStack` call explicit serialization helpers instead of
  reaching through runtime tool methods.
- Legacy `manual_patch_rect` is loaded as migration metadata and moved to
  `Layer.patch_rect`; it no longer becomes runtime tool state.

Tasks:

- Keep runtime tool classes/dataclasses focused on current editable state. Done
  for the existing tool classes.
- Move tool serialization/deserialization to `document/tool_serialization.py`.
  Done.
- Move legacy migration helpers out of normal runtime paths where possible.
  Done for legacy `manual_patch_rect`.
- Add explicit migration tests for old tool formats. Done for nested mask files,
  manual patch rects, and missing legacy layer ids.
- Avoid indefinite fallback compatibility during active development when a
  clean migration is better.

Remaining follow-up:

- Finish moving project/layer serialization into the document package.
- If ControlNet or richer reference inputs add substantial data shape, introduce
  typed serializable settings DTOs instead of expanding ad hoc dict builders.

Success criteria:

- Adding a new tool/reference setting does not require scattering migration code.
- Runtime tool code is not dominated by serialization concerns.
- Legacy support is documented and tested.

## Phase 7: Reorganize Packages After Boundaries Exist

Only move modules after earlier phases make ownership clear.

Current status:

- `diffusion_editor/document/` now owns the document model, document commands,
  history, layer stack, masks, tiles, tool runtime models, layer rendering, and
  document/archive/tool serialization.
- Application code and tests import document types through
  `diffusion_editor.document.*`; no top-level compatibility wrappers were kept
  for the moved document modules.
- `diffusion_editor/canvas/` now owns canvas interaction, brush settings, canvas
  tool state, overlays, CPU/GPU composition, and pixel stroke helpers.
- Application code and tests import canvas types through
  `diffusion_editor.canvas.*`; no top-level compatibility wrappers were kept
  for the moved canvas modules.
- `diffusion_editor/generation/` now owns generation DTOs, request building,
  source/reference resolution, generation controllers, patch extraction helpers,
  and result-to-command mapping.
- Application code and tests import generation types through
  `diffusion_editor.generation.*`; no top-level compatibility wrappers were kept
  for the moved generation modules.
- `diffusion_editor/engines/` now owns backend adapters for diffusion,
  InstructPix2Pix, LaMa, segmentation, and Grounding.
- Application code imports engines through `diffusion_editor.engines.*`; no
  top-level compatibility wrappers were kept for the moved engine modules.
- `diffusion_editor/ui/` now owns editor panels and dialogs.
- Application code and tests import panels/dialogs through
  `diffusion_editor.ui.*`; no top-level compatibility wrappers were kept for
  the moved UI modules.
- `diffusion_editor/app/` now owns process entry points, application settings,
  and the `EditorWindow` composition root.
- `diffusion_editor/agent/` now owns agent chat integration, threaded runner,
  and the editor tool registry exposed to the agent.
- `diffusion_editor/grounding/` now owns Grounding workflow DTOs and controller.

Possible final layout:

```text
diffusion_editor/
  app/
    editor_window.py
    main.py
    settings.py
  agent/
    chat_panel.py
    runner.py
    tools.py
  grounding/
    controller.py
    types.py
  document/
    layer.py
    layer_stack.py
    commands.py
    document_service.py
    history.py
    serialization.py
  canvas/
    editor_canvas.py
    canvas_tools.py
    brush.py
    gpu_compositor.py
  generation/
    types.py
    patch_image.py
    patch_resolver.py
    reference_resolver.py
    diffusion_request_builder.py
    diffusion_controller.py
    instruct_controller.py
    lama_controller.py
    segmentation_controller.py
    result_mapper.py
  engines/
    diffusion_engine.py
    instruct_engine.py
    lama_engine.py
    segmentation_engine.py
    grounding_engine.py
  ui/
    panels/
    dialogs/
```

Tasks:

- Move files in small batches. Done for document, canvas, generation, engine,
  UI, and app packages.
- Update imports mechanically. Done for document, canvas, generation, engine,
  UI, and app packages.
- Run full tests after each batch. Done locally for document, canvas,
  generation, engine, UI, and app packages; run central tests before committing
  each batch.
- Avoid combining package moves with behavior changes.

Success criteria:

- Package layout communicates subsystem ownership.
- Imports become more directional.
- No behavior changes are hidden inside file moves.

## Testing Strategy

- Pure helpers: fast unit tests.
- Commands/document: undo/redo and serialization roundtrip tests.
- Controllers: fake engines/builders and event lifecycle tests.
- Engines: smoke/unit tests around request dispatch and error logging where
  feasible.
- Canvas: focused interaction tests for extracted behaviors.
- UI dialogs: test pure candidate/formatting/selection helpers, not pixel UI.
- Full `./run-tests.sh` after behavior-affecting changes.

## Priority Order

1. Add `SetLayerNameCommand` and move rename to command/history.
2. Extract Instruct/LaMa generation controllers.
3. Split Grounding into engine/controller/dialog/types.
4. Add `submit_request(DiffusionRequest)` and unify engine APIs.
5. Decompose `EditorCanvas`.
6. Split tool runtime model from serialization/migration.
7. Reorganize packages once boundaries are real.

## Current Progress

- Phase 1 started.
- Added `SetLayerNameCommand`.
- Moved layer rename from direct `LayerPanel` mutation to the
  `EditorWindow`/`DocumentService` command path.
- Added `LayerStack.set_layer_name()` as the document-model mutation point used
  by the command and by non-editor fallback panel usage.
- Added undo/redo coverage for layer rename.
- Audited remaining direct writes matching document fields:
  - document mutations are now concentrated in `LayerStack`, commands,
    serialization/migration, and canvas snapshot callbacks;
  - visible direct writes in panels/windows are widget visibility, not layer
    visibility.
- Phase 2 implemented.
- Added `InstructGenerationController`.
- Added `LamaGenerationController`.
- Moved LaMa submit/poll/pending-layer state out of `EditorWindow`.
- Moved InstructPix2Pix load/apply/poll/pending-layer state out of
  `EditorWindow`.
- Added controller tests with fake engines for LaMa and Instruct lifecycle.
- Phase 3 implemented for current generation engines.
- Added typed `InstructRequest` and `LamaRequest` DTOs next to
  `DiffusionRequest`.
- Added `submit_request(...)` methods to Diffusion, InstructPix2Pix, LaMa, and
  segmentation engines.
- Updated Diffusion, InstructPix2Pix, and LaMa controllers to submit typed
  request objects instead of long argument lists.
- Added typed `SegmentationRequest` and `SegmentationResult` DTOs.
- Added `EnginePollEvent` and typed inference result DTOs for Diffusion,
  InstructPix2Pix, LaMa, and segmentation.
- Updated generation controllers to consume `poll_event()` instead of raw
  engine tuples.
- Added `SegmentationGenerationController` and moved segmentation pending-layer
  state out of `EditorWindow`.
- Removed legacy low-level `submit(...)` and tuple `poll()` methods from
  generation engines after all callers migrated to typed request/event APIs.
- Phase 4 implemented for the current Grounding workflow.
- Added `grounding/types.py` for model options, request params, detections,
  results, and engine events.
- Added `GroundingEngine`; DINO/SAM model caches, loading, inference, and the
  worker thread moved out of `grounding_dialog.py`.
- Added `GroundingController`; pending-layer state and request submission moved
  out of `EditorWindow`.
- `GroundingDialog` now only builds UI and emits typed `GroundingParams`.
- Grounding result application now goes through `map_grounding_result(...)` and
  document commands.
- Preserved DINO box fallback when SAM 2.1 segmentation fails, while surfacing
  the SAM error through status events and logs.
- Added controller and result-mapper tests for Grounding.
- Phase 5 started.
- Added `canvas_geometry.py` and moved canvas/layer rectangle clipping,
  coordinate conversion, visible layer rect calculation, and rect union helpers
  behind pure functions.
- Added focused geometry tests for clipped canvas/layer rectangles and offset
  layers.
- Added `canvas_rect_drag.py` for Selection/Patch rectangle drag state.
- `EditorCanvas` now delegates rectangle drag begin/move/finish/preview logic
  to `CanvasRectDrag` while preserving the existing Selection/Patch final rect
  semantics.
- Added focused rect drag tests.
- Added `soft_mask_stroke.py` for pure soft-brush dab/line operations on float
  masks.
- `EditorCanvas` now reuses the same tested soft-stroke implementation for
  layer masks, mask erase previews, and selection painting instead of carrying
  duplicated pixel math.
- Added focused soft mask stroke tests.
- Added `canvas_composite.py` with `CanvasCompositeBridge`.
- Moved CPU/GPU composite ownership, stale GPU readback, dirty layer refresh,
  move-transform refresh, and mask-erase preview composition out of
  `EditorCanvas`.
- `EditorCanvas` keeps compatibility properties for the remaining private
  callers/tests, but delegates composite updates and GPU compositor lifecycle to
  `CanvasCompositeBridge`.
- `CanvasStrokeTool` implementations now ask the canvas to refresh transformed
  layers instead of duplicating compositor invalidation logic.
- Added focused composite bridge tests for CPU refresh, move transforms, and
  GPU invalidation.
- Added `canvas_paint_stroke.py` with `PaintStrokeBuffer` and tested paint
  blending for live brush strokes.
- Removed paint stroke runtime fields from `EditorCanvas`; the canvas now asks
  the stroke buffer to apply dirty regions and only coordinates compositor
  refresh.
- Added `canvas_edit_session.py` with `CanvasEditSession` for transient mouse
  edit state, dirty accumulation, last pointer position, and edit callback
  metadata.
- Removed the scattered `_painting`/`_last_paint_pos`/edit metadata fields from
  `EditorCanvas` mouse handlers.
- Added `canvas_smudge.py` with `SmudgeStrokeBuffer` and brush-rect clipping
  helpers for smudge carry-buffer state, dab application, and line
  interpolation.
- Removed smudge carry-buffer ownership and smudge pixel math from
  `EditorCanvas`; the canvas now delegates smudge operations and refreshes the
  returned dirty region.
- Added `canvas_overlay.py` with `CanvasOverlayBridge` and tested overlay pixel
  composition for selection, layer masks, offset layers, region updates, and
  mask erase previews.
- Removed mask/selection overlay buffer ownership and full overlay composition
  logic from `EditorCanvas`; the canvas now delegates overlay rebuilds and
  incremental mask-preview updates.
- Added `canvas_mask_erase.py` with `MaskEraseStrokeBuffer` for mask erase
  preview mask ownership, dirty accumulation, preview region calculation, and
  final mask application.
- Removed direct `_mask_erase_stroke` array and `_mask_erase_dirty` ownership
  from `EditorCanvas`/`MaskEraserTool`; the tool now uses the runtime buffer
  API and clears it after applying the stroke.
- Fixed GPU render preview detection to test whether a mask erase stroke is
  active, not whether the runtime buffer object exists.
- Added `canvas_selection_paint.py` with `CanvasSelectionPainter` for selection
  brush settings, eraser state, dab operations, and line stroke operations.
- Removed selection brush size/hardness/flow/eraser fields and selection paint
  wrapper methods from `EditorCanvas`; mouse handlers now delegate selection
  pixel edits to `CanvasSelectionPainter`.
- Added `canvas_mask_paint.py` with `CanvasMaskPainter` for layer-mask brush
  settings, eraser state, dab operations, and line stroke operations.
- Removed layer-mask brush size/hardness/flow/eraser fields and
  `_dab_mask`/`_stroke_mask_line` wrapper methods from `EditorCanvas`;
  `MaskPaintTool` and `MaskEraserTool` now delegate mask pixel edits directly
  to `CanvasMaskPainter`.
- Moved paint stroke begin/apply/end lifecycle into `PaintTool`; removed
  `_begin_stroke`, `_end_stroke`, `_update_stroke_region`, and the obsolete
  `_brush_eraser` state from `EditorCanvas`.
- Moved smudge lifecycle calls into `SmudgeTool`; removed `_begin_smudge`,
  `_end_smudge`, `_smudge_dab`, and `_smudge_stroke_line` compatibility
  wrappers from `EditorCanvas`.
- Added `canvas_image_erase.py` for image-alpha erase dab/line operations and
  alpha-region erasing used by mask erase finalization.
- Removed image eraser pixel math, `_begin_mask_erase`, `_erase_layer_rect`,
  and the dead `_composite_rect_below` compatibility wrapper from
  `EditorCanvas`.
- Mask erase finalization now clips the affected layer-local rect before
  sampling the erase mask and refreshing the compositor, avoiding shape
  mismatches for layers partially outside the project canvas.
- Added `canvas_tool_context.py` with `CanvasToolContext`, an explicit runtime
  host API for brush-like canvas tools.
- `EditorCanvas` now calls active tools through `CanvasToolContext`; tools no
  longer receive the canvas object or reach into `EditorCanvas` private fields.
- Moved paint/smudge/mask overlay/mask erase preview/finalization/move
  invalidation coordination behind the tool context.
- Removed stale geometry/invalidation compatibility helpers from `EditorCanvas`
  after tool callers migrated to `CanvasToolContext`.
- Added focused coverage for mask erase finalization on a partially off-canvas
  layer.
- Added `SelectionPaintTool` and moved selection brush strokes into the same
  begin/move/end tool lifecycle used by image, mask, smudge, erase, and move
  tools.
- `EditorCanvas` now uses common `_begin_tool_edit`, `_move_tool_edit`, and
  `_finish_tool_edit` helpers for active stroke tools instead of carrying a
  separate selection-paint branch inside mouse move/up.
- Added `CanvasRectDragController` to coordinate Selection and Patch rectangle
  drags through one begin/move/finish path.
- `EditorCanvas` no longer owns separate selection/patch drag state or branches
  mouse-up handling by drag type; it dispatches the controller result to the
  relevant callback.
- Phase 5 final pass completed for current scope.
- Kept `CanvasToolContext` as a single facade for now. Splitting it into
  image/mask/selection/transform hosts is deferred until package moves or new
  tool families create clearer pressure; splitting now would mostly add wiring
  without reducing current tool complexity.
- Removed remaining direct `EditorWindow`/`LayerPanel` access to private
  `LayerStack` internals for layer offset dirty rects, root layer lists, and
  renderer cache memory.
- Added `LayerStack.cache_memory_bytes()` as the public cache-memory diagnostic
  used by the status bar.
- Remaining Phase 5 work: no required blocker before Phase 6. Optional cleanup:
  reduce compatibility properties in `EditorCanvas` once tests stop needing
  them, or split `CanvasToolContext` during package reorganization.
- Phase 6 implemented for the current tool model.
- Added `document/tool_serialization.py` for tool dict/zip serialization and
  legacy tool migrations.
- Added `document/archive_serialization.py` for low-level archive array/PIL
  helpers used by layer, selection, and tool serialization.
- Removed `to_dict()`, `from_dict()`, `save_assets_to_zip()`, and archive helper
  ownership from runtime tool classes in `tool.py`.
- `Layer` and `LayerStack` now use explicit archive/tool serialization helpers
  instead of depending on runtime tool methods.
- Legacy `manual_patch_rect` now loads as `ToolLoadResult.legacy_patch_rect` and
  migrates to `Layer.patch_rect` without leaving a `manual_patch_rect` field on
  the runtime tool.
- Remaining Phase 6 follow-up: finish folding project/layer serialization into
  the document package during Phase 7, and consider typed settings DTOs when
  ControlNet/multi-reference data shapes grow.
- Phase 7 started.
- Created `diffusion_editor/document/` as the first package reorganization
  batch.
- Moved document-owned modules under the document package:
  `layer.py`, `layer_stack.py`, `commands.py`, `document_service.py`,
  `history.py`, `mask.py`, `tiles.py`, `tool.py`, `layer_renderer.py`,
  `archive_serialization.py`, and `tool_serialization.py`.
- Updated application and test imports to use `diffusion_editor.document.*`.
- Kept no top-level compatibility wrappers for the moved document modules,
  because this is still active development and stale import paths should fail
  loudly.
- Created `diffusion_editor/canvas/` as the second package reorganization batch.
- Moved canvas-owned modules under the canvas package:
  `editor_canvas.py`, `brush.py`, `canvas_composite.py`,
  `canvas_edit_session.py`, `canvas_geometry.py`, `canvas_image_erase.py`,
  `canvas_mask_erase.py`, `canvas_mask_paint.py`, `canvas_overlay.py`,
  `canvas_paint_stroke.py`, `canvas_rect_drag.py`,
  `canvas_selection_paint.py`, `canvas_smudge.py`,
  `canvas_tool_context.py`, `canvas_tools.py`, `gpu_compositor.py`, and
  `soft_mask_stroke.py`.
- Updated application and test imports to use `diffusion_editor.canvas.*`.
- Kept no top-level compatibility wrappers for the moved canvas modules, for the
  same active-development reason as document modules.
- Created `diffusion_editor/generation/` as the third package reorganization
  batch.
- Moved generation-owned modules under the generation package:
  `types.py`, `diffusion_controller.py`, `instruct_controller.py`,
  `lama_controller.py`, `segmentation_controller.py`,
  `diffusion_request_builder.py`, `patch_resolver.py`,
  `reference_resolver.py`, and `result_mapper.py`.
- Split the old mixed `diffusion_brush.py`:
  `generation/patch_image.py` now owns source patch extraction helpers, while
  `document/result_paste.py` owns applying generated image patches to layer
  pixels.
- Updated engines, application code, and tests to use
  `diffusion_editor.generation.*`.
- Kept no top-level compatibility wrappers for the moved generation modules.
- Created `diffusion_editor/engines/` as the fourth package reorganization
  batch.
- Moved backend adapters under the engines package:
  `diffusion_engine.py`, `instruct_engine.py`, `lama_engine.py`,
  `segmentation_engine.py`, and `grounding_engine.py`.
- Renamed old `segmentation.py` to `engines/segmentation_engine.py` so the module
  name matches its role.
- Updated generation controllers, editor window, and engine-internal imports to
  use `diffusion_editor.engines.*`.
- Kept no top-level compatibility wrappers for the moved engine modules.
- Created `diffusion_editor/ui/` as the fifth package reorganization batch.
- Moved editor panels under `ui/panels/`:
  `brush_panel.py`, `diffusion_panel.py`, `instruct_panel.py`,
  `lama_panel.py`, `layer_panel.py`, and `selection_panel.py`.
- Moved dialogs and dialog helpers under `ui/dialogs/`:
  `file_dialog.py`, `grounding_dialog.py`, and
  `ip_adapter_reference_dialog.py`.
- Updated editor window and dialog tests to use `diffusion_editor.ui.*`.
- Kept no top-level compatibility wrappers for the moved UI modules.
- Created `diffusion_editor/app/` as the sixth package reorganization batch.
- Moved app composition modules under the app package:
  `editor_window.py`, `main.py`, and `settings.py`.
- Updated launch scripts to run `python -m diffusion_editor.app.main`.
- Updated tests and agent chat settings import to use `diffusion_editor.app.*`.
- Kept no top-level compatibility wrappers for the moved app modules.
- Created `diffusion_editor/agent/` and `diffusion_editor/grounding/` as the
  final feature-package cleanup batch.
- Moved agent modules under `agent/`: `chat_panel.py`, `runner.py`, and
  `tools.py`.
- Moved Grounding workflow modules under `grounding/`: `controller.py` and
  `types.py`.
- Updated app, engine, generation, UI dialog, and test imports to use the new
  feature packages.
- Kept no top-level compatibility wrappers for the moved agent/Grounding
  modules.
- Phase 7 is complete for the current module set.

## Architectural Smell Checklist

Use this checklist during future work:

- Does a UI widget directly mutate document state?
- Does a dialog access private fields of `EditorWindow`?
- Does a workflow keep pending async state in `EditorWindow`?
- Does a function accept a long list of loosely related parameters where a typed
  request object would be clearer?
- Does a module mix UI, model loading, inference, and document application?
- Does serialization fallback logic leak into normal runtime behavior?
- Does a pixel operation live inside UI code?
- Does a new feature require editing `EditorWindow` in multiple unrelated
  places?

If the answer is yes, prefer extracting a boundary before adding more behavior
on top.
