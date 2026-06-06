# Diffusion Generation Refactor Plan

## Problem

`EditorWindow` currently acts as UI shell, callback hub, domain workflow,
generation request builder, reference-image resolver, and async task coordinator.
That made the IP-Adapter layer-reference feature land in `editor_window.py`,
because the window is the only object that has access to `LayerStack`,
`EditorCanvas`, `DiffusionEngine`, dialogs, status messages, and document
commands at the same time.

This is workable for one feature, but it is the wrong shape for adding
ControlNet, multi-reference inputs, model-specific reference settings, or richer
generation workflows. The main goal is to move generation data preparation and
workflow state out of the window without doing a risky whole-editor rewrite.

## Target Boundaries

`EditorWindow` should remain a composition root and UI shell:

- construct panels and services;
- wire callbacks;
- show dialogs and message boxes;
- update status text;
- execute document commands;
- call generation/application services.

`EditorWindow` should not know:

- how Patch is converted into a source crop;
- how mask bbox/center becomes a generation patch;
- how an IP-Adapter reference layer is cropped;
- how RGBA reference data is converted to RGB;
- how masks are cropped for inpainting;
- how request dimensions are resized for model resolution;
- how pending generation is resumed after model/IP-Adapter loading.

## Proposed Modules

### `generation_types.py`

Shared DTOs for generation preparation:

- `GenerationError`
- `PatchSource`
- `ReferenceImage`
- `DiffusionRequest`

These types make request-building explicit and testable.

### `patch_resolver.py`

Pure helpers for source-patch preparation:

- clip canvas rects against a composite image;
- resolve a source patch from layer-local `Patch`;
- resolve a source patch from layer mask bbox/center;
- fall back to an existing tool patch when appropriate.

This should remove duplicated Patch/mask crop logic from Diffusion and
InstructPix2Pix flows.

### `reference_resolver.py`

Pure helpers for reference-image preparation:

- resolve `DiffusionTool.ip_adapter_layer_id` through `LayerStack`;
- prefer reference layer `Patch`;
- otherwise use alpha bbox;
- otherwise use full layer;
- composite RGBA over a stable background and return RGB PIL image;
- return structured errors instead of silently disabling reference input.

This is the immediate home for IP-Adapter reference behavior.

### `diffusion_request_builder.py`

Build a complete `DiffusionRequest` from:

- target `Layer`;
- its `DiffusionTool`;
- current composite below the target;
- `LayerStack` for reference lookup.

Responsibilities:

- refresh source patch for img2img/inpaint;
- extract inpaint mask patch;
- attach resolved IP-Adapter image;
- apply model-resolution resize policy;
- produce either request or `GenerationError`.

### Later: `generation_controller.py`

Move async workflow out of `EditorWindow`:

- load model if needed;
- load IP-Adapter if needed;
- resume pending request using a structured pending operation, not only a layer;
- submit inference;
- map results back through document commands.

This is a second phase. It should happen after request building is already
isolated and tested.

### Later: UI Dialog Extraction

Move IP-Adapter reference picking UI into a small dialog helper:

- list available layers;
- preserve current selection;
- return chosen layer.

This is less urgent than request building because it is UI-specific and small.

## Migration Sequence

1. Add `generation_types.py`, `patch_resolver.py`, `reference_resolver.py`.
2. Add tests for reference resolving:
   - Patch crop wins over alpha bbox;
   - alpha bbox is used when Patch is absent;
   - full layer is used for fully transparent/empty alpha;
   - missing reference layer returns a structured error.
3. Add `diffusion_request_builder.py`.
4. Replace `EditorWindow._ip_adapter_reference_rect()` and
   `_make_ip_adapter_reference_image()` with the resolver.
5. Replace most of `EditorWindow._submit_regenerate()` with request builder
   usage, keeping engine submission and status messages in the window.
6. Move duplicated Diffusion/Instruct patch-refresh logic into `patch_resolver`.
7. Add/adjust tests for request-building behavior.
8. In a later pass, introduce `generation_controller.py` and replace
   `_pending_request` with a structured pending operation.

## Non-Goals For The First Pass

- Do not rewrite `DiffusionEngine` internals yet.
- Do not change project file compatibility beyond the already introduced
  `ip_adapter_layer_id`.
- Do not implement multi-reference inputs yet.
- Do not change UI layout beyond callback rewiring required by extracted logic.

## Success Criteria

- `EditorWindow` no longer contains pixel-level IP-Adapter reference extraction.
- Diffusion source patch and mask preparation are testable without constructing
  the UI.
- Missing/invalid reference inputs still log and surface errors.
- Existing editor behavior and tests remain green.

## Current Progress

- Added typed generation DTOs.
- Extracted source patch and mask preparation into `patch_resolver.py`.
- Extracted IP-Adapter reference image preparation into `reference_resolver.py`.
- Extracted diffusion request construction into `diffusion_request_builder.py`.
- Extracted diffusion load/inference/pending workflow into
  `diffusion_generation_controller.py`.
- Moved Diffusion, InstructPix2Pix, and LaMa source patch preparation onto the
  shared patch resolver.
- Moved layer-local mask crop extraction for Diffusion inpaint and LaMa onto the
  shared patch resolver.
- Extracted IP-Adapter reference layer picking into
  `ip_adapter_reference_dialog.py`.
- `EditorWindow` still owns UI wiring, dialogs, status updates, and document
  command execution, but no longer owns diffusion request construction,
  diffusion pending task state, pixel-level Patch/mask extraction, or
  IP-Adapter reference dialog construction.

## Remaining Refactor Targets

- Consider moving InstructPix2Pix and LaMa load/inference/pending workflows into
  controller objects matching `diffusion_generation_controller.py`.
- Consider replacing `DiffusionEngine.submit(...)`'s long argument list with
  `submit_request(DiffusionRequest)` after the current controller boundary
  settles.
