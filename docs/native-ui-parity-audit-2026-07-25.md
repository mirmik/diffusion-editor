# Legacy/native UI parity audit — 2026-07-25

This audit compares the retained `tcgui` UI (`--ui legacy`) with the
`termin-gui-native` UI (`--ui native`) before the native cutover. It records
differences only; parity fixes are intentionally split into follow-up tasks.

## Baseline

- Window content: 1280 × 800.
- Document: the same imported 640 × 480 high-contrast test image.
- Backend used for comparable captures: OpenGL/llvmpipe under Xvfb.
- States inspected: main editor, Settings, Grounding, empty generation host,
  and attached-tool construction.
- Evidence: visual captures, `TcDocument.inspect_snapshot()` geometry,
  legacy runtime widget geometry, and application/framework source inspection.

The command inventory is equivalent in both UIs. Menu commands, shortcuts and
toolbar actions are present. The dual launch contract is also in place:

```sh
./run.sh --ui legacy [document-or-image]
./run.sh --ui native [document-or-image]
```

## Required fixes

| Priority | Area | Legacy expectation | Native observation | Disposition |
| --- | --- | --- | --- | --- |
| P0 | Numeric sliders | Brush, selection, layer opacity, generation and Grounding sliders show a track, thumb and editable value. | `SliderEdit` paints only its caption. Its dynamically adopted Slider and SpinBox children do not appear in the rendered tree, so the control is unusable. | Fix in `termin-gui-native`; add a focused render/input regression test. |
| P0 | Left panel allocation | Brush and selection controls occupy their preferred compact heights; an attached generation panel is placed below them. | Canvas controls and the generation scroll host each receive half of the full left column. Selection occupies y=382…676 while generation starts at y=638, producing a 38 px overlap even with no attached tool. The empty-state text is clipped. | Fix application shell ownership/sizing after the framework slider defect is covered. |
| P1 | Main shell | Four horizontal regions: controls, Canvas, Layers, Agent Chat. At 1280 px their widths are approximately 260 / 465 / 220 / 320. | Three regions: controls, Canvas, and a 285 px right column in which Layers and Agent Chat are vertically stacked. The Canvas grows to about 732 px and Agent Chat loses most of its height. | Restore the legacy horizontal arrangement as the parity baseline; reconsider only as a separate redesign. |
| P1 | Canvas status | Pointer motion shows document size, coordinates, active layer, tool/brush size, history and cache memory. | `EditorCanvasController.on_mouse_moved` is never wired by `NativeEditorRoot`; the status bar remains an operation message. | Add a toolkit-neutral status projection and bind it in native UI. |
| P1 | Settings dialog | Numeric agent fields have captions; the history-limit explanation is visible; the dialog is compact around its contents. | Temperature, max-token and timeout values have no captions. The history explanation is omitted. A fixed 520 × 430 content size leaves a large empty lower area. | Restore captions/help text and content-driven compact geometry. |
| P1 | Layer rows | Every row exposes visibility directly and includes the layer color swatch; selected-layer opacity remains below the tree. | Visibility/solo are encoded as `[V]`/`[S]` text and duplicated as controls below the tree. Layer color is absent. | Restore row-level affordances. A Termin tree-row extension may be required for a true swatch/control renderer. |
| P2 | Brush modes | Mutually exclusive tool buttons make the current mode visually explicit. | Six checkboxes are used as a radio group by application logic. | Replace with a native exclusive-choice/button presentation; no framework blocker is known. |
| P2 | Toolbar and chrome | Compact legacy labels and 24/40/18 px menu, toolbar and status heights. | More verbose labels and 28/36/24 px chrome. Group boxes also lose the legacy header-bar/collapse treatment. | Accept the native metrics unless they cause narrow-window clipping; keep label wording as a small parity cleanup. |

## Dialog parity

Settings has the concrete omissions listed above. Grounding preserves the
parameter set, defaults and result mapping, but all of its numeric fields are
affected by the same native `SliderEdit` rendering defect. Some explanatory
legacy wording is shortened (`experimental`, tighter-mask and disabled-area
hints); this is lower priority than restoring usable controls.

Native file dialogs are intentionally framework-native. Their behavioral
contract—mode, initial path, filters, cancel/accept callback and modality—is
covered by application tests and the desktop checklist; pixel parity with the
legacy dialog is not required.

## Behavior that is already equivalent

- Menus, shortcuts, command enabled/checked projection and toolbar actions.
- Layer add/remove/flatten, selection, rename, visibility, solo, opacity,
  drag/drop and attach/detach intents.
- Brush color picking and bracket-driven brush-size changes synchronize back
  into the native control state.
- Patch rectangle, selection rectangle and generation-mask intents are routed
  through toolkit-neutral coordinators.
- Settings and Grounding acceptance/cancel values map to the same application
  state even where presentation differs.
- Agent Chat has the same coordinator-owned conversation actions; native also
  exposes connection state explicitly.

## Shared rendering warning and test gap

In the OpenGL/llvmpipe comparison capture, both UIs imported a valid saturated
640 × 480 image but displayed no source pixels. Native Canvas held a borrowed
640 × 480 compositor texture, while a direct compositor readback was all zero.
Because the defect reproduces in both UIs, it is not a migration-parity
difference. It still blocks a trustworthy visual Canvas comparison.

The windowed native smoke currently proves texture ownership and dimensions,
but does not assert that the imported source contributes recognizable pixels
to the borrowed compositor output. A focused application/render task must
determine whether this is an llvmpipe-only compositor issue or a general
regression, and strengthen the gate so a dimensionally valid blank texture
cannot pass.

## Exact shell geometry

Legacy:

| Region | Bounds |
| --- | --- |
| Menu / toolbar / status | y=0 h=24 / y=24 h=40 / y=782 h=18 |
| Left controls | x=0, w=260, h=718 |
| Canvas | x=265, w=465, h=718 |
| Layers | x=735, w=220, h=718 |
| Agent Chat | x=960, w=320, h=718 |

Native:

| Region | Bounds |
| --- | --- |
| Menu / toolbar / status | y=0 h=28 / y=28 h=36 / y=776 h=24 |
| Left controls | x=0, w=255.2, h=712 |
| Canvas | x=259.2, w=732.1, h=712 |
| Layers | x=995.3, y=64, w=284.7, h=410.6 |
| Agent Chat | x=995.3, y=478.6, w=284.7, h=297.4 |

These values explain the largest visible difference: native did not merely
restyle the original shell; it changed its panel topology.

## Cutover conclusion

The feature wiring is substantially present, but the native UI is not yet a
safe default. The cutover should wait for the two P0 issues, horizontal shell
parity, Canvas status, and a non-blank source-pixel gate. The remaining P1/P2
items can then be reviewed side by side on a real desktop before deleting the
legacy path.

## Resolution update

The audit findings were implemented and covered by focused regressions:

- `SliderEdit` now paints and routes input to its slider and editable numeric
  field.
- The shell again uses four horizontal regions, while the left column follows
  preferred control heights and scrolls without overlap.
- Pointer motion now projects document, layer, tool, history and cache details
  into the native status bar.
- Settings and Grounding dialogs restore the missing captions and explanatory
  text with compact sizing.
- Canvas tools use exclusive checkable toolbar buttons. Layer visibility and
  solo are direct row affordances, backed by reusable native TreeWidget
  toggles; ToolBar also has keyboard navigation.
- The blank Canvas was traced to GUI state leaking back-face culling into the
  compositor pass. Every pass now establishes its full render state, and the
  smoke test compares deterministic imported pixels with the final Canvas
  texture while reporting stage signatures on failure.

The application suite passes with 334 tests. The source-pixel gate also passes
in a windowed OpenGL/Xvfb run. Windowed Vulkan remains a real-desktop gate on
this host because Xvfb exposes neither DRI3 nor a Vulkan present queue; the
headless Vulkan path may use the CPU fallback and is not counted as Vulkan GPU
compositor evidence.
