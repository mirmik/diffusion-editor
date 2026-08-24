# Character asset inventory — 2026-08-23

This inventory supports the canonical point-map head experiments. The first
training expansion should increase identity and proportion diversity without
changing the task to arbitrary non-humanoid reconstruction.

## Downloaded Blender Studio cohort

All twelve assets below were downloaded from Blender Studio under CC-BY. Exact
source pages and SHA-256 hashes are stored in
`data/character-assets/blender-studio/manifest.json`. The binary `.blend` files
and generated previews are intentionally ignored by Git.

| Cohort | Characters | Decision |
| --- | --- | --- |
| Primary | Snow, Ellie, Jay, Phil, Rex, Victoria | Include in the first expanded training run. |
| Secondary extreme | Gabby, Lunte, Elder Sprite | Keep for a later controlled ablation or sample at a lower rate. |
| Needs cleanup | Einar | Visually valid, but the source scene exposes 76 render-visible meshes and five armatures; select the character collection explicitly before dataset rendering. |
| Non-humanoid holdout | Phileas, Pip | Keep downloaded, but exclude from the first humanoid training run. |

The four-view contact sheet is generated at
`data/character-assets/blender-studio/previews/contact-sheet.png`.

## Existing local stash

The local MakeHuman/export stash was also rendered from four azimuths. It
contains many duplicated exports rather than distinct identities.

| Group | Assessment |
| --- | --- |
| Atom, Barret, CorpGuard, Kate | Useful distinct identities after provenance/license verification. |
| Binesh variants | One identity; select a single clean export, not every variant. |
| Tamara variants | One identity; select a single clean export. |
| Arthur variants | One identity with export/clothing variants; do not count them as independent characters. |
| `arthur_model_full_set` | Reject for automatic rendering: overloaded/abnormal silhouette. |
| `BineshSlave` | Reject until cleaned: an outlying elongated object corrupts automatic framing. |
| Termin variants | Preserve as held-out identity; never add to the training split. |

The MakeHuman stash has no provenance record in this repository. It must not be
treated as redistributable or production-ready until its source and license are
recorded.

## Downloaded external CC0 candidates

The raw assets are stored under
`/home/mirmik/mnt-nvme/character-assets/incoming`; source URLs, licenses and
integrity hashes are recorded in
`data/character-assets/external-manifest.json`.

| Source | Downloaded | Visual assessment |
| --- | ---: | --- |
| Quaternius Ultimate Animated Character Pack | 52 Blend files | Clean and directly renderable, but predominantly one stylized body family with clothing, hair and gender variants. Use a small weighted subset rather than treating all 52 as independent anatomical diversity. |
| OpenGameArt `character pack1` | 20 Blend scenes | Useful proportion diversity. Each scene contains four side-by-side variants of one identity, so one mesh/rig pair must be isolated before dataset rendering. |
| OpenGameArt Low Poly Human Pack | 4 rigged plus 2 unrigged figures in one scene | Very coarse geometry. Keep as a possible stress-test/control source, not a primary training cohort. |

Generated visual inventories:

- `data/character-assets/quaternius-ultimate/previews/contact-sheet.png`;
- `data/character-assets/opengameart-character-pack1/previews/contact-sheet.png`;
- `data/character-assets/opengameart-low-poly-human/previews/all/az000.png`.

## Proposed first expansion

Use the six primary Blender Studio characters plus the distinct, provenance-
checked local identities. Keep Termin exclusively as the held-out evaluation
character. Do not multiply samples merely by including duplicate Arthur,
Binesh, or Tamara exports: pose, camera and appearance variation should be
generated explicitly and tracked as such.
