# Ladder Template Tooling For Agents

This is the compact code-oriented companion to [docs/ladder_template_tooling.md](./ladder_template_tooling.md).

Read that document first for the user-facing workflow. Use this one to reload the code structure quickly when working on the tooling again.

## Main code map

### Annotator UI

- `docs/ladder_template_annotator.html:480-542`
  - Main metadata and image controls.
  - Important recent additions:
    - `targetEdgeInput` defaults to `red_bottom`
    - `imageOffsetXInput`
    - `imageOffsetYInput`
    - `placeImageBelowButton`
- `docs/ladder_template_annotator.html:689-760`
  - Main browser-side state object.
  - The exported JSON contract is defined implicitly here:
    - `metadata`
    - `grid`
    - `image_overlay`
    - `cells`
- `docs/ladder_template_annotator.html:812-869`
  - Core hex-grid geometry:
    - `hexCenter(...)`
    - `hexPoints(...)`
    - `hexVertices(...)`
    - `edgeMidpoint(...)`
    - `computeStageBounds(...)`
- `docs/ladder_template_annotator.html:994-1148`
  - Main rendering path:
    - source image placement
    - target-edge painting
    - cell overlays
    - marker rendering for `plus` / `minus`
- `docs/ladder_template_annotator.html:1166-1215`
  - JSON import/apply path:
    - `applyAnnotationPayload(...)`
  - This is the browser-side normalization point for imported JSON.
- `docs/ladder_template_annotator.html:1242-1266`
  - Image utility actions:
    - `clearImage()`
    - `placeImageBelowBoard()`
- `docs/ladder_template_annotator.html:1374-...`
  - Event wiring:
    - form sync
    - paint interactions
    - keyboard shortcuts

Concise role:

- This file is the manual review and correction tool.
- It is the source of truth for the current annotation JSON shape as used by humans.

### Annotation loader / compiler

- `hex_ai/ladder_templates.py:27-112`
  - Core dataclasses:
    - `LadderTemplateCell`
    - `LadderTemplateMetadata`
    - `LadderTemplateGrid`
    - `LadderTemplateImageOverlay`
    - `LadderTemplateAnnotation`
    - `MaterializedLadderTemplate`
- `hex_ai/ladder_templates.py:115-217`
  - Validation helpers:
    - `_expect_mapping`
    - `_read_string`
    - `_read_bool`
    - `_read_int`
    - `_read_float`
    - `_normalize_target_edge`
    - `_normalize_attacker`
    - `_normalize_cell_state`
    - `_validate_unique_cells`
- `hex_ai/ladder_templates.py:218-293`
  - `ladder_template_annotation_from_dict(...)`
  - Validates and normalizes raw JSON into typed dataclasses.
- `hex_ai/ladder_templates.py:296-301`
  - `load_ladder_template_annotation(...)`
- `hex_ai/ladder_templates.py:304-363`
  - `materialize_ladder_template(...)`
  - Embeds local template cells into a concrete square board.
  - Important current behavior:
    - `red` / `blue` become actual board stones
    - `empty` stays empty but is tracked in metadata
    - `plus` / `minus` are preserved as boundary metadata only
    - `shaded` is preserved as metadata only

Concise role:

- This module is the Python-side contract for reviewed template JSON.
- It is the bridge from annotation data to concrete Snowflake board state.

### Prefill geometry helper

- `hex_ai/ladder_template_prefill.py:10-35`
  - Prefill-side dataclasses:
    - `DetectedImageCell`
    - `InferredTemplateCell`
    - `InferredGridGeometry`
- `hex_ai/ladder_template_prefill.py:53-128`
  - `infer_offset_grid(...)`
  - Converts detected image centers into local template `(row, col)` coordinates using the offset-grid geometry.
- `hex_ai/ladder_template_prefill.py:131-185`
  - `build_prefill_annotation_payload(...)`
  - Emits annotator-compatible JSON.

Concise role:

- This module is intentionally pure and reusable.
- It isolates the non-OpenCV logic so grid inference can be tested directly.

### OpenCV prefill CLI

- `scripts/prefill_ladder_template_from_image.py:20-29`
  - `_require_cv2()`
  - Explicit optional dependency boundary.
- `scripts/prefill_ladder_template_from_image.py:47-89`
  - `_detect_hex_candidates(...)`
  - Contour-based candidate cell detection.
  - Uses adaptive thresholding plus simple geometric filters.
- `scripts/prefill_ladder_template_from_image.py:92-154`
  - `_patch_with_mask(...)`
  - `_classify_detected_cell(...)`
  - Patch-level classification for:
    - `red`
    - `blue`
    - `plus`
    - `minus`
    - `shaded`
    - fallback `empty`
- `scripts/prefill_ladder_template_from_image.py:157-172`
  - `detect_image_cells(...)`
- `scripts/prefill_ladder_template_from_image.py:175-209`
  - `_draw_debug_overlay(...)`
  - Useful when trying to understand failures on real diagrams.
- `scripts/prefill_ladder_template_from_image.py:212-301`
  - CLI surface and end-to-end execution.

Concise role:

- This is the current image-to-JSON draft generator.
- It should always be treated as a prefill for review, not final truth.

### Preview CLI

- `scripts/preview_ladder_template.py:19-23`
  - `_format_coords(...)`
- `scripts/preview_ladder_template.py:26-44`
  - CLI arguments
- `scripts/preview_ladder_template.py:47-75`
  - End-to-end preview

Concise role:

- This is the quickest way to sanity-check a reviewed annotation as a concrete board embedding.

## Tests worth checking first

- `hex_ai/test_ladder_templates.py:54-99`
  - Loader normalization
  - Materialization semantics
  - `HexGameState` conversion
- `hex_ai/test_ladder_template_prefill.py:8-66`
  - Offset-grid inference
  - Prefill payload generation

If something breaks in the JSON shape or grid inference, these are the first tests to inspect or extend.

## Current behavior to remember

- The template JSON is local-fragment oriented, not a full-game encoding.
- `plus` and `minus` do not become stones or edge-forced cells in the board array.
- The compiler is intentionally conservative: it embeds what was annotated and preserves the rest as metadata.
- The annotator and the prefill script currently share the same JSON contract informally rather than through a generated schema file.

## Limitations

- There is no dedicated schema file or versioned migration layer yet.
- The annotator is a standalone HTML document rather than part of the main web app.
- The OpenCV prefill is heuristic and currently tuned for clean, stylized diagrams.
- There is not yet a downstream pipeline that:
  - takes a reviewed template,
  - generates a family of concrete board embeddings,
  - runs Snowflake/MCTS/other verification automatically,
  - and records the result back into structured template metadata.

## Good next steps

- Add a small corpus directory for reviewed template JSON files and a naming convention.
- Add a harvesting script for ladder-related source images/pages.
- Add a stronger “template instantiation” layer:
  - edge attachment semantics
  - open-boundary handling
  - multiple anchor/rotation/reflection embeddings
- Add a downstream evaluator that can batch:
  - materialize template variants
  - run Snowflake
  - store summarized outcomes
- If the JSON contract starts changing more often, introduce a single shared schema module or JSON Schema file.
