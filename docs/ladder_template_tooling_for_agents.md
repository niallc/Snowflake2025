# Ladder Template Tooling For Agents

This is the compact code-oriented companion to [docs/ladder_template_tooling.md](./ladder_template_tooling.md).

Read that document first for the user-facing workflow. Use this one to reload the code structure quickly when working on the tooling again.

## Main code map

### Annotator UI

- `docs/ladder_template_annotator.html:480-690`
  - Metadata, image, grid, paint, and export/import controls.
- `docs/ladder_template_annotator.html:689-1215`
  - Browser-side annotation state, geometry helpers, rendering path, and JSON import/apply logic.
- `docs/ladder_template_annotator.html:1242-...`
  - Image utilities, event wiring, and keyboard shortcuts.

Concise role:

- This file is the manual review and correction tool.
- It is the source of truth for the current annotation JSON shape as used by humans.

### Ladder-template package

All ladder-template Python tooling now lives under:

- `hex_ai/utils/ladder_templates/`

#### Core annotation loader / compiler

- `hex_ai/utils/ladder_templates/core.py:27-112`
  - Core dataclasses:
    - `LadderTemplateCell`
    - `LadderTemplateMetadata`
    - `LadderTemplateGrid`
    - `LadderTemplateImageOverlay`
    - `LadderTemplateAnnotation`
    - `MaterializedLadderTemplate`
- `hex_ai/utils/ladder_templates/core.py:115-218`
  - Validation helpers and normalization utilities.
- `hex_ai/utils/ladder_templates/core.py:218-293`
  - `ladder_template_annotation_from_dict(...)`
- `hex_ai/utils/ladder_templates/core.py:296-363`
  - `load_ladder_template_annotation(...)`
  - `materialize_ladder_template(...)`

Concise role:

- This module is the Python-side contract for reviewed template JSON.
- It is the bridge from annotation data to concrete Snowflake board state.

#### Shared prefill geometry

- `hex_ai/utils/ladder_templates/prefill_geometry.py:10-35`
  - Prefill-side dataclasses:
    - `DetectedImageCell`
    - `InferredTemplateCell`
    - `InferredGridGeometry`
- `hex_ai/utils/ladder_templates/prefill_geometry.py:53-128`
  - `infer_offset_grid(...)`
- `hex_ai/utils/ladder_templates/prefill_geometry.py:131-185`
  - `build_prefill_annotation_payload(...)`

Concise role:

- This module is intentionally pure and reusable.
- It isolates the grid inference and annotation-payload builder so both raster and SVG paths can reuse them.

#### Raster-image prefill

- `hex_ai/utils/ladder_templates/prefill_from_image.py:20-172`
  - OpenCV dependency boundary, contour detection, patch classification, and image-cell detection.
- `hex_ai/utils/ladder_templates/prefill_from_image.py:175-209`
  - `_draw_debug_overlay(...)`
- `hex_ai/utils/ladder_templates/prefill_from_image.py:212-301`
  - CLI surface and end-to-end raster prefill.

Concise role:

- This is the OpenCV image-to-JSON draft generator.
- It should always be treated as a prefill for review, not final truth.

#### HexWiki SVG scrape path

- `hex_ai/utils/ladder_templates/hexwiki_svg_scraper.py:12-52`
  - Dataclasses for section specs, headings, and extracted SVG records.
- `hex_ai/utils/ladder_templates/hexwiki_svg_scraper.py:55-170`
  - HTML section parsing and SVG normalization.
  - Important recent behavior:
    - injects shared HexWiki SVG styles / defs into standalone files
    - keeps extracted SVGs render-faithful outside the original page
- `hex_ai/utils/ladder_templates/hexwiki_svg_scraper.py:173-280`
  - section extraction, manifest writing, and review-page generation.
- `hex_ai/utils/ladder_templates/scrape_hexwiki_example_svgs.py:22-109`
  - CLI for scraping example sections from a HexWiki page.

Concise role:

- This path turns a source HexWiki page into standalone SVG diagram files plus a scrape manifest.
- It is the preferred first step when the source diagrams are inline SVG rather than raster images.

#### Direct SVG-to-JSON conversion

- `hex_ai/utils/ladder_templates/svg_prefill.py:21-72`
  - SVG dataclass and core geometry parsing helpers.
- `hex_ai/utils/ladder_templates/svg_prefill.py:85-150`
  - marker extraction and marker-to-cell assignment.
- `hex_ai/utils/ladder_templates/svg_prefill.py:153-207`
  - `parse_ladder_template_svg(...)`
  - `build_prefill_annotation_payload_from_svg(...)`
- `hex_ai/utils/ladder_templates/prefill_from_svg.py:13-143`
  - batch CLI that converts one SVG or a whole directory into annotator-compatible JSON.

Concise role:

- This is the current preferred HexWiki conversion path.
- It bypasses OpenCV and reads the hex geometry and markers directly from vector data.

#### Preview CLI

- `hex_ai/utils/ladder_templates/preview.py:19-23`
  - `_format_coords(...)`
- `hex_ai/utils/ladder_templates/preview.py:26-44`
  - CLI arguments
- `hex_ai/utils/ladder_templates/preview.py:47-79`
  - End-to-end preview

Concise role:

- This is the quickest way to sanity-check a reviewed annotation as a concrete board embedding.

## Tests worth checking first

- `hex_ai/utils/ladder_templates/test_core.py:54-99`
  - Loader normalization
  - Materialization semantics
  - `HexGameState` conversion
- `hex_ai/utils/ladder_templates/test_prefill_geometry.py:8-66`
  - Offset-grid inference
  - Prefill payload generation
- `hex_ai/utils/ladder_templates/test_hexwiki_svg_scraper.py:31-96`
  - section heading parsing
  - SVG extraction boundaries
  - standalone SVG normalization
  - manifest / contact-sheet writing
- `hex_ai/utils/ladder_templates/test_svg_prefill.py:26-64`
  - direct SVG geometry parse
  - SVG-to-annotation payload generation

If something breaks in the JSON shape, grid inference, SVG scrape, or direct SVG conversion, these are the first tests to inspect or extend.

## Current behavior to remember

- The template JSON is local-fragment oriented, not a full-game encoding.
- `plus` and `minus` do not become stones or edge-forced cells in the board array.
- The compiler is intentionally conservative: it embeds what was annotated and preserves the rest as metadata.
- The annotator and the prefill helpers still share the same JSON contract informally rather than through a generated schema file.
- The HexWiki scrape path depends on inline SVG structure plus shared page-level defs. The scraper now injects those defs into each extracted standalone SVG.
- The current direct SVG parser targets the current HexWiki diagram structure:
  - hex body path
  - black plus/minus line markers
  - `vertcirc` / `horizcirc` circles

## Current pilot output locations

- `hex_ai/utils/ladder_templates/library/hexwiki/theory_of_ladder_escapes/generated/`
  - committed imported JSON corpus
- `hex_ai/utils/ladder_templates/library/hexwiki/theory_of_ladder_escapes/reviewed/`
  - reserved canonical reviewed set
- `temp/ladder_templates/hexwiki_pilot/scraped_images/`
  - standalone SVGs
  - scrape `manifest.json`
  - review `index.html`
- `temp/ladder_templates/hexwiki_pilot/svg_json/`
  - direct SVG-derived annotation JSON files
  - batch `manifest.json`
- `temp/ladder_templates/hexwiki_pilot/prefill/`
  - earlier raster-image prefills
- `temp/ladder_templates/hexwiki_pilot/reviewed/`
  - manually reviewed samples

## Limitations

- There is no dedicated schema file or versioned migration layer yet.
- The annotator is a standalone HTML document rather than part of the main web app.
- The OpenCV raster prefill is heuristic and currently tuned for clean, stylized diagrams.
- The direct SVG path is not yet a general-purpose diagram parser for arbitrary SVG sources.
- There is not yet a downstream pipeline that:
  - takes a reviewed template,
  - generates a family of concrete board embeddings,
  - runs Snowflake / MCTS / other verification automatically,
  - and records the result back into structured template metadata.

## Good next steps

- Review the committed `generated/` corpus and promote good outputs into `reviewed/`.
- Improve preview placement so target-edge templates embed more usefully by default.
- Add a stronger “template instantiation” layer:
  - edge attachment semantics
  - open-boundary handling
  - multiple anchor / rotation / reflection embeddings
- Add a downstream evaluator that can batch:
  - materialize template variants
  - run Snowflake
  - store summarized outcomes
- If the JSON contract starts changing more often, introduce a single shared schema module or JSON Schema file.
