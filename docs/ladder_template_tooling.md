# Ladder Template Tooling

This document describes the current ladder-template tooling in Snowflake2025.

Agent-oriented code map:

- See [docs/ladder_template_tooling_for_agents.md](./ladder_template_tooling_for_agents.md) for the compact code-level companion document.

The goal is to turn ladder-escape diagrams and related template fragments into reviewed structured data that can later be:

- materialized into concrete board positions,
- evaluated by Snowflake,
- and eventually used for auxiliary training data or template libraries.

## Current package layout

The ladder-template Python tooling now lives under:

- `hex_ai/utils/ladder_templates/`

Key pieces:

- `docs/ladder_template_annotator.html`
  - Manual / semi-automatic review UI.
  - Loads an image, overlays a hex grid, and exports annotation JSON.
- `hex_ai/utils/ladder_templates/core.py`
  - Loads, validates, and materializes annotation JSON.
- `hex_ai/utils/ladder_templates/prefill_geometry.py`
  - Shared offset-grid inference and annotation-payload builder.
- `hex_ai/utils/ladder_templates/prefill_from_image.py`
  - OpenCV prefill from raster images.
- `hex_ai/utils/ladder_templates/hexwiki_svg_scraper.py`
  - Extracts standalone SVG diagrams from HexWiki sections.
- `hex_ai/utils/ladder_templates/scrape_hexwiki_example_svgs.py`
  - CLI for HexWiki section scraping.
- `hex_ai/utils/ladder_templates/svg_prefill.py`
  - Direct `SVG -> annotation JSON` converter.
- `hex_ai/utils/ladder_templates/prefill_from_svg.py`
  - Batch CLI for converting SVG files to annotation JSON.
- `hex_ai/utils/ladder_templates/preview.py`
  - CLI preview for a reviewed annotation.

## Annotation JSON model

The annotation format is deliberately template-oriented rather than game-oriented.

Important fields:

- `metadata.attacker`
  - Which side the template is for.
- `metadata.target_edge`
  - Intended target edge, such as `red_bottom`.
- `metadata.open_left` / `metadata.open_right`
  - Whether the template is open on one or both sides.
- `cells`
  - Local template cells with states:
  - `empty`
  - `red`
  - `blue`
  - `plus`
  - `minus`
  - `shaded`

`plus` and `minus` are boundary metadata, not board stones.

## Environment

The Python tooling in this repo expects the project venv to be active.

Typical setup:

```bash
cd /Users/niallHome/Documents/programming/Snowflake2025
source hex_ai_env/bin/activate
```

OpenCV is optional and is only needed for the raster-image prefill path:

```bash
pip install opencv-python
```

## Current pilot outputs

The current committed library lives under:

- `hex_ai/utils/ladder_templates/library/hexwiki/theory_of_ladder_escapes/generated/`
  - Direct SVG-derived annotation JSON files committed as the current imported corpus.
- `hex_ai/utils/ladder_templates/library/hexwiki/theory_of_ladder_escapes/reviewed/`
  - Reserved for manually reviewed canonical files promoted from `generated/`.

Scratch and regeneration artifacts remain under:

- `temp/ladder_templates/hexwiki_pilot/scraped_images/`
  - Standalone SVGs scraped from the `Theory_of_ladder_escapes` example sections.
  - Includes:
    - `index.html`
    - `manifest.json`
- `temp/ladder_templates/hexwiki_pilot/svg_json/`
  - Annotation JSON generated directly from those SVGs.
  - Includes:
    - one JSON file per scraped SVG
    - `manifest.json`
- `temp/ladder_templates/hexwiki_pilot/prefill/`
  - Earlier OpenCV-based raster prefills.
- `temp/ladder_templates/hexwiki_pilot/reviewed/`
  - Manually reviewed JSON samples.

At the time of writing, the HexWiki scrape produced:

- 59 standalone SVG files in `scraped_images/`
- 59 annotation JSON files in `svg_json/`
- 59 committed generated JSON files in `hex_ai/utils/ladder_templates/library/hexwiki/theory_of_ladder_escapes/generated/`

## Main workflows

### 1. Manual annotation from an image

Open the annotator in a browser:

```bash
open /Users/niallHome/Documents/programming/Snowflake2025/docs/ladder_template_annotator.html
```

Practical process:

1. Load a source image or SVG.
2. If the image obscures the board, either:
   - adjust `Image X` / `Image Y`, or
   - click `Place below board`.
3. Align the grid using `Radius`, `Origin X`, and `Origin Y`.
4. Paint cells as `red`, `blue`, `empty`, `plus`, `minus`, or `shaded`.
5. Export the JSON.

Use this for:

- reviewing wiki diagrams by hand,
- fixing prefills,
- creating canonical template JSON for a library.

### 2. OpenCV prefill from a raster image

Generate a first-pass annotation JSON from a diagram image:

```bash
source hex_ai_env/bin/activate
python -m hex_ai.utils.ladder_templates.prefill_from_image \
  /path/to/wiki_diagram.png \
  --output /tmp/wiki_diagram_prefill.json \
  --family hexwiki_prefill \
  --template-name fourth_row_escape_example \
  --target-edge red_bottom \
  --debug-image /tmp/wiki_diagram_debug.png
```

What this does:

- detects likely hex cells,
- classifies simple cell contents,
- infers a local offset-grid,
- writes annotator-compatible JSON.

Expected use:

1. Run the prefill.
2. Open the JSON in the annotator.
3. Review and correct mistakes.
4. Save the reviewed JSON as the canonical template.

This is still a prefill only. It is not trusted ground truth.

### 3. Scrape standalone HexWiki SVG diagrams

For HexWiki pages that embed diagrams as inline SVG, extract them directly:

```bash
source hex_ai_env/bin/activate
python -m hex_ai.utils.ladder_templates.scrape_hexwiki_example_svgs \
  --output-dir temp/ladder_templates/hexwiki_pilot/scraped_images
```

This:

- fetches the page,
- slices out the requested section anchors,
- writes standalone SVG files,
- writes a scrape manifest,
- and creates a lightweight `index.html` review page.

Review the current pilot scrape in a browser:

```bash
open /Users/niallHome/Documents/programming/Snowflake2025/temp/ladder_templates/hexwiki_pilot/scraped_images/index.html
```

### 4. Convert scraped SVGs directly to annotation JSON

The current HexWiki diagrams are better handled as vector graphics than as raster screenshots.

Convert a directory of scraped SVGs directly to JSON:

```bash
source hex_ai_env/bin/activate
python -m hex_ai.utils.ladder_templates.prefill_from_svg \
  temp/ladder_templates/hexwiki_pilot/scraped_images \
  --output-dir temp/ladder_templates/hexwiki_pilot/svg_json
```

What this does:

- reads the hex geometry directly from the SVG path data,
- detects `plus` markers from the SVG line paths,
- detects `red` / `blue` stones from SVG circles,
- infers the local offset-grid,
- and writes annotator-compatible JSON plus a batch manifest.

This is currently the preferred path for the `Theory_of_ladder_escapes` pilot.

To refresh the committed generated corpus after regeneration, copy the reviewed batch into:

```bash
cp temp/ladder_templates/hexwiki_pilot/svg_json/*.json \
  /Users/niallHome/Documents/programming/Snowflake2025/hex_ai/utils/ladder_templates/library/hexwiki/theory_of_ladder_escapes/generated/
```

### 5. Preview a reviewed template on a Snowflake board

Preview a template after review:

```bash
source hex_ai_env/bin/activate
python -m hex_ai.utils.ladder_templates.preview \
  /path/to/reviewed_template.json \
  --board-size 13 \
  --anchor-row 2 \
  --anchor-col 4 \
  --summary-json
```

This prints:

- the board embedding,
- the stone coordinates,
- boundary coordinates,
- and a compact JSON summary.

Use this for:

- checking that a reviewed template embeds where you expect,
- testing anchor placement,
- preparing concrete positions for later analysis.

## Specific tasks this tooling supports

### Build a reviewed template library

Process:

1. Collect relevant ladder diagrams.
2. Prefer direct SVG conversion when the source page exposes vector diagrams.
3. Fall back to raster-image prefill when needed.
4. Review in the annotator.
5. Commit bulk-import results into `library/.../generated/`.
6. Promote hand-checked files into `library/.../reviewed/`.

Outcome:

- a consistent machine-readable library of ladder/switchback templates.

### Turn one reviewed template into concrete test positions

Process:

1. Load the template JSON with `hex_ai.utils.ladder_templates.core`.
2. Materialize it at a chosen anchor on a 13x13 board.
3. Create one or more `HexGameState` objects from it.

Outcome:

- reproducible board positions suitable for MCTS or other analysis.

### Build a source-page-to-review pipeline

Process:

1. Scrape SVG diagrams from a source page where possible.
2. Convert the standalone SVGs directly to annotation JSON.
3. Review the JSON in the annotator.
4. Commit bulk-import JSON into `library/.../generated/`.
5. Promote canonical reviewed JSON into `library/.../reviewed/`.

Outcome:

- a scalable path from source page to reviewed structured data.

### Prepare for later solver or MCTS verification

Process:

1. Review a template carefully.
2. Materialize it into one or more concrete boards.
3. Add downstream code that evaluates the position with Snowflake and/or a bounded search.

Outcome:

- a clean separation between:
  - extracting template structure,
  - reviewing correctness,
  - and verifying game-theoretic behavior.

## Current limitations

- The annotator is a local standalone HTML file, not yet integrated into the main Snowflake web app.
- The OpenCV raster prefill is heuristic and should be treated as a draft generator.
- The direct SVG path currently targets the structure used by the current HexWiki diagrams; it is not yet a generic SVG diagram parser.
- `plus` / `minus` boundary cells are preserved as metadata; they are not yet compiled into richer ladder semantics.
- There is not yet a full template-to-search pipeline that automatically derives all concrete verification positions for Snowflake.

## Suggested next steps

- Review the committed `generated/` corpus and start promoting checked files into `reviewed/`.
- Improve preview placement so bottom-edge templates default closer to the target edge.
- Extend the template compiler with stronger semantics for open boundaries, edge attachment, and ladder-family-specific instantiation rules.
- Add a downstream evaluator that takes a reviewed template and generates concrete MCTS verification jobs.
