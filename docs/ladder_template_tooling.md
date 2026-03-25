# Ladder Template Tooling

This document describes the current ladder-template tooling in Snowflake2025.

Agent-oriented code map:

- See [docs/ladder_template_tooling_for_agents.md](./ladder_template_tooling_for_agents.md) for the compact code-level companion document.

The goal is to turn ladder-escape diagrams and related template fragments into reviewed structured data that can later be:

- materialized into concrete board positions,
- evaluated by Snowflake,
- and eventually used for auxiliary training data or template libraries.

## Current pieces

- `docs/ladder_template_annotator.html`
  - Manual / semi-automatic review UI.
  - Loads an image, overlays a hex grid, and exports annotation JSON.
- `scripts/prefill_ladder_template_from_image.py`
  - First-pass OpenCV prefill.
  - Detects likely cells in a diagram image and emits annotator-compatible JSON.
- `hex_ai/ladder_templates.py`
  - Loads and validates annotation JSON.
  - Materializes template cells onto a concrete Snowflake board.
- `scripts/preview_ladder_template.py`
  - CLI preview for a reviewed annotation.
  - Prints an ASCII board plus boundary/stone metadata.

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

OpenCV is optional and is only needed for the image-prefill script:

```bash
pip install opencv-python
```

## Main workflows

### 1. Manual annotation from an image

Open the annotator in a browser:

```bash
open /Users/niallHome/Documents/programming/Snowflake2025/docs/ladder_template_annotator.html
```

Practical process:

1. Load a source image.
2. If the image obscures the board, either:
   - adjust `Image X` / `Image Y`, or
   - click `Place below board`.
3. Align the grid using `Radius`, `Origin X`, and `Origin Y`.
4. Paint cells as `red`, `blue`, `empty`, `plus`, `minus`, or `shaded`.
5. Export the JSON.

Use this for:

- reviewing wiki diagrams by hand,
- fixing OpenCV prefills,
- creating canonical template JSON for a library.

### 2. OpenCV prefill from an image

Generate a first-pass annotation JSON from a diagram image:

```bash
source hex_ai_env/bin/activate
python scripts/prefill_ladder_template_from_image.py \
  /path/to/wiki_diagram.png \
  --output /tmp/wiki_diagram_prefill.json \
  --family hexwiki_prefill \
  --template-name fourth_row_escape_example \
  --target-edge red_bottom \
  --debug-image /tmp/wiki_diagram_debug.png
```

What this currently does:

- detects likely hex cells,
- classifies simple cell contents,
- infers a local offset-grid,
- writes annotator-compatible JSON.

Expected use:

1. Run the prefill.
2. Open the JSON in the annotator.
3. Review and correct mistakes.
4. Save the reviewed JSON as the canonical template.

This is a prefill only. It is not trusted ground truth.

### 3. Preview a reviewed template on a Snowflake board

Preview a template after review:

```bash
source hex_ai_env/bin/activate
python scripts/preview_ladder_template.py \
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
2. Prefill with OpenCV where useful.
3. Review in the annotator.
4. Save reviewed JSON files in a dedicated template directory.

Outcome:

- a consistent machine-readable library of ladder/switchback templates.

### Turn one reviewed template into concrete test positions

Process:

1. Load the template JSON with `hex_ai.ladder_templates`.
2. Materialize it at a chosen anchor on a 13x13 board.
3. Create one or more `HexGameState` objects from it.

Outcome:

- reproducible board positions suitable for MCTS or other analysis.

### Build an image-to-review pipeline

Process:

1. Start from stored wiki images.
2. Run `prefill_ladder_template_from_image.py`.
3. Save JSON and optional debug overlay.
4. Review in the annotator.

Outcome:

- a scalable path from raw diagrams to reviewed structured data.

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
- The OpenCV prefill is heuristic and should be treated as a draft generator.
- `plus` / `minus` boundary cells are preserved as metadata; they are not yet compiled into richer ladder semantics.
- There is not yet a full template-to-search pipeline that automatically derives all concrete verification positions for Snowflake.

## Suggested next steps

- Add a small corpus directory for reviewed ladder-template JSON files.
- Add scripts to harvest or catalogue ladder images from source pages.
- Add a downstream evaluator that takes a reviewed template and generates concrete MCTS verification jobs.
- Extend the template compiler with stronger semantics for open boundaries, edge attachment, and ladder-family-specific instantiation rules.
