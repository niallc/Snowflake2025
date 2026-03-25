# Theory Of Ladder Escapes Library

This directory is the committed ladder-template corpus derived from the HexWiki page:

- `https://www.hexwiki.net/index.php/Theory_of_ladder_escapes`

Layout:

- `generated/`
  - Direct SVG-derived annotation JSON files.
  - This is the current bulk-import state.
  - These files are suitable for code consumption, but not all have been manually reviewed yet.
- `reviewed/`
  - Canonical hand-checked files promoted from `generated/` after review.

Current state:

- The `generated/` directory contains the current import corpus from the `Examples`, `Examples_2`, `Examples_3`, and `Examples_4` sections.
- Scratch scrape artifacts, standalone SVGs, and experimental manifests remain under:
  - `temp/ladder_templates/hexwiki_pilot/`

Promotion policy:

1. Generate or regenerate bulk JSON into `generated/`.
2. Review selected files in the annotator.
3. Copy reviewed files into `reviewed/` once they are considered canonical.
