# Ladder Certificate Tooling

**Last updated:** 2026-03-25

This document tracks the current implementation status of the ladder-certificate
tooling that will feed a future auxiliary training head.

## Goal

Build a deterministic bridge between the existing ladder-template corpus and
spatial supervision that the network can learn from.

For now, the implementation target is:

- use the HexWiki `generated/` corpus directly
- match local certificate fragments on concrete boards
- emit dense spatial labels
- expose timing stats and an optimization hook for per-game incremental use

## New modules

- `hex_ai/utils/ladder_templates/library_loader.py`
  - loads the generated library
  - infers row class from the current filename convention
  - pre-indexes carrier / attacker / empty / boundary cells
- `hex_ai/utils/ladder_templates/matcher.py`
  - deterministic phase-1 matcher
  - normalizes board inputs from `(N, N)`, `(2, N, N)`, or `(3, N, N)`
  - returns structured match objects plus timing/search-volume stats
- `hex_ai/utils/ladder_templates/labeler.py`
  - converts matches into canonical 8-plane spatial masks
  - currently emits:
    - `*_template_origin`
    - carrier maps
- `hex_ai/utils/ladder_templates/benchmark.py`
  - synthetic benchmark CLI for measuring raw matcher throughput

## Current phase-1 semantics

### Library source

For now, code uses:

- `hex_ai/utils/ladder_templates/library/hexwiki/theory_of_ladder_escapes/generated/`

This is intentionally temporary. Once the reviewed set exists, loading policy
should flip to reviewed-by-default.

### Matchable template subset

The current matcher is deliberately narrow:

- it directly matches templates authored as:
  - `attacker = red`
  - `target_edge = red_bottom`
- it derives `blue_right` supervision by applying the existing long-diagonal
  color-swapping symmetry to the board before matching

This keeps the first implementation explicit and testable.

### Boundary semantics

`plus` / `minus` are treated as boundary metadata, not board occupancy
constraints.

This matches both:

- the existing loader/materializer behavior
- the HexWiki definition where boundary cells are deleted when patterns are
  glued together

### Carrier semantics

Carrier cells currently mean:

- template `red`
- template `blue`
- template `empty`

`shaded` is treated as excluded / ignored, not as carrier.

### Monotone attacker closure

Phase 1 enables the following by default:

- if a template carrier cell is annotated `empty`, the matcher still accepts an
  attacker stone on that cell

Rationale:

- adding attacker stones cannot invalidate an attacker certificate
- this avoids teaching the head that only minimal instances are positive

This behavior is controlled by:

- `allow_attacker_superset_on_empty`

## Current labels

The canonical plane set is:

- `red_bottom_row2`
- `red_bottom_row3`
- `red_bottom_row4`
- `red_bottom_row5`
- `blue_right_row2`
- `blue_right_row3`
- `blue_right_row4`
- `blue_right_row5`

The current emitted maps are:

- `*_template_origin`
  - marks where local template cell `(0, 0)` lands on the board
- carrier maps
  - marks every carrier cell in a matched certificate

## Important open issue: true anchor semantics

The current template JSON does **not** explicitly encode the “ladder stone
anchor” that the aux-head proposal originally discussed.

What exists today:

- a local fragment `P`
- boundary cells `+`
- top-left local grid coordinates

What does **not** yet exist:

- an explicit schema field for the tactical ladder-stone anchor to supervise

Because of that, phase 1 currently emits `template_origin` maps instead of
claiming they are true ladder-stone anchor maps.

This should be resolved before final aux-head target semantics are frozen.

## Incremental / per-game optimization hook

The matcher now supports:

- `must_include_cell=(row, col)`

This limits candidate embeddings to templates whose carrier contains that cell.

That is the intended hook for a faster per-game labeling pass:

1. step through positions in move order
2. use the last move as `must_include_cell`
3. only search certificates whose carrier could have changed because of that move

This does not yet implement a full cached incremental labeler, but it is the
first clean interface needed for that direction.

## Timing

Synthetic timing CLI:

```bash
source hex_ai_env/bin/activate
python -m hex_ai.utils.ladder_templates.benchmark --iterations 5
```

Useful variants:

```bash
python -m hex_ai.utils.ladder_templates.benchmark --iterations 5 --must-include-carrier-tail
python -m hex_ai.utils.ladder_templates.benchmark --iterations 5 --fill-empty-with-attacker
```

The matcher also returns per-call stats:

- elapsed milliseconds
- templates considered
- embeddings considered
- matches found

## Next likely steps

1. Add a real-board benchmark path over sampled self-play positions.
2. Decide and encode true anchor semantics.
3. Add a small synthetic dataset generator for held-out supervision tests.
4. Thread optional ladder targets into the training data path.
5. Add the auxiliary head and loss once target semantics are stable enough.
