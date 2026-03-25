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
- `hex_ai/utils/ladder_templates/data_targets.py`
  - resolves last moves from processed example metadata
  - builds optional ladder targets directly from processed-example dicts
- `hex_ai/utils/ladder_templates/sequence_labeler.py`
  - exact incremental label generation over ordered move sequences
  - keeps active matches and only rechecks embeddings touching the latest move
- `hex_ai/utils/ladder_templates/sidecar.py`
  - JSONL sidecar contract for per-game ladder-label sequences
  - compressed dense storage for `template_origin` and carrier maps

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

That initial hook is now used by:

- `build_ladder_certificate_label_sequence_for_trmph_game(...)`

Current incremental semantics:

1. do one opening full-board scan at position 0
2. maintain the active full-match set
3. after each move, only rescan embeddings whose carrier contains that move
4. remove stale touched matches and add newly valid touched matches
5. keep all untouched matches

Under the current phase-1 matcher semantics, this is exact because only carrier
cells impose board-occupancy constraints.

Important scope limit:

- this exactness claim is tied to the current phase-1 semantics where boundary
  metadata is not an occupancy constraint
- if richer boundary semantics are added later, the incremental invalidation
  rule must be revisited

At the processed-example boundary, the new helper path is:

- `resolve_last_move_for_training_example(...)`
- `find_ladder_matches_for_training_example(...)`
- `build_ladder_certificate_target_for_training_example(...)`

These work directly on the dict format produced by the processed training
shards, so ladder labels can be generated at the data boundary before any
trainer/model integration.

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

Processed-shard benchmark:

```bash
python -m hex_ai.utils.ladder_templates.benchmark \
  --processed-shard data/processed/ordered_positions_selfplay_20251016_125528/cleaned_chunk_000_processed.pkl.gz \
  --sample-size 128 \
  --compare-last-move-filter
```

In processed-shard mode the benchmark:

- measures ordinary full-board matching
- optionally compares it against last-move-filter matching
- reports how often last-move reconstruction succeeded versus fell back
- note that last-move-filter match counts are intentionally lower by design,
  because the filter only searches certificates whose carrier contains the
  reconstructed last move

### Current interpretation of the processed-shard benchmark

Important caveat:

- the processed-shard benchmark is useful for measuring matcher cost on real
  boards
- it is **not** the place where sequence-order optimization should actually be
  applied in production

Reason:

- shuffled processed shards no longer preserve game-sequence order
- using the reconstructed last move on a shuffled example is only a benchmark
  trick for estimating how much work the `must_include_cell` path removes
- if sequence-order acceleration is adopted for real label generation, it needs
  to happen earlier, while positions are still processed in game order

### Current recommendation on label generation timing

Given the current cost profile, the working recommendation is:

- do **not** plan on computing ladder labels on demand during every training run
- prefer pre-generating ladder labels before the later striping/shuffling stages
- if the last-move filter is used as a real optimization, apply it while
  iterating positions in game order during preprocessing or self-play export

The current measured cost is still too high to casually recompute for large
training runs:

- rough order of magnitude: milliseconds per position, not microseconds
- at million-position scale that becomes hours of preprocessing if done naively

## Ordered-game sidecar path

The repo now has a direct backfill path over complete `.trmph` game records:

- `scripts/backfill_ladder_certificate_sidecars.py`

This path:

- scans `.trmph` files
- runs the exact ordered-game incremental labeler
- writes one JSONL sidecar per source file:
  - `<basename>.ladder_certificates.jsonl`
- stores compressed dense arrays for:
  - `template_origin` maps
  - carrier maps

Default file semantics intentionally mirror the current training-preprocessing
contract:

- parseable TRMPH lines are considered
- winnerless lines are skipped unless `--include-winnerless` is passed

Example:

```bash
source hex_ai_env/bin/activate
python scripts/backfill_ladder_certificate_sidecars.py \
  --data-dir data/sf25 \
  --max-files 10
```

## On-demand targets vs pre-generated targets

There are two distinct integration choices:

1. generate ladder targets on demand at the dataset / trainer boundary
2. pre-generate ladder targets earlier and treat them as part of the processed
   training data

The current recommendation is:

- use the on-demand path only for:
  - prototyping
  - small-scale experiments
  - correctness checks
  - benchmark measurements
  - backfill / regeneration tooling
- treat pre-generation as the expected production path for real training runs

Reasoning:

- the on-demand path is useful because it is simple and keeps experimentation
  local
- but the measured matcher cost is too high to assume it belongs in the hot
  training loop for large runs
- pre-generation moves that cost out of repeated training execution
- pre-generation also gives a natural place to exploit game-order information
  and last-move filtering correctly

This also answers the sequencing question of “dataset feature now, or aux head
first?”:

- do **not** start by wiring the auxiliary head to expensive on-demand label
  computation in the main training loop
- first decide how ladder labels will be pre-generated and stored
- then expose those labels through the dataset boundary
- then add the aux head and loss against those stable targets

So the current intended role of `hex_ai/utils/ladder_templates/data_targets.py`
is:

- debug / prototype target generation
- benchmark support
- future regeneration and migration tooling

It remains useful even if the final production path becomes fully
pre-generated.

The matcher also returns per-call stats:

- elapsed milliseconds
- templates considered
- embeddings considered
- matches found

## Next likely steps

1. Decide the pre-generation insertion point.
2. Decide whether the current TRMPH-sidecar format should remain an
   intermediate/backfill format or also become the canonical preprocessing
   source.
3. Decide how shuffled processed data should carry or derive aligned ladder
   sidecars.
4. Decide and encode true anchor semantics.
5. Add a small synthetic dataset generator for held-out supervision tests.
6. Add the auxiliary head and loss once target semantics are stable enough.

For the dated handoff / recommended next-coding plan, see:

- `write_ups/ladder_certificate_handoff_2026-03-25.md`
