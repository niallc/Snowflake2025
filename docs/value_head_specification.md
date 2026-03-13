# Value Head Specification

**Last updated:** 2026-03-13

## Overview

The value head predicts a **signed value** in `[-1, 1]` from the board state.

- `+1` means certain **Red** win
- `-1` means certain **Blue** win
- `0` means neutral

The model output is `value_signed`, not a probability.

## Current runtime contract

The family-standard board API is:

- `model.forward_from_boards(boards, move_stage=None, batch_idx=None)`
- `model.forward_value_only_from_boards(boards, move_stage=None)`

The helper boundary used by training/inference is:

- `hex_ai.model_interface.forward_model(...)`
- `hex_ai.model_interface.forward_value_only_model(...)`

Forward returns:

- `policy_logits`: shape `(B, board_area)`
- `value_signed`: shape `(B, 1)`, tanh-activated, in `[-1, 1]`

For current models, `board_area = BOARD_SIZE * BOARD_SIZE` for the configured
runtime board size.

## Current architecture

The current value head in `hex_ai/models.py` uses:

- trunk feature preprocessing via `1x1 conv + BN + ReLU`
- global average pooling
- concatenation with normalized `move_stage`
- `LayerNorm`
- MLP
- `K` parallel scalar outputs with learned linear combination
- final `tanh`

Implementation reference:

- `/Users/niallHome/Documents/programming/Snowflake2025/hex_ai/models.py`

## Planned next-family change

The active network-design note recommends keeping the same value semantics while
upgrading the pooled features for the next family:

- replace GAP-only pooled features with `mean + max`
- keep `move_stage`
- keep the MLP + `K`-output combination pattern

See:

- `/Users/niallHome/Documents/programming/Snowflake2025/write_ups/Current_Network_vs_KataGo_Gumbel_2026-03-13.md`
- `/Users/niallHome/Documents/programming/Snowflake2025/write_ups/Training_Architecture_Change_Discussion_2026-03-13.md`

## Training targets and loss

Training data stores winner targets in probability space:

- Blue win = `0.0`
- Red win = `1.0`

For training, targets are converted to signed space:

- `value_signed_target = 2.0 * value_prob_target - 1.0`

Value loss is computed in signed space with label smoothing and log-cosh:

- `compute_value_loss(pred, target, smooth=0.95)`
- internally clamps targets to `[-0.95, 0.95]`
- uses numerically guarded log-cosh loss

Implementation references:

- `/Users/niallHome/Documents/programming/Snowflake2025/hex_ai/data_pipeline.py`
- `/Users/niallHome/Documents/programming/Snowflake2025/hex_ai/models.py`
- `/Users/niallHome/Documents/programming/Snowflake2025/hex_ai/training.py`

## Probability conversion

Convert signed output to Red win probability:

- `p_red = (value_signed + 1.0) / 2.0`

Use centralized helpers:

- `ValuePredictor.model_output_to_probability(...)`
- `ValuePredictor.get_win_probability(..., player)`
- `ValuePredictor.get_win_probability_for_winner(..., winner)`

Implementation reference:

- `/Users/niallHome/Documents/programming/Snowflake2025/hex_ai/value_utils.py`

## MCTS semantics

MCTS internally uses signed values and reference-frame transforms.

- signed values remain in `[-1,1]` for search logic
- convert to probability only at API/output boundaries

For current MCTS termination/output contracts, see:

- `/Users/niallHome/Documents/programming/Snowflake2025/docs/mcts_early_termination.md`

## Minimal examples

### Inference

```python
from hex_ai.model_interface import forward_model
from hex_ai.value_utils import ValuePredictor

policy_logits, value_signed = forward_model(model, board_batch)
p_red = ValuePredictor.model_output_to_probability(float(value_signed[0].item()))
```

### Player-specific probability

```python
from hex_ai.value_utils import ValuePredictor
from hex_ai.enums import Player

p_current = ValuePredictor.get_win_probability(float(value_signed), Player.RED)
```

### Root-player signed value

```python
from hex_ai.value_utils import ValuePredictor

root_ref_signed = ValuePredictor.convert_to_minimax_value(
    red_ref_signed=value_signed,
    root_player=root_player,
)
```

## Board-size note

The project direction is toward board-size-parameterized play/training, even
though some runtime/data components still assume the configured default board
size today.

For value-head work, prefer wording and interfaces that depend on:

- the runtime board tensor shape
- or the configured board size

Do not describe the value head as inherently `13x13`-only unless a specific
implementation path truly still is.

## Consistency rules

1. Treat model value output as signed `[-1,1]`, not as probability.
2. Use shared conversion helpers in `value_utils`.
3. Prefer `forward_from_boards(...)` or `model_interface.forward_model(...)`
   over calling family-specific `forward(...)` signatures directly.
4. Name variables explicitly (`value_signed`, `win_probability`) to avoid
   mixing semantics.
5. Fail fast on invalid ranges at interface boundaries.
