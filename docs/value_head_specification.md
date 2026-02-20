# Value Head Specification (Current)

**Last updated:** 2026-02-16

## Overview

The value head predicts a **signed value** in `[-1, 1]` from the board state.

- `+1` means certain **Red** win
- `-1` means certain **Blue** win
- `0` means neutral

The model output is `value_signed` (not a probability). Convert to probability only when needed.

## Output Contract

Model forward returns:

- `policy_logits`: shape `(B, BOARD_SIZE*BOARD_SIZE)`
- `value_signed`: shape `(B, 1)`, tanh-activated, in `[-1, 1]`

## Architecture

Current architecture uses a KataGo-inspired value head:

- trunk feature preprocessing (`1x1 conv + BN + ReLU`)
- global average pooling
- concatenation with normalized move-stage feature (`[0,1]`)
- MLP
- `K` parallel scalar outputs + learned linear combination
- final `tanh` to enforce signed range

Implementation reference:

- `/Users/niallHome/Documents/programming/Snowflake2025/hex_ai/models.py`

## Training Targets and Loss

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

## Probability Conversion

Convert signed output to Red win probability:

- `p_red = (value_signed + 1.0) / 2.0`

Use centralized helpers:

- `ValuePredictor.model_output_to_probability(...)`
- `ValuePredictor.get_win_probability(..., player)`
- `ValuePredictor.get_win_probability_for_winner(..., winner)`

Implementation reference:

- `/Users/niallHome/Documents/programming/Snowflake2025/hex_ai/value_utils.py`

## MCTS Semantics

MCTS internally uses signed values and reference-frame transforms.

- signed values remain in `[-1,1]` for search logic
- convert to probability at API/output boundaries

For current MCTS termination/output contracts, see:

- `/Users/niallHome/Documents/programming/Snowflake2025/docs/mcts_early_termination.md`

## Minimal Examples

### Inference

```python
policy_logits, value_signed = model(board_batch)  # value_signed in [-1, 1]
p_red = ValuePredictor.model_output_to_probability(float(value_signed[0].item()))
```

### Player-specific probability

```python
from hex_ai.value_utils import ValuePredictor
from hex_ai.enums import Player

p_current = ValuePredictor.get_win_probability(float(value_signed), Player.RED)
```

### Minimax-style signed value from root-player perspective

```python
from hex_ai.value_utils import ValuePredictor

root_ref_signed = ValuePredictor.convert_to_minimax_value(red_ref_signed=value_signed, root_player=root_player)
```

## Rules to Keep Code Consistent

1. Treat model value output as signed `[-1,1]`, not as probability.
2. Use shared conversion helpers in `value_utils` (do not re-implement conversions ad hoc).
3. Name variables explicitly (`value_signed`, `win_probability`) to avoid mixing semantics.
4. Fail fast on invalid ranges in interfaces carrying probabilities.
