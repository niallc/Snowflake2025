# Value Head Specification

**Date:** 2024-12-19  
**Version:** 1.0

## Overview

The value head is a component of the two-headed ResNet architecture that predicts the probability of Red winning from a given board position.

## Architecture

- **Input**: 512-dimensional feature vector from the ResNet backbone (after global average pooling)
- **Output**: Single raw logit (unbounded value)
- **Activation**: Sigmoid is applied during training and inference to convert logit to probability

```python
# Architecture
self.value_head = nn.Linear(512, 1)  # Single linear layer
value_logit = self.value_head(features)  # Raw logit output
value_prob = torch.sigmoid(value_logit)  # Probability in [0, 1]
```

## Training

### Target Labels
- **Red wins**: Target = `1.0`
- **Blue wins**: Target = `0.0`

### Loss Function
```python
# MSE loss on sigmoid probabilities
value_loss = MSE(sigmoid(value_logit), target)
```

### Rationale
The value head predicts Red's win probability because:
1. Red wins are labeled as `1.0` in the training data
2. This creates a direct mapping: high logit → high probability → Red likely wins
3. Blue's win probability can be computed as `1.0 - red_win_probability`

## Inference

### Raw Output
The model outputs a raw logit that must be converted to a probability:
```python
value_logit = model(board)[1]  # Raw logit (unbounded)
red_win_prob = torch.sigmoid(value_logit)  # Probability Red wins
blue_win_prob = 1.0 - red_win_prob  # Probability Blue wins
```

### Interpretation
- **High positive logit** → High probability Red wins
- **Low negative logit** → Low probability Red wins (high probability Blue wins)
- **Logit near 0** → ~50% probability for both players

## Usage Examples

### Basic Inference
```python
policy_logits, value_logit = model(board)
red_win_prob = torch.sigmoid(value_logit).item()
blue_win_prob = 1.0 - red_win_prob
```

### Using Utility Functions
```python
from hex_ai.value_utils import get_win_prob_from_model_output, Winner

# Get probability for specific player
red_prob = get_win_prob_from_model_output(value_logit, Winner.RED)
blue_prob = get_win_prob_from_model_output(value_logit, Winner.BLUE)
```

### Minimax Search
```python
from hex_ai.value_utils import ValuePredictor

# Convert to minimax-friendly value from root player's reference frame
minimax_value = ValuePredictor.convert_to_minimax_value(value_signed, root_player)
```

## Consistency Rules

1. **Always apply sigmoid** to convert logits to probabilities
2. **Red's win probability** is the primary output (sigmoid(logit))
3. **Blue's win probability** is derived (1.0 - red_win_prob)
4. **Training targets** are 0.0 for Blue wins, 1.0 for Red wins
5. **All code** should be consistent with this interpretation

## Common Pitfalls

1. **Using raw logits directly** - Always apply sigmoid for probabilities
2. **Inconsistent interpretation** - The value head always predicts Red's win probability
3. **Forgetting the conversion** - Blue's probability is 1.0 - Red's probability
4. **Mixing perspectives** - Be clear about which player's perspective you're using

## Testing

Test cases should verify:
- Empty board gives ~50% probability for both players
- Clear winning positions give high confidence (>90%) for the winning player
- Training and inference are consistent (same sigmoid application)
- Utility functions return correct probabilities for both players

## Future Improvements

### Architecture Enhancements

The current value head uses a single linear layer (512 → 1), which follows AlphaZero/Leela patterns. For potentially better performance, consider upgrading to a KataGo-style architecture:

```python
# Current architecture (AlphaZero-style)
self.value_head = nn.Linear(512, 1)

# Proposed KataGo-style architecture
self.value_head = nn.Sequential(
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 1)
)
```

**Benefits:**
- More capacity to learn complex value patterns
- Dropout for regularization
- Could help resolve value head performance issues

**Trade-offs:**
- Slightly slower training/inference
- More hyperparameters to tune
- More prone to overfitting

**When to consider:**
- If current value head performance is insufficient
- After resolving current value head debugging issues
- When moving to larger datasets or more complex training regimes 