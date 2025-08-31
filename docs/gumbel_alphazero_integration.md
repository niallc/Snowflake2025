# Gumbel-AlphaZero Integration for MCTS

This document describes the integration of Gumbel-AlphaZero root selection into the MCTS implementation for improved efficiency with small simulation budgets.

## Overview

Gumbel-AlphaZero is a variant of AlphaZero that uses Gumbel-Top-k sampling and Sequential Halving for more efficient root action selection, particularly beneficial when using small numbers of simulations (50-500).

**Reference**: "Gumbel AlphaZero" by Danihelka et al. (2022)

## Key Benefits

- **Efficient for small simulation budgets**: Optimized for 50-500 simulations
- **Automatic fallback**: Uses standard MCTS for larger budgets
- **Configurable parameters**: Tunable for different use cases
- **Root-only modification**: Non-root nodes use standard PUCT selection

## Configuration

The Gumbel-AlphaZero integration is controlled by parameters in `BaselineMCTSConfig`:

```python
from hex_ai.inference.mcts import BaselineMCTSConfig

config = BaselineMCTSConfig(
    sims=100,  # Number of simulations
    enable_gumbel_root_selection=True,  # Enable Gumbel selection
    gumbel_sim_threshold=200,  # Use Gumbel when sims <= this value
    gumbel_c_visit=50.0,  # Gumbel parameter (default: 50.0)
    gumbel_c_scale=1.0,   # Gumbel parameter (default: 1.0)
    gumbel_m_candidates=None  # Number of candidates (None for auto)
)
```

### Parameters

- **`enable_gumbel_root_selection`**: Enable/disable Gumbel selection
- **`gumbel_sim_threshold`**: Maximum simulation count for using Gumbel (default: 200)
- **`gumbel_c_visit`**: Gumbel parameter controlling exploration (default: 50.0)
- **`gumbel_c_scale`**: Gumbel parameter controlling scaling (default: 1.0)
- **`gumbel_m_candidates`**: Number of candidate actions to consider (None for auto)

## Usage Examples

### Basic Usage

```python
from hex_ai.inference.mcts import BaselineMCTS, BaselineMCTSConfig
from hex_ai.inference.game_engine import HexGameEngine
from hex_ai.inference.model_wrapper import ModelWrapper

# Create configuration with Gumbel selection
config = BaselineMCTSConfig(
    sims=100,
    enable_gumbel_root_selection=True,
    gumbel_sim_threshold=200
)

# Create MCTS instance
engine = HexGameEngine()
model = ModelWrapper("path/to/checkpoint.pt")
mcts = BaselineMCTS(engine, model, config)

# Run search
state = engine.reset()
result = mcts.run(state)
print(f"Selected move: {result.move}")
```

### Comparison with Standard MCTS

```python
# Standard MCTS
config_standard = BaselineMCTSConfig(
    sims=100,
    enable_gumbel_root_selection=False
)

# Gumbel MCTS
config_gumbel = BaselineMCTSConfig(
    sims=100,
    enable_gumbel_root_selection=True,
    gumbel_sim_threshold=200
)

# Both will work, but Gumbel may be more efficient for small budgets
```

### Parameter Tuning

```python
# Conservative settings (more exploration)
config_conservative = BaselineMCTSConfig(
    sims=100,
    enable_gumbel_root_selection=True,
    gumbel_c_visit=100.0,  # Higher exploration
    gumbel_c_scale=1.0
)

# Aggressive settings (less exploration)
config_aggressive = BaselineMCTSConfig(
    sims=100,
    enable_gumbel_root_selection=True,
    gumbel_c_visit=25.0,   # Lower exploration
    gumbel_c_scale=0.5     # Less scaling
)

# Limited candidates
config_limited = BaselineMCTSConfig(
    sims=100,
    enable_gumbel_root_selection=True,
    gumbel_m_candidates=16  # Consider only top 16 actions
)
```

## Algorithm Details

### When Gumbel Selection is Used

Gumbel-AlphaZero root selection is used when:
1. `enable_gumbel_root_selection=True`
2. `sims <= gumbel_sim_threshold`
3. Root node is expanded and non-terminal

### How It Works

1. **Gumbel-Top-k Sampling**: Selects candidate actions using Gumbel noise
2. **Sequential Halving**: Allocates simulations efficiently among candidates
3. **Root-only**: Only affects root node selection; deeper nodes use PUCT
4. **Automatic Fallback**: Uses standard MCTS for larger simulation budgets

### Mathematical Foundation

The algorithm uses:
- **Gumbel distribution**: For sampling without replacement
- **Sequential Halving**: For efficient simulation allocation
- **Policy improvement**: Based on completed Q-values

## Performance Characteristics

### Simulation Budgets

- **50-200 simulations**: Gumbel selection typically more efficient
- **200+ simulations**: Standard MCTS often better
- **Automatic threshold**: Configurable via `gumbel_sim_threshold`

### Computational Complexity

- **Gumbel selection**: O(m log m) where m is number of candidates
- **Standard MCTS**: O(n log n) where n is number of simulations
- **Memory usage**: Similar to standard MCTS

## Integration with Existing Code

The Gumbel-AlphaZero integration is designed to be:

- **Backward compatible**: Existing code works unchanged
- **Modular**: Can be enabled/disabled per configuration
- **Transparent**: Same API as standard MCTS
- **Configurable**: Parameters can be tuned for specific use cases

## Best Practices

1. **Start with defaults**: Use recommended parameter values initially
2. **Tune for your domain**: Adjust `gumbel_sim_threshold` based on your simulation budget
3. **Monitor performance**: Compare with standard MCTS for your specific use case
4. **Consider candidate count**: Use `gumbel_m_candidates` to limit exploration if needed

## Troubleshooting

### Common Issues

1. **Gumbel not being used**: Check that `sims <= gumbel_sim_threshold`
2. **Poor performance**: Try adjusting `gumbel_c_visit` and `gumbel_c_scale`
3. **Too many candidates**: Set `gumbel_m_candidates` to a smaller value

### Debugging

Enable verbose output to see when Gumbel selection is used:

```python
result = mcts.run(state, verbose=1)
# Look for: "Using Gumbel-AlphaZero root selection for X simulations"
```

## References

- Danihelka, J., et al. "Gumbel AlphaZero." arXiv preprint arXiv:2202.13545 (2022)
- Original AlphaZero paper: Silver, D., et al. "Mastering the game of Go without human knowledge." Nature 550.7676 (2017): 354-359
