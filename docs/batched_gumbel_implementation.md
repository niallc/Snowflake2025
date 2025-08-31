# Batched Gumbel-AlphaZero Implementation

## Overview

This document describes the implementation of a batched Gumbel-AlphaZero root selection algorithm that reuses the existing MCTS batching infrastructure for maximum efficiency.

## Problem

The original Gumbel-AlphaZero implementation was inefficient because it called `run_one_sim()` in a loop, which didn't take advantage of the sophisticated batching of neural network calls in the rest of the codebase. This resulted in poor performance with small numbers of simulations.

## Solution

The new implementation makes Gumbel drive the existing batched leaf pipeline instead of calling `run_one_sim()` in a loop. This keeps all evaluation/expansion/backprop and caching exactly as-is while providing full batching benefits.

## Key Changes

### 1. Enhanced MCTS Selection (`hex_ai/inference/mcts.py`)

#### Modified `_select_leaves_batch` method
- Added optional `forced_root_actions` parameter
- When provided, forces specific root actions for each descent
- Maps full action indices to local child indices using vectorized search
- Falls back to normal PUCT if forced action is illegal

#### New `_run_forced_root_batch` method
- Runs exactly `len(actions)` simulations, forcing each root action once
- Uses the existing batched pipeline infrastructure

#### New `run_forced_root_actions` method
- Public entry-point for Gumbel root coordinator
- Respects `batch_cap` internally
- Processes actions in batches for optimal performance

### 2. New Batched Gumbel Function (`hex_ai/utils/gumbel_utils.py`)

#### `gumbel_alpha_zero_root_batched` function
- Replaces per-sim loop with batched forced action execution
- Builds per-round schedule of forced actions
- Calls `run_forced_root_actions` once per round
- Re-scores after each round using Sequential Halving

## Implementation Details

### Forced Root Action Selection

```python
# Inside _select_leaves_batch
if node is root and forced_a_full is not None:
    # Map full action index -> local child idx
    try:
        # Fast path: vectorized search
        li = np.asarray(node.legal_indices)
        loc_idx = int(np.where(li == forced_a_full)[0][0])
    except Exception:
        # Fallback if forced action is illegal: normal PUCT
        loc_idx = self._select_child_puct(node)
else:
    loc_idx = self._select_child_puct(node)
```

### Batched Gumbel Coordinator

```python
# Build the forced-action list for THIS ROUND and run in batches
actions_this_round = []
for a in cand:
    actions_this_round.extend([a] * per_arm)
rng.shuffle(actions_this_round)

if actions_this_round:
    mcts.run_forced_root_actions(root, actions_this_round, verbose=0)
    sims_used += len(actions_this_round)
```

## Benefits

### 1. DRY (Don't Repeat Yourself)
- Zero duplication of evaluation/expansion/backprop/caching
- Reuses `_select_leaves_batch` → `_prepare_leaf_evaluations` → `_run_neural_network_batch` → `_backpropagate_batch`
- Only one new codepath inside selection: single "if root and forced action" branch

### 2. Full Batching
- Each Gumbel round composes a list of forced actions and calls `run_forced_root_actions()` once
- Neural network sees big batches again
- Maintains the same batching performance as standard MCTS

### 3. Backward Compatibility
- Existing code works unchanged
- Can be enabled/disabled via configuration
- Same API as standard MCTS

## Usage

### Configuration

```python
from hex_ai.inference.mcts import BaselineMCTSConfig

config = BaselineMCTSConfig(
    sims=100,
    enable_gumbel_root_selection=True,
    gumbel_sim_threshold=200,  # Use Gumbel when sims <= this value
    gumbel_c_visit=50.0,       # Gumbel parameter
    gumbel_c_scale=1.0,        # Gumbel parameter
    gumbel_m_candidates=None   # Number of candidates (None for auto)
)
```

### Automatic Usage

The batched Gumbel implementation is automatically used when:
1. `enable_gumbel_root_selection=True`
2. `sims <= gumbel_sim_threshold`
3. Root node is expanded and non-terminal

### Manual Usage

```python
from hex_ai.utils.gumbel_utils import gumbel_alpha_zero_root_batched

selected_action = gumbel_alpha_zero_root_batched(
    mcts=mcts,
    root=root,
    policy_logits=policy_logits,
    total_sims=100,
    legal_actions=legal_actions,
    q_of_child=q_of_child,
    n_of_child=n_of_child,
    m=10,
    c_visit=50.0,
    c_scale=1.0
)
```

## Performance Characteristics

### Simulation Budgets
- **50-200 simulations**: Batched Gumbel typically more efficient
- **200+ simulations**: Standard MCTS often better
- **Automatic threshold**: Configurable via `gumbel_sim_threshold`

### Computational Complexity
- **Batched Gumbel**: O(m log m) where m is number of candidates
- **Standard MCTS**: O(n log n) where n is number of simulations
- **Memory usage**: Similar to standard MCTS

## Testing

The implementation includes comprehensive tests that verify:
- Correct action selection
- Legal action validation
- Performance characteristics
- Integration with existing MCTS infrastructure

## Gotchas and Considerations

1. **Dirichlet root noise**: Apply before running Gumbel (when first expanding root)
2. **Gumbel vector consistency**: Same Gumbel vector `g` used for candidate selection and final ranking
3. **Illegal action fallback**: If forced action becomes illegal, falls back to normal PUCT
4. **Stability at tiny budgets**: Can set `m = min(#legal, total_sims)` for ~1 sim per arm in round 1

## Future Improvements

1. **Parameter tuning**: Optimize `gumbel_c_visit` and `gumbel_c_scale` for specific domains
2. **Adaptive thresholds**: Dynamic adjustment of `gumbel_sim_threshold` based on position complexity
3. **Performance monitoring**: Add detailed performance metrics for Gumbel vs standard MCTS
4. **Parallelization**: Extend to support parallel MCTS with proper synchronization

## References

- Danihelka, J., et al. "Gumbel AlphaZero." arXiv preprint arXiv:2202.13545 (2022)
- Original AlphaZero paper: Silver, D., et al. "Mastering the game of Go without human knowledge." Nature 550.7676 (2017): 354-359
