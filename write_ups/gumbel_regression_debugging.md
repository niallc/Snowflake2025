# Gumbel MCTS Regression Debugging

## Summary

This document chronicles the debugging of a critical performance regression in the Gumbel-AlphaZero MCTS implementation. The regression reduced Gumbel MCTS win rate from ~83% to ~50% against the policy network, essentially making it equivalent to random play.

## Timeline and Key Commits

### Working Version
- **Commit**: `950eec2bf09f43d585a73769e06c3d8cdee65746` (tagged as `last_gumbel_working`)
- **Date**: August 31, 2025
- **Performance**: ~83% win rate against policy network
- **Files Changed**: Only `scripts/run_deterministic_tournament.py` (timing statistics)

### Broken Version
- **Commit**: `9ad04e1ceeb5d6e40eec1758c5e6f2f8504e62a5` (tagged as `first_gumbel_broken`)
- **Date**: August 31, 2025 (40 minutes later)
- **Performance**: ~50% win rate against policy network (essentially random)
- **Files Changed**: `hex_ai/inference/mcts.py`, `hex_ai/utils/gumbel_utils.py`, `scripts/run_deterministic_tournament.py`

### Fix Attempts
1. **Attempt 1**: `14d18182a17f9d43e7fe08126aab6eb7787757e4` - Failed
2. **Attempt 2**: `8ac4d8f2915bab98131c35761014fc4d56e301fe` - Failed  
3. **Attempt 3**: `f1214aa1a1db19973768230fb57d161ad9865aa4` - **SUCCESS** ✅

## Investigation Process

### Initial Binary Search

The debugging began with a systematic binary search through commits to isolate the regression:

1. **Identified the gap**: Between `last_gumbel_working` and `first_gumbel_broken`
2. **Saved the diff**: `git diff last_gumbel_working first_gumbel_broken > temp/gumbel_regression_diff.patch`
3. **Created debug branch**: `git checkout -b debug-gumbel-regression last_gumbel_working`

### Early Investigation: Dictionary Optimization

The initial analysis of the diff revealed a **dictionary optimization** in `mcts.py` that was suspected to be the culprit:

```python
# BROKEN VERSION (from the diff):
try:
    # OPTIMIZATION: Use dictionary lookup instead of linear search
    if not hasattr(node, '_legal_indices_dict'):
        node._legal_indices_dict = {idx: i for i, idx in enumerate(node.legal_indices)}
    loc_idx = node._legal_indices_dict[forced_a_full]
except (KeyError, AttributeError):
    # Fallback if forced action is illegal: normal PUCT
    loc_idx = self._select_child_puct(node)

# WORKING VERSION (original):
try:
    # Fast path: vectorized search
    li = np.asarray(node.legal_indices)
    loc_idx = int(np.where(li == forced_a_full)[0][0])
except Exception:
    # Fallback if forced action is illegal: normal PUCT
    loc_idx = self._select_child_puct(node)
```

**Initial Hypothesis**: The dictionary optimization was modifying node state in a way that interfered with MCTS.

**Testing**: Applied the dictionary optimization change in isolation - **confirmed it caused the regression**.

**Why it failed**: The dictionary optimization was adding `_legal_indices_dict` attributes to MCTS nodes, which may have interfered with node reuse or state management in the MCTS tree.

### The Real Culprit: Temperature Cutoff

However, the dictionary optimization was just **one symptom** of a larger issue. The real problem was the **shared temperature cutoff** that prevented Gumbel from running at all.

## Root Cause Analysis

### The Original Problem

The regression was caused by a **shared temperature cutoff** that prevented the Gumbel algorithm from running. The code had:

```python
# Line 1084 in mcts.py
if tau <= self.cfg.temperature_deterministic_cutoff:  # 0.02
    # Pick argmax over priors among legal actions
    selected_tensor_action = int(np.argmax(np.where(legal_mask, priors_full, -np.inf)))
    return timing_tracker.get_final_stats()  # Early return, skips Gumbel!
```

**The Issue**: Even when `use_gumbel=True`, the algorithm was taking the "deterministic" path and just picking the highest policy probability move, completely bypassing the Gumbel algorithm.

### Why This Happened

1. **Tournament Configuration**: The tournament was configured with `temperature=0.0` for deterministic play
2. **Shared Cutoff**: Both vanilla MCTS and Gumbel used the same `temperature_deterministic_cutoff = 0.02`
3. **Early Return**: When `tau=0.0 <= 0.02`, the code returned early with just policy argmax
4. **No Gumbel Execution**: The `gumbel_alpha_zero_root_batched()` function was never called

### Evidence of the Problem

- **Win Rate**: Dropped from 83% to 50% (random performance)
- **Duplicate Games**: 20/30 games were identical between policy and Gumbel MCTS
- **Debug Output**: `use_gumbel=True` but no Gumbel function calls
- **Performance**: Fast execution (155s vs 1051s) but poor results

## The Fix

### Solution: Separate Temperature Cutoffs

The fix involved creating separate temperature cutoffs for Gumbel vs vanilla MCTS:

```python
# Added to BaselineMCTSConfig
gumbel_temperature_deterministic_cutoff: float = -1.0  # TODO: TEMPORARY - Allow Gumbel to always run

# Updated Gumbel path
if tau <= self.cfg.gumbel_temperature_deterministic_cutoff:  # -1.0
    # Deterministic path (now never taken)
    return timing_tracker.get_final_stats()

# Vanilla MCTS path (unchanged)
if temp <= self.cfg.temperature_deterministic_cutoff:  # 0.02
    # Vanilla MCTS deterministic path
```

### Additional Fix: Temperature Validation

The Gumbel function also needed to allow `temperature=0.0`:

```python
# Before
if temperature <= 0:
    raise ValueError(f"temperature must be positive, got {temperature}")

# After  
if temperature < 0:
    raise ValueError(f"temperature must be non-negative, got {temperature}")
```

## Technical Details

### How Gumbel-AlphaZero Works

The Gumbel algorithm uses:
1. **Gumbel-Top-k sampling** to select candidate actions
2. **Sequential Halving** to allocate simulations efficiently
3. **Q-value updates** between rounds for ranking

The critical insight was that **Sequential Halving requires intermediate MCTS calls** to update Q-values used for ranking between rounds.

### The Batching Optimization Issue

An intermediate "fix" attempted to batch all MCTS calls together:

```python
# BROKEN: Collect all actions, call MCTS once
all_actions = []
for r in range(R):
    all_actions.extend(actions_this_round)
# Execute all MCTS simulations in one call
stats = mcts.run_forced_root_actions(root, all_actions, verbose=0)
```

**Problem**: This broke Sequential Halving because Q-values weren't updated between rounds, making ranking equivalent to policy-only decisions.

### The Working Approach

```python
# WORKING: Call MCTS after each round
for r in range(R):
    actions_this_round = schedule_round(...)
    if actions_this_round:
        stats = mcts.run_forced_root_actions(root, actions_this_round, verbose=0)
        # Q-values are updated here for next round's ranking
    # Rank by updated Q-values
    cand.sort(key=rank_key, reverse=True)
```

## Performance Results

### Before Fix (Broken)
```
Win Rates:
  policy: 50.0% (30/60 games)
  mcts_80_b8_gumbel: 50.0% (30/60 games)
```
- 20/30 games were identical (duplicate games)
- Fast execution but poor performance

### After Fix (SUCCESS!)
```
Win Rates:
  mcts_80_b8_gumbel: 86.7% (52/60 games)
  policy: 13.3% (8/60 games)
```
- Gumbel algorithm actually running ✅
- Q-values being updated ✅
- Proper Sequential Halving with 5 rounds ✅
- **Performance restored and improved** ✅ (86.7% vs original 83%)
- **No duplicate games** ✅

## Key Lessons

1. **Shared Configuration Can Be Dangerous**: Using the same temperature cutoff for different algorithms can cause unexpected interactions

2. **Early Returns Hide Bugs**: The deterministic cutoff was masking the fact that Gumbel wasn't running

3. **Algorithm Dependencies Matter**: Sequential Halving requires intermediate Q-value updates, not just final batching

4. **Debug Output Is Crucial**: Without debug statements, it's nearly impossible to tell if an algorithm is actually running

5. **Temperature = 0.0 Is Valid**: Deterministic play should still run the algorithm, just without randomness

## Performance Considerations

### Current Status
- **Functionality**: ✅ Gumbel algorithm working correctly
- **Performance**: ⚠️ Slower than original (but still functional)
- **Win Rate**: ✅ 86.7% (improved from original 83%)

### Performance Issues
The current implementation is slower than the original because:
1. **Per-round MCTS calls**: Instead of batching all actions together
2. **Debug overhead**: Temporary debug statements add computational cost
3. **Dictionary optimization removed**: Reverted to vectorized search for stability

### Future Improvements

1. **Clean Up Debug Statements**: Remove temporary debug output after confirming stability
2. **Proper Temperature Control**: Implement proper Gumbel temperature scaling instead of the temporary -1.0 cutoff
3. **Performance Optimization**: 
   - Re-implement safe dictionary optimization (without node state modification)
   - Optimize the per-round MCTS calls while preserving Sequential Halving
   - Consider hybrid approaches that batch within rounds but call MCTS between rounds
4. **Documentation**: Add clear comments about the separate temperature cutoffs

## Code References

- **Working Commit**: `950eec2bf09f43d585a73769e06c3d8cdee65746`
- **Broken Commit**: `9ad04e1ceeb5d6e40eec1758c5e6f2f8504e62a5`
- **Fixed Commit**: `f1214aa1a1db19973768230fb57d161ad9865aa4`
- **Key Files**: `hex_ai/inference/mcts.py`, `hex_ai/utils/gumbel_utils.py`

## Conclusion

The regression was caused by a seemingly innocent shared configuration that prevented the Gumbel algorithm from running at all. The fix required understanding both the algorithm's requirements (Sequential Halving needs intermediate updates) and the configuration system (separate cutoffs for different algorithms). The debugging process involved systematic binary search through commits, careful analysis of the algorithm flow, and extensive debug output to trace execution.
