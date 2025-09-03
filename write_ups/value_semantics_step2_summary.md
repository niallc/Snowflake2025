# Value-Semantics Migration: Step 2 Summary

## Overview
Successfully switched MCTS internals to use **signed values in [-1, 1]** end-to-end while keeping probability conversions only at the edges. This step transforms the core MCTS pipeline to use the canonical AlphaZero-style signed value convention throughout the search tree.

## Key Accomplishments

### 1. Core MCTS Pipeline Transformed to Signed Values

#### Neural Network Interface
- **`_get_neural_network_value()`**: Now returns signed values directly from model output
- **`_get_terminal_value()`**: Now returns signed values (+1.0 for Red win, -1.0 for Blue win)
- **Cache storage**: Model's `tanh` values stored as signed values in [-1,1] range

#### Backpropagation Pipeline
- **`_backpropagate_path()`**: Now works with signed values throughout
- **Perspective conversion**: Uses `signed_for_player_to_move()` to flip to player-to-move perspective
- **W/Q values**: Maintained as signed averages in [-1,1] range
- **Depth discounting**: Applied in signed space using `apply_depth_discount_signed()` (shrink toward 0)

### 2. Edge-Only Probability Conversions

#### Confidence Termination
- **`_get_root_win_probability()`**: Converts signed to probability only for confidence termination
- **Edge conversion**: Uses `signed_for_player_to_move()` + `to_prob()` at the boundary

#### External API
- **`_compute_win_probability()`**: Converts signed root value to probability for external consumption
- **UI compatibility**: Maintains probability interface for external callers

### 3. Depth Discounting Improvements

#### Signed Space Discounting
- **Before**: `v_node *= gamma^depth` (drags values toward 0)
- **After**: `apply_depth_discount_signed(v_node, gamma, depth)` (shrink toward 0)
- **Benefit**: Avoids biases from direct multiplication, maintains proper neutral point

#### Distance Metric
- Added TODO for future `distance_to_leaf` implementation
- Current implementation uses absolute depth (functional but could be optimized)

### 4. Hyperparameter Tuning Preparation

#### TODO Markers Added
- **PUCT tuning**: `TODO(step: tune): retune c_puct for signed Q values in [-1,1] range`
- **Depth discounting**: `TODO(step: tune): retune depth_discount_factor for signed space`
- **Distance metric**: `TODO(step: tune): consider using distance_to_leaf instead of absolute depth`

## Technical Details

### Value Flow Transformation

#### Before (Probability-Based)
```
Model Output (tanh) → Probability [0,1] → Backprop → W/Q [0,1] → Selection
```

#### After (Signed-Based)
```
Model Output (tanh) → Signed [-1,1] → Backprop → W/Q [-1,1] → Selection
```

### Edge Conversions
- **Internal**: All values in signed space [-1,1]
- **External**: Convert to probability [0,1] only at API boundaries
- **Confidence termination**: Convert signed to probability for threshold comparison
- **UI/display**: Convert signed to probability for user-facing interfaces

### Perspective Handling
- **Red-centric**: Model outputs Red's perspective by default
- **Player-to-move**: Convert to player-to-move perspective using `signed_for_player_to_move()`
- **Consistent**: All internal operations use player-to-move perspective

## Files Modified

### Primary Changes
- `hex_ai/inference/mcts.py`: Complete transformation to signed value pipeline

### Key Method Updates
- `_get_terminal_value()`: Returns signed values (+1.0/-1.0)
- `_get_neural_network_value()`: Returns signed values directly
- `_backpropagate_path()`: Works with signed values throughout
- `_get_root_win_probability()`: Edge conversion for confidence termination
- `_compute_win_probability()`: Edge conversion for external API

## Verification

### Test Results
- ✅ All signed value conversions work correctly
- ✅ Edge conversions maintain proper probability semantics
- ✅ Depth discounting works correctly in signed space
- ✅ MCTS signed value flow produces expected results
- ✅ Equivalence with old flow verified for depth 0 cases

### Behavior Preservation
- **External API**: No changes to public interface
- **Probability outputs**: Maintained for UI and external consumers
- **Search behavior**: Core algorithm logic preserved
- **Performance**: No significant performance impact

## Benefits Achieved

### 1. **Canonical Value Representation**
- Consistent signed value convention throughout MCTS
- Eliminates confusion between probability and signed spaces
- Aligns with AlphaZero-style implementations

### 2. **Improved Mathematical Properties**
- Avoids biases from probability multiplication
- Proper neutral point handling (0.0 in signed space)
- More intuitive value semantics

### 3. **Cleaner Architecture**
- Clear separation between internal signed values and external probabilities
- Centralized edge conversions
- Reduced cognitive load for developers

### 4. **Future-Ready**
- Foundation for hyperparameter tuning
- Prepared for distance-to-leaf optimization
- Ready for advanced MCTS improvements

## Next Steps (Future Tuning)

The signed value foundation is now in place for Step 3, which will involve:

1. **Hyperparameter Tuning**
   - Retune `c_puct` for signed Q values in [-1,1] range
   - Optimize `depth_discount_factor` for signed space
   - Consider `distance_to_leaf` vs absolute depth

2. **Performance Optimization**
   - Evaluate impact of signed value operations
   - Optimize edge conversions if needed
   - Consider caching strategies for conversions

3. **Advanced Features**
   - Implement distance-to-leaf discounting
   - Add signed value visualization tools
   - Consider signed value training targets

## Acceptance Criteria Met

✅ MCTS internals use signed values in [-1,1] end-to-end  
✅ Cache stores model's `tanh` values as signed  
✅ Backpropagate signed values with perspective correction  
✅ Maintain W/Q as signed averages in [-1,1]  
✅ Apply depth discount in signed space (shrink toward 0)  
✅ Probability conversions only at edges  
✅ TODO markers added for hyperparameter tuning  
✅ External API compatibility maintained  

## Impact Assessment

### Performance
- **Minimal impact**: Signed value operations are computationally equivalent
- **Memory usage**: Unchanged (same data types)
- **Cache efficiency**: Improved (no conversion overhead in cache)

### Compatibility
- **External APIs**: Fully compatible (edge conversions maintain interface)
- **UI/display**: No changes required
- **Training pipeline**: Unaffected (model outputs unchanged)

### Maintainability
- **Code clarity**: Significantly improved
- **Debugging**: Easier with consistent value semantics
- **Future development**: Cleaner foundation for improvements

The MCTS module now uses the canonical signed value convention throughout its internal operations while maintaining full compatibility with existing external interfaces. The foundation is ready for the next phase of hyperparameter tuning and optimization.
