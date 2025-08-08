# MCTS Batch Inference Optimization Plan

## Overview

This document outlines a concrete plan to optimize the batched MCTS inference code, based on GPT-5's suggestions and analysis of the current implementation. The goal is to systematically identify and fix performance bottlenecks while maintaining code quality and correctness.

## Current State Analysis

### Identified Bottlenecks (from code TODOs and GPT-5 analysis)

1. **State Cloning Overhead** (CRITICAL)
   - `HexGameState.make_move()` creates new objects via `copy.deepcopy()`
   - Estimated 10-20x slowdown vs optimized approach
   - Location: `hex_ai/inference/game_engine.py:108-133`

2. **Tree Traversal Inefficiency**
   - `_select_child_with_puct()` uses Python loops over dicts
   - No vectorization of UCT calculations
   - Location: `hex_ai/inference/batched_mcts.py:280-320`

3. **Batch Underutilization**
   - Current batching may not fill batches optimally
   - Small model calls reduce GPU efficiency
   - Location: `hex_ai/inference/batch_processor.py`

4. **Tensor Allocation Overhead**
   - New tensors created for each evaluation
   - No pre-allocation of input buffers
   - Location: `hex_ai/inference/simple_model_inference.py`

5. **Backpropagation Inefficiency**
   - Python loops with repeated attribute lookups
   - No inlining of hot paths
   - Location: `hex_ai/inference/batched_mcts.py:450-470`

## Phase 1: Organize for Profiling (Low Risk, Immediate Payoff)

### 1.1 Performance Monitoring Infrastructure ✅ COMPLETED
- **Status**: Implemented `hex_ai/utils/perf.py`
- **Key Features**:
  - Thread-safe timing and counters
  - Rate-limited logging to avoid console spam
  - JSON output for easy parsing
  - Context managers for clean instrumentation

### 1.2 Instrument Key Boundaries
**Priority**: HIGH - Implement immediately

Add performance instrumentation to:
- `BatchedNeuralMCTS.search()` - Overall search timing
- `_execute_selection_phase()` - Selection timing
- `_expand_and_evaluate()` - Expansion timing  
- `BatchProcessor.process_batch()` - NN inference timing
- `_backpropagate()` - Backpropagation timing

**Implementation**:
```python
from hex_ai.utils.perf import PERF

# In search method
with PERF.timer("mcts.search"):
    # existing search code

# In selection
with PERF.timer("mcts.select"):
    leaf_node = self._select(root)

# In expansion
with PERF.timer("mcts.expand"):
    self._expand_and_evaluate(leaf_node)

# In batch processing
with PERF.timer("nn.infer"):
    # actual model inference

# Counters
PERF.inc("mcts.sim")  # per simulation
PERF.inc("nn.batch")  # per batch
```

### 1.3 Unified Evaluator Interface
**Priority**: HIGH - Implement next

Create `BatchedEvaluator` class to centralize all NN calls:
- Single choke point for all evaluations
- Consistent timing and batching
- Pre-allocated input tensors
- Background thread for queue management

**Location**: `hex_ai/inference/batched_evaluator.py`

### 1.4 Sanity Checks and Model Setup
**Priority**: MEDIUM - Implement after evaluator

- Ensure model is in eval mode once
- Set `torch.inference_mode()` globally
- Pin device and record metadata
- Validate tensor shapes and dtypes

## Phase 2: Iterative Profiling Loop

### 2.1 Baseline Profiling
**Priority**: HIGH - Run immediately after Phase 1

Run profiling on current codebase:
```bash
# CPU profiling
python -m cProfile -o prof.out scripts/run_tournament.py --one-move
python -c "import pstats; s=pstats.Stats('prof.out'); s.sort_stats('cumtime').print_stats(30)"

# Per-move profiling (using PERF)
# Run several moves and analyze PERF.snapshot() output
```

**Expected Outcomes**:
- Identify if selection, expansion, or NN dominates
- Measure batch utilization (avg_batch_size)
- Quantify deepcopy overhead

### 2.2 Performance Analysis Framework
**Priority**: MEDIUM - Implement after baseline

Create analysis scripts to:
- Parse PERF JSON output
- Generate performance reports
- Track optimization progress
- Compare before/after metrics

## Phase 3: Targeted Optimizations (In Order of Impact)

### 3.1 Replace Deepcopy with Apply/Undo Pattern ⭐ CRITICAL
**Priority**: CRITICAL - Highest impact optimization
**Estimated Gain**: 10-20x speedup

**Implementation Plan**:
1. Add `apply_move()` method to `HexGameState`
   - Mutate board in place
   - Update Union-Find incrementally
   - Push minimal inverse operations to stack

2. Add `undo_last()` method
   - Pop and revert operations
   - Restore previous state

3. Update MCTS to use apply/undo pattern
   - For encoding: apply → encode → undo
   - For child creation: use fast_copy() when needed

**Location**: `hex_ai/inference/game_engine.py`

### 3.2 Vectorize Child Statistics and UCT
**Priority**: HIGH - Significant impact
**Estimated Gain**: 2-5x speedup in selection

**Implementation Plan**:
1. Store child stats in NumPy arrays
   - `N`, `W`, `P`, `Q` arrays sized for max legal moves
   - Align with legal_moves for direct indexing

2. Vectorize UCT calculation
   - Compute all U values at once
   - Use `np.argmax()` for selection
   - Cache `sqrt(total_visits)` per call

3. Optimize backpropagation
   - Inline loops, use local variables
   - Avoid repeated attribute lookups

**Location**: `hex_ai/inference/batched_mcts.py`

### 3.3 Optimize Batch Utilization
**Priority**: HIGH - GPU efficiency
**Estimated Gain**: 1.5-3x speedup

**Implementation Plan**:
1. Tune batch collection parameters
   - Adjust `max_wait_ms` (1-5ms)
   - Pre-seed rollouts to fill queue
   - Monitor avg_batch_size vs target

2. Pre-allocate input tensors
   - Large tensor pool for batching
   - Write into views when stacking
   - Keep on CPU until batching

3. Optimize device transfers
   - Batch CPU→GPU transfers
   - Avoid per-state `.cpu()` calls

**Location**: `hex_ai/inference/batch_processor.py`

### 3.4 Device Performance Bake-off
**Priority**: MEDIUM - Platform dependent
**Estimated Gain**: 0.5-2x speedup

**Implementation Plan**:
1. Instrument device performance
   - CPU vs MPS vs CUDA comparison
   - Same batch tensor, different devices
   - Measure throughput and latency

2. Optimize for winning device
   - Keep model and inputs on device
   - Use `torch.inference_mode()`
   - Avoid sync points in loops

**Location**: `hex_ai/inference/simple_model_inference.py`

## Phase 4: Advanced Optimizations

### 4.1 Memory Pool Management
**Priority**: LOW - After core optimizations
- Pre-allocate node objects
- Reuse tensor buffers
- Minimize garbage collection

### 4.2 Parallel Tree Traversal
**Priority**: LOW - Complex, high risk
- Multi-threaded MCTS
- Lock-free data structures
- Requires careful synchronization

## Implementation Timeline

### Week 1: Foundation
- [x] Performance monitoring infrastructure
- [ ] Instrument key boundaries
- [ ] Baseline profiling
- [ ] Unified evaluator interface

### Week 2: Critical Optimizations
- [ ] Apply/undo pattern implementation
- [ ] Vectorized UCT calculations
- [ ] Performance measurement and validation

### Week 3: Batch and Device Optimization
- [ ] Batch utilization tuning
- [ ] Device performance bake-off
- [ ] Pre-allocated tensor management

### Week 4: Integration and Testing
- [ ] End-to-end performance testing
- [ ] Regression testing
- [ ] Documentation and cleanup

## Success Metrics

### Performance Targets
- **Simulations per second**: 2-5x improvement
- **Batch utilization**: >80% of optimal batch size
- **Memory usage**: <50% increase
- **Latency**: <20% increase for single moves

### Code Quality Targets
- **Test coverage**: Maintain >90%
- **Documentation**: Update all changed interfaces
- **Backward compatibility**: Maintain existing APIs

## Risk Mitigation

### High-Risk Changes
1. **Apply/undo pattern**: Complex state management
   - Mitigation: Extensive testing, gradual rollout
   - Fallback: Keep deepcopy as option

2. **Vectorized UCT**: Potential for bugs
   - Mitigation: Unit tests for all edge cases
   - Validation: Compare results with original

### Medium-Risk Changes
1. **Batch optimization**: May affect latency
   - Mitigation: Configurable parameters
   - Monitoring: Track both throughput and latency

## Monitoring and Validation

### Continuous Monitoring
- Performance regression tests
- Automated profiling on CI
- Alert on significant performance drops

### Validation Strategy
- Compare move quality before/after
- A/B testing in tournaments
- Gradual rollout with monitoring

## Conclusion

This plan provides a systematic approach to MCTS optimization, starting with low-risk profiling infrastructure and progressing to high-impact optimizations. The apply/undo pattern is expected to provide the largest performance gain, while the profiling infrastructure will enable data-driven optimization decisions.

The phased approach minimizes risk while maximizing learning and validation opportunities. Each phase builds on the previous one, ensuring that optimizations are based on actual performance data rather than assumptions.
