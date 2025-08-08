# MCTS Optimization Quick Start Guide

## Overview

This guide provides immediate steps to start MCTS optimization work using the new performance monitoring infrastructure.

## Prerequisites

1. **Environment Setup** (already done):
   ```bash
   source hex_ai_env/bin/activate
   export PYTHONPATH=.
   ```

2. **Performance Monitoring** (already implemented):
   - `hex_ai/utils/perf.py` - Performance monitoring utility
   - Tested and working: `python scripts/test_performance_monitoring.py`

## Immediate Next Steps

### Step 1: Add Performance Instrumentation (Phase 1.2)

Add timing instrumentation to key MCTS boundaries:

**File**: `hex_ai/inference/batched_mcts.py`

```python
# Add import at top
from hex_ai.utils.perf import PERF

# In search method (around line 200)
def search(self, root_state_or_node, num_simulations: int) -> BatchedMCTSNode:
    with PERF.timer("mcts.search"):
        # ... existing code ...
        
        # Run simulations
        for sim_idx in range(num_simulations):
            # Execute selection phase to get leaf node
            with PERF.timer("mcts.select"):
                leaf_node = self._execute_selection_phase(root)
            
            # Expand and evaluate the leaf node
            with PERF.timer("mcts.expand"):
                self._expand_and_evaluate(leaf_node)
            
            PERF.inc("mcts.sim")  # Count simulations
            
            # ... rest of existing code ...
        
        # Log performance at end of search
        PERF.log_snapshot(clear=True, force=True)
        return root
```

**File**: `hex_ai/inference/batch_processor.py`

```python
# Add import at top
from hex_ai.utils.perf import PERF

# In process_batch method (around line 138)
def process_batch(self, force: bool = False) -> int:
    with PERF.timer("nn.infer"):
        # ... existing batch processing code ...
        
        # After successful inference
        PERF.inc("nn.batch")
        PERF.add_sample("nn.batch_size", len(batch_boards))
```

### Step 2: Run Baseline Profiling

**Command**: Run a tournament or self-play game to get baseline performance data:

```bash
# Run a quick tournament
python scripts/run_tournament.py --num_games 1 --moves_per_game 10

# Or run self-play
python scripts/run_large_selfplay.py --num_games 1 --moves_per_game 10
```

**Expected Output**: Look for PERF JSON lines in the logs:
```json
{
  "counters": {"mcts.sim": 1600, "nn.batch": 25},
  "timings_s": {
    "mcts.select": 0.120,
    "mcts.expand": 0.280,
    "nn.infer": 0.310,
    "mcts.backprop": 0.060
  },
  "samples": {"nn.batch_size": [25, 1560.0]},
  "meta": {"device": "cpu", "dtype": "torch.float32"}
}
```

### Step 3: Analyze Performance Data

Calculate key metrics from the PERF output:

- **Simulations per second**: `total_sims / total_time`
- **Average batch size**: `batch_size_sum / batch_count`
- **Time distribution**: `% time in each phase`
- **Bottleneck identification**: Which phase dominates?

### Step 4: Implement First Optimization

Based on the profiling data, implement the highest-impact optimization:

**If expansion dominates** → Implement apply/undo pattern (Phase 3.1)
**If selection dominates** → Vectorize UCT calculations (Phase 3.2)  
**If inference dominates** → Optimize batch utilization (Phase 3.3)

## Quick Performance Analysis

### Example Analysis Script

Create `scripts/analyze_performance.py`:

```python
#!/usr/bin/env python3
import json
import sys

def analyze_perf_line(line):
    """Analyze a single PERF JSON line."""
    if not line.startswith('INFO:hex_ai.utils.perf:PERF:'):
        return
    
    # Extract JSON part
    json_str = line.split('PERF: ')[1]
    data = json.loads(json_str)
    
    # Calculate metrics
    total_time = sum(data['timings_s'].values())
    sims_per_sec = data['counters'].get('mcts.sim', 0) / total_time if total_time > 0 else 0
    
    avg_batch_size = 0
    if 'nn.batch_size' in data['samples']:
        count, total = data['samples']['nn.batch_size']
        avg_batch_size = total / count if count > 0 else 0
    
    print(f"Simulations/sec: {sims_per_sec:.1f}")
    print(f"Avg batch size: {avg_batch_size:.1f}")
    print(f"Time distribution:")
    for phase, time in data['timings_s'].items():
        pct = (time / total_time * 100) if total_time > 0 else 0
        print(f"  {phase}: {pct:.1f}%")

if __name__ == "__main__":
    for line in sys.stdin:
        analyze_perf_line(line.strip())
```

**Usage**:
```bash
python scripts/run_tournament.py --num_games 1 | python scripts/analyze_performance.py
```

## Common Performance Patterns

### Pattern 1: Expansion Dominates (>50% time)
- **Symptom**: `mcts.expand` is the largest timing
- **Cause**: Deepcopy overhead in `make_move()`
- **Solution**: Implement apply/undo pattern (Phase 3.1)

### Pattern 2: Selection Dominates (>30% time)
- **Symptom**: `mcts.select` is large, many simulations
- **Cause**: Python loops in UCT calculation
- **Solution**: Vectorize child statistics (Phase 3.2)

### Pattern 3: Inference Dominates (>40% time)
- **Symptom**: `nn.infer` is large, small batch sizes
- **Cause**: Underutilized batches, device transfers
- **Solution**: Optimize batch utilization (Phase 3.3)

### Pattern 4: Low Batch Utilization (<60% of target)
- **Symptom**: `avg_batch_size` much smaller than `optimal_batch_size`
- **Cause**: Batch collection parameters too conservative
- **Solution**: Tune `max_wait_ms`, pre-seed rollouts

## Validation Checklist

After each optimization:

- [ ] Performance improved (sims/sec increased)
- [ ] No regressions in move quality
- [ ] Tests still pass
- [ ] Memory usage reasonable
- [ ] Logs show expected timing distribution

## Troubleshooting

### Performance Monitoring Not Working
```bash
# Test the infrastructure
python scripts/test_performance_monitoring.py
```

### No PERF Output in Logs
- Check that `PERF.log_snapshot()` is called
- Verify logging level is INFO or lower
- Ensure `force=True` for immediate output

### Unexpected Performance Results
- Run multiple games for statistical significance
- Check for outliers in timing data
- Verify device and model configuration

## Next Phase Planning

After baseline profiling:

1. **Week 1**: Implement highest-impact optimization
2. **Week 2**: Measure improvement, implement second optimization
3. **Week 3**: Device optimization and batch tuning
4. **Week 4**: Integration testing and documentation

## Resources

- **Full Plan**: `write_ups/mcts_optimization_plan.md`
- **Performance Utility**: `hex_ai/utils/perf.py`
- **Test Script**: `scripts/test_performance_monitoring.py`
- **Current TODOs**: Updated in source files with implementation plans
