# Mixed Shard Dataset Implementation Plan

## Overview

Replace `StreamingSequentialShardDataset` with a new `StreamingMixedShardDataset` that maintains a mixed pool of positions from multiple shards to eliminate blockwise learning. The new dataset will load shards proportionally from multiple directories and maintain a large in-memory pool of positions.

## Design Decisions

### Core Architecture
- **Pool Size**: Default to ~2 million positions (expandable up to 40M if needed)
- **Memory Limit**: Hard limit of 5GB with graceful shutdown
- **Shard Loading**: Proportional loading based on directory sizes, sequential within each directory
- **No Deduplication**: Trust that shards are pre-shuffled and don't contain duplicates
- **Error Handling**: Crash loudly on any unexpected issues
- **No Backward Compatibility**: Replace existing dataset entirely

### Shard Management
- **Directory Weights**: Calculate based on number of shards in each directory's range
- **Sequential Loading**: Load shards sequentially within each directory (they're pre-shuffled)
- **Exhaustion Tracking**: Track which shards have been loaded to avoid duplicates
- **Refill Strategy**: Load new shards when pool drops below threshold (e.g., 1.5M positions)

### Memory Management
- **Pool Size**: 2M positions default (~600MB), expandable to 40M if system can handle it
- **Memory Monitoring**: Check memory usage and shutdown gracefully if >5GB
- **Shard Size**: Assume 50k-500k positions per shard, load until pool is full

## Implementation Steps

### Phase 1: Core Dataset Class
- [ ] **Step 1.1**: Create `StreamingMixedShardDataset` class skeleton
  - [ ] Inherit from `torch.utils.data.IterableDataset`
  - [ ] Define constructor with directory lists, shard ranges, pool size, etc.
  - [ ] Add memory monitoring utility function
  - [ ] Add shard tracking data structures

- [ ] **Step 1.2**: Implement shard discovery and weighting
  - [ ] Parse shard ranges for each directory
  - [ ] Calculate proportional weights based on shard counts
  - [ ] Create shard queues for each directory
  - [ ] Track loaded shards to prevent duplicates

- [ ] **Step 1.3**: Implement position pool management
  - [ ] Create in-memory position pool (list of examples)
  - [ ] Implement pool refilling logic
  - [ ] Add shuffling when pool is refilled
  - [ ] Add memory usage monitoring

### Phase 2: Data Loading Logic
- [ ] **Step 2.1**: Implement proportional shard loading
  - [ ] Load shards from each directory according to weights
  - [ ] Handle sequential loading within each directory
  - [ ] Stop loading when pool is full or shards exhausted
  - [ ] Add logging for shard loading progress

- [ ] **Step 2.2**: Implement position extraction
  - [ ] Extract positions from loaded shards
  - [ ] Add positions to pool
  - [ ] Handle augmentation factor calculation
  - [ ] Add position counting and validation

- [ ] **Step 2.3**: Implement pool consumption
  - [ ] Yield positions from pool in batches
  - [ ] Remove consumed positions from pool
  - [ ] Trigger refill when pool drops below threshold
  - [ ] Handle end-of-data scenarios

### Phase 3: Integration
- [ ] **Step 3.1**: Update data discovery functions
  - [ ] Modify `discover_and_split_multiple_data()` to return directory weights
  - [ ] Update shard range parsing to work with new dataset
  - [ ] Ensure compatibility with existing data source info

- [ ] **Step 3.2**: Update dataset creation
  - [ ] Replace `StreamingSequentialShardDataset` in `create_datasets()`
  - [ ] Update constructor calls with new parameters
  - [ ] Ensure DataLoader compatibility

- [ ] **Step 3.3**: Update training orchestration
  - [ ] Modify `run_hyperparameter_tuning_current_data()` to use new dataset
  - [ ] Update parameter passing for pool size, memory limits, etc.
  - [ ] Ensure logging and monitoring work correctly

### Phase 4: Testing and Validation
- [ ] **Step 4.1**: Create test data and scenarios
  - [ ] Generate test shards with known sizes
  - [ ] Test with multiple directories and different shard ranges
  - [ ] Test memory limits and graceful shutdown
  - [ ] Test edge cases (empty directories, single shard, etc.)

- [ ] **Step 4.2**: Performance testing
  - [ ] Benchmark memory usage with 2M position pool
  - [ ] Test shuffling performance with large pools
  - [ ] Compare training speed vs. old dataset
  - [ ] Test I/O performance with multiple directories

- [ ] **Step 4.3**: Integration testing
  - [ ] Test with real data directories
  - [ ] Verify position mixing is working correctly
  - [ ] Test with different augmentation settings
  - [ ] Verify training results are consistent

### Phase 5: Cleanup and Documentation
- [ ] **Step 5.1**: Remove old dataset code
  - [ ] Delete `StreamingSequentialShardDataset` class
  - [ ] Remove any references to old dataset
  - [ ] Clean up unused imports and functions

- [ ] **Step 5.2**: Update documentation
  - [ ] Update docstrings for new dataset
  - [ ] Update training pipeline documentation
  - [ ] Add examples of new dataset usage
  - [ ] Document memory requirements and limits

## Technical Specifications

### Class Interface
```python
class StreamingMixedShardDataset(torch.utils.data.IterableDataset):
    def __init__(self, 
                 data_dirs: List[str],
                 shard_ranges: List[str], 
                 pool_size: int = 2_000_000,
                 refill_threshold: int = 1_500_000,
                 max_memory_gb: float = 5.0,
                 enable_augmentation: bool = True,
                 max_examples_unaugmented: Optional[int] = None,
                 verbose: bool = False,
                 random_seed: Optional[int] = None):
```

### Key Methods
- `_discover_shards()`: Find and weight shards from all directories
- `_load_shards_proportionally()`: Load shards according to weights
- `_refill_pool()`: Add new positions to pool and shuffle
- `_monitor_memory()`: Check memory usage and shutdown if needed
- `__iter__()`: Main iteration logic

### Memory Management
- **Pool Size**: 2M positions default (~600MB)
- **Memory Limit**: 5GB hard limit with graceful shutdown
- **Refill Threshold**: 1.5M positions (75% of pool size)
- **Shard Size**: Assume 50k-500k positions per shard

### Error Handling
- **File Errors**: Crash immediately with detailed error message
- **Memory Errors**: Graceful shutdown with memory usage report
- **Data Errors**: Crash with position and shard information
- **Configuration Errors**: Crash with parameter validation details

## Success Criteria

1. **Functional**: Dataset loads positions from multiple directories proportionally
2. **Performance**: Memory usage stays under 5GB, shuffling completes in <5 seconds
3. **Correctness**: No duplicate positions, proper augmentation, consistent results
4. **Reliability**: Crashes loudly on errors, no silent failures
5. **Maintainability**: Clean, single-purpose code with good logging

## Risks and Mitigations

### Risk: Memory Usage Exceeds Limits
- **Mitigation**: Implement 5GB hard limit with graceful shutdown
- **Monitoring**: Add memory usage logging and alerts

### Risk: I/O Performance Degradation
- **Mitigation**: Load shards in background, maintain large pool
- **Monitoring**: Add I/O timing logs and performance metrics

### Risk: Position Mixing Not Working
- **Mitigation**: Add logging to track which directories positions come from
- **Validation**: Add tests to verify proportional mixing

### Risk: Training Performance Degradation
- **Mitigation**: Benchmark against old dataset, optimize if needed
- **Monitoring**: Add training speed and convergence metrics

## Notes and Discoveries

*This section will be updated as we implement and discover complications*

- **Date**: [To be filled]
- **Step**: [To be filled]
- **Discovery**: [To be filled]
- **Resolution**: [To be filled]
