# Data Shuffling Specification

## Overview

This document specifies the data shuffling process to address value head fingerprinting issues in the Hex AI training pipeline. The current ordered data structure allows the value network to memorize game-specific patterns rather than learning general evaluation principles.

## Problem Statement

### Current Issues
1. **Game Fingerprinting**: Each game contributes 40-150 positions with identical value targets
2. **Temporal Correlation**: Positions from the same game appear together in training batches
3. **Value Network Overfitting**: Network learns to recognize specific games rather than board patterns
4. **Memory Constraints**: Full dataset (~97M positions) is 5x too large for memory

### Evidence
- Value loss of 0.07 in epoch 0 indicates immediate overfitting
- Policy network generalizes while value network does not
- Data augmentation shows no significant impact

## Solution Design

### Two-Phase Shuffling Process

#### Phase 1: Distribution Bucketing
**Goal**: Distribute games evenly across k buckets to break game-level correlations

**Process**:
1. Loop through each `.pkl.gz` file in `data/processed/data/`
2. For each file, distribute games evenly across k buckets (k=500 recommended)
3. Create intermediate files: `{input_filename}_bucket_{i}.pkl.gz` (i=0 to k-1)
4. Each bucket file contains games from multiple source files

**Memory Management**:
- Process one input file at a time
- Write bucket files incrementally
- Close files immediately after writing

#### Phase 2: Bucket Consolidation and Shuffling
**Goal**: Consolidate each bucket and shuffle positions within memory constraints

**Process**:
1. For each bucket i (0 to k-1):
   - Load all `*_bucket_{i}.pkl.gz` files
   - Concatenate all examples into single list
   - Shuffle examples in memory
   - Write to final file: `shuffled_{i}.pkl.gz`
2. Result: k final files, each containing ~200MB of shuffled data

**Memory Management**:
- Process one bucket at a time
- Each bucket fits in memory (~200MB)
- Clean up intermediate bucket files after processing

## Data Structure

### Input Format
```python
{
    'examples': [
        {
            'board': np.ndarray,        # (2, 13, 13) board state
            'policy': np.ndarray,       # (169,) policy target or None
            'value': float,             # 0.0 or 1.0
            'metadata': {
                'game_id': None,        # Will be populated during processing
                'position_in_game': int,
                'total_positions': int,
                'value_sample_tier': int,
                'winner': str           # "BLUE" or "RED"
            }
        },
        # ... more examples
    ],
    'source_file': str,
    'processing_stats': dict,
    'processed_at': str,
    'file_size_bytes': int
}
```

### Output Format
```python
{
    'examples': [
        {
            'board': np.ndarray,        # (2, 13, 13) board state
            'policy': np.ndarray,       # (169,) policy target or None
            'value': float,             # 0.0 or 1.0
            'metadata': {
                'game_id': (file_idx, line_idx),  # Populated during processing
                'position_in_game': int,
                'total_positions': int,
                'value_sample_tier': int,
                'winner': str,          # "BLUE" or "RED"
                'original_source': str  # Original source file
            }
        },
        # ... more examples
    ],
    'shuffling_stats': {
        'num_buckets': int,
        'bucket_id': int,
        'total_examples': int,
        'shuffled_at': str,
        'source_files': List[str]
    }
}
```

## Implementation Details

### Configuration Parameters
```python
SHUFFLING_CONFIG = {
    'num_buckets': 500,              # Number of buckets for distribution
    'input_dir': 'data/processed/data',
    'output_dir': 'data/processed/shuffled',
    'temp_dir': 'data/processed/temp_buckets',
    'chunk_size': 1000,              # Examples per write operation
    'resume_enabled': True,          # Resume interrupted processing
    'cleanup_temp': True,            # Clean up intermediate files
    'validation_enabled': True       # Validate output data
}
```

### Phase 1: Distribution Algorithm
```python
def distribute_to_buckets(input_files, num_buckets):
    """
    Distribute games from input files across buckets.
    
    Args:
        input_files: List of .pkl.gz file paths
        num_buckets: Number of buckets to distribute across
    
    Returns:
        List of bucket file paths
    """
    bucket_files = [f"bucket_{i}.pkl.gz" for i in range(num_buckets)]
    bucket_data = [[] for _ in range(num_buckets)]
    
    for file_idx, input_file in enumerate(input_files):
        data = load_pkl_gz(input_file)
        
        # Distribute games evenly across buckets
        for game_idx, game_examples in enumerate(data['examples']):
            bucket_idx = (file_idx * len(data['examples']) + game_idx) % num_buckets
            bucket_data[bucket_idx].extend(game_examples)
    
    # Write bucket files
    for bucket_idx, examples in enumerate(bucket_data):
        write_bucket_file(bucket_files[bucket_idx], examples, bucket_idx)
    
    return bucket_files
```

### Phase 2: Consolidation Algorithm
```python
def consolidate_and_shuffle_bucket(bucket_idx, bucket_files):
    """
    Consolidate all files for a bucket and shuffle.
    
    Args:
        bucket_idx: Index of bucket to process
        bucket_files: List of bucket file paths for this bucket
    
    Returns:
        Path to shuffled output file
    """
    all_examples = []
    
    # Load all examples for this bucket
    for bucket_file in bucket_files:
        if bucket_file.endswith(f"_bucket_{bucket_idx}.pkl.gz"):
            data = load_pkl_gz(bucket_file)
            all_examples.extend(data['examples'])
    
    # Shuffle examples
    random.shuffle(all_examples)
    
    # Write shuffled file
    output_file = f"shuffled_{bucket_idx}.pkl.gz"
    write_shuffled_file(output_file, all_examples, bucket_idx)
    
    return output_file
```

## Error Handling and Recovery

### Resume Functionality
- Track progress in `shuffling_progress.json`
- Resume from last completed bucket
- Validate partial results before continuing

### Validation Checks
- Verify all examples have required fields
- Check that games are properly distributed across buckets
- Validate that shuffled files contain expected number of examples
- Ensure no duplicate examples in final output

### Error Recovery
- Handle corrupted input files gracefully
- Retry failed operations with exponential backoff
- Log all errors for debugging

## Performance Considerations

### Memory Usage
- Phase 1: ~200MB per bucket (manageable)
- Phase 2: ~200MB per bucket consolidation (manageable)
- Total peak memory: ~400MB

### Disk Usage
- Intermediate bucket files: ~50GB total
- Final shuffled files: ~50GB total
- Temporary storage required: ~100GB

### Processing Time
- Estimated time: 2-4 hours for full dataset
- Can be parallelized by processing multiple buckets simultaneously

## Testing Strategy

### Unit Tests
- Test game distribution algorithm
- Test bucket consolidation
- Test data validation functions

### Integration Tests
- Test full pipeline on small dataset
- Verify game dispersion across buckets
- Check memory usage under load

### Validation Tests
- Verify no game fingerprinting in shuffled data
- Check that value targets are properly distributed
- Validate metadata preservation

## Monitoring and Logging

### Progress Tracking
- Log files processed
- Log buckets completed
- Log memory usage
- Log processing time

### Quality Metrics
- Game distribution statistics
- Value target distribution
- Position correlation analysis

## Success Criteria

1. **Game Dispersion**: No more than 2% of examples in any bucket come from the same game
2. **Memory Efficiency**: Peak memory usage < 1GB
3. **Data Integrity**: All examples preserved with metadata intact
4. **Performance**: Processing completes within 4 hours
5. **Validation**: Shuffled data passes all validation checks

## Future Enhancements

### Advanced Shuffling Strategies
- Stratified sampling by game length
- Position-based sampling within games
- Value target balancing across buckets

### Parallel Processing
- Multi-threaded bucket processing
- Distributed processing across multiple machines
- GPU-accelerated data loading

### Real-time Shuffling
- Streaming dataset with continuous shuffling
- Dynamic bucket adjustment based on memory usage
- Online learning with shuffled data streams 