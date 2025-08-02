# Update Training for Shuffled Data - Plan

## What We've Updated

We have successfully implemented a two-phase data shuffling process to address value head fingerprinting issues in the Hex AI training pipeline. The shuffling process breaks game-level correlations that were causing the value network to overfit to specific games rather than learning general evaluation principles.

## New Data Location and Structure

### Location
- **New shuffled data**: `data/processed/shuffled/`
- **Temporary bucket files**: `data/processed/temp_buckets/` (can be cleaned up after completion)
- **Progress tracking**: `data/processed/shuffled/shuffling_progress.json`

### File Naming Convention
- **Shuffled files**: `shuffled_{bucket_id:04d}.pkl.gz` (e.g., `shuffled_0000.pkl.gz`, `shuffled_0001.pkl.gz`, etc.)
- **Total files**: 500 shuffled files (one per bucket)
- **File sizes**: ~200MB each (approximately 200,000 examples per file)

### Data Structure
Each shuffled file contains:
```python
{
    'examples': [
        {
            'board': np.ndarray,        # (2, 13, 13) board state
            'policy': np.ndarray,       # (169,) policy target or None
            'value': float,             # 0.0 or 1.0
            'metadata': {
                'game_id': str,         # Original game identifier
                'position_in_game': int,
                'total_positions': int,
                'value_sample_tier': int,
                'winner': str           # "BLUE" or "RED"
            }
        },
        # ... ~200,000 examples per file
    ],
    'shuffling_stats': {
        'num_buckets': 500,
        'bucket_id': int,              # 0-499
        'total_examples': int,         # ~200,000
        'shuffled_at': str,            # ISO timestamp
        'source_files': List[str]      # Original source files
    }
}
```

## Key Benefits of Shuffled Data

1. **Game Dispersion**: Each bucket contains examples from hundreds of different games
2. **Temporal Separation**: Positions from the same game are separated by ~200,000-600,000 records
3. **Memory Efficiency**: Each bucket is ~200MB, easily manageable for training
4. **No Fingerprinting**: Value network cannot memorize specific games due to dispersion

## Required Changes to Hyperparameter Sweep

### 1. Update Data Directory Path

**Current**: `data_dir="data/processed"`
**New**: `data_dir="data/processed/shuffled"`

### 2. Update File Discovery Pattern

**Current**: Discovers `*_processed.pkl.gz` files
**New**: Discovers `shuffled_*.pkl.gz` files

### 3. Update Data Loading Logic

The current `discover_processed_files()` function in `hex_ai/data_pipeline.py` needs to be updated to handle the new file pattern:

```python
def discover_processed_files(data_dir: str = "data/processed") -> List[Path]:
    """
    Discover all processed data files in the specified directory.
    
    Args:
        data_dir: Directory containing processed data files
        
    Returns:
        List of paths to processed data files
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory {data_dir} not found")
    
    # Check if this is shuffled data directory
    if (data_path / "shuffling_progress.json").exists():
        # Shuffled data: look for shuffled_*.pkl.gz files
        data_files = list(data_path.glob("shuffled_*.pkl.gz"))
        logger.info(f"Found {len(data_files)} shuffled data files")
    else:
        # Original processed data: look for *_processed.pkl.gz files
        data_files = list(data_path.glob("*_processed.pkl.gz"))
        logger.info(f"Found {len(data_files)} processed data files")
    
    if not data_files:
        raise FileNotFoundError(f"No data files found in {data_dir}")
    
    return data_files
```

## Implementation Steps

### Step 1: Update Data Pipeline
1. Modify `hex_ai/data_pipeline.py` to handle both original and shuffled data formats
2. Add detection logic for shuffled data directory
3. Update file discovery pattern

### Step 2: Update Hyperparameter Sweep Script
1. Change `data_dir` from `"data/processed"` to `"data/processed/shuffled"`

### Step 3: Update Training Utilities
1. Ensure `run_hyperparameter_tuning_current_data` works with shuffled data
2. Verify that `StreamingProcessedDataset` handles the new data format correctly
3. Test that data augmentation still works with shuffled data

### Step 4: Validation and Testing
1. Run a small test sweep to verify everything works
2. Run existing tests and possibly write new tests to very use of the new data.

## Files to Modify

1. `scripts/hyperparam_sweep.py` - Update data directory and experiment naming
2. `hex_ai/data_pipeline.py` - Update file discovery logic
3. `hex_ai/training_utils_legacy.py` - Verify compatibility (likely no changes needed)

## Testing Strategy

1. **Small-scale test**: Run with limited samples to verify functionality
2. **Performance comparison**: Compare training metrics with original data
3. **Memory usage**: Monitor memory consumption during training
4. **Validation**: Ensure no data corruption or loss during loading 