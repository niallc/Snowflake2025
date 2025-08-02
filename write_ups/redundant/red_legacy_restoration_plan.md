# Legacy Code Restoration Plan

**Date:** 2024-07-18

## Overview

The goal is to restore the legacy Hex AI code (from before the player-to-move channel addition) to a working state so we can test it against the current problematic version. This will help us understand what changed and why the current version is performing poorly.

## Current Situation

### What We Have
- `hex_ai/models_legacy.py` - Old 2-channel ResNet18 model (before player-to-move channel)
- `hex_ai/training_legacy.py` - Old training code that expects different data loading
- `scripts/hyperparameter_tuning_legacy.py` - Old training script
- `hex_ai/data_processing_legacy.py` - Now added, contains `ProcessedDatasetLegacy` and `create_processed_dataloader_legacy`

### What's Missing
The legacy training code previously tried to import:
- `ProcessedDataset` class
- `create_processed_dataloader` function

These are now provided in `hex_ai/data_processing_legacy.py` as `ProcessedDatasetLegacy` and `create_processed_dataloader_legacy`.

### Key Differences Between Legacy and Current
1. **Input channels**: Legacy uses 2 channels (blue, red), current uses 3 channels (blue, red, player-to-move)
2. **Data loading**: Legacy expects `ProcessedDataset`, current uses `StreamingProcessedDataset`
3. **Model architecture**: Legacy has 3x3 first conv, current has 5x5 first conv
4. **Data format**: Legacy expects different data loading patterns

## Plan

### Phase 1: Identify Missing Components
1. **Check GitHub history** for any other missing legacy dependencies if errors arise
2. **Identify any other missing imports** or dependencies in the legacy code
3. **Document the exact data format** the legacy code expected

### Phase 2: Create Legacy Data Loading Layer
1. **`hex_ai/data_processing_legacy.py` now exists** with the missing functions
2. **`ProcessedDatasetLegacy` class** works with the old 2-channel format
3. **`create_processed_dataloader_legacy` function** is available
4. **Ensure compatibility** with existing `.pkl.gz` files (may need format conversion)

### Phase 3: Update Legacy Files
1. **Update imports** in `training_legacy.py` to use `data_processing_legacy` and the `_legacy` suffixes
2. **Add 'Legacy' suffixes** to class names in legacy files to avoid conflicts
3. **Update `hyperparameter_tuning_legacy.py`** to use legacy imports

### Phase 4: Create Test Infrastructure
1. **Create `scripts/test_legacy_training.py`** - simple script to test legacy training
2. **Create data format conversion utilities** if needed
3. **Add validation** to ensure legacy code works with current data files

### Phase 5: Validation and Testing
1. **Test legacy training** on a small dataset
2. **Compare performance** between legacy and current versions
3. **Document findings** about what changed and potential causes of performance regression

## Immediate Next Steps

1. **Update imports and class/function names in legacy code** to use the `_legacy` suffix for clarity and to avoid conflicts
2. **Create a simple test script** to verify the legacy model can load and run
3. **Document the exact data format differences** between legacy and current

## Questions for User

1. **Are there any other files or functions that the legacy code references that might be missing?**
2. **Do you have any old training runs or checkpoints from the working legacy version that we could use for comparison?**

## Success Criteria

- Legacy code can train on current data files (with format conversion if needed)
- Legacy training achieves similar performance to previous successful runs
- We can directly compare legacy vs current performance on the same data
- Clear documentation of what changed and potential causes of regression

## Files to Create/Modify

### New Files
- `hex_ai/data_processing_legacy.py` - Legacy data loading functions
- `scripts/test_legacy_training.py` - Test script for legacy training
- `write_ups/legacy_vs_current_comparison.md` - Results and analysis

### Modified Files
- `hex_ai/training_legacy.py` - Update imports and add Legacy suffixes
- `hex_ai/models_legacy.py` - Add Legacy suffixes to class names
- `scripts/hyperparameter_tuning_legacy.py` - Update imports

## Notes

- The goal is minimal disruption to the main codebase
- All legacy code should be clearly marked with 'Legacy' suffixes
- We should be able to run both legacy and current versions side by side
- Focus on getting a working baseline first, then optimize 