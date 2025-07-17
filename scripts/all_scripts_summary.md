# All Scripts Summary

This document summarizes the scripts and utilities available in the `scripts/` directory, including their purpose and usage. (Updated 2024-07-18)

---

## Directory Structure

```
scripts/
  # Legacy and Migration Tools (NEW)
  hyperparameter_tuning_legacy.py        # Legacy hyperparameter tuning with 2-channel model
  test_legacy_incremental.py             # Test incremental changes from legacy to modern (PLANNED)
  compare_legacy_vs_modified.py          # Compare performance between legacy and modified versions (PLANNED)
  
  # Data Analysis and Debugging
  extract_error_sample_from_pkl.py       # Extract and inspect a single record from a .pkl.gz file
  inspect_training_batch.py              # Inspect and check a batch of records from a .pkl.gz file
  compare_loader_vs_raw.py               # Compare StreamingProcessedDataset output to raw records
  compare_training_pipelines.py          # Compare main vs overfit training pipelines, test training config impact
  analyze_gradient_norms.py              # Analyze/print gradient norms during training for debugging
  overfit_tiny_dataset.py                # Overfit a tiny dataset (e.g., one game) to test model/data sanity
  
  # Training and Hyperparameter Tuning
  hyperparam_sweep.py                    # Run hyperparameter sweeps, launching multiple training jobs
  run_training.py                        # Main training entry point (single run, all config via CLI)
  quick_test_training.py                 # Quick test training with improved hyperparameters
  
  # Inference and Testing
  simple_inference_cli.py                # Run inference on a board or TRMPH string using a trained model
  
  # Data Processing and Analysis
  analyze_raw_data.py                    # Analyze raw TRMPH data files
  analyze_trmph_data.py                  # Analyze processed TRMPH data
  analyze_training.py                    # Analyze training results and checkpoints
  analyze_error_samples.py               # Analyze error samples from data processing
  find_invalid_game_starts.py            # Find games with invalid starting positions
  find_problematic_samples.py            # Find problematic samples in processed data
  identify_bad_files.py                  # Identify files with data quality issues
  search_board_in_trmph.py               # Search for specific board states in TRMPH files
  verify_color_swap.py                   # Verify color swapping in processed data
  
  # Training Analysis and Debugging
  analyze_gradient_norms.py              # Analyze gradient norms during training
  test_channel_impact.py                 # Test impact of player-to-move channel
  test_fresh_init_value_head.py          # Test value head initialization
  extract_error_sample_from_pkl.py       # Extract specific error samples for analysis
  
  # Utility Scripts
  monitor_resources.py                   # Monitor system resources during training
  process_all_trmph.py                   # Process all TRMPH files (legacy, may need updates)
  
  lib/
    data_loading_utils.py                # Shared: load .pkl.gz files with 'examples' key
    board_viz_utils.py                   # Shared: board visualization and policy decoding utilities
    consistency_checks.py                # Shared: consistency checks for board, policy, and player-to-move
```

---

## Legacy and Migration Tools (NEW)

### hyperparameter_tuning_legacy.py
- **Purpose**: Run hyperparameter tuning using the legacy 2-channel model architecture
- **Key features**: Uses `TwoHeadedResNetLegacy`, `ProcessedDatasetLegacy`, and legacy training pipeline
- **Usage**: `python -m scripts.hyperparameter_tuning_legacy`
- **Status**: ✅ **WORKING** - Successfully restored and confirmed to perform better than current version
- **Performance**: Achieves good policy loss reduction, unlike the current 3-channel version

### test_legacy_incremental.py (PLANNED)
- **Purpose**: Test incremental changes from legacy to modern architecture
- **Key features**: 
  - Step 2.1: Test player-to-move channel addition
  - Step 2.2: Test 5x5 first convolution
  - Step 2.3: Test current training pipeline
  - Step 2.4: Test current data pipeline
- **Usage**: `python -m scripts.test_legacy_incremental --step 2.1`
- **Status**: 🔄 **PLANNED** - Part of incremental migration strategy

### compare_legacy_vs_modified.py (PLANNED)
- **Purpose**: Compare performance curves between legacy and modified versions
- **Key features**: Automated comparison of policy loss, value loss, and training dynamics
- **Usage**: `python -m scripts.compare_legacy_vs_modified --legacy-results path --modified-results path`
- **Status**: 🔄 **PLANNED** - Part of incremental migration strategy

---

## Shared Utilities (`scripts/lib/`)

- **data_loading_utils.py**
  - `load_examples_from_pkl(file_path)`: Load examples from a .pkl.gz file with an 'examples' key.

- **board_viz_utils.py**
  - `decode_policy_target(policy)`: Decode a one-hot policy vector to (row, col, trmph_move).
  - `visualize_board_with_policy(board, policy, file=None)`: Display the board and highlight the policy target move if present.

- **consistency_checks.py**
  - `policy_on_empty_cell(board, highlight_move)`: Check if the policy target move is on an empty cell.
  - `player_to_move_channel_valid(player_channel)`: Check if the player-to-move channel contains only 0.0 or 1.0.

These modules are imported by the main scripts to avoid code duplication and ensure consistent logic for loading, visualization, and checking.

---

## Main Scripts and Utilities

### Data Analysis and Debugging

- **extract_error_sample_from_pkl.py**
  - Extract and display a record from a .pkl.gz file by index or search string.
  - Uses shared utilities for loading, visualization, and consistency checks.

- **inspect_training_batch.py**
  - Inspect a batch of records from a .pkl.gz file (sequentially or randomly).
  - Visualize each board, highlight the policy target, and run consistency checks.
  - Prints a summary of any issues found in the batch.

- **compare_loader_vs_raw.py**
  - Compare the output of StreamingProcessedDataset to the raw records in a .pkl.gz file, visualizing and checking for any discrepancies.

- **compare_training_pipelines.py**
  - Compare the main training pipeline to a simplified overfit pipeline on the same data/model.
  - Systematically test the impact of training config (gradient clipping, weight decay, mixed precision, etc.).
  - Use `--test-parameters` to run a sweep and print a summary of results.
  - Useful for isolating causes of poor training performance.

- **analyze_gradient_norms.py**
  - Analyze and print gradient norms during training.
  - Helps diagnose issues with gradient clipping and learning dynamics.
  - Prints per-batch stats and overall summary.

- **overfit_tiny_dataset.py**
  - Overfit the model on a tiny dataset (e.g., one game or 100 samples).
  - Useful for sanity-checking the model, loss, and data pipeline.
  - Should be able to drive policy loss near zero if everything is correct.

### Training and Hyperparameter Tuning

- **hyperparam_sweep.py**
  - Launches multiple training jobs with different hyperparameters (learning rate, batch size, etc.).
  - Uses subprocesses to run `run_training.py` for each config.
  - Appends a timestamp to experiment names to avoid overwriting results.
  - Summarizes results in a sweep_summary.json file.
  - **Best practice:** Always use unique experiment names to avoid clobbering old results.

- **run_training.py**
  - Main entry point for training a model on a single config.
  - All config via CLI flags; results and config saved to a unique directory.
  - Per-epoch metrics are logged to a CSV in `checkpoints/bookkeeping/` (or per-experiment if configured).

- **quick_test_training.py**
  - Quick test training with improved hyperparameters.
  - Useful for rapid iteration and testing of training changes.

### Inference and Testing

- **simple_inference_cli.py**
  - Run inference on a board or TRMPH string using a trained model checkpoint.
  - Useful for quick sanity checks of model predictions after training.
  - Supports both current and legacy models.

### Data Processing and Analysis

- **analyze_raw_data.py**
  - Analyze raw TRMPH data files for quality and consistency.

- **analyze_trmph_data.py**
  - Analyze processed TRMPH data for training suitability.

- **analyze_training.py**
  - Analyze training results and checkpoints for performance insights.

- **find_invalid_game_starts.py**
  - Find games with invalid starting positions in the data.

- **find_problematic_samples.py**
  - Find problematic samples in processed data that may cause training issues.

- **identify_bad_files.py**
  - Identify files with data quality issues that should be excluded from training.

- **search_board_in_trmph.py**
  - Search for specific board states in TRMPH files for debugging.

- **verify_color_swap.py**
  - Verify color swapping in processed data to ensure consistency.

---

## Legacy Code Status

### Working Legacy Components
- ✅ `hex_ai/models_legacy.py` - 2-channel ResNet18 model
- ✅ `hex_ai/training_legacy.py` - Legacy training pipeline
- ✅ `hex_ai/data_processing_legacy.py` - Legacy data loading
- ✅ `hex_ai/training_utils_legacy.py` - Legacy training utilities
- ✅ `scripts/hyperparameter_tuning_legacy.py` - Legacy hyperparameter tuning

### Legacy vs Current Performance
- **Legacy (2-channel)**: ✅ Good policy loss reduction, stable training
- **Current (3-channel)**: ❌ Poor policy loss, barely improves from random

### Migration Strategy
- **Incremental testing**: Test each change individually (player-to-move channel, 5x5 conv, training pipeline, data pipeline)
- **Rollback capability**: Can always return to working legacy version
- **Systematic approach**: Identify exact cause of performance regression

---

## Notes

- Any truly generic library code (e.g., board display, format conversion) remains in `hex_ai/`.
- The `scripts/lib/` directory is for utilities specific to data inspection, file handling, and script-level tasks.
- All scripts should import from `scripts/lib/` to avoid duplication and ensure maintainability.
- **CSV logging:** Per-epoch metrics are logged to `checkpoints/bookkeeping/training_metrics.csv` by default. For sweeps, you may want to configure per-experiment CSVs.
- **Legacy code**: All legacy components are clearly marked with `_legacy` suffixes to avoid conflicts with current code.
- **Migration tools**: New tools are being developed to systematically test incremental changes from legacy to modern architecture.

---

(End of summary) 