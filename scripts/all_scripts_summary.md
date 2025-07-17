# All Scripts Summary

This document summarizes the scripts and utilities available in the `scripts/` directory, including their purpose and usage. (Updated 2024-07-17)

---

## Directory Structure

```
scripts/
  extract_error_sample_from_pkl.py      # Extract and inspect a single record from a .pkl.gz file
  inspect_training_batch.py             # Inspect and check a batch of records from a .pkl.gz file
  compare_loader_vs_raw.py              # Compare StreamingProcessedDataset output to raw records
  compare_training_pipelines.py         # Compare main vs overfit training pipelines, test training config impact
  analyze_gradient_norms.py             # Analyze/print gradient norms during training for debugging
  overfit_tiny_dataset.py               # Overfit a tiny dataset (e.g., one game) to test model/data sanity
  hyperparam_sweep.py                   # Run hyperparameter sweeps, launching multiple training jobs
  simple_inference_cli.py               # Run inference on a board or TRMPH string using a trained model
  run_training.py                       # Main training entry point (single run, all config via CLI)
  ... (other scripts omitted for brevity)
  lib/
    data_loading_utils.py               # Shared: load .pkl.gz files with 'examples' key
    board_viz_utils.py                  # Shared: board visualization and policy decoding utilities
    consistency_checks.py               # Shared: consistency checks for board, policy, and player-to-move
```

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

- **hyperparam_sweep.py**
  - Launches multiple training jobs with different hyperparameters (learning rate, batch size, etc.).
  - Uses subprocesses to run `run_training.py` for each config.
  - Appends a timestamp to experiment names to avoid overwriting results.
  - Summarizes results in a sweep_summary.json file.
  - **Best practice:** Always use unique experiment names to avoid clobbering old results.

- **simple_inference_cli.py**
  - Run inference on a board or TRMPH string using a trained model checkpoint.
  - Useful for quick sanity checks of model predictions after training.

- **run_training.py**
  - Main entry point for training a model on a single config.
  - All config via CLI flags; results and config saved to a unique directory.
  - Per-epoch metrics are logged to a CSV in `checkpoints/bookkeeping/` (or per-experiment if configured).

---

## Notes

- Any truly generic library code (e.g., board display, format conversion) remains in `hex_ai/`.
- The `scripts/lib/` directory is for utilities specific to data inspection, file handling, and script-level tasks.
- All scripts should import from `scripts/lib/` to avoid duplication and ensure maintainability.
- **CSV logging:** Per-epoch metrics are logged to `checkpoints/bookkeeping/training_metrics.csv` by default. For sweeps, you may want to configure per-experiment CSVs.
---

(End of summary) 