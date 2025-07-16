# All Scripts Summary

This document summarizes the scripts and utilities available in the `scripts/` directory, including their purpose and usage. (Updated 2024-07-17)

---

## Directory Structure

```
scripts/
  extract_error_sample_from_pkl.py      # Extract and inspect a single record from a .pkl.gz file
  inspect_training_batch.py             # Inspect and check a batch of records from a .pkl.gz file
  compare_loader_vs_raw.py              # (Planned) Compare StreamingProcessedDataset output to raw records
  lib/
    data_loading_utils.py               # Shared: load .pkl.gz files with 'examples' key
    board_viz_utils.py                  # Shared: board visualization and policy decoding utilities
    consistency_checks.py               # Shared: consistency checks for board, policy, and player-to-move
  ... (other scripts omitted for brevity)
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

## Main Scripts

- **extract_error_sample_from_pkl.py**
  - Extract and display a record from a .pkl.gz file by index or search string.
  - Uses shared utilities for loading, visualization, and consistency checks.

- **inspect_training_batch.py**
  - Inspect a batch of records from a .pkl.gz file (sequentially or randomly).
  - Visualize each board, highlight the policy target, and run consistency checks.
  - Prints a summary of any issues found in the batch.

- **compare_loader_vs_raw.py** (Planned)
  - Will compare the output of StreamingProcessedDataset to the raw records in a .pkl.gz file, visualizing and checking for any discrepancies.

---

## Notes

- Any truly generic library code (e.g., board display, format conversion) remains in `hex_ai/`.
- The `scripts/lib/` directory is for utilities specific to data inspection, file handling, and script-level tasks.
- All scripts should import from `scripts/lib/` to avoid duplication and ensure maintainability.

---

(End of summary) 