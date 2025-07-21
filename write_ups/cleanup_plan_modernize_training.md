# Cleanup Plan: Modernizing Hex AI Training Pipeline

## Overview
With the successful migration to a fully modernized training pipeline (confirmed via `scripts/hyperparam_sweep.py` calling `run_hyperparameter_tuning_current_data`), it's time to clean up legacy code and consolidate the codebase for maintainability and clarity.

---

## Cleanup Steps

### 1. Remove or Archive Legacy Code
- Remove or archive:
  - `hex_ai/training_legacy.py`
  - `hex_ai/models_legacy.py` and `models_legacy_with_player_channel.py`
  - `hex_ai/data_processing_legacy.py`
  - Any legacy dataset or trainer classes not used in the modern flow
- Clearly mark any archived files as deprecated.

### 2. Consolidate Orchestration/Experiment Logic
- Move `run_hyperparameter_tuning_current_data` and related experiment functions to a new, modern utility (e.g., `hex_ai/experiment_utils.py`).
- Update all scripts to import from the new location.
- Remove the `_legacy` suffix from scripts and functions that are now modern.

### 3. Unify All Training Scripts
- Ensure all entry points (`run_training.py`, `hyperparam_sweep.py`, etc.) use the unified, modern pipeline.
- Remove or refactor scripts that duplicate functionality (e.g., legacy test scripts, old hyperparameter tuning scripts).

### 4. Clean Up Logging and Configuration
- Standardize logging setup (file and terminal output, consistent format).
- Ensure all scripts and modules use a shared logging configuration.
- Remove ad-hoc print statements in favor of logger usage where appropriate.

### 5. Remove Duplicated or Obsolete Files
- Delete or archive scripts and modules that are no longer needed.
- Update documentation and script summaries to reflect the new structure.

### 6. Final Testing and Documentation
- Run end-to-end tests on all main scripts to confirm performance and correctness.
- Update `README.md` and `write_ups/debugging_journey_2024-07-16.md` to reflect the new, unified pipeline.
- Document any remaining known issues or TODOs.

### 7. (Optional) Future Improvements
- Refactor experiment orchestration for parallelism or distributed runs if needed.
- Add more robust CLI/config support for all scripts.
- Further modularize data augmentation and preprocessing utilities.
- Add more automated tests and CI for training and data pipeline.

---

**Goal:** A single, modern, maintainable training pipeline with no legacy code paths, clear logging, and up-to-date documentation. 