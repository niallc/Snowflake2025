# Code Duplication and Refactoring Log

This document tracks known areas of code duplication, technical debt, and major refactoring efforts in the project. Items are marked as **[RESOLVED]** or **[TODO]** as appropriate.

---

## Resolved

### Policy Processing & Move Selection Refactoring (**2025-07-23**) **[RESOLVED]**

- **Critical Bug Fixed:** Tree search now uses temperature-scaled softmax for policy logits, fixing a major play-quality issue.
- **Centralized Utilities:** All policy processing and move selection logic is now handled by canonical utilities in `hex_ai/value_utils.py`.
- **Files Updated:**
    - `hex_ai/value_utils.py` (new utilities)
    - `hex_ai/inference/fixed_tree_search.py` (bug fix, refactor)
    - `scripts/play_vs_model_cli.py`, `hex_ai/inference/tournament.py`, `hex_ai/web/app.py`, `hex_ai/inference/simple_model_inference.py`, `scripts/simple_inference_cli.py` (refactored to use utilities)
    - `tests/test_policy_utilities.py` (comprehensive tests)
- **Benefits:**
    - Bug fix, code deduplication, maintainability, consistency, and easier future improvements.

### Web Move-Making Code Modularization (**2025-07-23**) **[RESOLVED]**

- **Moved `_select_policy_move` to shared utility:** The private helper function in `hex_ai/web/app.py` has been moved to `hex_ai/value_utils.py` as the public `select_policy_move` function.
- **Updated all call-sites:** Web app, CLI scripts, and tournament code now use the shared utility.
- **Added comprehensive tests:** New test for `select_policy_move` function with various scenarios.
- **Benefits:**
    - Eliminated duplication between web and CLI move selection logic.
    - Improved testability and maintainability.
    - Consistent move selection behavior across all code paths.

### Move Application Utility Refactoring (**2025-07-23**) **[RESOLVED]**

- **Created core tensor-based function:** `apply_move_to_tensor()` for efficient 3-channel tensor manipulation
- **Created wrapper functions:** `apply_move_to_state()`, `apply_move_to_state_trmph()`, `apply_move_to_tensor_trmph()`
- **Updated all call sites:** Web app, CLI scripts, and tournament code now use centralized utilities
- **Added comprehensive tests:** New test suite in `tests/test_move_application_utilities.py` with 19 test cases
- **Benefits:**
    - Eliminated code duplication across move application logic
    - Improved efficiency with direct tensor manipulation
    - Better error handling and consistency
    - Easier to test and maintain
    - All tests pass (23/23)

### TRMPH Function Duplication (**2025-07-23**) **[RESOLVED]**

- **Problem:** TRMPH conversion functions were duplicated between `hex_ai/data_utils.py` and `hex_ai/utils/format_conversion.py`
- **Solution:** 
  - Removed duplicate functions from `data_utils.py` (`strip_trmph_preamble`, `split_trmph_moves`, `trmph_move_to_rowcol`, `rowcol_to_trmph`, `tensor_to_rowcol`, `rowcol_to_tensor`, `tensor_to_trmph`, `trmph_to_tensor`)
  - Updated imports in `hex_ai/inference/game_engine.py` to use `format_conversion` instead of `data_utils`
  - Updated imports in `tests/test_advanced_winner_detection.py`
- **Benefits:** Eliminated circular import issues, single source of truth for TRMPH functions
- **Files Modified:** `hex_ai/data_utils.py`, `hex_ai/inference/game_engine.py`, `tests/test_advanced_winner_detection.py`

---

## Still To Do

### 1. Dataset Classes

#### StreamingAugmentedProcessedDataset vs. StreamingSequentialShardDataset
- **Difference:** Random-access/in-memory vs. streaming/sequential. Both exist for different use-cases, but code could be unified or better documented.

### 2. Trainer Methods

#### Trainer.train vs. Trainer.train_on_batches
- **Difference:** Monolithic vs. modular training loop. Consider unifying or clearly documenting intended usage.

### 3. Inference and CLI Duplication
- `hex_ai/inference/simple_model_inference.py` and `scripts/simple_inference_cli.py` may have legacy or redundant code paths. Review for deletion or further refactoring.
- Model loading and inference logic is now centralized in `ModelWrapper`, but older scripts/utilities may still use custom or legacy code paths.
- Data preprocessing for inference vs. training should be unified; update all code to use `preprocess_example_for_model` where possible.

### 4. Player Identification Code: BLUE / RED, Constants and Enums
- Both constants (in `hex_ai/config.py`) and enums/utilities (in `hex_ai/value_utils.py`) are used. Decide on a single system for clarity.

### 5. Training Example Extraction
- Both `extract_training_examples_from_game` and `extract_training_examples_with_selector_from_game` exist; one may be dead code.

### 6. StreamingSequentialShardDataset __len__ Hack (**Technical Debt**)
- Dummy `__len__` returns a huge value for PyTorch compatibility. Remove when possible.

### 7. Test Suite Cleanup
- The test suite does not fully pass; needs to be cleaned up and modernized.

---

## Notes
- As the codebase is modernized, update this document to reflect new refactors, resolved issues, and remaining technical debt.
- Use this as a living checklist for ongoing code health and cleanup. 
