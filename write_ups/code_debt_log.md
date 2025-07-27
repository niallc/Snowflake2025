# Code Debt and Refactoring Log

This document tracks remaining areas of code duplication, technical debt, and refactoring efforts that need attention.

---

## Remaining Work

### 1. Trainer Methods

#### Trainer.train vs. Trainer.train_on_batches
- **Difference:** Monolithic vs. modular training loop. Consider unifying or clearly documenting intended usage.

### 2. Inference and CLI Duplication
- `hex_ai/inference/simple_model_inference.py` and `scripts/simple_inference_cli.py` may have legacy or redundant code paths. Review for deletion or further refactoring.
- Model loading and inference logic is now centralized in `ModelWrapper`, but older scripts/utilities may still use custom or legacy code paths.
- Data preprocessing for inference vs. training should be unified; update all code to use `preprocess_example_for_model` where possible.

### 3. Player Identification Code: BLUE / RED, Constants and Enums
- Both constants (in `hex_ai/config.py`) and enums/utilities (in `hex_ai/value_utils.py`) are used. Decide on a single system for clarity.

### 4. Training Example Extraction
- Both `extract_training_examples_from_game` and `extract_training_examples_with_selector_from_game` exist; one may be dead code.

### 5. StreamingSequentialShardDataset __len__ Hack (**Technical Debt**)
- Dummy `__len__` returns a huge value for PyTorch compatibility. Remove when possible.

### 6. Test Suite Cleanup
- **Remaining Issues:** Some tests still fail due to data format changes and API differences that need deeper investigation
- **Action:** Investigate and fix remaining test failures, particularly in:
  - `tests/test_data_shuffling.py`
  - `tests/test_mini_epoch_orchestrator.py`
  - `tests/test_policy_utilities.py`

### 7. Mock Model Code Debt
- **Location**: `tests/test_minimax_debug.py`
- **Issue**: Mock models with hardcoded heuristics (center preference, piece counting)
- **Problem**: These heuristics are not based on real Hex strategy
- **Debt**: Temporary test code that could mislead about actual performance
- **Action**: Replace with proper test cases using real game positions

---

## Notes
- Use this as a living checklist for ongoing code health and cleanup.
- Remove items as they are resolved to keep the document focused on actionable work. 
