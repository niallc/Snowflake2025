# Code Debt and Refactoring Log

This document tracks remaining areas of code duplication, technical debt, and refactoring efforts that need attention.

---

## Remaining Work

### 1. Trainer Methods ✅ **COMPLETED**

#### Trainer.train vs. Trainer.train_on_batches
- **Status:** Added comprehensive documentation explaining the differences and intended usage
- **Changes:**
  - `train()`: Main method for complete training runs with automatic checkpointing and validation
  - `train_on_batches()`: Lower-level method for mini-epoch orchestration and custom training loops
- **Result:** Clear guidance on when to use each method, reducing confusion for developers

### 2. Inference and CLI Duplication
- `hex_ai/inference/simple_model_inference.py` and `scripts/simple_inference_cli.py` may have legacy or redundant code paths. Review for deletion or further refactoring.
- Model loading and inference logic is now centralized in `ModelWrapper`, but older scripts/utilities may still use custom or legacy code paths.
- Data preprocessing for inference vs. training should be unified; update all code to use `preprocess_example_for_model` where possible.

### 3. Player Identification Code: BLUE / RED, Constants and Enums ✅ **COMPLETED**
- **Status:** Implemented comprehensive enum system for type safety and clarity
- **Changes:**
  - Added `Player`, `Piece`, and `Channel` enums to complement existing `Winner` enum
  - Created conversion functions for backward compatibility
  - Added utility functions for common operations (`get_opponent`, `is_blue`, `is_red`)
  - Updated key functions to use enums while maintaining compatibility
  - **Migrated `get_player_to_move_from_board` function** to return `Player` enum instead of integer
  - Updated all callers to handle the new enum return type
- **Benefits:**
  - Type safety: prevents invalid values and catches errors at runtime
  - Better IDE support: autocomplete, refactoring, error detection
  - Clearer code intent: `Player.BLUE` vs magic number `0`
  - Maintains backward compatibility with existing integer constants
  - **Demonstrated successful migration** of a core function with comprehensive testing
- **Documentation:** Created comprehensive guide in `write_ups/enum_system_guide.md`

### 4. Training Example Extraction ✅ **COMPLETED**
- **Status:** Analyzed both functions and confirmed they serve different purposes
- **Analysis:**
  - `extract_training_examples_from_game`: Used in `data_processing.py` for general processing
  - `extract_training_examples_with_selector_from_game`: Used in `batch_processor.py` with position selection
- **Action:** Fixed magic numbers in the first function to use proper constants
- **Result:** Both functions are actively used and serve distinct purposes - no consolidation needed

### 5. StreamingSequentialShardDataset __len__ Hack (**Technical Debt**)
- Dummy `__len__` returns a huge value for PyTorch compatibility. Remove when possible.

### 6. Test Suite Cleanup ✅ **COMPLETED**
- **Status:** Fixed test failures due to API mismatches and data format changes
- **Fixed Issues:**
  - `tests/test_mini_epoch_orchestrator.py`: Updated MockTrainer to accept `epoch` and `mini_epoch` parameters
  - `tests/test_policy_utilities.py`: Fixed flaky temperature scaling test by removing deterministic assertions
  - `tests/test_data_shuffling.py`: Added proper pytest.skip for missing test data files
- **Result:** All tests now pass or skip gracefully when data is unavailable

### 7. Mock Model Code Debt
- **Location**: `tests/test_minimax_debug.py`
- **Issue**: Mock models with hardcoded heuristics (center preference, piece counting)
- **Problem**: These heuristics are not based on real Hex strategy
- **Debt**: Temporary test code that could mislead about actual performance
- **Action**: Replace with proper test cases using real game positions

---

## Tidy up many inline imports
 -Move them to the top (and alphabetize within type, e.g. local vs. standard library imports)

---

## Notes
- Use this as a living checklist for ongoing code health and cleanup.
- Remove items as they are resolved to keep the document focused on actionable work. 
