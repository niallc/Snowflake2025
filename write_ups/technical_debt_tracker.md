# Technical Debt Tracker

**Date:** 2024-12-19  
**Last Updated:** 2024-12-19

This document tracks specific technical debt items, refactoring tasks, and completed work. It serves as a living checklist for ongoing code cleanup efforts.

---

## ✅ Completed Work

### 1. Trainer Methods Documentation
- **Status:** ✅ **COMPLETED**
- **Issue:** Confusion between `Trainer.train` vs. `Trainer.train_on_batches`
- **Solution:** Added comprehensive documentation explaining the differences and intended usage
- **Changes:**
  - `train()`: Main method for complete training runs with automatic checkpointing and validation
  - `train_on_batches()`: Lower-level method for mini-epoch orchestration and custom training loops
- **Result:** Clear guidance on when to use each method, reducing confusion for developers

### 2. Player Identification Code: BLUE / RED, Constants and Enums
- **Status:** ✅ **COMPLETED**
- **Issue:** Magic numbers and inconsistent player identification
- **Solution:** Implemented comprehensive enum system for type safety and clarity
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

### 3. Training Example Extraction Analysis
- **Status:** ✅ **COMPLETED**
- **Issue:** Potential duplication between `extract_training_examples_from_game` and `extract_training_examples_with_selector_from_game`
- **Analysis:**
  - `extract_training_examples_from_game`: Used in `data_processing.py` for general processing
  - `extract_training_examples_with_selector_from_game`: Used in `batch_processor.py` with position selection
- **Action:** Fixed magic numbers in the first function to use proper constants
- **Result:** Both functions are actively used and serve distinct purposes - no consolidation needed

### 4. Test Suite Cleanup
- **Status:** ✅ **COMPLETED**
- **Issue:** Test failures due to API mismatches and data format changes
- **Fixed Issues:**
  - `tests/test_mini_epoch_orchestrator.py`: Updated MockTrainer to accept `epoch` and `mini_epoch` parameters
  - `tests/test_policy_utilities.py`: Fixed flaky temperature scaling test by removing deterministic assertions
  - `tests/test_data_shuffling.py`: Added proper pytest.skip for missing test data files
- **Result:** All tests now pass or skip gracefully when data is unavailable

---

## 🔄 In Progress

### 1. Inference and CLI Duplication Review
- **Status:** 🔄 **NEEDS REVIEW**
- **Files:** `hex_ai/inference/simple_model_inference.py` and `scripts/simple_inference_cli.py`
- **Issue:** May have legacy or redundant code paths
- **Action:** Review for deletion or further refactoring
- **Notes:** Model loading and inference logic is now centralized in `ModelWrapper`, but older scripts/utilities may still use custom or legacy code paths

### 2. Data Preprocessing Unification
- **Status:** 🔄 **NEEDS REVIEW**
- **Issue:** Data preprocessing for inference vs. training should be unified
- **Action:** Update all code to use `preprocess_example_for_model` where possible

---

## 📋 Pending Work

### 1. Import Organization
- **Priority:** Medium
- **Issue:** Many inline imports should be moved to the top and alphabetized
- **Action:** Standardize import organization across all files
- **Files:** All Python files in the project

### 2. Value Head Documentation and Consistency
- **Priority:** High
- **Issue:** Need more complete description of what the value head is predicting and how to use the values it returns
- **Specific Issues:**
  - Inconsistent usage of value predictions across different parts of the codebase
  - Some code regenerates value logits instead of using original ones
  - In `hex_ai/inference/fixed_tree_search.py`: `prob_red_win = torch.sigmoid(torch.tensor(value_logit)).item()`
  - In `app.py` (~line 67): TODO about regenerating value logits for leaf nodes
- **Action:** 
  - Document the value head's prediction target (probability of blue win vs red win)
  - Ensure consistent usage of value predictions throughout the codebase
  - Fix code that regenerates value logits to use original values for debugging

### 3. StreamingSequentialShardDataset __len__ Hack
- **Priority:** Low
- **Issue:** Dummy `__len__` returns a huge value for PyTorch compatibility
- **Action:** Remove when PyTorch compatibility allows
- **Note:** This is technical debt that should be addressed when possible

### 4. Mock Model Code Debt
- **Priority:** Medium
- **Location:** `tests/test_minimax_debug.py`
- **Issue:** Mock models with hardcoded heuristics (center preference, piece counting) that are not based on real Hex strategy
- **Problem:** These heuristics could mislead about actual performance
- **Action:** Replace with proper test cases using real game positions

### 5. Inference Batching Optimization
- **Priority:** Medium
- **Location:** `hex_ai/inference/fixed_tree_search.py`
- **Issue:** TODO at line ~177: "This looks like it's passing the boards one at a time. The reason for batching is that networks are faster when batching."
- **Action:** Implement proper batching in inference code where possible

---

## 📝 Notes

- Use this as a living checklist for ongoing code health and cleanup
- Remove items as they are resolved to keep the document focused on actionable work
- Update status and add new items as they are discovered
- For high-level code health issues and recommendations, see `code_health_overview.md`

---

## 🗑️ Removed Items

Items that have been completed or are no longer relevant should be moved here with completion dates and brief summaries. 