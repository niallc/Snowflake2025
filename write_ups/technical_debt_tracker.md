# Technical Debt Tracker

**Date:** 2024-12-19  
**Last Updated:** 2024-12-19

This document tracks specific technical debt items and refactoring tasks that need to be addressed. Focus is on actionable items that improve code quality and maintainability.

---

## 🔄 High Priority - Needs Attention

### 1. Value Head Documentation and Consistency
- **Priority:** High
- **Issue:** Inconsistent value head interpretation across codebase
- **Action:** 
  - Document what the value head predicts (blue win probability vs red win)
  - Audit all value head usage across codebase
  - Fix inconsistent value interpretation
  - Resolve TODO in `app.py` about regenerating value logits
  - Update `fixed_tree_search.py` value interpretation if needed

### 2. Inference and CLI Duplication Review
- **Priority:** Medium
- **Files:** `hex_ai/inference/simple_model_inference.py` and `scripts/simple_inference_cli.py`
- **Issue:** May have legacy or redundant code paths
- **Action:** Review for deletion or further refactoring
- **Notes:** Model loading and inference logic is now centralized in `ModelWrapper`, but older scripts/utilities may still use custom or legacy code paths

### 3. Data Preprocessing Unification
- **Priority:** Medium
- **Issue:** Data preprocessing for inference vs. training should be unified
- **Action:** Update all code to use `preprocess_example_for_model` where possible

---

## 📋 Medium Priority - Code Quality

### 4. Import Organization
- **Priority:** Medium
- **Issue:** Many inline imports should be moved to the top and alphabetized
- **Action:** 
  - Audit all Python files for inline imports
  - Move all imports to top of files
  - Alphabetize imports within categories (standard library, third-party, local)
  - Add import organization to linting rules if possible

### 5. Inference Batching Optimization
- **Priority:** Medium
- **Location:** `hex_ai/inference/fixed_tree_search.py`
- **Issue:** TODO at line ~177: "This looks like it's passing the boards one at a time. The reason for batching is that networks are faster when batching."
- **Action:** Implement proper batching in inference code where possible

### 6. Mock Model Replacement
- **Priority:** Medium
- **Issue:** Mock models used in testing may not reflect real model behavior
- **Action:** Replace mock models with lightweight real models or improve mock fidelity

---

## 📋 Low Priority - Documentation & Organization

### 7. Documentation Organization
- **Priority:** Low
- **Issue:** Multiple documentation locations (`write_ups/`, `docs/`, `hex_ai/`) without clear organization
- **Action:** 
  - Define documentation structure and purpose for each location
  - Create project map or documentation index
  - Consider migrating files to appropriate locations
  - Document the organization system

### 8. Enhanced Testing Coverage
- **Priority:** Medium
- **Issue:** Some areas lack comprehensive test coverage
- **Action:** 
  - Add tests for critical inference paths
  - Improve integration test coverage
  - Add property-based testing for data transformations

### 9. Dependency Management
- **Priority:** Low
- **Issue:** Requirements may be outdated or include unnecessary dependencies
- **Action:** 
  - Audit and update requirements.txt
  - Remove unused dependencies
  - Pin critical dependency versions

---

## ✅ Recently Completed (Archive)

*Note: Completed work has been moved to separate documentation files to keep this tracker focused on actionable items.*
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

### 6. Documentation Organization
- **Priority:** Low
- **Issue:** Multiple documentation locations (`write_ups/`, `docs/`, `hex_ai/`) without clear organization
- **Action:** 
  - Define documentation structure and purpose for each location
  - Create project map or documentation index
  - Consider migrating files to appropriate locations
  - Document the organization system

---

## 📝 Notes

- Use this as a living checklist for ongoing code health and cleanup
- Remove items as they are resolved to keep the document focused on actionable work
- Update status and add new items as they are discovered
- For high-level code health issues and recommendations, see `code_health_overview.md`

---

## 🗑️ Removed Items

Items that have been completed or are no longer relevant should be moved here with completion dates and brief summaries. 