# Code Improvement Plan

**Date:** 2024-12-19  
**Last Updated:** 2024-12-19

This document outlines a prioritized plan for improving the Hex AI codebase. Focus is on actionable items that need attention.

---

## 🎯 High Priority - Critical Issues

### 1. Value Head Documentation and Consistency
**Estimated Time:** 1-2 days  
**Impact:** High - affects inference accuracy and debugging

**Tasks:**
- [ ] Document what the value head predicts (blue win probability vs red win)
- [ ] Audit all value head usage across codebase
- [ ] Fix inconsistent value interpretation
- [ ] Resolve TODO in `app.py` about regenerating value logits
- [ ] Update `fixed_tree_search.py` value interpretation if needed

**Files to modify:**
- `hex_ai/inference/fixed_tree_search.py`
- `hex_ai/web/app.py`
- `hex_ai/models.py` (value head documentation)
- Create: `docs/value_head_specification.md`

---

## 🔧 Medium Priority - Code Quality

### 2. Import Organization
**Estimated Time:** 1 day  
**Impact:** Medium - improves code readability and maintainability

**Tasks:**
- [ ] Audit all Python files for inline imports
- [ ] Move all imports to top of files
- [ ] Alphabetize imports within categories (standard library, third-party, local)
- [ ] Add import organization to linting rules if possible

**Files to audit:**
- All Python files in `hex_ai/`
- All Python files in `scripts/`
- All Python files in `tests/`

### 3. Inference and CLI Duplication Review
**Estimated Time:** 1-2 days  
**Impact:** Medium - reduces code duplication and maintenance burden

**Tasks:**
- [ ] Compare `hex_ai/inference/simple_model_inference.py` and `scripts/simple_inference_cli.py`
- [ ] Identify redundant code paths
- [ ] Consolidate or remove duplicate functionality
- [ ] Ensure all inference uses centralized `ModelWrapper`

**Files to review:**
- `hex_ai/inference/simple_model_inference.py`
- `scripts/simple_inference_cli.py`
- `hex_ai/inference/model_wrapper.py`

### 4. Data Preprocessing Unification
**Estimated Time:** 1 day  
**Impact:** Medium - ensures consistent data handling

**Tasks:**
- [ ] Audit all data preprocessing code
- [ ] Identify where `preprocess_example_for_model` should be used
- [ ] Update code to use unified preprocessing
- [ ] Add tests for preprocessing consistency

**Files to audit:**
- `hex_ai/data_processing.py`
- `hex_ai/batch_processor.py`
- `hex_ai/inference/` (all inference files)

---

## ⚡ Medium Priority - Performance & Testing

### 5. Inference Batching Optimization
**Estimated Time:** 1-2 days  
**Impact:** Medium - improves inference performance

**Tasks:**
- [ ] Implement proper batching in `fixed_tree_search.py`
- [ ] Optimize model inference for batch processing
- [ ] Add batch size configuration options
- [ ] Measure performance improvements

**Files to modify:**
- `hex_ai/inference/fixed_tree_search.py`
- `hex_ai/inference/model_wrapper.py`

### 6. Mock Model Replacement
**Estimated Time:** 1-2 days  
**Impact:** Medium - improves test quality

**Tasks:**
- [ ] Replace hardcoded heuristics in `test_minimax_debug.py`
- [ ] Create proper test cases with real game positions
- [ ] Ensure tests reflect actual model behavior

**Files to modify:**
- `tests/test_minimax_debug.py`

### 7. Enhanced Testing Coverage
**Estimated Time:** 2-3 days  
**Impact:** Medium - improves code reliability

**Tasks:**
- [ ] Add tests for data loading
- [ ] Add tests for model instantiation
- [ ] Add tests for checkpoint loading
- [ ] Add tests for inference pipeline
- [ ] Add integration tests for training pipeline

**Files to create/modify:**
- `tests/test_data_loading.py`
- `tests/test_model_instantiation.py`
- `tests/test_checkpoint_loading.py`
- `tests/test_inference_pipeline.py`

---

## 📚 Low Priority - Documentation & Organization

### 8. Documentation Organization
**Estimated Time:** 1-2 days  
**Impact:** Low - improves developer experience

**Tasks:**
- [ ] Define documentation structure and purpose for each location
- [ ] Create project map or documentation index
- [ ] Consider migrating files to appropriate locations
- [ ] Document the organization system

### 9. Dependency Management
**Estimated Time:** 1 day  
**Impact:** Low - ensures reproducible builds

**Tasks:**
- [ ] Audit and update `requirements.txt`
- [ ] Add version pinning where appropriate
- [ ] Create environment validation script
- [ ] Document dependency update process

**Files to modify:**
- `requirements.txt`
- Create: `scripts/validate_environment.py`

---

## 📝 Notes

- This plan should be updated as work progresses
- Some tasks may reveal additional issues that should be added to the plan
- Consider breaking large tasks into smaller, more manageable pieces
- Regular code reviews should be conducted to prevent new technical debt from accumulating

**For detailed tracking of individual tasks, see `technical_debt_tracker.md`.** 