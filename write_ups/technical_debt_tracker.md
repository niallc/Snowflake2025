# Technical Debt Tracker & Improvement Plan

**Date:** 2024-12-19  
**Last Updated:** 2024-12-19

This document tracks specific technical debt items and refactoring tasks that need to be addressed. Focus is on actionable items that improve code quality and maintainability.

---

## 🎯 High Priority - Critical Issues

### 1. Value Head Documentation and Consistency
**Priority:** High  
**Estimated Time:** 1-2 days  
**Impact:** High - affects inference accuracy and debugging

**Issue:** Inconsistent value head interpretation across codebase

**Tasks:**
- [x] Document what the value head predicts (Red's win probability)
- [x] Audit all value head usage across codebase
- [x] Fix inconsistent value interpretation
- [x] Resolve TODO in `fixed_tree_search.py` about value interpretation
- [x] Create: `docs/value_head_specification.md`

**Files to modify:**
- `hex_ai/inference/fixed_tree_search.py` ✅
- `hex_ai/value_utils.py` ✅
- `hex_ai/models.py` ✅
- `hex_ai/training.py` ✅
- `docs/value_head_specification.md` ✅

---

## 🔧 Medium Priority - Code Quality

### 2. Import Organization
**Priority:** Medium  
**Estimated Time:** 1 day  
**Impact:** Medium - improves code readability and maintainability

**Issue:** Many inline imports should be moved to the top and alphabetized

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
**Priority:** Medium  
**Estimated Time:** 1-2 days  
**Impact:** Medium - reduces code duplication and maintenance burden

**Issue:** May have legacy or redundant code paths between inference modules

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
**Priority:** Medium  
**Estimated Time:** 1 day  
**Impact:** Medium - ensures consistent data handling

**Issue:** Data preprocessing for inference vs. training should be unified

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
**Priority:** Medium  
**Estimated Time:** 1-2 days  
**Impact:** Medium - improves inference performance

**Issue:** TODO at line ~177 in `fixed_tree_search.py`: "This looks like it's passing the boards one at a time. The reason for batching is that networks are faster when batching."

**Tasks:**
- [ ] Implement proper batching in `fixed_tree_search.py`
- [ ] Optimize model inference for batch processing
- [ ] Add batch size configuration options
- [ ] Measure performance improvements

**Files to modify:**
- `hex_ai/inference/fixed_tree_search.py`
- `hex_ai/inference/model_wrapper.py`

### 6. Mock Model Replacement
**Priority:** Medium  
**Estimated Time:** 1-2 days  
**Impact:** Medium - improves test quality

**Issue:** Mock models used in testing may not reflect real model behavior

**Tasks:**
- [ ] Replace hardcoded heuristics in `test_minimax_debug.py`
- [ ] Create proper test cases with real game positions
- [ ] Ensure tests reflect actual model behavior

**Files to modify:**
- `tests/test_minimax_debug.py`

### 7. Enhanced Testing Coverage
**Priority:** Medium  
**Estimated Time:** 2-3 days  
**Impact:** Medium - improves code reliability

**Issue:** Some areas lack comprehensive test coverage

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
**Priority:** Low  
**Estimated Time:** 1-2 days  
**Impact:** Low - improves developer experience

**Issue:** Multiple documentation locations (`write_ups/`, `docs/`, `hex_ai/`) without clear organization

**Tasks:**
- [ ] Define documentation structure and purpose for each location
- [ ] Create project map or documentation index
- [ ] Consider migrating files to appropriate locations
- [ ] Document the organization system

### 9. Dependency Management
**Priority:** Low  
**Estimated Time:** 1 day  
**Impact:** Low - ensures reproducible builds

**Issue:** Requirements may be outdated or include unnecessary dependencies

**Tasks:**
- [ ] Audit and update `requirements.txt`
- [ ] Add version pinning where appropriate
- [ ] Create environment validation script
- [ ] Document dependency update process

**Files to modify:**
- `requirements.txt`
- Create: `scripts/validate_environment.py`

### 10. Deprecated Scripts Cleanup
**Priority:** Low  
**Estimated Time:** 1-2 days  
**Impact:** Low - improves codebase organization

**Issue:** Large number of deprecated scripts in `scripts/deprecated/` directory

**Tasks:**
- [ ] Review all deprecated scripts for any still-useful functionality
- [ ] Extract any reusable utilities to appropriate modules
- [ ] Remove scripts that are no longer needed
- [ ] Update documentation to reflect current state
- [ ] Consider archiving important historical scripts

**Files to review:**
- All files in `scripts/deprecated/` (30+ files)
- `write_ups/redundant/` directory for related documentation

### 11. Redundant Documentation Cleanup
**Priority:** Low  
**Estimated Time:** 1 day  
**Impact:** Low - improves documentation organization

**Issue:** Large number of redundant documentation files in `write_ups/redundant/` directory

**Tasks:**
- [ ] Review all redundant documentation for any still-relevant information
- [ ] Extract any useful content to appropriate documentation files
- [ ] Remove files that are completely outdated
- [ ] Consider creating a historical archive for important completed work
- [ ] Update main documentation to reference any preserved historical information

**Files to review:**
- All files in `write_ups/redundant/` (20+ files)

---

## 📝 Notes

- Use this as a living checklist for ongoing code health and cleanup
- Remove items as they are resolved to keep the document focused on actionable work
- Update status and add new items as they are discovered
- For high-level code health issues and recommendations, see `code_health_overview.md`
- Consider breaking large tasks into smaller, more manageable pieces
- Regular code reviews should be conducted to prevent new technical debt from accumulating

---

## 🗑️ Removed Items

Items that have been completed or are no longer relevant should be moved here with completion dates and brief summaries. 