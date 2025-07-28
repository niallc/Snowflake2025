# Code Improvement Plan

**Date:** 2024-12-19  
**Last Updated:** 2024-12-19

This document outlines a prioritized plan for improving the Hex AI codebase based on the issues identified in the code health overview and technical debt tracker.

---

## 🎯 Phase 1: Critical Infrastructure (High Priority)

### 1.1 Model Checkpointing Standardization
**Estimated Time:** 2-3 days  
**Impact:** High - affects all model loading/saving operations

**Tasks:**
- [ ] Audit all checkpoint saving code to identify inconsistencies
- [ ] Define and document standard checkpoint format
- [ ] Create migration script for old checkpoints
- [ ] Update all save/load code to use standard format
- [ ] Add validation utilities for checkpoint files
- [ ] Update documentation in README

**Files to modify:**
- `hex_ai/models.py` (checkpoint loading logic)
- `hex_ai/training.py` (checkpoint saving)
- `hex_ai/inference/model_wrapper.py` (model loading)
- Create: `scripts/validate_checkpoints.py`
- Create: `scripts/migrate_checkpoints.py`

### 1.2 Value Head Documentation and Consistency
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

## 🔧 Phase 2: Code Quality and Organization (Medium Priority)

### 2.1 Import Organization
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

### 2.2 Inference and CLI Duplication Review
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

### 2.3 Data Preprocessing Unification
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

## ⚡ Phase 3: Performance and Optimization (Medium Priority)

### 3.1 Inference Batching Optimization
**Estimated Time:** 2-3 days  
**Impact:** Medium - improves inference performance

**Tasks:**
- [ ] Investigate TODO in `fixed_tree_search.py` about board batching
- [ ] Implement proper batching for inference where possible
- [ ] Profile performance improvements
- [ ] Add configuration options for batch sizes

**Files to modify:**
- `hex_ai/inference/fixed_tree_search.py`
- `hex_ai/inference/model_wrapper.py`

### 3.2 Streaming Dataset Optimization
**Estimated Time:** 1 day  
**Impact:** Low - removes technical debt

**Tasks:**
- [ ] Investigate PyTorch compatibility for proper `__len__` implementation
- [ ] Remove hack when possible
- [ ] Add proper documentation about the limitation

**Files to modify:**
- `hex_ai/data_processing.py` (or wherever StreamingSequentialShardDataset is defined)

---

## 🧪 Phase 4: Testing and Quality Assurance (Medium Priority)

### 4.1 Mock Model Replacement
**Estimated Time:** 1-2 days  
**Impact:** Medium - improves test quality

**Tasks:**
- [ ] Replace hardcoded heuristics in `test_minimax_debug.py`
- [ ] Create proper test cases with real game positions
- [ ] Ensure tests reflect actual model behavior

**Files to modify:**
- `tests/test_minimax_debug.py`

### 4.2 Enhanced Testing Coverage
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

## 📚 Phase 5: Documentation and Project Setup (Low Priority)

### 5.1 Project Documentation
**Estimated Time:** 1-2 days  
**Impact:** Low - improves developer experience

**Tasks:**
- [ ] Update README with environment setup instructions
- [ ] Add dependency management documentation
- [ ] Create development setup guide
- [ ] Add contribution guidelines

**Files to modify:**
- `README.md`
- Create: `docs/development_setup.md`
- Create: `docs/contributing.md`

### 5.2 Dependency Management
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

## 🎯 Implementation Strategy

### Prioritization Criteria
1. **Critical Infrastructure** - Fixes that prevent or significantly impact core functionality
2. **Code Quality** - Improvements that make the codebase more maintainable
3. **Performance** - Optimizations that improve user experience
4. **Testing** - Improvements that increase reliability
5. **Documentation** - Improvements that help developers

### Recommended Order
1. Start with Phase 1 (Critical Infrastructure) - these are blocking issues
2. Move to Phase 2 (Code Quality) - these make future work easier
3. Continue with Phase 3 (Performance) - these improve user experience
4. Complete with Phase 4 (Testing) and Phase 5 (Documentation) - these improve long-term maintainability

### Success Metrics
- [ ] All tests pass consistently
- [ ] No critical TODOs remain in codebase
- [ ] Checkpoint loading/saving is reliable and well-documented
- [ ] Value head usage is consistent across codebase
- [ ] Import organization is standardized
- [ ] Performance bottlenecks are identified and addressed

---

## 📝 Notes

- This plan should be updated as work progresses
- Some tasks may reveal additional issues that should be added to the plan
- Consider breaking large tasks into smaller, more manageable pieces
- Regular code reviews should be conducted to prevent new technical debt from accumulating

**For detailed tracking of individual tasks, see `technical_debt_tracker.md`.** 