# Technical Debt Tracker & Improvement Plan

**Date:** 2024-12-19  
**Last Updated:** 2024-12-19

This document tracks specific technical debt items and refactoring tasks that need to be addressed. Focus is on actionable items that improve code quality and maintainability.

---

## 🎯 High Priority - Critical Issues



---

## 🔧 Medium Priority - Code Quality

### 2. Import Organization
**Priority:** Medium  
**Estimated Time:** 1 day  
**Impact:** Medium - improves code readability and maintainability

**Issue:** Many inline imports should be moved to the top and alphabetized

**Tasks:**
- [x] Audit all Python files for inline imports
- [x] Move all imports to top of files
- [ ] Alphabetize imports within categories (standard library, third-party, local)
- [ ] Add import organization to linting rules if possible

**Files to audit:**
- All Python files in `hex_ai/` ✅ **COMPLETED**
- All Python files in `scripts/` ✅ **COMPLETED**
- All Python files in `tests/` (not part of main scope)

**Completed:** Moved all inline imports to the top of files in `scripts/` and `hex_ai/` directories. The remaining inline imports are primarily in test files which were not part of the main scope.

### 2. Inference and CLI Duplication Review
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

### 3. Data Preprocessing Unification
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

### 4. Inference Batching Optimization
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

### 5. Mock Model Replacement
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

### 6. Enhanced Testing Coverage
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

### 7. Documentation Organization
**Priority:** Low  
**Estimated Time:** 1-2 days  
**Impact:** Low - improves developer experience

**Issue:** Multiple documentation locations (`write_ups/`, `docs/`, `hex_ai/`) without clear organization

**Tasks:**
- [ ] Define documentation structure and purpose for each location
- [ ] Create project map or documentation index
- [ ] Consider migrating files to appropriate locations
- [ ] Document the organization system

### 8. Dependency Management
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

### 9. Rationalize hex_ai/config.py and hex_ai/value_utils.py
 - These two files strongly overlap in their goals.
 - We should likely switch to a single system based primarily off Enums, for safety.
 - As we do that, (see 10., below) we'll want to update app.js, as well as other python code.

### 10. Brittle reliance of javascript code on config.py
 - Non critical (web code is just user convenience)
 - static/app.js relies on config.py to get constants from the python
 - config.py may well be deprecated for the hex_ai/value_utils.py approach which uses Enums and is therefore likely safer.

 ### 11. There are two functions called get_top_k_legal_moves (with different signatures)
  - Need to decide on one, delete the other (or if they're very different rename), and update code that calls the renamed / deleted version.

### 12. Update on moves in winning games more?
 - Feature request rather than a bug: Do we want the policy head to update more on moves from the winning player?

### 13. More batching in self-play code.
 - Currently the self-play engine creates games one at a time, which 

### 14. Multi-model and Search_widths flexibility in Self Play
 - Implement ability to play model A vs. model B in self-play
 - Implement ability to have different move-making strategies.

### 15. Implement MCTS

### 16. Remove _generate_single_game is unused
 - hex_ai/selfplay/selfplay_engine.py:158:
 - def _generate_single_game(self, board_size: int) -> Dict[str, Any]:

### 17. Too many functions to figure out the GPU situation
 - hex_ai/training_utils.py:116:def get_device() -> str:
 - hex_ai/inference/model_wrapper.py:37:        return get_device()
 - hex_ai/training_orchestration.py:222:def select_device():
 - hex_ai/training_orchestration.py:288:    device = select_device()
 - scripts/validate_checkpoints.py:38:        self.device = get_device()

---

## 📝 Notes

- Use this as a living checklist for ongoing code health and cleanup
- Remove items as they are resolved to keep the document focused on actionable work
- Update status and add new items as they are discovered
- For high-level code health issues and recommendations, see `code_health_overview.md`
- Consider breaking large tasks into smaller, more manageable pieces
- Regular code reviews should be conducted to prevent new technical debt from accumulating 