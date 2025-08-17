# Technical Debt Tracker & Improvement Plan

**Last Updated:** 2025-08-14

This document tracks specific technical debt items and refactoring tasks that need to be addressed. Focus is on actionable items that improve code quality and maintainability.

---

## 🔧 Medium Priority - Code Quality


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

### 8. Dependency Management ✅ COMPLETED
**Priority:** Low  
**Status:** ✅ **COMPLETED** - Updated requirements.txt with version pinning and better organization.

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

### 18. Self-play Data Format ✅ COMPLETED
**Priority:** Medium  
**Status:** ✅ **COMPLETED** - Removed .pkl.gz format and CSV output. Self-play now writes only TRMPH text files.

### 19. Self-play CSV Output ✅ COMPLETED  
**Priority:** Low  
**Status:** ✅ **COMPLETED** - Removed detailed CSV output functionality as it was not needed.

### 20. self_play_engine.py has relative paths to its imports
 - Also
  -  # Type annotation for PositionCollector (imported locally to avoid circular imports)
  - from typing import TYPE_CHECKING
  - if TYPE_CHECKING:
  - from ..inference.fixed_tree_search import PositionCollector
 - This all needs to be cleaned up.

### 21. Lots of duplication in methods to get the top moves from the policy network:
 - hex_ai/value_utils.py:279:def select_top_k_moves
 - hex_ai/value_utils.py:310:def get_top_k_moves_with_probs
 - hex_ai/inference/fixed_tree_search.py:118:def get_topk_moves_from_policy
 - hex_ai/inference/simple_model_inference.py:344:    def get_top_k_moves
 - hex_ai/inference/simple_model_inference.py:355:    def get_top_k_legal_moves

### 22. Update scripts/run_large_selfplay.py to handle new self-play & tournament data.
 - The new data from self play has metadata
 - Data files are in multiple directories
 - Tournament may use '1' & '2' for winner annotation instead of the newer (and preferred) 'b' and 'r'.

**Status:** PARTIALLY RESOLVED - Removed duplicate `_get_top_policy_moves` from selfplay_engine.py and renamed `_get_policy_move_values` to `_get_batched_policy_move_values` for clarity. The remaining functions serve different purposes:
- `select_top_k_moves`: Core utility for selecting top-k moves from legal policy array
- `get_top_k_moves_with_probs`: Main function that handles policy logits → legal moves → top-k with probabilities
- `get_topk_moves_from_policy`: Wrapper that extracts just moves (no probabilities) for tree search
- `get_top_k_moves`: Returns TRMPH format moves (different output format)
- `get_top_k_legal_moves`: Direct wrapper around `get_top_k_moves_with_probs` (could be removed)

---

## 📝 Notes

- Use this as a living checklist for ongoing code health and cleanup
- Remove items as they are resolved to keep the document focused on actionable work
- Update status and add new items as they are discovered
- For high-level code health issues and recommendations, see `code_health_overview.md`
- Consider breaking large tasks into smaller, more manageable pieces
- Regular code reviews should be conducted to prevent new technical debt from accumulating 