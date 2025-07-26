# Code Duplication and Refactoring Log

This document tracks known areas of code duplication, technical debt, and major refactoring efforts in the project. Items are marked as **[RESOLVED]** or **[TODO]** as appropriate.

---

## Resolved

### Policy Processing & Move Selection Refactoring (**2025-07-23**) **[RESOLVED]**

- **Centralized Utilities:** All policy processing and move selection logic is now handled by canonical utilities in `hex_ai/value_utils.py`.

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
