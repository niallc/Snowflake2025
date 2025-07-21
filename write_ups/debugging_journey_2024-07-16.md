# Debugging Journey: Hex AI Model Training and Data Issues

**Date:** 2024-07-16 to present

## Overview

This document tracks the investigation into poor policy head performance after a major Hex AI model and training pipeline redesign. It summarizes the main findings, useful utilities, and the incremental migration/testing process that led to isolating the root cause(s) and restoring good training performance.

---

## 1. Key Findings and Progress

- **Initial problem:** After updating the network and training pipeline, policy loss barely improved from random, despite using the same data that previously worked well.
- **Data quality:** Diagnostic scripts confirmed that most data files are clean, but a minority contain systematic errors (e.g., color-swapped games due to repeated moves in TRMPH records). However, these errors are not new and do not explain the regression, since previous training succeeded with the same data.
- **Data pipeline:** Extensive comparison between raw records and loader output (using `scripts/compare_loader_vs_raw.py`) showed no discrepancies. The data pipeline is not introducing bugs.
- **Overfitting sanity checks:** The model and loss function can overfit tiny datasets, confirming that learning is possible in principle.
- **Legacy code restoration:** Restoring the legacy model and training pipeline (2-channel, 3x3 conv) led to rapid policy loss reduction, confirming that the data and basic architecture are sound.
- **Incremental migration:** By adding back changes one at a time (player-to-move channel, 5x5 conv, modern Trainer, streaming data loader), each step was validated in isolation. At every stage, policy loss continued to drop rapidly, matching legacy performance.
- **Modern Trainer with legacy model:** The most recent step was to run the modern Trainer (from `hex_ai/training.py`) with the legacy model (with player-to-move channel). Training performance remained good, confirming that the Trainer and data pipeline are not the source of the regression.

---

## 2. Useful Utilities and Scripts

- **Data analysis and error detection:**
    - `scripts/identify_bad_files.py` — scan for and exclude problematic files
    - `scripts/find_problematic_samples.py` — extract and inspect error samples
    - `scripts/compare_loader_vs_raw.py` — verify loader output matches raw data
- **Training/testing:**
    - `scripts/overfit_tiny_dataset.py` — overfit on a single game for sanity checks
    - `scripts/hyperparameter_tuning_legacy.py` — run hyperparameter sweeps with legacy/modern Trainer
    - `scripts/compare_training_pipelines.py` — compare different training configurations
    - `scripts/test_3channel_legacy.py` — verify 3-channel legacy model
- **Comparison and analysis:**
    - `scripts/compare_legacy_vs_modified.py` — compare performance curves

---

## 3. Incremental Migration and Testing Plan (2024-07-18)

**Goal:** Identify which change(s) broke policy head learning by testing each modification individually.

### Steps:
1. **Restore legacy baseline** — Confirmed legacy model and pipeline work (good policy loss)
2. **Add player-to-move channel** — 3-channel legacy model, still good performance
3. **Switch to 5x5 first convolution** — Still good performance
4. **Switch to modern Trainer** — Still good performance (recently validated)
5. **Switch to streaming data loader** — Still good performance
6. **Final step: Swap in modern model** — (Next to test)

At each step, if performance regressed, the change would be isolated and debugged. So far, all steps have passed.

---

## 4. Current Status and Next Steps

- **Current status:**
    - The modern Trainer and data pipeline have been validated with the legacy (player-channel) model. Training performance is as good as with the legacy pipeline.
    - This rules out the Trainer and data pipeline as the source of the regression.
- **Next step:**
    - Swap in the modern model (with all architectural changes) and test training performance.
    - If regression reappears, the issue is isolated to the model architecture or its interaction with the data/loss.
    - If not, the migration is complete and the modern pipeline is validated.

---

## 5. Summary

- The debugging process has systematically ruled out data, data pipeline, and Trainer as the source of poor policy head learning.
- The incremental migration approach has restored good performance at every step except the final model swap.
- The next experiment will definitively isolate whether the modern model architecture is responsible for the regression.

---

(Older, obsolete, or overly detailed debugging notes have been removed for clarity. See previous git history for full details if needed.) 

## Legacy Usage and Migration Status (2024-07-18)

- The current hyperparameter tuning flow (`scripts/hyperparameter_tuning_legacy.py`) now uses only modern model, data pipeline, and Trainer code. The only remaining legacy aspect is the orchestration function (`run_hyperparameter_tuning_current_data`), which lives in a legacy-named file but does not use any legacy model, dataset, or training logic.
- DataLoader shuffling is now harmonized (`shuffle=False`), and performance remains good, confirming that upstream shuffling is sufficient.
- There is no urgent need to migrate the orchestration code out of the legacy file, as it does not affect training or results. When the codebase is further consolidated, duplicated or obsolete legacy code can be deleted.
- **Next step:** Run a full experiment via `scripts/hyperparam_sweep.py` (which now uses the fully modernized pipeline) and observe whether the performance issue persists. If it does, further comparison or migration decisions can be made; if not, the migration is effectively complete. 