# Debugging Journey: Hex AI Model Training and Data Issues

**Date:** 2024-07-16

## Overview

This document summarizes our ongoing debugging and investigation into the recent issues encountered with Hex AI model training, following a major network and training pipeline redesign. It covers the symptoms, hypotheses, investigative steps, tools developed, and current open questions.

---

## 1. Background and Motivation

- We updated the network architecture and training pipeline as described in `refine_model_and_training.md`.
- The new approach was expected to improve performance, but initial results were disappointing: the loss, especially the policy loss, was much worse than with the previous network.
- Previous training runs (using the old network) produced a policy network that could confidently and accurately predict moves in simple positions (e.g., the canonical g1a7g2b7... sequence). The `.pkl.gz` data files used for those runs are still dated July 13th, suggesting they have not been altered.

## 2. Symptoms and Observations

- Training with the new setup yields much higher policy loss than before, even on small runs (e.g., 50k samples). Earlier 50k runs achieved policy loss well below 5, but now this is not the case.
- There were warnings during data loading: in some files, the 2nxn data showed an empty board except for a solitary red piece (i.e., red appears to go first). These games continued with colors reversed throughout.
- Most data files had zero such errors, but a few had a nontrivial number of these anomalies.
- Training on these data files was previously successful, so it is unclear if these errors are new, or if they are responsible for the current poor performance.

## 3. Investigation and Tools Developed

- We wrote a suite of diagnostic and data analysis scripts (see `scripts/all_scripts_summary.md`) to:
    - Scan for and identify problematic files and samples
    - Analyze color distributions and board state validity
    - Extract and inspect error samples
    - Search for specific board states or move sequences in the original TRMPH files
    - Summarize and visualize error patterns
- These tools allowed us to:
    - Quantify the prevalence of errors (e.g., files with color-swapped games)
    - Isolate problematic files and generate exclusion lists
    - Attempt to trace problematic samples back to their source TRMPH records

## 4. What We Have Learned

- The majority of data files are clean, but a minority contain systematic errors (e.g., color-swapped games where red goes first).
- The errors are not random: in affected files, entire games have colors swapped, starting from the first move.
- We have not yet found a specific game record in the processed data that can be definitively matched to a record in the original TRMPH file to diagnose the root cause.
- The presence of these errors does not fully explain the poor training performance, since:
    - They are a minority of the data
    - Previous training runs (with the same data) were successful
- Nonetheless, it is prudent to fix or exclude these errors to ensure data quality.

## 5. Hypotheses and Open Questions

- **Data corruption may be related to TRMPH file anomalies:**
    - One hypothesis is that TRMPH files with errors (e.g., duplicate moves) may cause the pre-processing code to skip moves or misalign color assignment, possibly causing color swaps that persist into subsequent games.
    - There may be a bug in the data processing pipeline that mishandles edge cases, especially when encountering malformed or unexpected input.
- **Training pipeline or loss computation changes:**
    - The new training setup may have inadvertently introduced issues (e.g., loss weighting, data shuffling, or batch construction) that degrade learning, independent of the data corruption.
- **Are the errors new or pre-existing?**
    - The `.pkl.gz` files are dated July 13th, matching the successful earlier runs, suggesting the data has not changed. This raises the question of why the new training is so much worse.

## 6. Tools and Scripts

- A comprehensive set of scripts for data analysis, error extraction, and TRMPH file inspection is available and summarized in `scripts/all_scripts_summary.md`.
- These tools allow for:
    - Automated scanning for errors
    - Manual and programmatic inspection of raw and processed data
    - Tracing of problematic samples through the data pipeline

## 7. Confirmed Cause of Color-Swapped Games

- **We have now confirmed that problematic games with color/channel errors are associated with repeated moves in the TRMPH records.**
    - For example, the reconstructed move sequence for a problematic sample contains the substring `b12b12`, indicating a repeated move.
    - This repeated move appears to trigger a bug in the pre-processing pipeline: the repeated move is skipped, but the color alternation logic is not properly reset, causing all subsequent moves to be assigned the wrong color.
    - As a result, every second move in the affected game is invalid (e.g., the board state checker sees equal numbers of red and blue pieces in positions where this should not be possible).
- **The exact mechanism by which the color swap occurs is not fully known, but the repeated move is a clear trigger for the error.**

### Evidence and File Locations
- Problematic processed data file: `data/processed/twoNetGames_13x13_mk2_d2b6_1963k_processed.pkl.gz`
- Analysis outputs: 
    - Problematic sample summary: `analysis/debugging/problematic_samples_twoNetGames_13x13_mk2_d2b6.txt`
    - Verbose move sequence output: `analysis/debugging/problematic_samples_twoNetGames_13x13_mk2_d2b6_verbose.txt`
- Exclusion list: `analysis/debugging/exclude_files.txt`

### Reproducing the Data Trail
To reproduce the analysis and confirm the issue, run the following steps:

1. **Identify problematic files:**
   ```sh
   PYTHONPATH=. python scripts/identify_bad_files.py --data-dir data/processed --generate-exclusion-list --output-file analysis/debugging/exclude_files.txt
   ```
2. **Find problematic samples and reconstruct move sequences:**
   ```sh
   PYTHONPATH=. python scripts/find_problematic_samples.py data/processed/twoNetGames_13x13_mk2_d2b6_1963k_processed.pkl.gz > analysis/debugging/problematic_samples_twoNetGames_13x13_mk2_d2b6_verbose.txt
   ```
3. **Inspect the output:**
   - Open `analysis/debugging/problematic_samples_twoNetGames_13x13_mk2_d2b6_verbose.txt` and look for repeated moves (e.g., `b12b12`) in the reconstructed move sequences for problematic samples.
   - The presence of such repeated moves confirms the link between TRMPH file anomalies and color/channel errors in the processed data.

---

**Summary:**
We have confirmed that color-swapped or otherwise corrupted games in a minority of the data are associated with repeated moves in the original TRMPH records. This triggers a bug in the pre-processing pipeline, leading to systematic color/channel errors in the processed data. While the data errors are concerning, they do not fully explain the training issues, and further investigation is needed on the training code front. Our growing suite of diagnostic tools is helping to systematically narrow down the possible causes.

---

## 8. Next Steps

- Investigate the new training pipeline for possible bugs or misconfigurations that could explain the poor performance, independent of data quality.
- Document any further findings and update this write-up as the investigation progresses. 

---

## 9. Next Debugging Steps: Data and Training Sanity Checks (2024-07-17)

Based on the current investigation, the model and training code appear structurally sound, and the player-to-move logic is correct. The most likely sources of the poor policy head performance are subtle data/label mismatches, issues in the data pipeline, or loss weighting. To systematically debug this, we propose the following steps:

### A. Sanity-Check the Data Pipeline
- **Goal:** Ensure that the data passed to the network (board, player-to-move channel, policy target) matches expectations.
- **Actions:**
    - Develop a script/tool to visualize random training samples:
        - Print or plot the board state, player-to-move channel, and the move encoded by the policy target.
        - Confirm by hand that the player-to-move and policy target are correct for the given board state.
    - Use this tool to inspect a variety of samples, especially from files previously identified as problematic.

### B. Overfit a Tiny Subset
- **Goal:** Test whether the model can learn the policy on a very small dataset, which should be trivial if the data and loss are correct.
- **Actions:**
    - Train the model on a single batch or a tiny dataset (e.g., 100 samples).
    - The model should be able to drive the policy loss near zero (overfit).
    - If not, this strongly suggests a data or loss calculation bug.

### C. Check Policy Loss Calculation
- **Goal:** Ensure that the policy loss is being computed as expected and is not always zero, NaN, or dominated by invalid targets.
- **Actions:**
    - Add logging or assertions to confirm that the policy loss receives valid targets and gradients.
    - Check the distribution of valid/invalid policy targets in a typical batch.

### D. Compare with Old Pipeline
- **Goal:** Identify any differences in board/target pairs between the old and new data pipelines.
- **Actions:**
    - Run the old pipeline on the same data and compare a few samples to the new pipeline's output.
    - Look for systematic differences in the player-to-move channel or policy targets.

### E. Develop and Use Visualization/Inspection Tools
- **Goal:** Build reusable scripts to automate the above checks and make debugging easier in the future.
- **Actions:**
    - Extend the suite of scripts in `scripts/` to include data visualization and batch inspection tools.
    - Minimize code duplication by leveraging existing utilities and following the summary in `scripts/all_scripts_summary.md`.

---

**Summary:**
- The next phase of debugging will focus on verifying the integrity and correctness of the data pipeline and loss computation, using both visualization tools and controlled overfitting experiments. This should help isolate whether the policy head's poor learning is due to data/label mismatches, subtle bugs, or other issues not yet identified. 