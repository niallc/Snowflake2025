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

## 7. Next Steps

- Continue efforts to match problematic processed samples to their original TRMPH records to diagnose the root cause of color-swapped games.
- Consider reprocessing or excluding affected files to ensure clean training data.
- Investigate the new training pipeline for possible bugs or misconfigurations that could explain the poor performance, independent of data quality.
- Document any further findings and update this write-up as the investigation progresses.

---

**Summary:**
We are investigating two intertwined issues: (1) poor training performance after a network redesign, and (2) the presence of color-swapped or otherwise corrupted games in a minority of the data. While the data errors are concerning, they do not fully explain the training issues, and further investigation is needed both on the data and training code fronts. Our growing suite of diagnostic tools is helping to systematically narrow down the possible causes. 