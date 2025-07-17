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

## 10. Validation of Data Loader Consistency (2024-07-17)

To rule out bugs in the data loading pipeline, we compared the output of the `StreamingProcessedDataset` (used in training) to the raw records in the processed `.pkl.gz` files. This was done using the script `scripts/compare_loader_vs_raw.py`, which:
- Loads records from a `.pkl.gz` file both directly (raw) and via the loader.
- Compares the board state, policy target, and value for each record.
- Highlights any discrepancies, including mismatches in the board (first two channels), policy, or value.
- Can be run in a non-verbose mode to efficiently scan tens or hundreds of thousands of records.

**Result:**
- We examined over 200,000 records from multiple files and found **no discrepancies** between the raw records and the loader output.
- This strongly suggests that the data pipeline, including the logic for determining the next move and player-to-move channel, is functioning as intended and is not introducing bugs at this stage.

**Scripts and methodology:**
- Used `scripts/compare_loader_vs_raw.py` with the `--verbose` flag for spot checks and without it for large-scale automated comparison.
- The script automatically compared the board state (first two channels), policy target, and value between the raw and loader-processed records, reporting any mismatches.

**Next step:**
- Attempt to train the model on a very small dataset (e.g., one complete game from `twoNetGames_13x13_mk1_1_processed.pkl.gz`) to see if the network can overfit and drive the policy loss near zero. This will help determine if the model and loss function are working as expected in the absence of data pipeline issues.

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

## 11. Overfitting on a Single Game: Progress and Open Questions (2024-07-18)

As part of our ongoing debugging, we set up an experiment to overfit the model on a single complete game (extracted as a small `.pkl.gz` file with 55 examples). The goal is to verify that the model and training loop can learn a trivial dataset, which would indicate that the core architecture and loss are functioning as expected.

### Progress
- We developed and iteratively refined a script (`scripts/overfit_tiny_dataset.py`) to:
    - Load all samples from a single-game `.pkl.gz` file (handling both list-of-dicts and new tuple-in-dict formats).
    - Convert the data to PyTorch tensors, with careful handling of numpy arrays and shapes.
    - Train the model on this data as a single batch for many epochs, tracking policy accuracy and loss.
    - Implement early stopping and provide move-by-move analysis at the end.
    - Add detailed memory usage diagnostics (including tensor sizes and process memory) to investigate high memory usage.
- The script now successfully loads the single-game file and prepares the data for training.

### Current Observations
- The model is able to learn and improve accuracy on the tiny dataset, confirming that learning is possible in principle.
- However, the process is much slower than expected, and memory usage is extremely high (~12GB), even though the data and model are small.
- Diagnostics show that the in-memory tensors and model parameters account for only a small fraction of this memory, suggesting possible memory fragmentation, PyTorch/MPS backend issues, or other inefficiencies.
- The data format for single-game files is not fully standardized; we now handle both the legacy list-of-dicts and the new dict-with-'examples'-key format.

### Next Steps and Open Questions
- **Why is memory usage so high?**
    - Continue investigating PyTorch/MPS memory allocation and possible leaks or inefficiencies.
    - Try running the same script on CPU or CUDA (if available) to compare memory profiles.
    - Profile the script with external tools (e.g., `memory_profiler`, `Activity Monitor`) to pinpoint where memory is being consumed.
- **Why is training so slow?**
    - Investigate whether the MPS backend is causing slowdowns or if there are inefficiencies in the training loop.
    - Try running with different batch sizes, devices, or PyTorch versions.
- **Standardize data formats:**
    - Update data extraction and loading utilities to consistently use a single, well-documented format for all small/batch files.
- **If overfitting succeeds, move to more complex sanity checks:**
    - Try overfitting on a few games, or on a hand-crafted dataset with known solutions.
    - Use visualization tools to confirm that the model's predictions match expectations at each step.

**Summary:**
We have made progress in verifying that the model can learn from a single game, but are currently blocked by unexpectedly high memory usage and slow training, especially on Apple MPS. The next phase will focus on profiling and resolving these performance issues, and on standardizing the data formats for future experiments. 

---

## 12. Major Breakthrough: Training Pipeline Performance Differences (2024-07-18)

We have made a significant breakthrough in understanding the poor policy loss performance. Through systematic comparison of different training approaches, we discovered that **the main training pipeline performs significantly worse than a simplified overfit pipeline, even when using identical data and model architecture**.

### Key Findings

**Performance Gap Identified:**
- The main training pipeline (`run_trainer.py`) achieves much worse policy loss and accuracy than a simplified overfit pipeline
- This gap persists even when using the same data, model architecture, and basic hyperparameters
- The overfit pipeline can achieve policy loss ~0.5-1.0 and accuracy ~70-80% on small datasets, while the main pipeline struggles to improve beyond initial random performance

**Overfit Pipeline Success:**
- A simplified training script (`scripts/overfit_tiny_dataset.py`) can successfully overfit small datasets
- This pipeline runs much faster with GPU acceleration (MPS on Mac)
- Memory usage is reasonable (~0.6GB) when using appropriate batch sizes
- The model can learn and improve policy accuracy significantly on controlled datasets

**Systematic Comparison Framework:**
- Created `scripts/compare_training_pipelines.py` to systematically compare different training configurations
- This allows us to isolate which training settings cause the performance difference
- Initial tests show that mixed precision, gradient clipping, and weight decay may contribute to the gap, but don't fully explain it

### What We Know vs. What We Don't Know

**What We Know:**
1. **The model architecture is sound** - it can learn effectively in the overfit pipeline
2. **The data pipeline is correct** - validation shows no discrepancies between raw and loader output
3. **The performance gap is real** - systematic comparison confirms the main pipeline underperforms
4. **GPU acceleration works well** - MPS provides significant speedup for the overfit pipeline
5. **Memory usage can be controlled** - appropriate batch sizes and data loading prevent excessive memory use

**What We Don't Know:**
1. **Which specific training settings cause the performance gap** - mixed precision, gradient clipping, weight decay, data loading differences, or combinations thereof
2. **Whether the main pipeline has bugs** - subtle issues in the training loop, loss computation, or optimization
3. **How to fix the main pipeline** - whether to modify existing settings or adopt the overfit approach
4. **The root cause of the difference** - whether it's a configuration issue, implementation bug, or fundamental design problem

### Next Steps: Systematic Configuration Analysis

We need to systematically test different training configurations to identify what causes the performance difference:

1. **Create a comprehensive comparison framework** that can test various combinations of:
   - Mixed precision (on/off)
   - Gradient clipping (on/off, different thresholds)
   - Weight decay (on/off, different values)
   - Data loading approaches (LimitedDataset vs direct loading)
   - Learning rate schedules
   - Batch sizes and optimization settings

2. **Document results systematically** in `analysis/debugging/` to track which settings matter most

3. **Identify the critical differences** that explain the performance gap

4. **Fix the main pipeline** or adopt the working approach as the new standard

### Implications

This discovery suggests that:
- The poor training performance is likely due to training pipeline configuration, not fundamental model or data issues
- We may be able to significantly improve results by adjusting training settings rather than redesigning the model
- The overfit pipeline provides a working baseline that we can build upon
- GPU acceleration (MPS) is viable and should be used consistently

**Summary:**
We have identified that the main training pipeline underperforms compared to a simplified approach, despite using the same model and data. This is a major breakthrough that shifts our focus from data/model issues to training configuration. The next phase involves systematic testing of different training settings to identify and fix the root cause of the performance gap.

---

## 13. Next Steps

- Implement systematic configuration testing framework
- Document and analyze results of different training settings
- Identify critical differences between working and non-working pipelines
- Fix the main training pipeline or adopt the working approach
- Validate improvements on larger datasets 

---

## 13. Gradient Clipping, Pipeline Performance, and Simple Inference Test (2024-07-17)

### Gradient Clipping and Pipeline Comparison

- We systematically compared the main training pipeline and a simplified overfit pipeline using `scripts/compare_training_pipelines.py`.
- The script allows running both pipelines on the same data and model, and systematically toggling features like weight decay, gradient clipping, and mixed precision.
- **Key finding:** Gradient clipping (especially with a low max norm) and weight decay can significantly degrade policy head performance, but removing or adjusting them does not fully restore the old performance.
- To replicate: run
  ```sh
  PYTHONPATH=. python scripts/compare_training_pipelines.py data/processed/your_data.pkl.gz --max-samples 5000 --epochs 10 --device auto --test-parameters
  ```
  and review the output for policy loss and accuracy differences between the pipelines.
- See the script for details on how to add or modify parameter sweeps.

### Ongoing Training Issues

- Despite these findings, after 5 epochs of training on 100k samples with the new pipeline, the policy loss remains considerably higher than in previous runs with the old network design.
- This suggests that the root cause of poor policy learning is not solely due to gradient clipping or weight decay settings.

### Simple Inference Test

- We tested the trained model using `scripts/simple_inference_cli.py` on a trivial game:
  ```
  g1a7g2b7g3c7g4d7g5e7g6f7g8h7g9i7g10j7g11k7g12l7g13m7
  ```
  - In this position, the only reasonable move is g7 (the (6,6) point), which should be a win for either player.
  - The model failed to rank g7 in the top 3 moves, indicating a fundamental issue with policy learning.

### Summary

- Gradient clipping and weight decay can impact performance, but are maybe not the sole cause of the poor policy head learning.
- A new training run, even with 100k samples and 5 epochs, does not match the performance of the old network.
- Simple inference tests confirm that the model is not learning even simple positions correctly.
- Further investigation is needed into the training loop, loss computation, and data/label alignment.

--- 