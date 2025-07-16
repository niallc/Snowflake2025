# Hex AI Utility Scripts Manual

**Last updated:** 2024-07-16, time: 06.10 

This manual describes the diagnostic, training, and data analysis scripts in the `scripts/` directory. Each script is designed to help with a specific aspect of Hex AI model development, data quality assurance, or error investigation.

---

## Debugging Data Errors and Tracing Move Sequences: Best Starting Points

- **For finding and inspecting problematic samples in processed data:**
  - `find_problematic_samples.py` (best for quickly listing invalid board states and their indices in a .pkl.gz file)
  - `analyze_raw_data.py` (best for batch statistics, error rates, and examining specific samples in detail)
- **For extracting or reconstructing move sequences:**
  - `find_problematic_samples.py` (can be extended to reconstruct move sequences by replaying board states up to a sample index)
  - `extract_error_sample_from_pkl.py` (for extracting and displaying a specific record by index)
- **For searching or matching move sequences in raw TRMPH files:**
  - `search_board_in_trmph.py` (search for a move sequence or board state in a TRMPH file)
  - `analyze_trmph_data.py`, `find_invalid_game_starts.py`, `verify_color_swap.py` (for scanning TRMPH files for anomalies or color-swapped games)

---

## Table of Contents
- [analyze_trmph_data.py](#analyze_trmph_datapy)
- [identify_bad_files.py](#identify_bad_filespy)
- [find_problematic_samples.py](#find_problematic_samplespy)
- [analyze_raw_data.py](#analyze_raw_datapy)
- [extract_error_sample_from_pkl.py](#extract_error_sample_from_pklpy)
- [analyze_error_samples.py](#analyze_error_samplespy)
- [analyze_training.py](#analyze_trainingpy)
- [quick_test_training.py](#quick_test_trainingpy)
- [simple_inference_cli.py](#simple_inference_clipy)
- [test_fresh_init_value_head.py](#test_fresh_init_value_headpy)
- [run_training.py](#run_trainingpy)
- [find_invalid_game_starts.py](#find_invalid_game_startspy)
- [verify_color_swap.py](#verify_color_swappy)
- [search_board_in_trmph.py](#search_board_in_trmphpy)

---

## analyze_trmph_data.py
**Purpose:**
- Analyze TRMPH (raw game) files for anomalies that might cause color swapping or other data issues.
- Detects duplicate moves, short games, and invalid move strings.
- **Best for:** Scanning raw TRMPH files for structural anomalies that could propagate into processed data.

**Usage:**
```sh
python scripts/analyze_trmph_data.py <trmph_file> [--max-games N]
```

---

## identify_bad_files.py
**Purpose:**
- Scan processed `.pkl.gz` data files for high error rates (e.g., color channel swaps, invalid board states).
- Generates an exclusion list for problematic files.
- **Best for:** Quickly identifying which processed data files are most likely to contain systematic errors.

**Usage:**
```sh
python scripts/identify_bad_files.py [--data-dir DIR] [--max-files N] [--max-samples N] [--generate-exclusion-list] [--output-file FILE]
```

---

## find_problematic_samples.py
**Purpose:**
- Find and print details of specific problematic samples (invalid board states) in a processed data file.
- **Best for:** Quickly listing invalid samples and their indices. Can be extended to reconstruct move sequences for these samples (see plan in debugging write-up).
- **Limitation:** Does not currently output the full move sequence for a problematic sample, but can be modified to do so by replaying board states from the start of the file.

**Usage:**
```sh
python scripts/find_problematic_samples.py <file_path>
```

---

## analyze_raw_data.py
**Purpose:**
- Analyze raw `.pkl.gz` data files for color distribution, board state validity, and data quality.
- Can examine specific samples or scan multiple files for error rates and statistics.
- **Best for:** Batch analysis, error rate statistics, and detailed inspection of individual samples (including board state and piece positions).
- **Limitation:** Does not reconstruct move sequences, but can be used to find indices of problematic samples for further analysis.

**Usage:**
```sh
python scripts/analyze_raw_data.py [--data-dir DIR] [--max-files N] [--max-samples N] [--examine-file FILE --sample-index N] [--save-results]
```

---

## extract_error_sample_from_pkl.py
**Purpose:**
- Extract and display a record from a `.pkl.gz` file by index or search criteria (value or policy).
- **Best for:** Extracting and displaying a specific sample, including its board state and targets.
- **Limitation:** Does not reconstruct the full move sequence, but can be used in combination with other scripts.

**Usage:**
```sh
python scripts/extract_error_sample_from_pkl.py <file_path> [--index N] [--search-value V] [--search-policy P] [--max N]
```

---

## analyze_error_samples.py
**Purpose:**
- Analyze error samples saved during training (e.g., from error logging).
- Summarizes error types, board statistics, and can visualize or save plots of error samples.
- **Best for:** Summarizing and visualizing error types from training logs.

**Usage:**
```sh
python scripts/analyze_error_samples.py [--error-dir DIR] [--sample N] [--save-plots]
```

---

## analyze_training.py
**Purpose:**
- Analyze and visualize training results (loss curves, learning rates, gradients, etc.).
- Provides recommendations and warnings for common training issues.
- **Best for:** Post-training analysis and visualization.

**Usage:**
```sh
python scripts/analyze_training.py <training_results.json> [--plot] [--save-plot]
```

---

## quick_test_training.py
**Purpose:**
- Run a quick test training session with improved hyperparameters and compact logging.
- Useful for debugging and rapid iteration.

**Usage:**
```sh
python scripts/quick_test_training.py [--epochs N] [--batch-size N] [--learning-rate LR] ...
```

---

## simple_inference_cli.py
**Purpose:**
- Command-line interface for running inference on a Hex board position using a trained model.
- Displays board and top policy moves.
- **Best for:** Quick inference and model sanity checks.

**Usage:**
```sh
python scripts/simple_inference_cli.py --trmph <trmph_string_or_link> [--model_dir DIR] [--model_file FILE] [--topk N] [--device DEV]
```

---

## test_fresh_init_value_head.py
**Purpose:**
- Test the value head of a freshly initialized model on an empty board.
- Verifies that the value output is near 0.5 (as expected for a zero-initialized network).
- **Best for:** Sanity-checking value head initialization.

**Usage:**
```sh
python scripts/test_fresh_init_value_head.py
```

---

## run_training.py
**Purpose:**
- Main entry point for full or quick training runs using the modern pipeline.
- Supports flexible hyperparameters, checkpointing, and early stopping.
- **Best for:** Running and managing training jobs.

**Usage:**
```sh
python scripts/run_training.py [--epochs N] [--batch-size N] [--learning-rate LR] ...
```

---

## find_invalid_game_starts.py
**Purpose:**
- Quickly scan TRMPH files to print the first few moves of each game, helping to spot games that start with invalid or suspicious color sequences (e.g., color swapping or non-standard openings).
- Useful for manual inspection of opening move patterns across many games.
- **Best for:** Spot-checking TRMPH files for color swap patterns in openings.

**Usage:**
```sh
python scripts/find_invalid_game_starts.py <trmph_file>
```

---

## verify_color_swap.py
**Purpose:**
- Systematically check TRMPH files for games where the color order appears swapped or otherwise anomalous.
- Traces games through the data processing pipeline and highlights those matching known error patterns (e.g., red_count > blue_count early in the game).
- Useful for confirming and diagnosing systematic color/channel errors in datasets.
- **Best for:** Automated detection of color swap patterns in TRMPH files.

**Usage:**
```sh
python scripts/verify_color_swap.py <trmph_file>
```

---

## search_board_in_trmph.py
**Purpose:**
- Search for a specific board state or move sequence within a TRMPH file, either by providing moves directly or by specifying board positions for blue/red.
- Helps trace problematic samples or verify if a particular position or sequence exists in the raw data.
- **Best for:** Matching a move sequence or board state from processed data to a TRMPH record.

**Usage:**
```sh
python scripts/search_board_in_trmph.py --trmph-file <trmph_file> [--moves m1 m2 ...] [--blue-positions r1 c1 r2 c2 ...] [--red-positions r1 c1 ...]
```

---

**Note:** For all scripts, use `-h` or `--help` to see full argument options and descriptions. 