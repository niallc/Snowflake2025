# Data Preprocessing Guide for Hex AI Training Pipeline

This guide covers how to collect, preprocess, and prepare training data using the `training_pipeline.py` and related tools.

## Overview

The data preprocessing pipeline has three main stages:

1. **Collection**: Gather .trmph files from multiple sources and make them unique
2. **Processing**: Convert games to training positions (.pkl.gz files)  
3. **Shuffling**: Shuffle and organize positions into training shards

## Data Sources

### Tournament Games
Located in `data/tournament_play/` with subdirectories like:
- `deterministic_tournament_20250908_1625/`
- `tournament_50games_2models_250908_09/`

### Self-Play Games  
Located in `data/sf25/` with date-based subdirectories like:
- `sep13/` - contains multiple .trmph files from September 13th
- `sep14/` - contains multiple .trmph files from September 14th

## Collection Methods

### Method 1: Using `collect_and_clean_games.py` (Recommended for Complex Collection)

This script provides flexible collection options:

#### Collect All Data
```bash
python scripts/collect_and_clean_games.py collect \
  --source-dirs data/tournament_play data/sf25 \
  --output-dir data/collected/my_collection
```

#### Collect Tournament Data Since a Date
```bash
# Collect tournament data since September 3rd
python scripts/collect_and_clean_games.py tournament \
  --since-date 2025-09-03 \
  --output-dir data/collected/tournament_sep3_onwards

# Collect tournament data from last 7 days
python scripts/collect_and_clean_games.py tournament \
  --since-days 7 \
  --output-dir data/collected/tournament_last_week
```

#### Collect Specific Self-Play Directories
```bash
python scripts/collect_and_clean_games.py collect \
  --source-dirs data/sf25/sep13 data/sf25/sep14 \
  --output-dir data/collected/sep13_14_games
```

### Method 2: Using `training_pipeline.py` Game Collection

For simpler cases, you can collect directly in the training pipeline:

```bash
python scripts/training_pipeline.py \
  --use-current-best-model \
  --raw-trmph-data-dirs data/sf25/sep13 data/sf25/sep14 \
  --run-game-collection \
  --no-selfplay --no-preprocessing --no-trmph-processing --no-shuffling --no-training
```

## Use Cases

### 1. Collect and Process Only (No Training)

Use this when you want to prepare data for later training runs:

```bash
python scripts/training_pipeline.py \
  --use-current-best-model \
  --raw-trmph-data-dirs data/sf25/sep13 data/sf25/sep14 \
  --run-game-collection \
  --no-selfplay --no-training
```

This will:
- Collect and deduplicate games from specified directories
- Preprocess them into cleaned .trmph files
- Convert to training positions (.pkl.gz files)
- Shuffle into training shards

**Output**: Processed data in `data/processed/shuffled_*` directory

### 2. Train on Already Processed Data

Use this when you have existing processed data and want to train:

```bash
python scripts/training_pipeline.py \
  --use-current-best-model \
  --processed-data-dirs data/processed/sf18_shuffled data/processed/shuffled_my_new_data \
  --shard_ranges "221-250" "all" \
  --no-selfplay --no-preprocessing --no-trmph-processing --no-shuffling
```

**Key points**:
- `--shard_ranges` must match the number of `--processed-data-dirs`
- Use `"all"` for new data, specific ranges like `"221-250"` for existing data
- All processing steps are disabled with `--no-*` flags

### 3. Collect, Process, and Train in One Step

Use this for a complete pipeline run:

```bash
python scripts/training_pipeline.py \
  --use-current-best-model \
  --raw-trmph-data-dirs data/sf25/sep13 data/sf25/sep14 \
  --processed-data-dirs data/processed/sf18_shuffled \
  --shard_ranges "221-250" \
  --run-game-collection \
  --no-selfplay
```

This will:
- Collect new games from `data/sf25/sep13` and `data/sf25/sep14`
- Process them through the full pipeline
- Train using both the new processed data AND existing data from `data/processed/sf18_shuffled`

## Data Type Arguments

The pipeline uses explicit data type arguments to avoid confusion:

- `--raw-trmph-data-dirs`: Raw .trmph files to collect and clean (e.g., `data/sf25/sep13/`)
- `--cleaned-trmph-data-dirs`: Already cleaned .trmph files to process (e.g., `data/collected/my_data/`)
- `--processed-data-dirs`: Existing processed data directories for training (e.g., `data/processed/shuffled_*`)

**Important**: Use the correct data type argument for your data:
- If you collected data with `collect_and_clean_games.py`, use `--cleaned-trmph-data-dirs`
- If you have raw .trmph files, use `--raw-trmph-data-dirs`
- If you have processed .pkl.gz files, use `--processed-data-dirs`

## Validation and Error Prevention

The pipeline includes validation to catch common mistakes:

- **Missing data**: Will error if you specify collection but disable all processing steps
- **Mismatched shard ranges**: Will error if shard ranges don't match processed data directories
- **Type safety**: Clear error messages when wrong data types are specified

## Data Summary

After shard discovery, the pipeline displays a summary:

```
============================================================
TRAINING DATA SUMMARY
============================================================
Estimated total positions: ~12,500,000
Estimated total games: ~178,571
Total shards: 150
Data directories: 2
============================================================
```

## Practical Examples

### Example 1: Collect Recent Tournament Data
```bash
# Collect tournament data from last 3 days
python scripts/collect_and_clean_games.py tournament \
  --since-days 3 \
  --output-dir data/collected/recent_tournaments

# Process and train on it
python scripts/training_pipeline.py \
  --use-current-best-model \
  --cleaned-trmph-data-dirs data/collected/recent_tournaments \
  --no-selfplay --no-preprocessing --no-trmph-processing --no-shuffling
```

### Example 2: Add New Self-Play Data to Existing Training
```bash
# Collect new self-play data and train with existing data
python scripts/training_pipeline.py \
  --use-current-best-model \
  --raw-trmph-data-dirs data/sf25/sep15 data/sf25/sep16 \
  --processed-data-dirs data/processed/sf18_shuffled data/processed/previous_run \
  --shard_ranges "221-250" "all" \
  --run-game-collection \
  --no-selfplay
```

### Example 3: Process Multiple Data Sources Separately
```bash
# Step 1: Collect tournament data
python scripts/collect_and_clean_games.py tournament \
  --since-date 2025-09-01 \
  --output-dir data/collected/tournament_sep1_onwards

# Step 2: Collect self-play data  
python scripts/collect_and_clean_games.py collect \
  --source-dirs data/sf25/sep13 data/sf25/sep14 \
  --output-dir data/collected/sep13_14_games

# Step 3: Process both together
python scripts/training_pipeline.py \
  --use-current-best-model \
  --cleaned-trmph-data-dirs data/collected/tournament_sep1_onwards data/collected/sep13_14_games \
  --no-selfplay --no-preprocessing --no-trmph-processing --no-shuffling
```

## Troubleshooting

### Common Issues

1. **"Game collection enabled but all processing steps are disabled"**
   - Solution: Remove some `--no-*` flags or use `--cleaned-trmph-data-dirs` instead

2. **"Number of shard ranges must match number of processed data directories"**
   - Solution: Ensure `--shard_ranges` has the same number of entries as `--processed-data-dirs`

3. **"No data files found"**
   - Solution: Check that directories exist and contain .trmph or .pkl.gz files

### Getting Help

- Use `--help` flag with any script to see all options
- Check logs in `logs/` directory for detailed error information
- Use `python scripts/collect_and_clean_games.py test` to validate tournament pattern configuration
