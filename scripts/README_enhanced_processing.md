# Enhanced Data Processing Pipeline

This directory contains the enhanced data processing pipeline for Hex AI, designed to address value network overfitting through better data organization and correlation breaking.

## Overview

The enhanced processing pipeline addresses the core problem of **game fingerprinting** - where the value network overfits to temporal correlations between consecutive positions in the same game. The solution includes:

1. **Comprehensive metadata tracking** - Track game origins and position information
2. **Flexible value sampling tiers** - Control which positions per game are used for value training
3. **Memory-efficient stratified processing** - Break temporal correlations while managing memory usage
4. **Repeated moves handling** - Clean up corrupted games with repeated moves
5. **File lookup tables** - Map processed data back to original TRMPH files

## Files

- `process_enhanced_data.py` - Main processing script
- `test_enhanced_processing.py` - Test script for verification
- `README_enhanced_processing.md` - This file

## Quick Start

### 1. Test the Pipeline

First, test the enhanced processing with a small sample:

```bash
cd scripts
python test_enhanced_processing.py
```

This will verify that all components work correctly.

### 2. Process Your Data

Process your TRMPH files using the stratified approach (recommended):

```bash
python process_enhanced_data.py \
    --data_dir ../data \
    --output_dir ../data/enhanced_processed \
    --strategy stratified \
    --positions_per_pass 5 \
    --max_positions_per_game 200
```

Or use the chunked approach:

```bash
python process_enhanced_data.py \
    --data_dir ../data \
    --output_dir ../data/enhanced_processed \
    --strategy chunked \
    --games_per_chunk 10000
```

## Processing Strategies

### Stratified Processing (Recommended)

**How it works:**
- Pass 1: Process positions 0-4 from ALL games
- Pass 2: Process positions 5-9 from ALL games  
- Pass 3: Process positions 10-14 from ALL games
- etc.

**Benefits:**
- ✅ Breaks temporal correlations between consecutive positions
- ✅ Memory efficient - processes in small batches
- ✅ Maintains training diversity - each batch sees all game stages
- ✅ Simple to implement and debug

**Output files:**
- `pass_000_positions_0-5.pkl.gz` (empty boards + first few moves from all games)
- `pass_001_positions_5-10.pkl.gz` (middle game positions from all games)
- `pass_002_positions_10-15.pkl.gz` (later game positions from all games)

### Chunked Processing

**How it works:**
- Load all games into memory (just references, not positions)
- Shuffle game order globally
- Process in chunks of N games each
- Shuffle within each chunk

**Benefits:**
- ✅ Simpler implementation
- ✅ Good for very large datasets

**Trade-offs:**
- ❌ Less effective at breaking temporal correlations
- ❌ Requires loading all game references into memory

## Enhanced Data Format

The new format includes comprehensive metadata:

```python
{
    'board': torch.Tensor,      # Board state (unchanged)
    'policy': torch.Tensor,     # Policy target (unchanged)  
    'value': float,             # Value target (unchanged)
    'metadata': {
        'game_id': (file_idx, line_idx),  # Track game origin
        'position_in_game': int,          # Position within game
        'total_positions': int,           # Total positions in game
        'value_sample_tier': int,         # 0-3: priority for value training
        'winner': str,                    # "BLUE" or "RED"
        'trmph_game': str                 # Optional: full TRMPH string
    }
}
```

### Value Sampling Tiers

- **Tier 0** (High priority): 5 positions - Always used for value training
- **Tier 1** (Medium priority): 5 positions - Usually used for value training  
- **Tier 2** (Low priority): 10 positions - Sometimes used for value training
- **Tier 3** (Very low priority): 20+ positions - Rarely used for value training

This allows flexible control over how many positions per game are used for value network training while keeping all positions for policy training.

## File Lookup Table

A JSON file is created alongside the processed data:

```json
{
    "file_mapping": {
        "0": "data/file1.trmph",
        "1": "data/file2.trmph",
        "2": "data/subdir/file3.trmph"
    },
    "created_at": "2024-01-15T10:30:00",
    "total_files": 3,
    "format_version": "1.0"
}
```

This allows you to trace any processed example back to its original TRMPH file and line.

## Winner Format Mapping

- **TRMPH format**: "1" = BLUE win, "2" = RED win
- **Training format**: BLUE = 0.0, RED = 1.0 (subtract 1 from TRMPH values)
- **Enhanced metadata**: "BLUE" or "RED" (clear text)

## Command Line Options

### Basic Options
- `--data_dir`: Directory containing TRMPH files (default: "data")
- `--output_dir`: Directory to save processed files (default: "data/enhanced_processed")
- `--strategy`: Processing strategy - "stratified" or "chunked" (default: "stratified")
- `--include_trmph`: Include full TRMPH strings in metadata (default: False)
- `--log_level`: Logging level - DEBUG, INFO, WARNING, ERROR (default: "INFO")
- `--max_files`: Maximum number of files to process (for testing)

### Stratified Options
- `--positions_per_pass`: Number of positions to process per pass (default: 5)
- `--max_positions_per_game`: Maximum positions to consider per game (default: 200)

### Chunked Options
- `--games_per_chunk`: Number of games per chunk (default: 10000)

## Memory Usage

### Stratified Processing
- **Memory per pass**: ~(games × positions_per_pass × example_size) bytes
- **Example**: 1.2M games × 5 positions × 1KB = ~6GB per pass
- **Total memory**: Peak memory usage is manageable

### Chunked Processing  
- **Memory for game references**: ~(total_games × 100 bytes) = ~120MB
- **Memory per chunk**: ~(games_per_chunk × avg_positions × example_size) bytes
- **Example**: 10k games × 70 positions × 1KB = ~700MB per chunk

## Next Steps

After processing, you'll need to:

1. **Update training pipeline** to use the enhanced format
2. **Implement selective value training** based on sampling tiers
3. **Update data loading** to handle the new file structure
4. **Test with a small subset** before full training

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce `positions_per_pass` or `games_per_chunk`
2. **No TRMPH files found**: Check `--data_dir` path and file extensions
3. **Processing errors**: Check logs for specific game parsing issues
4. **Slow processing**: Consider using `--max_files` for testing first

### Log Files

Processing creates detailed logs:
- Console output with progress information
- `enhanced_processing.log` file with detailed logging

## Performance Tips

1. **Start small**: Use `--max_files 10` to test with a subset
2. **Monitor memory**: Watch memory usage during processing
3. **Use SSD storage**: Faster I/O for large datasets
4. **Adjust batch sizes**: Tune `positions_per_pass` based on your system 