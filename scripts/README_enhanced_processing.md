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

### Two-Step Mixed Strategy (Recommended)

**Step 1: Individual Position Stratified Processing** ✅ (Implemented, Updated)
- Creates individual position bands to break game correlations
- Pass 0: Process position 0 from ALL games → `pass_000_position_0.pkl.gz`
- Pass 1: Process position 1 from ALL games → `pass_001_position_1.pkl.gz`
- Pass 2: Process position 2 from ALL games → `pass_002_position_2.pkl.gz`
- etc. (up to position 168 for 13x13 board)

**Benefits of individual positions:**
- ✅ 169 bands instead of 34 = much finer granularity
- ✅ Better position diversity in final files
- ✅ More effective correlation breaking
- ✅ Simpler memory management (can load all bands at once)
- ✅ Each final file gets examples from all position ranges

**Step 2: Smart Re-aggregation** (Implementation phase)
- Choose k=500 final output files
- Load all 169 position bands into memory
- For each band:
  - Count examples in that band (N examples)
  - Divide into k chunks as evenly as possible: some chunks get ⌈N/k⌉ examples, others get ⌊N/k⌋
  - Randomly permute chunk assignments (0 to k-1) - this breaks game reassembly
- For each final output file j:
  - Extract chunk j from each band (some bands may contribute 0 examples)
  - Concatenate all chunks j into final file j
  - Shuffle the combined examples before writing

**Example with k=3 final files:**
```
Band 0 (position 0): [1,2,3,4,5,6,7,8] (8 examples)
Band 1 (position 1): [2,3,6,8]         (4 examples) 
Band 2 (position 2): [2]               (1 example)

Random chunk assignments:
Band 0: [1,3,5] → chunk_shuffled_0, [2,4,8] → chunk_shuffled_1, [6,7] → chunk_shuffled_2
Band 1: [2,8] → chunk_shuffled_0, [6] → chunk_shuffled_1, [3] → chunk_shuffled_2  
Band 2: [2] → chunk_shuffled_0, [] → chunk_shuffled_1, [] → chunk_shuffled_2

Result (after shuffling within each file):
chunk_shuffled_0: shuffle([1,3,5] + [2,8] + [2]) = 6 examples
chunk_shuffled_1: shuffle([2,4,8] + [6] + []) = 4 examples
chunk_shuffled_2: shuffle([6,7] + [3] + []) = 3 examples
```

**Key points:**
- Some final files may get more examples than others (due to uneven division)
- Some bands may contribute 0 examples to some final files (when N < k)
- Random permutation ensures games don't get reassembled
- Each final file gets a mix of all positions from different games
- Final shuffling within each file breaks any remaining correlations

**Benefits:**
- ✅ Breaks temporal correlations between consecutive positions (Step 1)
- ✅ Breaks temporal correlations between position bands (Step 2)
- ✅ Memory efficient - loads all bands at once (feasible with 169 bands)
- ✅ Random permutation prevents games from being reassembled
- ✅ Uniform position distribution in final files
- ✅ Scalable to any dataset size
- ✅ Maximum position diversity in each final file

**Output files:**
- `chunk_shuffled_000.pkl.gz` (mixed all positions, chunk 0 from each band, shuffled)
- `chunk_shuffled_001.pkl.gz` (mixed all positions, chunk 1 from each band, shuffled)
- etc.

**Why this works:**
1. **Game correlation breaking**: Random permutation ensures chunks from same game don't end up in same final file
2. **Position diversity**: Each final file gets examples from all 169 positions from different games
3. **Memory efficiency**: With 169 bands, can load all at once for processing
4. **Uniform distribution**: Each final file has roughly equal representation from all position ranges
5. **Final shuffling**: Breaks any remaining temporal correlations within each file

**Example file size distribution** (with your data):
- `chunk_shuffled_000.pkl.gz`: ~174 examples (gets chunk from all bands)
- `chunk_shuffled_100.pkl.gz`: ~170 examples (gets chunk from most bands)
- `chunk_shuffled_400.pkl.gz`: ~165 examples (gets chunk from fewer bands, missing late positions)

### Stratified Processing (Step 1 Only)

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

**Trade-offs:**
- ❌ Creates position bands that need further shuffling
- ❌ Training would proceed through early positions, then late positions

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

## Implementation Plan

### Current Status ✅
- **Step 1: Individual Position Stratified Processing** - Implemented and tested
- **Enhanced data format** - Implemented with metadata
- **Value sampling tiers** - Implemented (0-3 priority levels)
- **File lookup tables** - Implemented for game origin tracking

### Next Steps 🔄

#### Step 2: Smart Re-aggregation (Implementation phase)
**Goal**: Transform 169 individual position bands into 500 properly shuffled final files

**Updated Algorithm**:
```python
def create_chunk_shuffled_dataset(stratified_files, output_dir, k=500):
    """
    Step 2: Smart re-aggregation with random permutation and final shuffling
    """
    # Phase 1: Load all 169 position bands into memory
    band_info = []
    for band_file in stratified_files:
        with gzip.open(band_file, 'rb') as f:
            data = pickle.load(f)
        examples = data['examples']
        total_examples = len(examples)
        
        # Calculate chunk boundaries (handle uneven division)
        chunk_size = total_examples // k
        remainder = total_examples % k
        chunks = []
        start = 0
        for i in range(k):
            # First 'remainder' chunks get one extra example
            current_chunk_size = chunk_size + (1 if i < remainder else 0)
            end = start + current_chunk_size
            chunks.append((start, end))
            start = end
        
        # Randomly permute chunk assignments
        chunk_permutation = list(range(k))
        random.shuffle(chunk_permutation)
        
        band_info.append({
            'file': band_file,
            'examples': examples,  # Keep in memory
            'total_examples': total_examples,
            'chunks': chunks,
            'permutation': chunk_permutation
        })
    
    # Phase 2: Create final files with shuffling
    for final_file_idx in range(k):
        all_examples = []
        
        for band_data in band_info:
            # Find which chunk from this band goes to final_file_idx
            chunk_idx = band_data['permutation'].index(final_file_idx)
            start, end = band_data['chunks'][chunk_idx]
            
            # Extract chunk from memory (may be empty)
            if start < end:  # Only extract if chunk has examples
                chunk_examples = band_data['examples'][start:end]
                all_examples.extend(chunk_examples)
        
        # Shuffle the combined examples before writing
        random.shuffle(all_examples)
        
        # Save final file
        output_file = output_dir / f"chunk_shuffled_{final_file_idx:03d}.pkl.gz"
        save_processed_data(output_file, all_examples, final_file_idx)
```

**Key Implementation Notes**:
- Load all 169 bands into memory (feasible with individual positions)
- Handle uneven division: first `remainder` chunks get one extra example
- Some chunks may be empty (when N < k)
- Some final files may have fewer examples than others
- Final shuffling within each file breaks any remaining correlations
- Code must handle empty chunks gracefully

**Key Design Decisions**:
1. **k=500 final files** - Balance between file size and shuffling effectiveness
2. **Random permutation per band** - Ensures games don't get reassembled
3. **Memory-based processing** - Load all bands at once for efficiency
4. **Final shuffling** - Breaks any remaining temporal correlations
5. **chunk_shuffled naming** - Clear indication of the processing applied

**Expected Output**:
- `chunk_shuffled_000.pkl.gz` through `chunk_shuffled_499.pkl.gz`
- File sizes will vary: some files get more examples than others due to uneven division
- All bands (positions 0-168) will contribute to most final files
- Each file has mix of all positions from different games
- No temporal correlations between consecutive examples
- Maximum position diversity in each final file

**Example file size distribution** (with your data):
- `chunk_shuffled_000.pkl.gz`: ~174 examples (gets chunk from all bands)
- `chunk_shuffled_100.pkl.gz`: ~170 examples (gets chunk from most bands)
- `chunk_shuffled_400.pkl.gz`: ~165 examples (gets chunk from fewer bands, missing late positions)

#### Step 3: Training Pipeline Updates
1. **Update data loading** to handle new file structure
2. **Implement selective value training** based on sampling tiers
3. **Add value tier filtering** in training loop
4. **Test with small subset** before full training

### Testing Strategy
1. **Small-scale test**: Use `--max_files 2` to test complete pipeline
2. **Position distribution analysis**: Verify final files have mixed positions
3. **Game correlation test**: Verify no temporal correlations remain
4. **Memory usage monitoring**: Ensure processing stays within limits

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