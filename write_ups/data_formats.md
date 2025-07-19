# Hex AI Data Formats

This document describes the various data formats used in the Hex AI project, including how to read and work with them.

## Processed Data Files (.pkl.gz)

### Overview
Processed data files contain training examples extracted from raw TRMPH game files. They are stored as compressed pickle files with the extension `.pkl.gz`.

### File Structure

#### Version 1.0 (Current)
```python
{
    'examples': [(board, policy, value), ...],  # List of training examples
    'source_file': str,                         # Original TRMPH file path
    'processing_stats': dict,                   # Processing metadata
    'processed_at': str                         # Timestamp of processing
}
```

#### Version 2.0 (Enhanced - Planned)
```python
{
    'examples': [example_dict, ...],            # List of enhanced training examples
    'source_file': str,                         # Original TRMPH file path
    'processing_stats': dict,                   # Processing metadata
    'processed_at': str,                        # Timestamp of processing
    'format_version': '2.0'                     # Format version identifier
}
```

**Enhanced Example Structure (Version 2.0)**:
```python
{
    'board': np.ndarray,                        # (2, 13, 13) board state
    'policy': np.ndarray,                       # (169,) policy target or None
    'value': float,                             # 0.0 or 1.0
    'metadata': {
        'game_id': (file_idx, line_idx),        # (file_index, line_index) tuple
        'position_in_game': int,                # 0-based position index
        'total_positions': int,                 # Total positions in game
        'value_sample_tier': int,               # 0, 1, 2, 3 for sampling control
        'trmph_game': str,                      # Full TRMPH string (optional)
        'winner': str                           # "1" or "2"
    }
}
```

### Example Structure
```python
{
    'examples': [
        (board_1, policy_1, value_1),
        (board_2, policy_2, value_2),
        # ... more examples
    ],
    'source_file': 'data/twoNetGames/twoNetGames_13x13_mk45_d1b20_v1816_2s0_p2551k_vt25_pt10.trmph',
    'processing_stats': {
        'file_path': 'data/twoNetGames/twoNetGames_13x13_mk45_d1b20_v1816_2s0_p2551k_vt25_pt10.trmph',
        'games_processed': 24,
        'examples_generated': 1551,
        'corrupted_games': 0,
        'valid_games': 24,
        'error': None
    },
    'processed_at': '2025-07-13T17:32:09.452035'
}
```

### Data Types

#### Version 1.0 (Current)
- **board**: `numpy.ndarray` of shape `(2, 13, 13)` - 2-channel format (blue, red)
- **policy**: `numpy.ndarray` of shape `(169,)` - one-hot policy target, or `None` for final moves
- **value**: `float` - value target (0.0 or 1.0)

#### Version 2.0 (Enhanced - Planned)
- **board**: `numpy.ndarray` of shape `(2, 13, 13)` - 2-channel format (blue, red)
- **policy**: `numpy.ndarray` of shape `(169,)` - one-hot policy target, or `None` for final moves
- **value**: `float` - value target (0.0 or 1.0)
- **metadata**: `dict` - Game metadata including:
  - **game_id**: `tuple(int, int)` - (file_index, line_index) for tracking
  - **position_in_game**: `int` - 0-based position index within game
  - **total_positions**: `int` - Total number of positions in the game
  - **value_sample_tier**: `int` - Sampling tier (0-3) for value training control
  - **trmph_game**: `str` - Full TRMPH string (optional, for debugging)
  - **winner**: `str` - Game winner ("BLUE" or "RED")

### Winner Format Mapping

**Important**: There are multiple winner formats used throughout the pipeline:

- **TRMPH format**: "1" = BLUE win, "2" = RED win
- **Training format**: BLUE = 0.0, RED = 1.0  
- **Enhanced metadata**: "BLUE" or "RED" (clear text)

**BLUE definition**: 
- Goes from top to bottom in standard representation
- h1 is at the top edge
- Always goes first in un-augmented TRMPH data
- Represented as 0 in most training code (first channel)

**Note**: After data augmentation (reflections), colors may be swapped, so the "first player" concept becomes relative to the augmentation.

### Policy Target Handling
**Important**: Policy targets can be `None` for final moves in games where there is no next move to predict. This occurs when:
- A game has ended (someone won)
- The current position is the final position of the game

**Current handling in training**:
- `None` policies are converted to zero vectors `(169,)` filled with zeros
- This allows the model to train on final positions without policy loss

**Future improvement**: Consider using uniform distribution `(1/169, 1/169, ...)` instead of zeros for better training signal.

### How to Read

#### Version 1.0 (Current)
```python
import gzip
import pickle

# Load a processed data file
def load_processed_data(filepath):
    with gzip.open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['examples']

# Example usage
examples = load_processed_data('data/processed/your_file.pkl.gz')
for board, policy, value in examples[:5]:  # First 5 examples
    print(f"Board shape: {board.shape}")
    print(f"Policy shape: {policy.shape}")
    print(f"Value: {value}")
```

#### Version 2.0 (Enhanced - Planned)
```python
import gzip
import pickle

# Load a processed data file with format detection
def load_processed_data_v2(filepath):
    with gzip.open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    format_version = data.get('format_version', '1.0')
    examples = data['examples']
    
    if format_version == '2.0':
        # Enhanced format
        for example in examples[:5]:
            print(f"Board shape: {example['board'].shape}")
            print(f"Policy shape: {example['policy'].shape}")
            print(f"Value: {example['value']}")
            print(f"Game ID: {example['metadata']['game_id']}")
            print(f"Position: {example['metadata']['position_in_game']}/{example['metadata']['total_positions']}")
            print(f"Value tier: {example['metadata']['value_sample_tier']}")
    else:
        # Legacy format
        for board, policy, value in examples[:5]:
            print(f"Board shape: {board.shape}")
            print(f"Policy shape: {policy.shape}")
            print(f"Value: {value}")
    
    return examples, format_version
```

### Value Sample Tiers (Version 2.0)

The enhanced format introduces value sample tiers to control how many positions per game are used for value network training:

- **Tier 0**: High priority (5 positions) - Always used for value training
- **Tier 1**: Medium priority (5 positions) - Usually used for value training  
- **Tier 2**: Low priority (10 positions) - Sometimes used for value training
- **Tier 3**: Very low priority (20+ positions) - Rarely used for value training

This allows flexible control over value training while preserving all positions for policy training.

### File Naming Convention
Processed files follow the pattern:
```
{original_filename}_processed.pkl.gz
```

For example:
- Original: `twoNetGames_13x13_mk45_d1b20_v1816_2s0_p2551k_vt25_pt10.trmph`
- Processed: `twoNetGames_13x13_mk45_d1b20_v1816_2s0_p2551k_vt25_pt10_processed.pkl.gz`

---

## Checkpoint Files (.pt)

*[To be documented when we get around to it]*

### Overview
Model checkpoint files contain trained model weights and training metadata.

### File Structure
*[To be documented]*

### How to Load
*[To be documented]*

---

## Raw TRMPH Files (.trmph)

*[To be documented if needed]*

### Overview
Raw game files in TRMPH format.

### File Structure
*[To be documented]*

### How to Parse
*[To be documented]* 