# Hex AI Data Formats

This document describes the various data formats used in the Hex AI project, including how to read and work with them.

## Processed Data Files (.pkl.gz)

### Overview
Processed data files contain training examples extracted from raw TRMPH game files. They are stored as compressed pickle files with the extension `.pkl.gz`.

### File Structure
```python
{
    'examples': [(board, policy, value), ...],  # List of training examples
    'source_file': str,                         # Original TRMPH file path
    'processing_stats': dict,                   # Processing metadata
    'processed_at': str                         # Timestamp of processing
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
- **board**: `numpy.ndarray` of shape `(2, 13, 13)` - 2-channel format (blue, red)
- **policy**: `numpy.ndarray` of shape `(169,)` - one-hot policy target, or `None` for final moves
- **value**: `float` - value target (0.0 or 1.0)

### Policy Target Handling
**Important**: Policy targets can be `None` for final moves in games where there is no next move to predict. This occurs when:
- A game has ended (someone won)
- The current position is the final position of the game

**Current handling in training**:
- `None` policies are converted to zero vectors `(169,)` filled with zeros
- This allows the model to train on final positions without policy loss

**Future improvement**: Consider using uniform distribution `(1/169, 1/169, ...)` instead of zeros for better training signal.

### How to Read
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