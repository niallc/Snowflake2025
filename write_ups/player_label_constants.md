# Constants Centralization and Board Format Documentation

## Overview

This document describes the centralized constants system and the different board formats used throughout the Hex AI project. The goal is to eliminate magic numbers and make the code more readable and maintainable.

## Board Formats

### 1. N×N Format (Single Channel, Trinary)
**Shape**: `(BOARD_SIZE, BOARD_SIZE)`  
**Values**: 
- `0` = Empty position
- `1` = Blue piece  
- `2` = Red piece

**Use cases**: Display, game logic, debugging, coordinate calculations

**Constants**:
```python
EMPTY_PIECE = 0
BLUE_PIECE = 1  
RED_PIECE = 2
```

### 2. 2N×N Format (Two Channels, One-Hot Encoded)
**Shape**: `(2, BOARD_SIZE, BOARD_SIZE)`  
**Values**: 
- `0` = Empty position
- `1` = Occupied position (channel indicates color)

**Channels**:
- `[0]` = Blue pieces channel
- `[1]` = Red pieces channel

**Use cases**: Neural network input (legacy models), data augmentation

**Constants**:
```python
PIECE_ONEHOT = 1      # Value for occupied positions
EMPTY_ONEHOT = 0      # Value for empty positions
BLUE_CHANNEL = 0      # Channel index for blue pieces
RED_CHANNEL = 1       # Channel index for red pieces
```

### 3. 3N×N Format (Three Channels, One-Hot Encoded + Player-to-Move)
**Shape**: `(3, BOARD_SIZE, BOARD_SIZE)`  
**Values**: 
- `0` = Empty position
- `1` = Occupied position (channels 0,1 indicate color)
- `0` or `1` = Player-to-move (channel 2)

**Channels**:
- `[0]` = Blue pieces channel
- `[1]` = Red pieces channel  
- `[2]` = Player-to-move channel

**Use cases**: Neural network input (current models)

**Constants**:
```python
PIECE_ONEHOT = 1      # Value for occupied positions
EMPTY_ONEHOT = 0      # Value for empty positions
BLUE_CHANNEL = 0      # Channel index for blue pieces
RED_CHANNEL = 1       # Channel index for red pieces
PLAYER_CHANNEL = 2    # Channel index for player-to-move
```

## Player Constants

### Game Logic Players
**Values**: `0` = Blue, `1` = Red  
**Use cases**: Player-to-move logic, game state management

**Constants**:
```python
BLUE_PLAYER = 0
RED_PLAYER = 1
```

### TRMPH Format Winners (Legacy)
**Values**: `"1"` = Blue win, `"2"` = Red win  
**Use cases**: Parsing TRMPH data files

**Constants**:
```python
TRMPH_BLUE_WIN = "1"
TRMPH_RED_WIN = "2"
```

### Training Format Winners
**Values**: `0.0` = Blue win, `1.0` = Red win  
**Use cases**: Neural network training targets

**Constants**:
```python
TRAINING_BLUE_WIN = 0.0
TRAINING_RED_WIN = 1.0
```

## Conversion Patterns

### N×N ↔ 2N×N Conversion
```python
# N×N to 2N×N
board_2nxn[BLUE_CHANNEL] = (board_nxn == BLUE_PIECE).astype(np.float32)
board_2nxn[RED_CHANNEL] = (board_nxn == RED_PIECE).astype(np.float32)

# 2N×N to N×N  
board_nxn[board_2nxn[BLUE_CHANNEL] == PIECE_ONEHOT] = BLUE_PIECE
board_nxn[board_2nxn[RED_CHANNEL] == PIECE_ONEHOT] = RED_PIECE
```

### 2N×N ↔ 3N×N Conversion
```python
# 2N×N to 3N×N (add player-to-move channel)
player_channel = np.full((BOARD_SIZE, BOARD_SIZE), float(player_to_move), dtype=np.float32)
board_3nxn = np.concatenate([board_2nxn, player_channel[None, ...]], axis=0)
```

## Implementation Guidelines

### 1. Always Use Constants
Replace magic numbers with descriptive constants:
```python
# ❌ Bad
if board[row, col] == 1:
    # blue piece logic

# ✅ Good  
if board[row, col] == BLUE_PIECE:
    # blue piece logic
```

### 2. Format-Specific Constants
Use format-specific constants when working with one-hot encoded boards:
```python
# ❌ Bad
board_2nxn[0] == 1.0

# ✅ Good
board_2nxn[BLUE_CHANNEL] == PIECE_ONEHOT
```

### 3. Clear Comments
Add comments explaining format conversions:
```python
# Convert N×N format (0=empty, 1=blue, 2=red) to 2N×N one-hot format
board_2nxn[BLUE_CHANNEL] = (board_nxn == BLUE_PIECE).astype(np.float32)
```

### 4. Function Documentation
Always document the expected format in function docstrings:
```python
def process_board(board: np.ndarray) -> np.ndarray:
    """
    Process a board in N×N format.
    
    Args:
        board: N×N array with values 0=empty, 1=blue, 2=red
        
    Returns:
        Processed board in same format
    """
```

## Migration Status

- ✅ N×N format constants implemented
- ✅ Player constants implemented  
- ✅ TRMPH format constants implemented
- ✅ Training format constants implemented
- ✅ One-hot format constants implemented
- ✅ Format conversion functions updated
- ✅ All modules migrated to use constants

## Benefits

1. **Readability**: Code is self-documenting
2. **Maintainability**: Changes only need to be made in one place
3. **Error Prevention**: Eliminates magic number typos
4. **Consistency**: All code uses the same naming scheme
5. **Documentation**: Constants serve as living documentation of formats 