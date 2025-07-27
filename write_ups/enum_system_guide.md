# Enum System Guide: Type-Safe Player and Piece Constants

## Overview

This document explains the new enum-based system for handling player and piece constants in the Hex AI project. The system provides type safety, better IDE support, and prevents runtime errors while maintaining backward compatibility.

## Why Enums Are Better

### Problems with Simple Constants
```python
# Old approach in config.py
BLUE_PLAYER = 0
RED_PLAYER = 1
BLUE_PIECE = 1
RED_PIECE = 2
```

**Issues:**
- No type safety - can accidentally assign `BLUE_PLAYER = 42`
- No IDE autocomplete or intellisense
- No runtime validation of valid values
- Easy to make typos like `BLUE_PLAYER` vs `BLUE_PIECE`
- No clear relationship between related constants

### Benefits of Enums
```python
# New approach in value_utils.py
class Player(Enum):
    BLUE = 0
    RED = 1

class Piece(Enum):
    EMPTY = 0
    BLUE = 1
    RED = 2
```

**Benefits:**
- **Type Safety**: Can't accidentally assign invalid values
- **IDE Support**: Autocomplete, type checking, refactoring
- **Runtime Validation**: `Player(42)` raises an error
- **Clear Intent**: Makes it obvious what values are valid
- **Prevents Typos**: `Player.BLUE` vs `Player.RED` is much clearer
- **Immutable**: Can't accidentally reassign enum values

## Available Enums

### 1. Player Enum
```python
from hex_ai.value_utils import Player

class Player(Enum):
    BLUE = 0
    RED = 1
```

**Usage:**
- Game logic and player-to-move channel
- Replaces `BLUE_PLAYER` and `RED_PLAYER` constants
- Used for tracking whose turn it is

### 2. Piece Enum
```python
from hex_ai.value_utils import Piece

class Piece(Enum):
    EMPTY = 0
    BLUE = 1
    RED = 2
```

**Usage:**
- N×N board representation
- Replaces `EMPTY_PIECE`, `BLUE_PIECE`, `RED_PIECE` constants
- Used for board state representation

### 3. Channel Enum
```python
from hex_ai.value_utils import Channel

class Channel(Enum):
    BLUE = 0
    RED = 1
    PLAYER_TO_MOVE = 2
```

**Usage:**
- One-hot encoded board formats (2N×N and 3N×N)
- Replaces `BLUE_CHANNEL`, `RED_CHANNEL`, `PLAYER_CHANNEL` constants
- Used for tensor operations

### 4. Winner Enum (Existing)
```python
from hex_ai.value_utils import Winner

class Winner(Enum):
    BLUE = 0
    RED = 1
```

**Usage:**
- Game outcome representation
- Used in value head predictions and training targets

## Utility Functions

### Conversion Functions
```python
# Convert between enums and integers (for backward compatibility)
player_to_int(Player.BLUE)  # Returns 0
int_to_player(0)           # Returns Player.BLUE
piece_to_int(Piece.BLUE)   # Returns 1
int_to_piece(1)           # Returns Piece.BLUE
```

### Game Logic Functions
```python
# Get opponent
get_opponent(Player.BLUE)  # Returns Player.RED

# Check player color
is_blue(Player.BLUE)       # Returns True
is_red(Player.RED)         # Returns True

# Convert between Player and Winner
player_to_winner(Player.BLUE)  # Returns Winner.BLUE
winner_to_player(Winner.BLUE)  # Returns Player.BLUE
```

## Migration Guide

### 1. Immediate Benefits (No Code Changes Needed)
The system maintains backward compatibility, so existing code continues to work:
```python
# This still works
from hex_ai.config import BLUE_PLAYER, RED_PLAYER
if current_player == BLUE_PLAYER:
    next_player = RED_PLAYER
```

### 2. Gradual Migration (Recommended)
Replace constants with enums in new code or when touching existing code:

**Before:**
```python
from hex_ai.config import BLUE_PLAYER, RED_PLAYER

def get_next_player(current_player):
    if current_player == BLUE_PLAYER:
        return RED_PLAYER
    else:
        return BLUE_PLAYER
```

**After:**
```python
from hex_ai.value_utils import Player, get_opponent

def get_next_player(current_player):
    if isinstance(current_player, int):
        current_player = int_to_player(current_player)
    return get_opponent(current_player)
```

### 3. Tensor Operations
**Before:**
```python
from hex_ai.config import BLUE_CHANNEL, RED_CHANNEL, PLAYER_CHANNEL

# Unclear what these numbers mean
board_tensor[0, row, col] = 1  # Blue channel
board_tensor[1, row, col] = 1  # Red channel
board_tensor[2, :, :] = 0      # Player channel
```

**After:**
```python
from hex_ai.value_utils import Channel

# Clear intent
board_tensor[Channel.BLUE.value, row, col] = 1
board_tensor[Channel.RED.value, row, col] = 1
board_tensor[Channel.PLAYER_TO_MOVE.value, :, :] = 0
```

## Type Safety Examples

### 1. Runtime Validation
```python
# This raises an error
Player(42)  # ValueError: 42 is not a valid Player

# This works
Player(0)   # Returns Player.BLUE
```

### 2. IDE Support
```python
# IDE will autocomplete and show available options
player = Player.BLUE  # IDE shows: Player.BLUE, Player.RED

# IDE will catch typos
if player == Player.BLU  # IDE error: no such attribute
```

### 3. Refactoring Safety
```python
# If you rename Player.BLUE to Player.FIRST, IDE will update all references
# With constants, you'd have to manually find and replace all occurrences
```

## Best Practices

### 1. Use Enums for New Code
```python
# Good
def process_move(player: Player, position: Tuple[int, int]):
    if player == Player.BLUE:
        # Handle blue move
        pass

# Avoid (for new code)
def process_move(player: int, position: Tuple[int, int]):
    if player == 0:  # Magic number
        # Handle blue move
        pass
```

### 2. Use Utility Functions
```python
# Good
next_player = get_opponent(current_player)
color = winner_to_color(winner)

# Avoid
if current_player == Player.BLUE:
    next_player = Player.RED
else:
    next_player = Player.BLUE
```

### 3. Maintain Backward Compatibility
```python
# Good - handles both old and new formats
def process_player(player):
    if isinstance(player, int):
        player = int_to_player(player)
    # Now work with Player enum
    return get_opponent(player)
```

## Testing

The enum system includes comprehensive tests in `tests/test_enum_system.py` that demonstrate:
- Basic enum functionality
- Type safety benefits
- Conversion functions
- Utility functions
- Backward compatibility
- Practical usage examples

Run the tests to see the benefits in action:
```bash
python -m pytest tests/test_enum_system.py -v
```

## Future Improvements

1. **Gradual Migration**: Replace constants with enums throughout the codebase
2. **Type Hints**: Add proper type hints using the enums
3. **Documentation**: Update docstrings to reference enums instead of constants
4. **IDE Integration**: Configure IDE to suggest enums over constants

## Conclusion

The enum system provides significant benefits in terms of type safety, code clarity, and maintainability. While it maintains backward compatibility, gradually migrating to use enums will make the codebase more robust and easier to work with.

The key advantages are:
- **Prevents Runtime Errors**: Invalid values are caught immediately
- **Improves Code Clarity**: Intent is clear from the enum names
- **Better IDE Support**: Autocomplete, refactoring, and error detection
- **Maintains Compatibility**: Existing code continues to work
- **Future-Proof**: Easy to extend with new values if needed 