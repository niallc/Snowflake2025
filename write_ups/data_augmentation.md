# Data Augmentation for Hex AI

## Project Context

This is a Hex AI project using a two-headed neural network (policy and value heads) in the style of AlphaZero. The project is experiencing overfitting in the value head while the policy head needs more training. Data augmentation is being implemented to increase effective dataset size and improve generalization.

The training pipeline uses PyTorch with custom datasets that load from `.pkl.gz` files containing `(board_2ch, policy, value)` tuples. The model expects 3-channel input `(blue_channel, red_channel, player_to_move_channel)` where the player-to-move is computed on-the-fly from the board state.

## Goals

Data augmentation aims to increase the effective training dataset size by exploiting the symmetries of the Hex board. By applying board transformations that preserve the logical game state, we can create additional training examples without collecting new data.

## Board Symmetries

The Hex board has several symmetries that preserve the logical game state:

1. **180° Rotation**: Rotates the board 180 degrees without changing colors
2. **Long Diagonal Reflection + Color Swap**: Reflects across the long diagonal and swaps blue/red pieces
3. **Short Diagonal Reflection + Color Swap**: Reflects across the short diagonal and swaps blue/red pieces

## Current Implementation

### Board Transformations

The four board states produced by augmentation are:
1. **Original**: Unmodified board state
2. **180° Rotation**: Board rotated 180° (no color swap)
3. **Long Diagonal Reflection + Color Swap**: Board reflected across long diagonal with colors swapped
4. **Short Diagonal Reflection + Color Swap**: Board reflected across short diagonal with colors swapped

### Label Transformations

#### Policy Labels
Policy labels (one-hot vectors indicating the next move) must be transformed to match the board transformations:
- **180° Rotation**: Policy is rotated 180° (no color swap)
- **Long Diagonal Reflection**: Policy is reflected across long diagonal
- **Short Diagonal Reflection**: Policy is reflected across short diagonal

#### Value Labels
Value labels (game outcome: 0.0 or 1.0) must be swapped for color-swapping symmetries:
- **Original & 180° Rotation**: Value unchanged
- **Long Diagonal Reflection + Color Swap**: Value swapped (1.0 ↔ 0.0)
- **Short Diagonal Reflection + Color Swap**: Value swapped (1.0 ↔ 0.0)

## Player-to-Move Channel

### Current System Architecture

The player-to-move information is **not stored** in the processed data files. Instead, it's computed on-the-fly during training using the `get_player_to_move_from_board()` function.

#### How Player-to-Move is Determined

The function `get_player_to_move_from_board(board_2ch)` determines whose turn it is by counting pieces:
- If `blue_count == red_count`: Blue's turn (BLUE_PLAYER = 0)
- If `blue_count == red_count + 1`: Red's turn (RED_PLAYER = 1)
- Otherwise: Invalid board state (raises error or uses default)

#### Integration with Training Pipeline

In `hex_ai/training_utils_legacy.py`, the `__getitem__` method:
1. Loads the 2-channel board from the data file
2. Calls `get_player_to_move_from_board(board_np)` to determine whose turn it is
3. Creates a 3rd channel filled with the player-to-move value
4. Concatenates to create a 3-channel board: `(blue_channel, red_channel, player_to_move_channel)`

### Augmentation Impact on Player-to-Move

When we augment boards with color-swapping symmetries, the player-to-move must also change:

#### Rules for Player-to-Move Transformation
- **Original & 180° Rotation**: Player-to-move unchanged
- **Long Diagonal Reflection + Color Swap**: Player-to-move swapped (0 ↔ 1)
- **Short Diagonal Reflection + Color Swap**: Player-to-move swapped (0 ↔ 1)

#### Implementation Strategy

Since the player-to-move is computed on-the-fly, we have two options:

1. **Modify `get_player_to_move_from_board()`**: Add a parameter to handle augmented boards
2. **Create augmented player-to-move function**: Create a separate function that computes the correct player-to-move for each augmentation

The second approach is cleaner as it doesn't modify the core logic and makes the augmentation process more explicit.

### Proposed Implementation

```python
def create_augmented_player_to_move(player_to_move: int) -> list[int]:
    """
    Create player-to-move values for the 4 board augmentations.
    - For color-swapping symmetries, swap the player (0 <-> 1).
    """
    return [
        player_to_move,          # Original
        player_to_move,          # 180° rotation (no color swap)
        1 - player_to_move,      # Long diagonal reflection + color swap
        1 - player_to_move       # Short diagonal reflection + color swap
    ]
```

## Data Flow

### Current Training Flow
1. **Data Loading**: `NewProcessedDataset.__getitem__(idx)` or `StreamingProcessedDataset.__getitem__(idx)`
   - Loads 2-channel board from `.pkl.gz` file: `(blue_channel, red_channel)`
   - Loads policy target: `(169,)` one-hot vector
   - Loads value target: `float` (0.0 or 1.0)

2. **Player-to-Move Computation**: `get_player_to_move_from_board(board_2ch)`
   - Counts blue and red pieces
   - Returns `BLUE_PLAYER=0` or `RED_PLAYER=1`

3. **3-Channel Board Creation**: 
   - Creates player-to-move channel: `(13, 13)` filled with player value
   - Concatenates: `(blue_channel, red_channel, player_channel)` → `(3, 13, 13)`

4. **Tensor Conversion**:
   - Converts to PyTorch tensors: `board_state`, `policy_target`, `value_target`

5. **DataLoader Batching**:
   - PyTorch DataLoader batches multiple examples
   - Returns: `(boards, policies, values)` where each is a batch tensor

6. **Training Loop**:
   - Model forward pass: `policy_pred, value_pred = model(boards)`
   - Loss computation: `loss = criterion(policy_pred, value_pred, policies, values)`

### Proposed Augmented Training Flow

#### Option 1: Augmentation in `__getitem__` (Recommended)
**Pros**: Simple, works with existing DataLoader, no changes to training loop
**Cons**: Each call returns 4 examples, which changes the effective dataset size

**Implementation**:
```python
def __getitem__(self, idx):
    # Load original example
    board_2ch, policy, value = self.examples[idx]
    
    # Compute player-to-move
    player_to_move = get_player_to_move_from_board(board_2ch)
    
    # Create all 4 augmented examples
    augmented_examples = []
    for i in range(4):
        # Get augmented components
        aug_board_2ch = create_augmented_boards(board_2ch)[i]
        aug_policy = create_augmented_policies(policy)[i]
        aug_value = create_augmented_values(value)[i]
        aug_player = create_augmented_player_to_move(player_to_move)[i]
        
        # Create 3-channel board
        player_channel = np.full((13, 13), float(aug_player), dtype=np.float32)
        board_3ch = np.concatenate([aug_board_2ch, player_channel[None, ...]], axis=0)
        
        # Convert to tensors
        board_tensor = torch.from_numpy(board_3ch)
        policy_tensor = torch.FloatTensor(aug_policy)
        value_tensor = torch.FloatTensor([aug_value])
        
        augmented_examples.append((board_tensor, policy_tensor, value_tensor))
    
    # Return all 4 examples (DataLoader will handle batching)
    return augmented_examples
```

**DataLoader Impact**:
- Original dataset size: N examples
- Augmented dataset size: N examples (but each returns 4 samples)
- Effective training samples: 4N
- Batch size: If batch_size=32, you get 8 original examples × 4 augmentations = 32 samples

#### Option 2: Augmentation in Training Loop
**Pros**: Keeps dataset size unchanged, more control over augmentation
**Cons**: More complex, requires changes to training loop

**Implementation**:
```python
def train_epoch(self):
    for batch_idx, (boards, policies, values) in enumerate(self.train_loader):
        # Apply augmentation to each sample in the batch
        augmented_boards = []
        augmented_policies = []
        augmented_values = []
        
        for i in range(boards.size(0)):
            board_2ch = boards[i, :2]  # Extract 2-channel board
            policy = policies[i]
            value = values[i]
            
            # Apply augmentation (randomly choose 1 of 4)
            aug_idx = random.randint(0, 3)
            aug_board_2ch = create_augmented_boards(board_2ch)[aug_idx]
            aug_policy = create_augmented_policies(policy)[aug_idx]
            aug_value = create_augmented_values(value)[aug_idx]
            
            # Create 3-channel board
            player_to_move = get_player_to_move_from_board(aug_board_2ch)
            player_channel = np.full((13, 13), float(player_to_move), dtype=np.float32)
            board_3ch = np.concatenate([aug_board_2ch, player_channel[None, ...]], axis=0)
            
            augmented_boards.append(torch.from_numpy(board_3ch))
            augmented_policies.append(torch.FloatTensor(aug_policy))
            augmented_values.append(torch.FloatTensor([aug_value]))
        
        # Stack into batch tensors
        boards = torch.stack(augmented_boards)
        policies = torch.stack(augmented_policies)
        values = torch.stack(augmented_values)
        
        # Continue with training...
```

#### Option 3: Pre-augmented Dataset
**Pros**: Fastest training, no runtime augmentation overhead
**Cons**: 4x storage, 4x memory usage, requires preprocessing step

**Implementation**:
1. Create preprocessing script that generates augmented `.pkl.gz` files
2. Each original example becomes 4 examples in the augmented dataset
3. Use existing training pipeline unchanged

### Recommended Approach: Option 1

I recommend **Option 1** because:
1. **Minimal code changes**: Only modify `__getitem__` method
2. **Works with existing pipeline**: No changes to DataLoader or training loop
3. **Flexible**: Can easily disable augmentation by returning single example
4. **Memory efficient**: No preprocessing step, no 4x storage increase

### Implementation Details

#### Dataset Class Changes
```python
class AugmentedProcessedDataset(NewProcessedDataset):
    def __init__(self, data_files, enable_augmentation=True, **kwargs):
        super().__init__(data_files, **kwargs)
        self.enable_augmentation = enable_augmentation
    
    def __getitem__(self, idx):
        if not self.enable_augmentation:
            return super().__getitem__(idx)
        
        # Load original example
        board_2ch, policy, value = self.examples[idx]
        
        # Skip empty boards (no pieces to augment)
        if np.sum(board_2ch) == 0:
            return super().__getitem__(idx)
        
        # Create augmented examples...
        # (implementation as shown above)
```

#### Configuration
```python
# In training config
AUGMENTATION_CONFIG = {
    'enable_augmentation': True,
    'augmentation_probability': 1.0,  # Always augment
    'skip_empty_boards': True,  # Don't augment empty boards
}
```

## Next Steps for Implementation

### Immediate Tasks

1. **Integrate AugmentedProcessedDataset into Training Pipeline**
   - Update `run_hyperparameter_tuning_current_data()` in `hex_ai/training_utils_legacy.py`
   - Replace `NewProcessedDataset` with `AugmentedProcessedDataset`
   - Add `collate_fn=augmented_collate_fn` to DataLoader creation
   - Add configuration option to enable/disable augmentation

2. **Update Training Scripts**
   - Modify `scripts/hyperparam_sweep.py` to use augmented datasets
   - Add augmentation configuration to experiment parameters
   - Test with small datasets first

3. **Performance Validation**
   - Run training with and without augmentation to compare:
     - Training speed (samples/second)
     - Memory usage
     - Convergence behavior
   - Monitor for any issues with the augmentation pipeline

### Integration Example

```python
# In run_hyperparameter_tuning_current_data()
train_dataset = AugmentedProcessedDataset(
    train_files, 
    enable_augmentation=True,  # Add to experiment config
    max_examples=max_examples_per_split
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=hyperparams['batch_size'],
    shuffle=True,
    collate_fn=augmented_collate_fn,  # Required for augmentation
    num_workers=0
)
```

### Configuration Options

Add to experiment configuration:
```python
AUGMENTATION_CONFIG = {
    'enable_augmentation': True,
    'augmentation_probability': 1.0,  # Always augment
    'skip_empty_boards': True,  # Don't augment empty boards
}
```

### Testing Strategy

1. **Small-scale testing**: Use `max_examples=1000` to verify augmentation works
2. **Performance comparison**: Run identical experiments with/without augmentation
3. **Validation**: Compare augmented outputs with `scripts/visualize_board_augmentations.py`
4. **Memory monitoring**: Check for any memory issues with 4x data

### Expected Benefits

- **4x effective dataset size** without storage increase
- **Better generalization** through board symmetries
- **Reduced overfitting** in value head
- **Improved policy training** with more diverse examples

### Potential Issues to Monitor

- **Training speed**: Runtime augmentation may slow training
- **Memory usage**: 4x examples in memory during batching
- **Batch size effects**: Effective batch size is 4x larger
- **Learning rate**: May need adjustment with 4x more data

## TODO

- [x] **Integration**: Integrate `AugmentedProcessedDataset` into training pipeline
- [x] **Configuration**: Add enable/disable flag for augmentation
- [x] **Performance Testing**: Test training performance with augmentation enabled
- [x] **Validation**: Ensure validation data is not augmented (as intended)
- [x] **Documentation**: Update training documentation
- [x] **Error Handling**: Handle `None` policy targets for final moves
- [x] **Validation Set Size**: Optimize validation dataset size for faster training
- [x] **Cleanup**: Remove debug logging and polish code for production

## Future Improvements

- [ ] **Performance Optimization**: Profile and optimize augmentation performance
- [ ] **Memory Usage**: Monitor and optimize memory usage with 4x data
- [ ] **Hyperparameter Tuning**: Adjust learning rates/batch sizes for augmented data
- [ ] **Policy Target Handling**: Consider using uniform distribution `(1/169, ...)` instead of zeros for final moves
- [ ] **Visual Testing**: Add comprehensive visual validation of augmentations

## Implementation Summary

### Completed Components

1. **Core Augmentation Functions** (`hex_ai/data_utils.py`):
   - `create_augmented_boards()`: Creates 4 board transformations
   - `create_augmented_policies()`: Transforms policy labels accordingly
   - `create_augmented_values()`: Swaps values for color-swapping symmetries
   - `create_augmented_player_to_move()`: Swaps player-to-move for color-swapping symmetries
   - `create_augmented_example()`: Comprehensive function that handles all labels
   - `create_augmented_example_with_player_to_move()`: Includes player-to-move computation

2. **Augmented Dataset Class** (`hex_ai/training_utils_legacy.py`):
   - `AugmentedProcessedDataset`: Inherits from `NewProcessedDataset`
   - Implements augmentation in `__getitem__` method
   - Returns 4 augmented examples per original example
   - Handles empty boards gracefully (no augmentation)

3. **DataLoader Integration**:
   - `augmented_collate_fn()`: Custom collate function for DataLoader
   - Flattens 4 augmented examples into single batch
   - Maintains compatibility with existing training pipeline

4. **Testing and Validation**:
   - `scripts/visualize_board_augmentations.py`: Visual verification of all transformations
   - `scripts/test_augmentation.py`: End-to-end testing of augmentation pipeline
   - All transformations verified to work correctly

### Key Features

- **4x Data Augmentation**: Each training example becomes 4 examples through symmetries
- **Label Consistency**: All labels (board, policy, value, player-to-move) are correctly transformed
- **Empty Board Handling**: Empty boards are not augmented (no pieces to transform)
- **Memory Efficient**: No preprocessing step, augmentation happens on-the-fly
- **Backward Compatible**: Can disable augmentation to use original pipeline
- **PyTorch Compatible**: Works seamlessly with DataLoader and existing training code

### Usage Example

```python
# Create augmented dataset
dataset = AugmentedProcessedDataset(
    data_files=[Path("data/processed/example.pkl.gz")],
    enable_augmentation=True
)

# Create DataLoader with custom collate function
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,  # Will give ~128 effective samples (32 * 4 augmentations)
    shuffle=True,
    collate_fn=augmented_collate_fn
)

# Use in training (no changes to training loop needed)
for boards, policies, values in dataloader:
    # boards.shape: (batch_size, 3, 13, 13)
    # policies.shape: (batch_size, 169)
    # values.shape: (batch_size, 1)
    policy_pred, value_pred = model(boards)
    loss = criterion(policy_pred, value_pred, policies, values)
``` 