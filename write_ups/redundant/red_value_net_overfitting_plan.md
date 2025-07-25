# Value Network Overfitting Analysis and Strategic Plan

**Date:** July 2025, ~15th
**Status:** Active Investigation  
**Priority:** High

## References

- **Superseded by**: [write_ups/value_head_debugging_summary.md](value_head_debugging_summary.md) (**most up-to-date**)
- **Older Debugging Log**: [write_ups/debugging_value_head_performance.md](debugging_value_head_performance.md)

## 0. Executive Summary

### Project Overview
The Hex AI project is developing a neural network-based AI for the game of Hex using a two-headed ResNet architecture. The model consists of:
- **Policy Head**: Predicts the best next move (169 possible positions)
- **Value Head**: Predicts the probability of Blue winning (0.0 to 1.0)

### Current Progress
- **Data Pipeline**: Successfully processes 1.2M games from TRMPH format into training examples
- **Training Infrastructure**: Robust training pipeline with hyperparameter tuning, data augmentation, and monitoring
- **Policy Network**: Shows good performance and learning characteristics
- **Value Network**: **CRITICAL ISSUE** - Severe overfitting despite regularization attempts

### The Core Problem
The value network exhibits classic overfitting symptoms:
- **Training Loss**: Very low (0.03-0.04) indicating perfect fit to training data
- **Validation Loss**: Much higher (0.24-0.35) and increasing over time
- **Out-of-Sample Performance**: Poor on simple, obvious board positions
- **Specific Failure Case**: Cannot correctly evaluate simple winning positions (e.g., two straight lines meeting at center position (6,6))

**Key Observation**: The policy network correctly identifies (6,6) as the winning move, but the value network predicts ~50% (random) for which player should win, despite having a player-to-move channel.

## 1. Code Architecture and Overfitting Analysis

### Training Data Pipeline

#### Data Loading Strategy
```python
# From hex_ai/data_pipeline.py - StreamingProcessedDataset
class StreamingProcessedDataset(torch.utils.data.Dataset):
    def __init__(self, data_files: List[Path], chunk_size: int = 100000, 
                 shuffle_files: bool = True, max_examples: Optional[int] = None):
        self.data_files = data_files
        if shuffle_files:
            random.shuffle(self.data_files)  # File-level shuffling
        # ... chunk loading logic
    
    def _load_next_chunk(self):
        # ... load chunk from files
        if len(self.current_chunk) > 0:
            random.shuffle(self.current_chunk)  # Within-chunk shuffling
```

**Critical Issue**: The current shuffling regime only shuffles at the file and chunk level, not globally across all training examples. This means:

1. **Game Clustering**: Multiple positions from the same game appear together in training batches
2. **Temporal Correlation**: Early game positions (moves 1-15) likely uniquely identify most games
3. **Value Network Fingerprinting**: The network can learn to recognize specific games rather than general board patterns

#### Data Generation Process
```python
# From hex_ai/data_utils.py - extract_training_examples_from_game
def extract_training_examples_from_game(trmph_text: str, winner_from_file: str = None) -> List[Tuple]:
    # Creates training examples from ALL positions in a game
    for position in range(len(moves) + 1):
        board_state = create_board_from_moves(moves[:position])
        policy_target = None if position >= len(moves) else create_policy_target(next_move)
        value_target = 1.0 if winner_from_file == "1" else 0.0  # Same for all positions
        training_examples.append((board_state, policy_target, value_target))
```

**Problem**: Each game contributes 40-150 training examples, all with the same value target. This creates a perfect scenario for the value network to memorize game-specific patterns rather than learn general evaluation principles.

**Critical Insight**: The value loss of 0.07 in epoch 0 suggests the network is overfitting immediately, not from repeated exposure to the same data. This strongly indicates game fingerprinting is the root cause.

### Model Architecture and Training

#### Value Head Configuration
```python
# From hex_ai/training.py - Trainer.__init__
param_groups = [
    {
        'params': other_params,
        'lr': learning_rate,
        'weight_decay': weight_decay
    },
    {
        'params': value_head_params,
        'lr': learning_rate * value_learning_rate_factor,  # Slower learning
        'weight_decay': weight_decay * value_weight_decay_factor  # More regularization
    }
]
```

#### Current Hyperparameter Sweep
```python
# From scripts/hyperparam_sweep.py
SWEEP = {
    "learning_rate": [0.001, 0.01],
    "batch_size": [256],
    "max_grad_norm": [20],
    "dropout_prob": [0, 0.001],
    "weight_decay": [1e-4],
    "value_learning_rate_factor": [0.00001, 0.005],  # Value head learns slower
    "value_weight_decay_factor": [250.0, 10.0],  # Value head gets more regularization
}
```

**Analysis**: The current approach uses:
- **Reduced Learning Rate**: Value head learns 1000x-200x slower than policy head
- **Increased Weight Decay**: 10x-250x more regularization on value head
- **Data Augmentation**: 4x increase in training samples through board symmetries

### Data Augmentation Implementation
```python
# From hex_ai/data_utils.py - create_augmented_example_with_player_to_move
def create_augmented_example_with_player_to_move(board, policy, value, error_tracker=None):
    # Creates 4 augmented examples from each original example
    augmented_boards = create_augmented_boards(board)
    augmented_policies = create_augmented_policies(policy)
    augmented_values = create_augmented_values(value)
    # ... returns 4 augmented examples
```

**Impact**: While augmentation increases sample diversity, it doesn't address the fundamental issue of game clustering and temporal correlation in the training data.

## 2. Root Cause Analysis

### Primary Hypothesis: Game Fingerprinting
The value network is likely learning to recognize specific games rather than general board evaluation principles. Evidence:

1. **Training Loss**: Very low (0.03-0.04), indicating the network can perfectly fit the training data
2. **Validation Loss**: Much higher (0.24-0.35) and increasing, classic overfitting pattern
3. **Simple Position Failure**: Cannot evaluate obvious winning positions that weren't in training
4. **Game Structure**: Each game provides 40-150 examples with identical value targets
5. **Shuffling Regime**: Insufficient to break game-level correlations
6. **Immediate Overfitting**: Value loss of 0.07 in epoch 0 suggests game fingerprinting, not repeated exposure

### Secondary Hypothesis: Insufficient Regularization
While weight decay and reduced learning rates help, they may not be sufficient to prevent memorization of game-specific patterns.

### Tertiary Hypothesis: Data Quality Issues
Some games may have systematic errors or biases that the network learns to exploit.

## 3. Current Mitigation Strategies

### Implemented Solutions
1. **Value Head Regularization**: Reduced learning rate and increased weight decay
2. **Data Augmentation**: 4x increase in training samples through board symmetries
3. **Increased Dataset Size**: From 100k to 1.6M samples (16k games vs 1k games)

### Results
- **Policy Network**: Shows reasonable generalization (validation loss ~2.6 vs training ~2.2)
- **Value Network**: Severe overfitting (validation loss 0.24-0.35 vs training 0.03-0.04)
- **Training Loss**: Remains very low (0.03-0.04)
- **Out-of-Sample Performance**: Still poor on simple positions
- **Data Augmentation**: No significant impact on value network overfitting
- **Increased Dataset Size**: From 100k to 1.6M samples (16k games) - no improvement

## 4. Strategic Recommendations

### Immediate Actions (High Priority)

#### 4.1 Implement Global Data Shuffling
**Problem**: Current shuffling only operates at file and chunk level
**Solution**: Implement true global shuffling across all training examples

```python
# Proposed implementation in StreamingProcessedDataset
class GloballyShuffledDataset(StreamingProcessedDataset):
    def __init__(self, data_files, max_examples=None, shuffle_window=10000):
        super().__init__(data_files, max_examples)
        self.shuffle_window = shuffle_window
        self.global_buffer = []
    
    def _load_with_global_shuffling(self):
        # Load examples into global buffer
        # Shuffle within sliding window
        # Ensure no more than N examples from same game in window
```

**Expected Impact**: Break game-level correlations, force network to learn general patterns

#### 4.2 Implement Game-Level Sampling
**Problem**: Each game contributes many examples with identical value targets
**Solution**: Sample positions from each game more sparsely

```python
# Proposed modification to extract_training_examples_from_game
def extract_training_examples_from_game(trmph_text, winner_from_file, 
                                      max_positions_per_game=20, 
                                      sample_strategy='uniform'):
    # Sample only max_positions_per_game positions from each game
    # Use different sampling strategies (uniform, early-game biased, etc.)
```

**Expected Impact**: Reduce game fingerprinting, increase diversity of value targets

#### 4.3 Enhanced Regularization Techniques
**Problem**: Current regularization may be insufficient
**Solution**: Implement additional regularization methods

```python
# Proposed additions to Trainer
class EnhancedValueRegularization:
    def __init__(self, dropout_rate=0.3, label_smoothing=0.1):
        self.dropout_rate = dropout_rate
        self.label_smoothing = label_smoothing
    
    def apply_regularization(self, value_pred, value_target):
        # Apply dropout to value head
        # Apply label smoothing to value targets
        # Add noise to value targets during training
```

### Medium-Term Solutions

#### 4.4 Curriculum Learning
**Problem**: Network may be overwhelmed by complex positions early in training
**Solution**: Start with simple positions and gradually increase complexity

```python
# Proposed curriculum implementation
class CurriculumDataset:
    def __init__(self, datasets_by_complexity):
        self.datasets = datasets_by_complexity
        self.current_stage = 0
    
    def get_training_data(self, epoch):
        # Return appropriate complexity level based on training progress
```

#### 4.5 Self-Play Data Generation
**Problem**: Limited diversity in human game data
**Solution**: Generate additional training data through self-play

```python
# Proposed self-play implementation
class SelfPlayDataGenerator:
    def __init__(self, model, num_games=1000):
        self.model = model
    
    def generate_games(self):
        # Use current model to play games against itself
        # Extract training examples from self-play games
        # Mix with human game data
```

#### 4.6 Value Network Architecture Modifications
**Problem**: Current value head may be too complex for the task
**Solution**: Simplify value head architecture

**Current Architecture Analysis**:
```python
# Current value head (from hex_ai/models.py)
self.value_head = nn.Linear(CHANNEL_PROGRESSION[3], VALUE_OUTPUT_SIZE)
# This is: nn.Linear(512, 1) - a single linear layer from 512 features to 1 output
```

**Proposed Simplified Value Head**:
```python
# Proposed simplified value head with reduced capacity
class SimplifiedValueHead(nn.Module):
    def __init__(self, input_size, hidden_size=32):  # Much smaller hidden layer
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.5)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
    
    def forward(self, x):
        x = F.relu(self.batch_norm(self.fc1(x)))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x
```

**Alternative: Separate Feature Extraction**:
```python
# Proposed separate value feature extraction
class SeparateValueHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # Use fewer channels for value head
        self.value_conv = nn.Conv2d(512, 64, kernel_size=1)  # Reduce from 512 to 64 channels
        self.value_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.value_fc = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # x is the output from layer4 (512 channels)
        x = F.relu(self.value_conv(x))
        x = self.value_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = torch.sigmoid(self.value_fc(x))
        return x
```

### Long-Term Solutions

#### 4.7 Multi-Task Learning
**Problem**: Value network may benefit from auxiliary tasks
**Solution**: Add auxiliary prediction tasks

```python
# Proposed multi-task value head
class MultiTaskValueHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.shared = nn.Linear(input_size, 128)
        self.value_head = nn.Linear(128, 1)
        self.auxiliary_head = nn.Linear(128, 10)  # Predict game phase, etc.
```

#### 4.8 Ensemble Methods
**Problem**: Single value network may be unreliable
**Solution**: Train multiple value networks and ensemble predictions

```python
# Proposed ensemble implementation
class ValueEnsemble:
    def __init__(self, models):
        self.models = models
    
    def predict(self, board):
        predictions = [model(board) for model in self.models]
        return torch.mean(torch.stack(predictions), dim=0)
```

## 5. Evaluation and Monitoring

### Key Metrics to Track
1. **Value Loss**: Training vs validation loss curves
2. **Simple Position Accuracy**: Performance on obvious winning positions
3. **Game Fingerprinting Test**: Performance on positions from unseen games
4. **Policy-Value Consistency**: Agreement between policy and value predictions

### Testing Framework
```python
# Proposed evaluation script
class ValueNetworkEvaluator:
    def __init__(self, model):
        self.model = model
    
    def test_simple_positions(self):
        # Test on obvious winning positions
        # Test on positions with clear evaluation
    
    def test_game_independence(self):
        # Test on positions from games not in training set
        # Measure correlation between game ID and prediction accuracy
```

## 6. Implementation Priority

### Phase 1 (Week 1-2): Critical Fixes
1. **Implement global data shuffling** - Break game-level correlations
2. **Add game-level sampling** - Limit positions per game (e.g., max 20 positions per game)
3. **Simplify value head architecture** - Reduce from 512→1 to 512→64→1 or 512→32→1
4. **Test player-to-move channel** - Verify the 3rd channel is working correctly

### Phase 2 (Week 3-4): Advanced Techniques
1. Implement curriculum learning
2. Add self-play data generation
3. Simplify value head architecture

### Phase 3 (Month 2+): Research Directions
1. Multi-task learning approaches
2. Ensemble methods
3. Novel regularization techniques

## 7. Success Criteria

### Short-Term Goals (1-2 weeks)
- [ ] Value network achieves >80% accuracy on simple winning positions
- [ ] Training and validation loss curves show convergence
- [ ] No evidence of game fingerprinting in evaluation

### Medium-Term Goals (1-2 months)
- [ ] Value network generalizes well to unseen game types
- [ ] Policy and value predictions are consistent
- [ ] Model performs competitively in self-play tournaments

### Long-Term Goals (3+ months)
- [ ] Value network achieves human-level evaluation accuracy
- [ ] Robust performance across diverse game scenarios
- [ ] Scalable architecture for larger board sizes

## 8. Risk Assessment

### High Risk
- **Data Quality Issues**: Systematic errors in training data
- **Architecture Limitations**: Current model may be fundamentally limited
- **Computational Constraints**: Solutions may require significant computational resources

### Medium Risk
- **Over-regularization**: Too much regularization may hurt learning
- **Implementation Complexity**: Advanced solutions may introduce new bugs
- **Evaluation Bias**: Test positions may not be representative

### Low Risk
- **Hyperparameter Tuning**: Fine-tuning existing parameters
- **Data Augmentation**: Expanding current augmentation techniques
- **Monitoring**: Adding more comprehensive evaluation metrics

## 9. Specific Implementation Recommendations

### Immediate Actions (This Week)

#### 9.1 Test Player-to-Move Channel
**Priority**: HIGH - This could be a simple fix

```python
# Test script to verify player-to-move channel is working
def test_player_to_move_channel():
    # Create simple test positions
    # Position 1: Empty board, Blue to move (un-augmented: Blue goes first)
    # Position 2: Same board, Red to move (un-augmented: Red goes second)
    # Note: After augmentation, colors may be swapped for reflections
    # Verify value predictions are different
    pass
```

#### 9.2 Implement Game-Level Sampling
**Priority**: HIGH - Most likely to solve the core problem

**Option 1A: Enhanced Data Processing with Metadata (Recommended)**

This approach involves reprocessing the data with comprehensive metadata and flexible sampling strategies.

### Design Overview

#### 1. Enhanced Data Format
```python
# New processed data format with metadata
{
    'examples': [
        {
            'board': np.ndarray,           # (2, 13, 13) board state
            'policy': np.ndarray,          # (169,) policy target or None
            'value': float,                # 0.0 or 1.0
            'metadata': {
                'game_id': (file_idx, line_idx),  # (file_index, line_index) tuple
                'position_in_game': int,          # 0-based position index
                'total_positions': int,           # Total positions in game
                'value_sample_tier': int,         # 0, 1, 2, 3 for sampling control
                'trmph_game': str,                # Full TRMPH string (optional)
                'winner': str                     # "BLUE" or "RED" (clearer than "1"/"2")
            }
        },
        # ... more examples
    ],
    'source_file': str,
    'processing_stats': dict,
    'processed_at': str,
    'format_version': '2.0'  # Track format changes
}

# File lookup table (separate file)
{
    'file_mapping': {
        0: 'data/twoNetGames/twoNetGames_13x13_mk45_d1b20_v1816_2s0_p2551k_vt25_pt10.trmph',
        1: 'data/twoNetGames/twoNetGames_13x13_mk45_d1b20_v1816_2s0_p2551k_vt25_pt11.trmph',
        # ... more files
    },
    'created_at': str,
    'total_files': int
}
```

#### 2. Flexible Sampling Strategy
```python
def assign_value_sample_tiers(total_positions: int) -> List[int]:
    """
    Assign sampling tiers to positions in a game.
    
    Tiers:
    - 0: High priority (5 positions) - Always used for value training
    - 1: Medium priority (5 positions) - Usually used for value training  
    - 2: Low priority (10 positions) - Sometimes used for value training
    - 3: Very low priority (20+ positions) - Rarely used for value training
    
    This allows flexible control over how many positions per game are used
    for value network training while keeping all positions for policy training.
    """
    if total_positions <= 5:
        # Small games: all positions get tier 0
        return [0] * total_positions
    
    tiers = []
    positions_per_tier = [5, 5, 10, max(0, total_positions - 20)]
    
    for tier, count in enumerate(positions_per_tier):
        if count > 0:
            tiers.extend([tier] * min(count, total_positions - len(tiers)))
    
    # Shuffle within each tier to avoid bias
    tier_groups = {}
    for i, tier in enumerate(tiers):
        if tier not in tier_groups:
            tier_groups[tier] = []
        tier_groups[tier].append(i)
    
    # Shuffle each tier group
    for tier in tier_groups:
        random.shuffle(tier_groups[tier])
    
    # Reconstruct tiers list
    result = [0] * total_positions
    for tier, indices in tier_groups.items():
        for idx in indices:
            result[idx] = tier
    
    return result
```

#### 2.5 Memory-Efficient Shuffling Strategy

**Problem**: With 1.2M games and 87M+ positions, global shuffling exceeds memory capacity.

**Proposed Solution**: Multi-pass processing with position-based stratification

```python
def create_stratified_dataset(trmph_files, output_dir, positions_per_pass=5):
    """
    Create dataset in multiple passes to break game correlations while managing memory.
    
    Pass 1: Process positions 0-4 from all games
    Pass 2: Process positions 5-9 from all games  
    Pass 3: Process positions 10-14 from all games
    etc.
    
    This creates a 'striped' dataset where each file contains positions from
    different stages of games, breaking temporal correlations.
    """
    for pass_num in range(0, max_positions, positions_per_pass):
        start_pos = pass_num
        end_pos = min(pass_num + positions_per_pass, max_positions)
        
        # Process all games for this position range
        examples = []
        for file_idx, trmph_file in enumerate(trmph_files):
            games = load_trmph_file(trmph_file)
            for line_idx, game_line in enumerate(games):
                try:
                    trmph_url, winner = parse_trmph_game_record(game_line)
                    game_examples = extract_positions_range(
                        trmph_url, winner, start_pos, end_pos, 
                        game_id=(file_idx, line_idx)
                    )
                    examples.extend(game_examples)
                except Exception as e:
                    logger.warning(f"Error processing game: {e}")
        
        # Shuffle within this pass
        random.shuffle(examples)
        
        # Save pass-specific file
        output_file = output_dir / f"pass_{pass_num:03d}_positions_{start_pos}-{end_pos}.pkl.gz"
        save_processed_data(output_file, examples, pass_num)

def extract_positions_range(trmph_text, winner, start_pos, end_pos, game_id):
    """Extract only positions in the specified range from a game."""
    moves = parse_moves(trmph_text)
    examples = []
    
    for position in range(start_pos, min(end_pos, len(moves) + 1)):
        # Create example for this position
        example = create_example_for_position(moves, position, winner, game_id)
        examples.append(example)
    
    return examples
```

**Alternative: Chunked Processing with Smart Boundaries**
```python
def create_chunked_dataset(trmph_files, output_dir, games_per_chunk=10000):
    """
    Process games in chunks, but ensure chunks don't align with file boundaries.
    """
    all_games = []
    for file_idx, trmph_file in enumerate(trmph_files):
        games = load_trmph_file(trmph_file)
        for line_idx, game_line in enumerate(games):
            all_games.append((file_idx, line_idx, game_line))
    
    # Shuffle game order
    random.shuffle(all_games)
    
    # Process in chunks
    for chunk_idx in range(0, len(all_games), games_per_chunk):
        chunk_games = all_games[chunk_idx:chunk_idx + games_per_chunk]
        examples = []
        
        for file_idx, line_idx, game_line in chunk_games:
            # Process game normally
            examples.extend(process_single_game(game_line, file_idx, line_idx))
        
        # Shuffle within chunk
        random.shuffle(examples)
        
        # Save chunk
        output_file = output_dir / f"chunk_{chunk_idx:06d}.pkl.gz"
        save_processed_data(output_file, examples, chunk_idx)
```

**Recommended Approach**: Use the stratified multi-pass approach as it:
1. **Breaks temporal correlations** - positions from different game stages are mixed
2. **Manages memory** - processes in small batches
3. **Maintains diversity** - each training batch sees positions from all game stages
4. **Simple to implement** - straightforward processing logic

#### 3. Enhanced trmph → pkl.gz Processing Function
```python
def extract_training_examples_from_game_v2(
    trmph_text: str, 
    winner_from_file: str = None,
    game_id: Tuple[int, int] = None,  # (file_idx, line_idx)
    include_trmph: bool = False,       # Whether to include full TRMPH string
    shuffle_positions: bool = True
) -> List[Dict]:
    """
    Extract training examples with comprehensive metadata and flexible sampling.
    
    Args:
        trmph_text: Complete TRMPH string
        winner_from_file: Winner from file data ("1" for blue, "2" for red)
        game_id: Tuple of (file_index, line_index) for tracking
        include_trmph: Whether to include full TRMPH string in metadata
        shuffle_positions: Whether to shuffle position order within game
    """
    # Parse moves and validate
    bare_moves = strip_trmph_preamble(trmph_text)
    moves = split_trmph_moves(bare_moves)
    
    # Handle repeated moves
    moves = _remove_repeated_moves(moves)
    
    if not moves:
        raise ValueError("Empty game after removing repeated moves")
    
    # Validate winner and convert to clear format
    if winner_from_file not in ["1", "2"]:
        raise ValueError(f"Invalid winner format: {winner_from_file}")
    
    # Convert winner format: "1"=BLUE, "2"=RED
    winner_clear = "BLUE" if winner_from_file == "1" else "RED"
    value_target = 0.0 if winner_from_file == "1" else 1.0  # BLUE=0.0, RED=1.0
    
    total_positions = len(moves) + 1
    
    # Assign sampling tiers
    value_sample_tiers = assign_value_sample_tiers(total_positions)
    
    # Create position indices (shuffle if requested)
    position_indices = list(range(total_positions))
    if shuffle_positions:
        random.shuffle(position_indices)
    
    training_examples = []
    
    for i, position in enumerate(position_indices):
        # Create board state
        board_state = create_board_from_moves(moves[:position])
        
        # Create policy target
        policy_target = None if position >= len(moves) else create_policy_target(moves[position])
        
        # Create metadata
        metadata = {
            'game_id': game_id,
            'position_in_game': position,
            'total_positions': total_positions,
            'value_sample_tier': value_sample_tiers[i],
            'winner': winner_clear  # Store as "BLUE" or "RED"
        }
        
        if include_trmph:
            metadata['trmph_game'] = trmph_text
        
        # Create example
        example = {
            'board': board_state,
            'policy': policy_target,
            'value': value_target,
            'metadata': metadata
        }
        
        training_examples.append(example)
    
    return training_examples
```

#### 4. Training Pipeline Updates

**New Dataset Class for Enhanced Format**
```python
class EnhancedStreamingDataset(torch.utils.data.Dataset):
    """Dataset that handles the new enhanced data format with metadata."""
    
    def __init__(self, data_files, value_tier_threshold=1, **kwargs):
        """
        Args:
            data_files: List of processed data files
            value_tier_threshold: Maximum tier to include for value training
                                 (0=only tier 0, 1=0+1, 2=0+1+2, 3=all tiers)
        """
        super().__init__(data_files, **kwargs)
        self.value_tier_threshold = value_tier_threshold
    
    def __getitem__(self, idx):
        # Load example with metadata
        example = super().__getitem__(idx)
        
        # Check if this is enhanced format
        if isinstance(example, dict) and 'metadata' in example:
            # Enhanced format
            board = torch.FloatTensor(example['board'])
            policy = torch.FloatTensor(example['policy']) if example['policy'] is not None else torch.zeros(169)
            value = torch.FloatTensor([example['value']])
            metadata = example['metadata']
            
            # Determine if this example should be used for value training
            value_sample_tier = metadata.get('value_sample_tier', 0)
            use_for_value = value_sample_tier <= self.value_tier_threshold
            
            # Add player-to-move channel
            board_3ch = self._add_player_to_move_channel(board)
            
            return board_3ch, policy, value, use_for_value, metadata
        else:
            # Legacy format - backward compatibility
            board, policy, value = example
            board_3ch = self._add_player_to_move_channel(torch.FloatTensor(board))
            policy = torch.FloatTensor(policy) if policy is not None else torch.zeros(169)
            value = torch.FloatTensor([value])
            return board_3ch, policy, value, True, {}  # Always use for value training
```

**Enhanced Loss Function**
```python
class EnhancedPolicyValueLoss(nn.Module):
    """Loss function that handles selective value training."""
    
    def __init__(self, policy_weight=0.14, value_weight=0.86):
        super().__init__()
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.policy_loss = nn.CrossEntropyLoss()
        self.value_loss = nn.MSELoss()
    
    def forward(self, policy_pred, value_pred, policy_target, value_target, use_for_value):
        """
        Args:
            use_for_value: Boolean tensor indicating which examples to use for value loss
        """
        # Policy loss (always computed)
        policy_loss = self.policy_loss(policy_pred, policy_target.argmax(dim=1))
        
        # Value loss (only for selected examples)
        if use_for_value.any():
            value_loss = self.value_loss(
                value_pred[use_for_value].squeeze(), 
                value_target[use_for_value].squeeze()
            )
        else:
            value_loss = torch.tensor(0.0, device=policy_pred.device)
        
        # Combined loss
        total_loss = self.policy_weight * policy_loss + self.value_weight * value_loss
        
        return total_loss, {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'value_examples_used': use_for_value.sum().item()
        }
```

#### 5. File Lookup Table and Winner Format

**File Lookup Table**
```python
# Create alongside processed data: file_lookup_YYYYMMDD_HHMMSS.json
{
    'file_mapping': {
        0: 'data/twoNetGames/twoNetGames_13x13_mk45_d1b20_v1816_2s0_p2551k_vt25_pt10.trmph',
        1: 'data/twoNetGames/twoNetGames_13x13_mk45_d1b20_v1816_2s0_p2551k_vt25_pt11.trmph',
        # ... more files
    },
    'created_at': '2025-01-XXTXX:XX:XX',
    'total_files': 42,
    'format_version': '1.0'
}
```

**Winner Format Clarification**
- **TRMPH format**: "1" = BLUE win, "2" = RED win
- **Training format**: BLUE = 1.0, RED = 0.0
- **Enhanced metadata**: Store as "BLUE" or "RED" for clarity
- **BLUE definition**: Goes from top to bottom, always goes first in un-augmented data
- **Note**: After augmentation, colors may be swapped for reflections

#### 6. Documentation Updates Required

**Update write_ups/data_formats.md**
- Document new enhanced format structure
- Add examples of metadata fields
- Update file structure documentation
- Add migration guide from v1.0 to v2.0
- Document winner format mapping and BLUE/RED definitions

**Update Training Code**
- Modify `hex_ai/data_pipeline.py` to handle enhanced format
- Update `hex_ai/training.py` to use enhanced loss function
- Add configuration options for value tier thresholds
- Update hyperparameter sweep to test different tier thresholds

**Note**: Alternative approaches (post-processing sampling and streaming with game tracking) were considered but rejected in favor of the enhanced data processing approach for its simplicity, maintainability, and effectiveness.

#### 9.3 Simplify Value Head Architecture
**Priority**: MEDIUM - Architectural change

```python
# Modify TwoHeadedResNet.__init__
# Replace current value head with:
self.value_head = nn.Sequential(
    nn.Linear(CHANNEL_PROGRESSION[3], 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 1)
)
```

### Quick Diagnostic Tests

#### Test 1: Game Fingerprinting Detection
```python
def test_game_fingerprinting():
    # Train on subset of games
    # Test on positions from held-out games
    # Compare performance vs positions from training games
    pass
```

#### Test 2: Simple Position Evaluation
```python
def test_simple_positions():
    # Create obvious winning positions
    # Test value network predictions
    # Should be close to 0.0 or 1.0, not 0.5
    pass
```

### Additional trmph → pkl.gz Processing Improvements

#### Fix Repeated Moves Issue
```python
# In scripts/process_all_trmph.py, modify extract_training_examples_from_game
def extract_training_examples_from_game(trmph_text: str, winner_from_file: str = None, 
                                      max_positions_per_game: int = 20,
                                      handle_repeated_moves: bool = True) -> List[Tuple]:
    """
    Extract training examples with improved validation.
    """
    # Parse moves
    bare_moves = strip_trmph_preamble(trmph_text)
    moves = split_trmph_moves(bare_moves)
    
    # Handle repeated moves
    if handle_repeated_moves:
        moves = self._remove_repeated_moves(moves)
    
    # ... rest of existing code ...

def _remove_repeated_moves(self, moves: List[str]) -> List[str]:
    """Remove repeated moves and all subsequent moves from the game."""
    seen_moves = set()
    clean_moves = []
    
    for move in moves:
        if move in seen_moves:
            # Found repeated move - discard this and all subsequent moves
            logger.warning(f"Repeated move {move} found, discarding game from this point")
            break
        seen_moves.add(move)
        clean_moves.append(move)
    
    return clean_moves
```

#### Add Randomization During trmph → pkl.gz Processing
```python
# In scripts/process_all_trmph.py, add randomization options
def process_single_file(self, file_path: Path, 
                       randomize_games: bool = True,
                       randomize_positions: bool = True) -> Dict[str, Any]:
    """Process file with optional randomization."""
    
    # Load games
    games = load_trmph_file(str(file_path))
    
    # Randomize game order if requested
    if randomize_games:
        random.shuffle(games)
    
    # Process games with position randomization
    all_examples = []
    for i, game_line in enumerate(games):
        try:
            trmph_url, winner = parse_trmph_game_record(game_line)
            examples = extract_training_examples_from_game(
                trmph_url, winner, 
                max_positions_per_game=20,  # Add sampling
                randomize_positions=randomize_positions
            )
            all_examples.extend(examples)
        except Exception as e:
            logger.warning(f"Error processing game {i+1}: {e}")
    
    # Randomize final example order
    if randomize_positions:
        random.shuffle(all_examples)
    
    return all_examples
```

## 10. Conclusion

The value network overfitting issue represents a significant challenge that requires a multi-faceted approach. The primary focus should be on breaking game-level correlations in the training data through global shuffling and game-level sampling, while simultaneously implementing enhanced regularization techniques.

**Key Insights from Your Data**:
1. **Immediate Overfitting**: Value loss of 0.07 in epoch 0 confirms game fingerprinting
2. **Policy vs Value**: Policy network generalizes reasonably, value network does not
3. **Data Augmentation**: No significant impact, suggesting the problem is structural
4. **Player-to-Move Channel**: May not be working correctly (50% predictions on obvious wins)

The success of the policy network suggests that the fundamental architecture and data pipeline are sound, and that the value network issues are likely solvable through improved training strategies rather than fundamental architectural changes.

**Next Steps**: 
1. **Test the player-to-move channel functionality** - Quick diagnostic
2. **Implement enhanced data processing** - New format with metadata and flexible sampling
3. **Fix repeated moves issue** - Add to data processing pipeline
4. **Update training pipeline** - Handle enhanced format with selective value training
5. **Simplify the value head architecture** - Reduce from 512→1 to 512→64→1
6. **Monitor results and iterate** based on empirical findings

### Complete Design Summary

**Enhanced Data Processing Approach**:
- **New Format**: Rich metadata including game tracking, position info, and sampling tiers
- **Flexible Sampling**: 4-tier system (0-3) for controlling value training intensity
- **Backward Compatibility**: Legacy format still supported
- **Game Tracking**: (file_idx, line_idx) tuples for easy debugging and validation
- **Repeated Moves**: Automatic detection and removal during processing

**Training Pipeline Updates**:
- **Enhanced Dataset**: Handles both legacy and enhanced formats
- **Selective Value Training**: Only selected positions used for value loss
- **Policy Training**: All positions still used for policy training
- **Configurable Tiers**: Easy adjustment of value training intensity

**Key Advantages**:
1. **Solves Root Cause**: Breaks game fingerprinting through flexible sampling
2. **Maintains Policy Quality**: All positions still used for policy training
3. **Future-Proof**: Easy to extend with additional metadata
4. **Debugging Friendly**: Rich metadata enables better analysis
5. **Backward Compatible**: Existing code continues to work

### Implementation Priority Summary

**Week 1: Enhanced Data Processing**
- **Day 1-2**: Implement `extract_training_examples_from_game_v2` with metadata
- **Day 3**: Add `assign_value_sample_tiers` function and repeated moves handling
- **Day 4**: Update `scripts/process_all_trmph.py` to use new format
- **Day 5**: Regenerate processed data with enhanced format

**Week 2: Training Pipeline Updates**
- **Day 1-2**: Implement `EnhancedStreamingDataset` and `EnhancedPolicyValueLoss`
- **Day 3**: Update training code to handle new format
- **Day 4**: Test with different value tier thresholds (0, 1, 2, 3)
- **Day 5**: Monitor results and iterate

**Week 3: Documentation and Optimization**
- **Day 1**: Update `write_ups/data_formats.md` with new format
- **Day 2**: Add configuration options for tier thresholds
- **Day 3**: Update hyperparameter sweep to test tier strategies
- **Day 4-5**: Fine-tune based on empirical results

### Key Benefits of This Approach

1. **Flexible Control**: Can easily adjust how many positions per game are used for value training
2. **Policy Training**: All positions still used for policy training (no information loss)
3. **Backward Compatibility**: Legacy format still supported
4. **Rich Metadata**: Game tracking enables better debugging and analysis
5. **Future-Proof**: Easy to add more metadata fields as needed 