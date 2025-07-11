# Hex Data Formats and Processing Pipeline (v0.1)

This document describes the initial data formats and processing pipeline for the Hex AI project. **These are v0.1 designs that will evolve based on performance analysis, modern best practices, and empirical findings.**

**Note:** This is a living document. Formats and approaches will be updated as we:
- Profile performance bottlenecks
- Discover more efficient representations
- Learn from modern ML practices
- Optimize for our specific use case

## Network Input/Output Formats (v0.1)

### Current Proposed Format

**Input:** `(batch_size, 2, 13, 13)` - 2 channels for blue/red players
**Policy Output:** `(batch_size, 169)` - flattened board positions  
**Value Output:** `(batch_size, 1)` - single win probability

### Modern Alternatives Under Consideration

**Input Format Options:**
```python
# Option 1: Current (2, 13, 13) - Good for CNNs
board = torch.zeros(2, 13, 13)

# Option 2: Single channel with values (1, 13, 13) - More compact
board = torch.zeros(1, 13, 13)  # 0=empty, 1=blue, 2=red

# Option 3: Multi-hot encoding (3, 13, 13) - Most explicit
board = torch.zeros(3, 13, 13)  # [empty, blue, red] channels

# Option 4: Attention-friendly format (169, 3) - For transformer models
board = torch.zeros(169, 3)  # Each position has [empty, blue, red] features
```

**Policy Output Options:**
```python
# Option 1: Current (169,) - Simple
policy = torch.zeros(169)

# Option 2: Logits vs probabilities
policy_logits = torch.zeros(169)  # Raw logits
policy_probs = torch.softmax(policy_logits, dim=-1)  # Probabilities

# Option 3: Move ranking (169,) - For MCTS integration
move_ranking = torch.zeros(169)  # Higher = better move
```

**Value Output Options:**
```python
# Option 1: Current (1,) - Win probability
value = torch.zeros(1)

# Option 2: Multi-class (3,) - Win/Draw/Loss probabilities
value = torch.zeros(3)  # [win, draw, loss]

# Option 3: Expected score (-1 to 1)
value = torch.zeros(1)  # -1 = certain loss, +1 = certain win
```

### Performance Considerations

**Training Scale:** 1-10M games (4M+ positions with augmentation)
**Hardware:** M1 Mac + occasional GPU access
**Real-time Requirements:** Yes (for game play)
**Memory Constraints:** Moderate (no data center resources)

### Modern Best Practices to Implement

1. **Mixed Precision Training:** Use `torch.float16` for efficiency
2. **Gradient Checkpointing:** For memory efficiency with larger models
3. **Efficient Data Loading:** Memory-mapped files, prefetching
4. **Model Parallelism:** For very large models (future consideration)

### Open Questions for Empirical Testing

1. **Model Architecture:** CNN vs Attention-based models
2. **Input Format:** 2-channel vs 3-channel vs attention-friendly
3. **Policy Output:** Logits vs probabilities vs rankings
4. **Value Output:** Binary vs multi-class vs expected score
5. **Data Augmentation:** Which symmetries are most effective?

---

## File Formats

**Description:** URL-based format used by www.trmph.com and the primary format for stored games.

**Structure:** 
- Preamble: `"http://www.trmph.com/hex/board#13,"` (for 13x13 boards)
- Moves: Sequence of letter+number pairs: `"a1b2c3d4e5f6..."`
- Example: `"http://www.trmph.com/hex/board#13,a1b2c3d4e5f6g7h8i9j10k11l12m13"`

**Coordinate System:**
- Letters: `a` through `m` (columns, left to right)
- Numbers: `1` through `13` (rows, top to bottom)
- Position `a1` = top-left corner
- Position `m13` = bottom-right corner

**Legacy Functions:**
- `StripMultiTrmphs()` - Remove preamble from multiple games
- `LookUpRowCol()` - Convert trmph to (row, col) coordinates
- `RowColToTrmph()` - Convert (row, col) to trmph format

### 2. DotHex Format (Matrix)

**Description:** 2D matrix representation, easier for neural network processing.

**Structure:**
- `0` = empty cell
- `1` = blue player (Player 1)
- `2` = red player (Player 2)
- Shape: `(13, 13)` for 13x13 boards

**Example:**
```
[[0, 1, 0, 0, ...],
 [0, 0, 2, 0, ...],
 [0, 0, 0, 1, ...],
 ...]
```

**Legacy Functions:**
- `trmphToDotHex()` - Convert trmph to matrix format
- `DotHexToOneHot()` - Convert matrix to one-hot encoding
- `OneHotToDotHex()` - Convert one-hot back to matrix

### 3. One-Hot Format (Vector)

**Description:** Vector representation where each position has 2 binary values.

**Structure:**
- Length: `13 * 13 * 2 = 338` for 13x13 boards
- Each position: `[blue_present, red_present]`
- `[0, 0]` = empty
- `[0, 1]` = blue player
- `[1, 0]` = red player

**Example:**
```
[0,0, 0,1, 1,0, 0,0, ...]  # 338 values total
```

**Legacy Functions:**
- `trmphToOneHotBoards()` - Convert trmph to one-hot
- `LookupOneHotPiece()` - Get piece at position
- `OneHotToFourOneHotBoards()` - Apply augmentations

## Data Augmentation

Hex has rich symmetries that can be exploited for data augmentation. The legacy code implements sophisticated augmentation strategies:

### Symmetry Types

1. **Rotational Symmetry (180°)**
   - Function: `RotateOneHotBy180()`
   - Effect: Rotates board 180° and swaps player colors
   - Preserves: Game outcome and strategic patterns

2. **Reflectional Symmetry (Long Diagonal)**
   - Function: `FlipColoursAlongDiag(diagonal="long")`
   - Effect: Reflects through main diagonal (top-left to bottom-right)
   - Preserves: Game outcome and strategic patterns

3. **Reflectional Symmetry (Short Diagonal)**
   - Function: `FlipColoursAlongDiag(diagonal="short")`
   - Effect: Reflects through secondary diagonal
   - Preserves: Game outcome and strategic patterns

### Augmentation Strategy

**Legacy Implementation:** `OneHotToFourOneHotBoards()`
- Takes one board position
- Returns 4 variants: original + rotation + 2 reflections
- **4x data multiplication** per training position

**Benefits:**
- Exploits Hex's natural symmetries
- Increases effective training data size
- Improves model generalization
- Reduces overfitting

## Processing Pipeline (Planned)

### Phase 1: File Reading
```python
# Input: .trmph files in data/twoNetGames/
# Output: List of trmph strings
def load_trmph_files(data_dir: str) -> List[str]:
    """Load all .trmph files from directory"""
    pass
```

### Phase 2: Format Conversion
```python
# Input: trmph strings
# Output: PyTorch tensors (batch_size, 2, 13, 13)
def trmph_to_tensor(trmph_text: str) -> torch.Tensor:
    """Convert trmph format to tensor representation"""
    pass
```

### Phase 3: Data Augmentation
```python
# Input: Single board tensor
# Output: List of 4 augmented tensors
def augment_board(board: torch.Tensor) -> List[torch.Tensor]:
    """Apply rotation and reflection augmentations"""
    pass
```

### Phase 4: Training Data Preparation
```python
# Input: Raw game data
# Output: Processed, augmented, batched data
def prepare_training_data(games: List[str]) -> HexDataset:
    """Convert games to training dataset with augmentation"""
    pass
```

## Legacy Code Analysis

### Worth Keeping/Adapting
- **`trmphToDotHex()`** - Core conversion, well-tested
- **`OneHotToFourOneHotBoards()`** - Proven augmentation strategy
- **`StripMultiTrmphs()`** - File format handling
- **Coordinate conversion functions** - `LookUpRowCol()`, `RowColToTrmph()`

### Needs Modernization
- **Code style** - Inconsistent naming, no type hints
- **Error handling** - Many `assert` statements
- **Performance** - Some inefficient loops
- **Documentation** - Minimal docstrings

### Performance Considerations
- Legacy code was profiled and optimized (2017-2019)
- Bottlenecks likely in I/O and data loading, not computation
- Modern PyTorch DataLoader with multiprocessing may help
- Consider memory-mapped files for large datasets

## Implementation Plan

1. **Document formats** (this document) ✅
2. **Create placeholder interfaces** for data pipeline
3. **Implement model architecture** to understand input requirements
4. **Design efficient data pipeline** based on model needs
5. **Implement modern conversion utilities** with type hints
6. **Add comprehensive testing** for data pipeline
7. **Optimize for performance** based on profiling

## File Structure

```
data/
├── twoNetGames/           # Raw .trmph files
│   ├── game1.trmph
│   ├── game2.trmph
│   └── ...
└── processed/             # Processed data (future)
    ├── train/
    ├── val/
    └── test/
```

## Modern Architecture Analysis

### CNN vs Attention-Based Models

**CNN (ResNet) Approach:**
- ✅ **Pros:** Proven for board games, efficient, good for spatial patterns
- ✅ **Pros:** Lower memory requirements, faster training
- ✅ **Pros:** Works well with 2D board representations
- ❌ **Cons:** May miss long-range dependencies
- ❌ **Cons:** Fixed receptive field size

**Attention-Based Models:**
- ✅ **Pros:** Can capture long-range dependencies
- ✅ **Pros:** More flexible for complex patterns
- ✅ **Pros:** State-of-the-art in many domains
- ❌ **Cons:** Higher memory requirements
- ❌ **Cons:** Slower training, more parameters
- ❌ **Cons:** May be overkill for board games

**Recommendation for Your Use Case:**
- **Start with ResNet:** Proven, efficient, good for 1-10M games
- **Consider attention later:** If you have GPU access and want to experiment
- **Hybrid approach:** CNN backbone + attention heads for policy/value

### Resource Requirements Comparison

**ResNet-18:**
- Parameters: ~11M
- Memory: ~2GB for training
- Training time: 1-2 days on M1 Mac
- Inference: Very fast

**ResNet-34:**
- Parameters: ~21M
- Memory: ~4GB for training
- Training time: 3-5 days on M1 Mac
- Inference: Fast

**Attention Model (Transformer):**
- Parameters: ~50M+
- Memory: ~8GB+ for training
- Training time: 1-2 weeks on M1 Mac
- Inference: Slower

### Modern Best Practices for Your Scale

1. **Start Simple:** ResNet-18 with proven architecture
2. **Use Mixed Precision:** `torch.float16` for 2x memory efficiency
3. **Gradient Checkpointing:** Trade compute for memory
4. **Efficient Data Loading:** Memory-mapped files, prefetching
5. **Model Parallelism:** Only if you get consistent GPU access

### Tree Search Integration

**Current Approach (Simple Tree Search):**
- ✅ Works with any model architecture
- ✅ Easy to implement and debug
- ✅ Good baseline for comparison

**Future MCTS/UCT Integration:**
- Requires policy/value outputs in specific format
- May benefit from move ranking outputs
- Transposition tables important for efficiency

**Recommendation:** Start with current approach, optimize model first, then improve search.

---

## Next Steps

1. **Phase 1:** Implement ResNet-18 model (proven, efficient)
2. **Phase 2:** Profile performance and memory usage
3. **Phase 3:** Experiment with attention if resources allow
4. **Phase 4:** Optimize tree search integration
5. **Phase 5:** Scale up with more data/GPU access

### Empirical Testing Plan

1. **Baseline:** ResNet-18 with current format
2. **Input Format Test:** 2-channel vs 3-channel
3. **Output Format Test:** Logits vs probabilities
4. **Architecture Test:** ResNet-18 vs ResNet-34 vs attention
5. **Augmentation Test:** Which symmetries help most?

**Success Metrics:**
- Training time per epoch
- Memory usage
- Inference speed
- Model accuracy
- Game play strength 