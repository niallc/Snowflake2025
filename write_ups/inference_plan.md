# Update 2025-07-25: Inference Plan Progress and Next Steps

## Current State
- **Model**: A trained model checkpoint is available at:
  - `checkpoints/hyperparameter_tuning/loss_weight_sweep_exp0_bs256_98f719_20250724_233408/epoch1_mini30.pt`
- **Board Size**: The model is currently trained for 13x13 Hex. Support for other sizes is possible in the future, but 13x13 is the default.
- **Simple Inference**: A basic CLI script (`scripts/simple_inference_cli.py`) exists for running inference on static positions. Both policy and value heads are performing well in spot checks.
- **Interactive Play**: A simple interactive-play mode is now available via `scripts/play_vs_model_cli.py`, allowing a human to play against the model in the terminal.

## New Plan: Interactive Play
- **Goal**: Enable interactive play against the trained model, starting with a simple CLI interface and progressing to a browser-based, point-and-click UI.
- **Current Implementation**:
  - CLI script to play human vs model (`scripts/play_vs_model_cli.py`).
  - Board displayed in ASCII (using `hex_ai/inference/board_display.py`).
  - After each move, output the board's `.trmph` representation (using `hex_ai/utils/format_conversion.py`).
  - Model move selection: take top-k (e.g., 20) policy moves, evaluate with value head, pick the best (with some randomness for variety).
- **Next Step**: Develop a browser-based, point-and-click interface (e.g., using Flask) for easier and more engaging play.

## Future Directions
- **Point-and-Click Web UI**: Build a browser-based interface for interactive play (not yet implemented).
- **Tournament System**: Implement a tournament structure to compare different model checkpoints and engine configurations (e.g., different search depths/breadths). Not yet implemented.
- **MCTS**: Implement Monte Carlo Tree Search for stronger move selection. This is a major planned upgrade, not yet implemented.
- **Self-Play Pipeline**: Develop a self-play pipeline to generate new training data and enable reinforcement learning. Not yet implemented.
- **Scalability**: Batch inference, efficient data pipelines, and distributed play for RL.
- **Advanced Features**: Pie rule, support for other board sizes, online deployment, and more.

## What Has Changed
- The codebase has evolved significantly since the original plan. Many utilities and data formats have been consolidated and modernized.
- The immediate focus is now on usability and interactive play, rather than just batch inference or self-play.
- The plan is more incremental: start with a simple CLI, then add a browser-based UI, then expand to tournaments, MCTS, and self-play.

## Next Steps
1. Develop a browser-based, point-and-click interface for interactive play.
2. Implement a tournament structure to compare models and engine configs.
3. Implement MCTS for stronger move selection.
4. Build a self-play pipeline for new data generation and RL.
5. Continue to expand scalability and advanced features as needed.

---

# Hex AI Inference/Generation Plan

## Overview

This document outlines the plan for implementing inference and generation capabilities for the Hex AI project. The system will support three main modes:

1. **Self-play**: Generate training data to improve the model
2. **Tournament play**: Evaluate model performance and compare configurations
3. **Human play**: Interactive games against human players or other agents

## Architecture Overview

```
hex_ai/
├── inference/
│   ├── __init__.py
│   ├── game_engine.py          # Core game logic and state management
│   ├── model_wrapper.py        # Model loading and inference
│   ├── search.py               # MCTS and other search algorithms
│   ├── self_play.py           # Self-play data generation
│   ├── tournament.py           # Tournament evaluation
│   ├── human_play.py          # Interactive human play
│   └── utils.py               # Common utilities
```

## Core Components

### 1. Game Engine (`game_engine.py`)

**Purpose**: Core game logic, state management, and move validation

**Key Functions**:
- `HexGameState`: Game state representation
- `is_valid_move()`: Validate move legality
- `make_move()`: Apply move to state
- `get_winner()`: Check for game end
- `get_legal_moves()`: Get available moves
- `get_board_tensor()`: Convert to model input format

**Integration with Legacy Code**:
- Use `legacy_code/BoardUtils.py` for winner detection (Union-Find logic)
- Port relevant functions but adapt to modern board format
- Maintain compatibility with existing data formats

**Coordinate System Consistency**:
- Use existing functions from `hex_ai/data_utils.py` for all coordinate conversions
- Ensure board format matches training data: `(2, 13, 13)` tensor
- Player encoding: Blue = 0 (first player), Red = 1 (second player)

### 2. Model Wrapper (`model_wrapper.py`)

**Purpose**: Load trained models and provide inference interface

**Key Functions**:
- `load_model()`: Load checkpoint and create model
- `predict()`: Get policy and value predictions
- `get_move_probabilities()`: Extract move probabilities
- `batch_predict()`: Efficient batch inference

**Features**:
- Support for different model configurations
- Automatic device detection (CPU/MPS/CUDA)
- Model ensemble support
- Caching for efficiency

### 3. Search Algorithms (`search.py`)

**Purpose**: Implement search-based move selection

**Key Algorithms**:
- **MCTS (Monte Carlo Tree Search)**: Primary search algorithm
- **Alpha-Beta**: Alternative for comparison
- **Greedy**: Baseline for testing

**MCTS Implementation**:
```python
class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.prior = 0.0

class MCTS:
    def __init__(self, model, num_simulations=800):
        self.model = model
        self.num_simulations = num_simulations
    
    def search(self, state):
        # MCTS implementation
        pass
    
    def select_move(self, state, temperature=1.0):
        # Select final move based on visit counts
        pass
```

## Implementation Modes

### 1. Self-Play (`self_play.py`)

**Purpose**: Generate training data for model improvement

**Features**:
- Parallel game generation
- Configurable game parameters
- Data format compatibility
- Progress tracking and logging

**Configuration**:
```python
class SelfPlayConfig:
    num_games = 1000
    num_parallel = 4
    mcts_simulations = 800
    temperature = 1.0
    save_format = "trmph"  # or "numpy"
    output_dir = "data/self_play/"
```

**Output**:
- Game records in trmph format
- Training positions with labels
- Performance metrics

### 2. Tournament (`tournament.py`)

**Purpose**: Evaluate model performance and compare configurations

**Features**:
- Round-robin tournaments
- Elo rating system
- Performance metrics
- Model comparison

**Tournament Types**:
- **Round-robin**: All models play each other
- **Swiss system**: For large model pools
- **Knockout**: Single elimination
- **Double elimination**: More robust ranking

**Metrics**:
- Win rate
- Elo rating
- Average game length
- Move quality analysis

### 3. Human Play (`human_play.py`)

**Purpose**: Interactive games against human players

**Features**:
- Command-line interface
- Graphical board display
- Move validation
- Game history
- Analysis tools

**Interface Options**:
- **CLI**: Simple command-line interface
- **Web**: Flask/FastAPI web interface
- **GUI**: Tkinter or PyQt interface

## Model Integration

### Loading Trained Models

```python
def load_model(checkpoint_path: str, device: str = None) -> TwoHeadedResNet:
    """Load a trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = create_model()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model
```

### Inference Pipeline

```python
def get_model_prediction(model, state: HexGameState) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get policy and value predictions for a game state"""
    board_tensor = state.get_board_tensor()
    with torch.no_grad():
        policy_logits, value_logit = model(board_tensor)
    return policy_logits, value_logit
```

## Search Integration

### MCTS with Neural Network

```python
def mcts_search(state: HexGameState, model: TwoHeadedResNet, 
                num_simulations: int = 800) -> Dict[int, float]:
    """Run MCTS search using neural network for evaluation"""
    root = MCTSNode(state)
    
    for _ in range(num_simulations):
        node = root
        # Selection
        while node.children and not node.state.is_terminal():
            node = select_child(node)
        
        # Expansion
        if not node.state.is_terminal():
            policy, value = model.predict(node.state)
            expand_node(node, policy)
        
        # Simulation and backpropagation
        value = simulate_and_backpropagate(node, model)
    
    return get_move_probabilities(root)
```

## Data Formats

### Input Format
- **Board representation**: `(2, 13, 13)` tensor
- **Channel 0**: Blue player pieces (1.0 where blue piece exists, 0.0 elsewhere)
- **Channel 1**: Red player pieces (1.0 where red piece exists, 0.0 elsewhere)
- **Values**: Float32 (0.0 or 1.0) to match training data format

### Output Format
- **Policy**: `(169,)` tensor (flattened board)
- **Value**: `(1,)` tensor (win probability)

### Game Records
- **Primary**: TRMPH format for compatibility
- **Secondary**: JSON for analysis
- **Training**: NumPy arrays for efficiency

## Performance Considerations

### Optimization Strategies
1. **Batch processing**: Process multiple positions together
2. **Model caching**: Cache loaded models
3. **Parallel search**: Use multiple processes for MCTS
4. **Memory management**: Efficient tensor operations
5. **Device optimization**: GPU/MPS acceleration

### Scalability
- **Self-play**: Parallel game generation
- **Tournament**: Distributed evaluation
- **Human play**: Real-time response requirements

## Testing Strategy

### Unit Tests
- Game state management
- Move validation
- Model loading and inference
- Search algorithms

### Integration Tests
- End-to-end game play
- Data generation pipeline
- Tournament evaluation
- Human interface

### Performance Tests
- Inference speed
- Memory usage
- Search efficiency
- Scalability

### Test Data and Validation
- **Winner Detection Tests**: Create known winning positions to validate Union-Find logic
- **Coordinate Consistency Tests**: Verify all coordinate conversions work correctly
- **Training Data Compatibility**: Test that game states can be converted to training format
- **TRMPH Format Tests**: Ensure game serialization matches expected format

### Test Board Generation
```python
# Example test boards for winner detection
TEST_BOARDS = {
    "red_winner_horizontal": {
        "moves": ["a1", "b1", "a2", "b2", "a3", "b3", "a4", "b4", "a5", "b5", "a6", "b6", "a7"],
        "winner": "red",
        "description": "Red wins with horizontal line from left to right"
    },
    "blue_winner_vertical": {
        "moves": ["a1", "a2", "b1", "b2", "c1", "c2", "d1", "d2", "e1", "e2", "f1", "f2", "g1"],
        "winner": "blue", 
        "description": "Blue wins with vertical line from top to bottom"
    },
    "no_winner_midgame": {
        "moves": ["a1", "b2", "c3", "d4", "e5", "f6", "g7"],
        "winner": None,
        "description": "Mid-game position with no winner yet"
    }
}
```

### Testing Phases
1. **Phase 1**: Unit tests for game engine components
2. **Phase 2**: Integration tests with existing training data
3. **Phase 3**: Performance and stress tests
4. **Phase 4**: End-to-end validation with model inference

## Development Phases

### Phase 1: Core Infrastructure (Week 1-2)
- [ ] Game engine implementation
  - [ ] HexGameState class with proper board representation
  - [ ] Move validation using existing coordinate functions
  - [ ] Union-Find winner detection ported from legacy code
  - [ ] Game state serialization/deserialization
- [ ] Model wrapper
- [ ] Basic MCTS implementation
- [ ] Unit tests
  - [ ] Test board generation for winner detection
  - [ ] Coordinate conversion consistency tests
  - [ ] Training data compatibility tests

### Phase 2: Self-Play (Week 3-4)
- [ ] Self-play data generation
- [ ] Parallel processing
- [ ] Data format compatibility
- [ ] Performance optimization

### Phase 3: Tournament (Week 5-6)
- [ ] Tournament framework
- [ ] Elo rating system
- [ ] Performance metrics
- [ ] Model comparison tools

### Phase 4: Human Play (Week 7-8)
- [ ] Command-line interface
- [ ] Board visualization
- [ ] Move validation
- [ ] Analysis tools

### Phase 5: Integration and Optimization (Week 9-10)
- [ ] End-to-end testing
- [ ] Performance optimization
- [ ] Documentation
- [ ] Deployment preparation

## Configuration Management

### Model Configuration
```python
@dataclass
class ModelConfig:
    checkpoint_path: str
    device: str = "auto"
    temperature: float = 1.0
    mcts_simulations: int = 800
    batch_size: int = 32
```

### Game Configuration
```python
@dataclass
class GameConfig:
    board_size: int = 13
    komi: float = 0.0
    time_limit: Optional[float] = None
    move_limit: Optional[int] = None
```

### Search Configuration
```python
@dataclass
class SearchConfig:
    algorithm: str = "mcts"  # "mcts", "alpha_beta", "greedy"
    num_simulations: int = 800
    exploration_constant: float = 1.414
    temperature: float = 1.0
```

## Integration with Existing Code

### Legacy Code Integration
- Use `legacy_code/BoardUtils.py` for winner detection (Union-Find logic)
- Port relevant functions but adapt to modern board format
- Maintain compatibility with existing data formats

### Training Pipeline Integration
- Use same model architecture as training
- Compatible checkpoint loading
- Consistent data formats

### Data Pipeline Integration
- Compatible with existing data processing
- Support for TRMPH format
- Integration with training data generation

### Coordinate System Consistency
- **Use existing functions** from `hex_ai/data_utils.py`:
  - `trmph_move_to_rowcol()` for move parsing
  - `rowcol_to_trmph()` for move serialization
  - `tensor_to_rowcol()` / `rowcol_to_tensor()` for tensor conversions
- **Board format**: `(2, 13, 13)` tensor exactly as in training
  - Channel 0: Blue pieces (1.0 where blue piece exists, 0.0 elsewhere)
  - Channel 1: Red pieces (1.0 where red piece exists, 0.0 elsewhere)
- **Player encoding**: Blue = 0 (first player), Red = 1 (second player)
- **Coordinate system**: `(row, col)` with row 0 = top, col 0 = left
- **TRMPH format**: Letters a-m for columns, numbers 1-13 for rows

## Future Enhancements

### Advanced Features
- **Model ensembles**: Combine multiple models
- **Adaptive search**: Adjust search depth based on position
- **Opening book**: Pre-computed opening moves
- **Endgame database**: Perfect play for endgame positions

### Performance Improvements
- **GPU acceleration**: Optimize for GPU inference
- **Model compression**: Quantization and pruning
- **Distributed search**: Multi-machine MCTS
- **Caching**: Position evaluation caching

### User Experience
- **Web interface**: Modern web-based UI
- **Analysis tools**: Move analysis and visualization
- **Learning features**: Interactive tutorials
- **Social features**: Online play and tournaments

## Conclusion

This plan provides a comprehensive framework for implementing inference and generation capabilities for the Hex AI project. The modular design allows for incremental development and testing, while the integration with existing code ensures compatibility and efficiency.

The three main modes (self-play, tournament, human play) cover the primary use cases, and the search-based approach with MCTS provides a strong foundation for move selection. The performance considerations and testing strategy ensure robust and scalable implementation.

Next steps:
1. Start with Phase 1 (Core Infrastructure)
2. Implement basic game engine and model wrapper
3. Test with existing checkpoints
4. Iterate and refine based on performance 

# TEMPORARY: Stepwise Plan for Browser-Based Point-and-Click Hex Interface (Flask)

## Rationale
- Start with a robust, modular backend API using Flask, reusing all existing game and inference logic.
- Separate backend and frontend concerns for maintainability and extensibility.
- Enable the "URL = TRMPH" feature for easy interoperability with other tools and representations.

## Step 1: Minimal Flask Backend API
- Expose endpoints:
  - `/` (GET): Serves the frontend (can be static HTML for now).
  - `/api/state` (POST): Accepts a TRMPH string, returns board state, win probabilities, etc.
  - `/api/move` (POST): Accepts a TRMPH string and a move, returns updated state (board, model move, win probabilities, new TRMPH string).
- Reuse `HexGameState`, `SimpleModelInference`, and TRMPH utilities for all game/model logic.
- Keep all game/model logic in backend modules, not in Flask routes.

## Step 2: Minimal HTML/JS Frontend
- Renders a clickable 13x13 Hex board.
- Reads/writes the TRMPH string from/to the URL (enabling "URL = TRMPH").
- Calls the backend API to process moves and update the board.

## Next Steps
- Scaffold the Flask backend and endpoints, reusing existing code.
- Once the backend is working, build the frontend to consume the API and implement the point-and-click interface.
- Iterate and expand features as needed (analysis, model selection, etc.). 