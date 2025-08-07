# Batched MCTS Implementation

## Overview

The batched MCTS implementation provides a significant performance improvement over the original MCTS by using batched neural network inference. This reduces GPU kernel launches from O(n) to O(1) where n is the number of positions evaluated during search.

## Architecture

### Core Components

1. **BatchProcessor** (`hex_ai/inference/batch_processor.py`)
   - Manages batched inference requests
   - Provides caching for repeated evaluations
   - Handles callback-based result distribution
   - Optimizes batch sizes for GPU utilization

2. **BatchedNeuralMCTS** (`hex_ai/inference/batched_mcts.py`)
   - State machine-based MCTS implementation
   - Uses BatchProcessor for efficient inference
   - Implements virtual loss for pending evaluations
   - Maintains clean separation of concerns

3. **BatchedMCTSSelfPlayEngine** (`hex_ai/selfplay/batched_mcts_selfplay_engine.py`)
   - Self-play engine using batched MCTS
   - Provides comprehensive statistics and monitoring
   - Optimized for training data generation

### State Machine Design

The implementation uses a state machine approach where each node can be in one of three states:

- **UNEXPANDED**: Node has not been evaluated by the neural network
- **EVALUATION_PENDING**: Node is waiting for neural network result
- **EXPANDED**: Node has been evaluated and children created

### Key Features

#### 1. Batched Inference
Instead of making individual neural network calls for each position, the implementation:
- Collects multiple evaluation requests
- Processes them in optimal batch sizes
- Distributes results via callbacks

#### 2. Virtual Loss
To prevent multiple simulations from exploring the same pending node:
- Nodes with pending evaluations receive virtual loss
- This makes them less attractive in PUCT selection
- Virtual loss is removed when evaluation completes

#### 3. Efficient Caching
- Results are cached by board state hash
- Cache hits return immediately without inference
- Significantly reduces redundant evaluations

#### 4. Performance Monitoring
- Comprehensive statistics tracking
- Cache hit rates, batch sizes, inference counts
- Real-time performance monitoring

## Usage

### Basic Usage

```python
from hex_ai.inference.batched_mcts import BatchedNeuralMCTS
from hex_ai.inference.simple_model_inference import SimpleModelInference

# Load model
model = SimpleModelInference("path/to/checkpoint.pt")

# Create batched MCTS engine
mcts = BatchedNeuralMCTS(
    model=model,
    exploration_constant=1.4,
    optimal_batch_size=64
)

# Run search
root = mcts.search(game_state, num_simulations=800)
selected_move = mcts.select_move(root, temperature=1.0)
```

### Self-Play Generation

```python
from hex_ai.selfplay.batched_mcts_selfplay_engine import BatchedMCTSSelfPlayEngine

# Create self-play engine
engine = BatchedMCTSSelfPlayEngine(
    model_path="path/to/checkpoint.pt",
    num_simulations=800,
    optimal_batch_size=64
)

# Generate games
games = engine.play_multiple_games(100)
```

## Performance Benefits

### Theoretical Improvements
- **GPU Utilization**: Better GPU utilization through larger batch sizes
- **Kernel Launch Overhead**: Reduced from O(n) to O(1) kernel launches
- **Memory Efficiency**: Better memory access patterns
- **Cache Efficiency**: Higher cache hit rates through batching

### Expected Performance
- **2-4x speedup** for typical MCTS searches
- **Higher cache hit rates** due to better batching
- **More consistent performance** with reduced variance

## Configuration

### Optimal Batch Size
The optimal batch size depends on:
- GPU memory capacity
- Model architecture
- Board size and complexity

Recommended values:
- **32-64**: Good balance for most setups
- **128+**: For high-end GPUs with large memory
- **16**: For memory-constrained environments

### Other Parameters
- `exploration_constant`: PUCT exploration constant (default: 1.4)
- `win_value`: Value for winning terminal states (default: 1.5)
- `discount_factor`: Move count penalty factor (default: 0.98)

## Testing

### Unit Tests
```bash
PYTHONPATH=. pytest tests/test_batched_mcts.py -v
```

### Performance Tests
```bash
python scripts/test_batched_mcts.py --model-path path/to/model.pt
```

### Self-Play Tests
```bash
python hex_ai/selfplay/batched_mcts_selfplay_engine.py path/to/model.pt --games 10
```

## Migration from Original MCTS

The batched implementation is designed to be a drop-in replacement for the original MCTS:

1. **Import Changes**:
   ```python
   # Old
   from hex_ai.inference.mcts import NeuralMCTS, MCTSNode
   
   # New
   from hex_ai.inference.batched_mcts import BatchedNeuralMCTS, BatchedMCTSNode
   ```

2. **API Compatibility**:
   - Same method signatures
   - Same return types
   - Same configuration parameters

3. **Performance Monitoring**:
   - Additional statistics available
   - Better performance insights
   - Cache hit rate tracking

## Future Enhancements

### Planned Improvements
1. **Multi-Game Batching**: Batch positions across multiple games
2. **Adaptive Batch Sizing**: Dynamic batch size optimization
3. **Memory Management**: Better memory usage optimization
4. **Parallel Processing**: Multi-threaded batch processing

### Research Directions
1. **Asynchronous Processing**: Non-blocking batch processing
2. **Predictive Batching**: Pre-emptive position evaluation
3. **Distributed Batching**: Multi-GPU batch distribution

## Troubleshooting

### Common Issues

1. **Low Cache Hit Rate**
   - Increase batch size
   - Check for position diversity
   - Verify cache configuration

2. **Memory Issues**
   - Reduce batch size
   - Clear cache periodically
   - Monitor memory usage

3. **Performance Degradation**
   - Check GPU utilization
   - Verify batch processing
   - Monitor inference times

### Debug Mode
Enable verbose logging for detailed debugging:
```python
mcts = BatchedNeuralMCTS(model, verbose=2)
```

## Contributing

When contributing to the batched MCTS implementation:

1. **Maintain State Machine Integrity**: Ensure proper state transitions
2. **Preserve Batching Efficiency**: Avoid breaking batch processing
3. **Add Comprehensive Tests**: Include unit and performance tests
4. **Update Documentation**: Keep this document current
5. **Performance Validation**: Verify improvements with benchmarks
