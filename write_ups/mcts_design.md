# Revised MCTS Design for Hex AI (Single-Machine Focus)

## 1. Overview

This document outlines a revised plan for implementing a Monte Carlo Tree Search (MCTS) agent for Hex. It replaces the previous fixed-depth search with a modern, AlphaZero-style algorithm. The design prioritizes a phased implementation, starting with a simple, correct core and incrementally adding performance optimizations suitable for a single-machine, single-GPU setup.

## 2. Core MCTS Architecture

The fundamental architecture remains a ResNet with policy and value heads, integrated into an MCTS search loop.

### 2.1. MCTS Node (`MCTSNode`)

The node structure is the building block of the search tree.

```python
@dataclass
class MCTSNode:
    """Represents a node in the MCTS search tree."""
    # Core state
    state: HexGameState  # The game state this node represents. MUST be a deep copy.
    parent: Optional['MCTSNode'] = None
    move: Optional[Tuple[int, int]] = None  # The move that led from parent to this node.

    # Search statistics
    visits: int = 0
    total_value: float = 0.0  # Sum of all evaluations from this node's subtree.

    # Neural network priors (cached from the first time this node is expanded)
    policy_priors: Optional[Dict[Tuple[int, int], float]] = None
    
    # Children management
    children: Dict[Tuple[int, int], 'MCTSNode'] = field(default_factory=dict)

    @property
    def mean_value(self) -> float:
        """The mean value (Q-value) of this node."""
        if self.visits == 0:
            return 0.0
        # The value is from the perspective of the player *who just moved* to reach this state.
        return self.total_value / self.visits
    
    def is_leaf(self) -> bool:
        """A node is a leaf if it has not been expanded yet."""
        return not self.children

    def is_terminal(self) -> bool:
        """A node is terminal if the game is over."""
        return self.state.game_over
```

### 2.2. MCTS Engine (`NeuralMCTS`)

This class orchestrates the search process.

```python
class NeuralMCTS:
    """MCTS engine guided by a neural network."""
    
    def __init__(self, model: SimpleModelInference, exploration_constant: float = 1.4):
        self.model = model
        self.exploration_constant = exploration_constant
        # Optimizations will be added later, e.g., transposition_table, batch_collector.

    def search(self, root_state: HexGameState, num_simulations: int) -> MCTSNode:
        """Run MCTS search from a root state to build up statistics."""
        root = MCTSNode(state=root_state)

        for _ in range(num_simulations):
            # 1. Selection: Traverse the tree using PUCT until a leaf node is found.
            leaf_node = self._select(root)
            
            # 2. Expansion & Evaluation: If the game is not over, expand the leaf and get its value from the NN.
            value = self._expand_and_evaluate(leaf_node)
            
            # 3. Backpropagation: Update statistics up the tree from the leaf.
            self._backpropagate(leaf_node, value)
            
        return root

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Traverse the tree from the root to a leaf node."""
        while not node.is_leaf():
            node = self._select_child_with_puct(node)
        return node

    def _select_child_with_puct(self, node: MCTSNode) -> MCTSNode:
        """Select the child with the highest PUCT score."""
        # PUCT = Q(s,a) + C * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        # Q(s,a) is the child's mean_value.
        # P(s,a) is the child's prior probability from the parent's policy.
        # N(s) is the parent's visit count.
        # N(s,a) is the child's visit count.
        
        best_score = -float('inf')
        best_child = None

        for move, child_node in node.children.items():
            # Note: child_node.mean_value is from the perspective of the player who made the move.
            # We must negate it to get the value from the current node's (parent's) perspective.
            q_value = -child_node.mean_value 
            
            prior = node.policy_priors[move]
            ucb_component = self.exploration_constant * prior * (math.sqrt(node.visits) / (1 + child_node.visits))
            
            puct_score = q_value + ucb_component
            
            if puct_score > best_score:
                best_score = puct_score
                best_child = child_node
                
        return best_child

    def _expand_and_evaluate(self, node: MCTSNode) -> float:
        """Expand a leaf node, create its children, and return the evaluated value."""
        if node.is_terminal():
            return node.state.get_winner_value() # e.g., 1.0 for win, -1.0 for loss

        # Get policy and value from the neural network
        # In the initial version, this is a single, blocking call.
        policy_logits, value = self.model.simple_infer(node.state.get_board_tensor())
        
        # Apply softmax and filter for legal moves
        legal_moves = node.state.get_legal_moves()
        # (Implementation detail: create a dictionary of move -> prior_probability)
        node.policy_priors = self._get_priors_for_legal_moves(policy_logits, legal_moves)

        # Create child nodes for all legal moves
        for move, prior in node.policy_priors.items():
            # IMPORTANT: The new state must be a deep copy.
            child_state = node.state.make_move(*move) 
            node.children[move] = MCTSNode(state=child_state, parent=node, move=move)
            
        # The value is from the perspective of the current player at 'node'.
        return value

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Update visit counts and values from a leaf node up to the root."""
        current_node = node
        while current_node is not None:
            current_node.visits += 1
            current_node.total_value += value
            # IMPORTANT: The value is from the perspective of the player at the child node.
            # For the parent, this outcome has the opposite value.
            value = -value 
            current_node = current_node.parent
```

## 3. Phased Implementation Plan

This plan prioritizes correctness before performance. Each phase should be completed and thoroughly tested before moving to the next.

### Phase 1: Core MCTS Correctness (The "Slow but Right" Version)
- **Goal**: Implement a functionally correct `NeuralMCTS` engine.
- **Tasks**:
    1.  Implement `MCTSNode` and `NeuralMCTS` as detailed above.
    2.  For network evaluation (`_expand_and_evaluate`), make a direct, blocking call to the model for each leaf node. Do not implement batching yet.
    3.  Do not implement a transposition table.
    4.  **Crucial Testing**:
        - **Backpropagation Test**: Write a unit test that manually creates a small tree, sets a leaf value, runs `_backpropagate`, and asserts that the parent's `total_value` is correctly negated and visits are incremented.
        - **State Integrity Test**: Write a test to ensure that when a node is expanded, modifying a child's game state does not affect its parent or siblings. This validates the use of deep copies.
        - **Integration Test**: Run a full search on a simple, known "win-in-1" position. Use a dummy network that returns fixed values. Assert that after a small number of simulations, the winning move has the highest visit count.
        - **Value Perspective Test**: Verify that values are correctly handled from the perspective of the player who just moved (child's perspective) vs parent's perspective.
    5.  **Implementation Notes**:
        - Use `copy.deepcopy()` for initial state copying to protect caller's state
        - Use `make_move()` directly for child state creation (it returns a new, independent state)
        - Add logging to track value propagation through the tree
        - Implement a simple move selection function that samples based on visit counts
        - Add basic statistics tracking (simulations per second, tree size, etc.)

### Phase 2: Transposition Table & Symmetries
- **Goal**: Avoid re-computing results for identical board positions.
- **Tasks**:
    1.  Implement **Zobrist Hashing** for your `HexGameState`. This provides a fast, incremental way to hash board states.
    2.  Implement a **canonicalization function** for board states. For Hex, this means checking the board's 180-degree rotation and always using the lexicographically smaller representation for hashing.
    3.  Add a dictionary (`transposition_table`) to the `MCTSEngine`. The key is the canonical Zobrist hash. The value is the `MCTSNode`.
    4.  Modify the search logic: before expanding a node, check if its canonical hash is already in the table. If so, reuse the existing node's statistics instead of re-evaluating.
    5.  **Implementation Notes**:
        - Zobrist hashing should be incremental: hash = hash ^ zobrist_table[position][piece]
        - For canonicalization, compare the board with its 180-degree rotation and use the lexicographically smaller one
        - Add transposition table hit rate statistics
        - Consider memory management: limit table size and implement LRU eviction
        - Test with positions that have obvious symmetries to verify canonicalization works

### Phase 3: Efficient Batching for GPU Utilization
- **Goal**: Maximize GPU throughput by batching network inferences.
- **Tasks - Option A (Single Game Batching)**:
    1.  Modify the main search loop. Instead of fully running one simulation at a time, collect leaf nodes that need evaluation.
    2.  When the number of collected nodes hits a batch size (e.g., 16 or 32), send the entire batch to the model.
    3.  Distribute the results and run the backpropagation step for each evaluated node.
- **Tasks - Option B (Multi-Game Orchestrator - Recommended)**:
    1.  Create an `Orchestrator` class that manages multiple, independent self-play games simultaneously (e.g., 16 games).
    2.  Each game runs its own MCTS search. When a search needs a network evaluation, it `yields` the leaf node to the orchestrator and pauses.
    3.  The orchestrator collects leaf nodes from all paused games.
    4.  When a full batch is ready, it's sent to the GPU.
    5.  Results are dispatched back to the correct game, which then resumes its MCTS search. This is the most effective way to keep the GPU busy.

### Phase 4: Future Work (Out of Scope for Initial Implementation)
- **Virtual Loss**: For parallel MCTS where multiple threads explore the *same* tree. Not needed for a single-threaded search or the multi-game orchestrator model.
- **KataGo-style Enhancements**: Adding new network heads (e.g., ownership map), changing the loss function, and using more complex PUCT variants. This is a significant "v2.0" project.
- **Advanced Time Management**: Dynamically adjusting simulation counts based on the game situation.

## 4. Integration with Existing Codebase

### Self-Play Integration
```python
class BatchedMCTSSelfPlayEngine(SelfPlayEngine):
    """Self-play engine using batched MCTS instead of minimax."""
    
    def __init__(self, *args, mcts_simulations: int = 800, **kwargs):
        super().__init__(*args, **kwargs)
        self.mcts_simulations = mcts_simulations
        self.mcts_engine = BatchedNeuralMCTS(
            model=self.model,
            exploration_constant=1.4
        )
    
    def _select_move(self, state: HexGameState) -> Tuple[int, int]:
        """Select move using MCTS instead of minimax."""
        root = self.mcts_engine.search(state, self.mcts_simulations)
        return self._select_move_from_visits(root)
    
    def _select_move_from_visits(self, root: BatchedMCTSNode) -> Tuple[int, int]:
        """Select move based on visit counts with temperature scaling."""
        moves = list(root.children.keys())
        visit_counts = [root.children[move].visits for move in moves]
        probabilities = self._temperature_scale(visit_counts, self.temperature)
        return moves[np.random.choice(len(moves), p=probabilities)]
```

### Tournament Integration
```python
class BatchedMCTSPlayer:
    """Player that uses batched MCTS for move selection."""
    
    def __init__(self, model_path: str, mcts_config: Dict[str, Any]):
        self.model = SimpleModelInference(model_path)
        self.mcts_engine = BatchedNeuralMCTS(
            model=self.model,
            exploration_constant=mcts_config.get('exploration_constant', 1.4)
        )
        self.num_simulations = mcts_config.get('num_simulations', 800)
    
    def select_move(self, state: HexGameState) -> Tuple[int, int]:
        """Select move using MCTS search."""
        root = self.mcts_engine.search(state, self.num_simulations)
        return self._select_best_move(root)
    
    def _select_best_move(self, root: BatchedMCTSNode) -> Tuple[int, int]:
        """Select the most visited move (deterministic)."""
        return max(root.children.items(), key=lambda x: x[1].visits)[0]
```

### Configuration
```python
# Example configuration for self-play
MCTS_CONFIG = {
    'algorithm': 'mcts',  # or 'minimax' for backward compatibility
    'num_simulations': 800,
    'exploration_constant': 1.4,
    'temperature': 1.0,
    'enable_transposition_table': False,  # Phase 2
    'enable_batching': False,  # Phase 3
}
```

## 5. Final Recommendations
- **Start with Phase 1.** Do not proceed until you are confident the core logic is bug-free. The tests are not optional; they are essential for managing the complexity.
- **Add optimizations one at a time.** After each new feature (transposition table, batching), re-run your tests and benchmark performance to ensure you've introduced a net benefit without breaking correctness.
- **Keep it single-threaded.** The multi-game orchestrator pattern (Phase 3B) provides all the benefits of parallelism for GPU utilization without the complexity of multi-threaded programming (locks, race conditions, etc.).
- **Maintain backward compatibility.** Keep the existing minimax implementation as a fallback option during development.
- **Add comprehensive logging.** Track performance metrics, tree statistics, and any anomalies for debugging.

## 6. Common Pitfalls and Debugging Tips

### Value Perspective Confusion
- **Problem**: Values getting confused between child and parent perspectives
- **Solution**: Always remember: child's value is from the perspective of the player who just moved
- **Debug**: Add logging to track value propagation: `log.debug(f"Node {move}: value={value}, visits={visits}")`

### State Mutation Bugs
- **Problem**: Modifying a child state affects parent/siblings
        - **Solution**: Use `copy.deepcopy()` for initial state copying, use `make_move()` for child states
- **Debug**: Write test that modifies child state and verifies parent remains unchanged

### PUCT Formula Errors
- **Problem**: Incorrect UCB component calculation
- **Solution**: Double-check the formula: `Q(s,a) + C * P(s,a) * sqrt(N(s)) / (1 + N(s,a))`
- **Debug**: Log PUCT scores for each child during selection

### Memory Leaks
- **Problem**: Tree nodes not being garbage collected
- **Solution**: Clear transposition table periodically, limit tree depth
- **Debug**: Monitor memory usage with `psutil.Process().memory_info()`

### Performance Issues
- **Problem**: Too many individual neural network calls
- **Solution**: Implement batching as soon as basic correctness is verified
- **Debug**: Track simulations per second and GPU utilization

### Testing Strategy
- **Unit Tests**: Test each component in isolation
- **Integration Tests**: Test full search on simple positions
- **Performance Tests**: Benchmark against existing minimax implementation
- **Stress Tests**: Run many games to catch edge cases
