"""
Batched Monte Carlo Tree Search (MCTS) implementation for Hex AI.

This module provides a neural network-guided MCTS implementation that uses batched
inference to significantly improve performance. The design uses a state machine
approach where nodes can be in different states (UNEXPANDED, EVALUATION_PENDING, EXPANDED)
to manage the batching of neural network calls.

Key Features:
- State machine-based node management
- Batched neural network inference
- Virtual loss for pending evaluations
- Efficient caching and reuse
- Clean separation of concerns
"""

# TODO: PERFORMANCE CRITICAL - Replace deepcopy with apply/undo pattern
# Current make_move() creates new HexGameState objects which is expensive for MCTS
# IMPLEMENTATION PLAN (Phase 3.1):
# 1) Add apply_move() method to HexGameState that mutates in place
# 2) Add undo_last() method that restores previous state
# 3) Update MCTS to use apply → encode → undo pattern for evaluations
# 4) Use fast_copy() only when child nodes need to be materialized
# Expected gain: 10-20x speedup in expansion phase

# TODO: PERFORMANCE - Vectorize child statistics and UCT calculations
# Current _select_child_with_puct() uses Python loops over dicts
# IMPLEMENTATION PLAN (Phase 3.2):
# 1) Store child stats in NumPy arrays: N, W, P, Q sized for max legal moves
# 2) Align arrays with legal_moves for direct indexing
# 3) Vectorize UCT: compute all U values at once, use np.argmax()
# 4) Cache sqrt(total_visits) per call, avoid repeated math operations
# Expected gain: 2-5x speedup in selection phase

# TODO: PERFORMANCE - Optimize batch utilization and tensor allocation
# Current batching may underfill batches, creating new tensors per evaluation
# IMPLEMENTATION PLAN (Phase 3.3):
# 1) Tune batch collection: adjust max_wait_ms (1-5ms), pre-seed rollouts
# 2) Pre-allocate input tensor pool, write into views when stacking
# 3) Batch CPU→GPU transfers, avoid per-state .cpu() calls
# 4) Monitor avg_batch_size vs target, aim for >80% utilization
# Expected gain: 1.5-3x speedup in inference phase

# TODO: PERFORMANCE - Optimize backpropagation with local variables
# Current _backpropagate() uses repeated attribute lookups
# IMPLEMENTATION PLAN (Phase 3.2):
# 1) Inline the backpropagation loop
# 2) Use local variables instead of repeated node.visits += 1
# 3) Cache frequently accessed values (depth, discount_factor)
# Expected gain: 1.2-2x speedup in backpropagation

# TODO: STRUCTURAL - Create unified BatchedEvaluator interface
# Current branching between batched and single evaluation creates complexity
# IMPLEMENTATION PLAN (Phase 1.3):
# 1) Create BatchedEvaluator class in hex_ai/inference/batched_evaluator.py
# 2) Centralize all NN calls with consistent timing and batching
# 3) Pre-allocate input tensors, background thread for queue management
# 4) Single choke point for all evaluations with performance monitoring
# This enables easier profiling and optimization

# TODO: INSTRUMENTATION - Add performance profiling using PERF utility
# IMPLEMENTATION PLAN (Phase 1.2):
# 1) Import from hex_ai.utils.perf import PERF
# 2) Add with PERF.timer("mcts.search"): around search method
# 3) Add with PERF.timer("mcts.select"): around selection phase
# 4) Add with PERF.timer("mcts.expand"): around expansion phase
# 5) Add with PERF.timer("nn.infer"): around batch processing
# 6) Add PERF.inc("mcts.sim") per simulation, PERF.inc("nn.batch") per batch
# 7) Call PERF.log_snapshot(clear=True) at end of each move
# This will provide detailed performance breakdown for optimization targeting

import copy
import math
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np

from hex_ai.inference.game_engine import HexGameState
from hex_ai.inference.simple_model_inference import SimpleModelInference
from hex_ai.inference.batch_processor import BatchProcessor
from hex_ai.inference.batched_evaluator import BatchedEvaluator
from hex_ai.value_utils import policy_logits_to_probs, player_to_winner
from hex_ai.config import BOARD_SIZE
from hex_ai.enums import Player
from hex_ai.utils.perf import PERF

logger = logging.getLogger(__name__)


class NodeState(Enum):
    """States that a node can be in during MCTS search."""
    UNEXPANDED = "unexpanded"  # Node has not been evaluated by neural network
    EVALUATION_PENDING = "evaluation_pending"  # Node is waiting for neural network result
    EXPANDED = "expanded"  # Node has been evaluated and children created


@dataclass
class BatchedMCTSNode:
    """Represents a node in the batched MCTS search tree."""
    # Core state
    state: HexGameState  # The game state this node represents
    parent: Optional['BatchedMCTSNode'] = None
    move: Optional[Tuple[int, int]] = None  # The move that led from parent to this node
    
    # Node state management
    node_state: NodeState = NodeState.UNEXPANDED
    
    # Search statistics
    visits: int = 0
    total_value: float = 0.0  # Sum of all evaluations from this node's subtree
    
    # Neural network priors (cached from the first time this node is expanded)
    policy_priors: Optional[Dict[Tuple[int, int], float]] = None
    
    # Children management
    children: Dict[Tuple[int, int], 'BatchedMCTSNode'] = field(default_factory=dict)
    
    # Depth tracking for move count penalty
    depth: int = 0
    
    # Virtual loss for pending evaluations
    virtual_loss: int = 0

    @property
    def mean_value(self) -> float:
        """The mean value (Q-value) of this node."""
        if self.visits == 0:
            return 0.0
        # The value is from the perspective of the player *who just moved* to reach this state.
        return self.total_value / self.visits
    
    def is_leaf(self) -> bool:
        """A node is a leaf if it has not been expanded yet."""
        return self.node_state != NodeState.EXPANDED

    def is_terminal(self) -> bool:
        """A node is terminal if the game is over."""
        return self.state.game_over
    
    def update_statistics(self, value: float) -> None:
        """Update node statistics after backpropagation."""
        self.visits += 1
        self.total_value += value
    
    def get_depth(self) -> int:
        """Calculate the depth of this node from the root."""
        if self.parent is None:
            return 0
        return self.parent.get_depth() + 1
    
    def detach_parent(self):
        """Detach this node from its parent, making it a new root."""
        self.parent = None
    
    def add_virtual_loss(self, amount: int = 1):
        """Add virtual loss to discourage exploration while evaluation is pending."""
        # Add bounds checking to prevent unbounded accumulation
        self.virtual_loss = min(self.virtual_loss + amount, 1000)  # Reasonable upper bound
    
    def remove_virtual_loss(self, amount: int = 1):
        """Remove virtual loss after evaluation is complete."""
        if amount == self.virtual_loss:
            # Remove all virtual loss (common case after evaluation)
            self.virtual_loss = 0
        else:
            # Remove specified amount (for partial removal if needed)
            self.virtual_loss = max(0, self.virtual_loss - amount)


class MCTSNodeManager:
    """
    Manages MCTS nodes and their lifecycle.
    
    This class handles node creation, state management, and cleanup,
    providing a clean interface for the MCTS engine. It centralizes
    node registry management and provides a consistent API for
    node operations.
    
    Key Features:
    - Node registry management for callback tracking
    - Consistent node creation with proper parent relationships
    - Automatic cleanup of unused nodes
    - Thread-safe node ID generation
    
    This separation of concerns makes the MCTS engine more modular
    and easier to test and maintain.
    """
    
    def __init__(self, verbose: int = 0):
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        
        # Node registry for callbacks (cleaner than lambda closures)
        self._node_registry: Dict[int, BatchedMCTSNode] = {}
        self._next_node_id = 0
    
    def _get_node_id(self, node: BatchedMCTSNode) -> int:
        """Get or create a unique ID for a node."""
        # Use id() of the node object as the unique identifier
        node_id = id(node)
        if node_id not in self._node_registry:
            self._node_registry[node_id] = node
        return node_id

    def _get_node_by_id(self, node_id: int) -> Optional[BatchedMCTSNode]:
        """Get a node by its ID, or None if not found."""
        return self._node_registry.get(node_id)

    def create_root_node(self, state: HexGameState) -> BatchedMCTSNode:
        """Create a new root node from a game state."""
        return BatchedMCTSNode(state=state.fast_copy())
    
    def create_child_node(self, parent: BatchedMCTSNode, move: Tuple[int, int], 
                         state: HexGameState) -> BatchedMCTSNode:
        """Create a child node with proper parent relationship."""
        child_depth = parent.get_depth() + 1
        return BatchedMCTSNode(
            state=state,
            parent=parent,
            move=move,
            depth=child_depth
        )
    
    def cleanup_node(self, node_id: int) -> None:
        """Clean up a node from the registry."""
        self._node_registry.pop(node_id, None)
    
    def cleanup_all(self) -> None:
        """Clean up all nodes in the registry."""
        self._node_registry.clear()


class BatchedNeuralMCTS:
    """Batched MCTS engine guided by a neural network."""
    
    def __init__(self, model: SimpleModelInference, exploration_constant: float = 1.4, 
                 win_value: float = 1.5, discount_factor: float = 0.98, 
                 optimal_batch_size: int = 64, verbose: int = 0, selection_wait_ms: int = 500):
        """
        Initialize the batched MCTS engine.
        
        Args:
            model: Neural network model for policy and value predictions
            exploration_constant: PUCT exploration constant (default: 1.4)
            win_value: Value assigned to winning terminal states (default: 1.5)
            discount_factor: Discount factor for move count penalty (default: 0.98)
            optimal_batch_size: Optimal batch size for neural network inference
            verbose: Verbosity level (0=quiet, 1=basic, 2=detailed, 3=debug, 4=extreme debug)
        """
        self.model = model
        self.exploration_constant = exploration_constant
        self.win_value = win_value
        self.discount_factor = discount_factor
        self.optimal_batch_size = optimal_batch_size
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        # Small default wait helps ensure at least one callback/backprop completes before selection
        self.selection_wait_ms = selection_wait_ms
        
        # Initialize batched evaluator (replaces direct batch processor usage)
        self.evaluator = BatchedEvaluator(
            model=model,
            optimal_batch_size=optimal_batch_size,
            verbose=verbose,
            max_wait_ms=3,
            enable_background_processing=True
        )
        
        # Initialize node manager for better separation of concerns
        self.node_manager = MCTSNodeManager(verbose=verbose)
        
        # Statistics tracking
        self.stats = {
            'total_simulations': 0,
            'total_inferences': 0,
            'total_batches_processed': 0,
            'start_time': None,
            'end_time': None
        }
        
        self.logger.info(
            f"Initialized BatchedNeuralMCTS with exploration_constant={exploration_constant}, "
            f"win_value={win_value}, discount_factor={discount_factor}, "
            f"optimal_batch_size={optimal_batch_size}, selection_wait_ms={self.selection_wait_ms}"
        )

    def _get_node_id(self, node: BatchedMCTSNode) -> int:
        """Get or create a unique ID for a node."""
        return self.node_manager._get_node_id(node)

    def _get_node_by_id(self, node_id: int) -> Optional[BatchedMCTSNode]:
        """Get a node by its ID, or None if not found."""
        return self.node_manager._get_node_by_id(node_id)

    def _cleanup_node_registry(self):
        """Clean up nodes that are no longer referenced."""
        # This is a simple cleanup - in practice, you might want more sophisticated
        # reference counting or garbage collection
        pass

    def search(self, root_state_or_node, num_simulations: int) -> BatchedMCTSNode:
        """
        Run MCTS search from a root state or existing node to build up statistics.
        
        Args:
            root_state_or_node: Either a HexGameState (for new search) or BatchedMCTSNode (for continuing search)
            num_simulations: Number of MCTS simulations to run
            
        Returns:
            Root node with populated search tree
        """
        with PERF.timer("mcts.search"):
            if self.verbose >= 1:
                self.logger.info(f"Starting batched MCTS search with {num_simulations} simulations")
            self.stats['start_time'] = time.time()
            self.stats['total_simulations'] = 0
            
            # Handle both HexGameState and BatchedMCTSNode inputs
            if isinstance(root_state_or_node, HexGameState):
                # Create new root node with fast copy of state
                root = self.node_manager.create_root_node(root_state_or_node)
                if self.verbose >= 2:
                    self.logger.debug(f"Created new root node from HexGameState")
            elif isinstance(root_state_or_node, BatchedMCTSNode):
                # Continue search from existing node
                root = root_state_or_node
                if self.verbose >= 2:
                    self.logger.debug(f"Continuing search from existing BatchedMCTSNode")
            else:
                raise ValueError(f"Expected HexGameState or BatchedMCTSNode, got {type(root_state_or_node)}")
            
            # Track root for detailed logging
            self._last_root = root
            
            if self.verbose >= 2:
                self.logger.debug(f"Root node state: {root.node_state}, terminal: {root.is_terminal()}, leaf: {root.is_leaf()}")
            
            # Request up to num_simulations evaluations, respecting async pending states
            evaluations_requested = 0
            pending_encounters = 0
            while evaluations_requested < num_simulations:
                if self.verbose >= 3:
                    self.logger.debug(f"Starting selection for request {evaluations_requested + 1}")

                # Execute selection phase to get leaf node
                with PERF.timer("mcts.select"):
                    leaf_node = self._execute_selection_phase(root)

                # If evaluation is already pending for this leaf, wait briefly for callback
                if getattr(leaf_node, 'node_state', None) == NodeState.EVALUATION_PENDING:
                    pending_encounters += 1
                    # Respect selection_wait_ms knob: when zero, skip this wait entirely (useful for tests)
                    max_wait_s = min(0.5, max(0.0, getattr(self, 'selection_wait_ms', 0) / 1000.0))
                    if max_wait_s > 0.0:
                        with PERF.timer("mcts.pending_wait"):
                            wait_start = time.time()
                            waited_any = False
                            while (
                                getattr(leaf_node, 'node_state', None) == NodeState.EVALUATION_PENDING
                                and (time.time() - wait_start) < max_wait_s
                            ):
                                waited_any = True
                                time.sleep(0.001)
                            if waited_any:
                                PERF.inc("mcts.pending_wait_events")
                        # After waiting, try again without consuming budget
                        continue
                    # No waiting configured; if we keep encountering pending without progress, break to avoid long hangs in tests
                    if pending_encounters >= 1:
                        self.logger.debug("Pending leaf encountered with selection_wait_ms=0; breaking out of search loop early for test speed.")
                        break
                    continue

                # Expand and queue evaluation for unexpanded leaf
                with PERF.timer("mcts.expand"):
                    self._expand_and_evaluate(leaf_node)

                evaluations_requested += 1
                PERF.inc("mcts.sim")  # Treat each queued evaluation as one simulation budget unit
                self.stats['total_simulations'] = evaluations_requested

                # Log progress every 50 requests
                if self.verbose >= 1 and (evaluations_requested % 50 == 0 or evaluations_requested == num_simulations):
                    elapsed = max(time.time() - self.stats['start_time'], 1e-6)
                    sims_per_sec = evaluations_requested / elapsed
                    queue_size = self.evaluator.get_queue_size()
                    cache_size = self.evaluator.get_cache_size()
                    self.logger.info(
                        f"Completed {evaluations_requested}/{num_simulations} eval requests "
                        f"({sims_per_sec:.0f} req/sec, queue: {queue_size}, cache: {cache_size})"
                    )
                if self.verbose >= 3 and (evaluations_requested % 50 == 0):
                    self._log_detailed_progress(evaluations_requested, num_simulations)
        
        # Wait until at least one evaluation callback has been applied to the root
        # This ensures priors/children exist and at least some visits can accrue.
        # Skip this wait entirely when selection_wait_ms == 0 (e.g., unit tests).
        if self.selection_wait_ms > 0:
            with PERF.timer("mcts.ensure_applied_wait"):
                ensure_applied_deadline = time.time() + 0.25  # up to 250ms
                waited_any = False
                while time.time() < ensure_applied_deadline:
                    if root.policy_priors and root.children:
                        break
                    waited_any = True
                    time.sleep(0.001)
                if waited_any:
                    PERF.inc("mcts.ensure_applied_wait_events")
        
        self.stats['end_time'] = time.time()
        total_time = max(self.stats['end_time'] - self.stats['start_time'], 1e-6)
        sims_per_sec = num_simulations / total_time
        
        # Get evaluator statistics
        evaluator_stats = self.evaluator.get_statistics()
        self.stats['total_inferences'] = evaluator_stats.get('batch_processor', {}).get('total_inferences', 0)
        
        if self.verbose >= 1:
            self.logger.info(f"Batched MCTS search completed: {num_simulations} simulations in {total_time:.3f}s ({sims_per_sec:.0f} sims/sec)")
            self.logger.info(f"Total neural network inferences: {self.stats['total_inferences']}")
            self.logger.info(f"Total batches processed: {self.stats['total_batches_processed']}")
            self.logger.info(f"Cache hit rate: {evaluator_stats.get('cache_hit_rate', 0):.1%}")
            self.logger.info(f"Average batch size: {evaluator_stats.get('average_batch_size', 0):.1f}")
        
        # More detailed summary for higher verbosity
        if self.verbose >= 2:
            self.logger.info(f"Performance summary: {evaluator_stats.get('inferences_per_second', 0):.1f} inferences/sec, "
                           f"{evaluator_stats.get('average_batch_size', 0):.1f} avg batch size")
        
        # Optional readiness wait to reduce zero-visit path at selection time
        # When selection_wait_ms == 0 this returns immediately.
        with PERF.timer("mcts.selection_ready_wait"):
            self._wait_for_root_ready(root, max_wait_ms=self.selection_wait_ms)
        # Enforce at least one non-zero child visit if configured to wait
        if self.selection_wait_ms > 0 and root.children:
            deadline = time.time() + (self.selection_wait_ms / 1000.0)
            while time.time() < deadline:
                if any(child.visits > 0 for child in root.children.values()):
                    break
                time.sleep(0.001)

        # Log performance snapshot at end of search
        PERF.log_snapshot(clear=True, force=True)
        
        return root

    def _execute_selection_phase(self, root: BatchedMCTSNode) -> BatchedMCTSNode:
        """
        Execute the selection phase of MCTS: traverse the tree from root to a leaf node.
        
        This method only performs selection and returns the selected leaf node.
        It does not perform backpropagation - that is handled by the evaluation callback.
        
        Args:
            root: Root node of the search tree
            
        Returns:
            Selected leaf node
        """
        # Selection: Traverse the tree using PUCT until a leaf node is found.
        leaf_node = self._select(root)
        return leaf_node

    def _wait_for_root_ready(self, root: BatchedMCTSNode, max_wait_ms: int = 0) -> None:
        """Spin briefly until root has priors and children or timeout."""
        if max_wait_ms <= 0:
            return
        deadline = time.time() + (max_wait_ms / 1000.0)
        while time.time() < deadline:
            if root.policy_priors and root.children:
                return
            time.sleep(0.001)

    def _run_simulation(self, root: BatchedMCTSNode) -> None:
        """
        DEPRECATED: This method is replaced by the new orchestration in search().
        
        The old implementation had a critical bug where placeholder values were
        backpropagated immediately, and true network values were never propagated
        to ancestors. This method is kept for backward compatibility but should
        not be used.
        
        Use the new approach where search() orchestrates the simulation loop:
        1. _execute_selection_phase() handles selection
        2. _expand_and_evaluate() handles expansion and queues evaluation
        3. Backpropagation is handled by the evaluation callback
        """
        # 1. Selection: Traverse the tree using PUCT until a leaf node is found.
        leaf_node = self._select(root)
        
        # 2. Expansion & Evaluation: If the game is not over, expand the leaf and get its value from the NN.
        # Note: This now returns None and handles backpropagation in the callback
        self._expand_and_evaluate(leaf_node)
        
        # 3. Backpropagation: This is now handled by the evaluation callback
        # No manual backpropagation needed here

    def _select(self, node: BatchedMCTSNode) -> BatchedMCTSNode:
        """Traverse the tree from the root to a leaf node."""
        # TODO (P3): Add a crisp explanation: I think it's to descend the tree until we find a leaf node.
        while not node.is_leaf():
            if self.verbose >= 3:
                self.logger.debug(f"Selecting child from node with {len(node.children)} children")
            node = self._select_child_with_puct(node)
        
        if self.verbose >= 2:
            self.logger.debug(f"Selected leaf node: state={node.node_state}, terminal={node.is_terminal()}, depth={node.get_depth()}")
        
        return node

    def _select_child_with_puct(self, node: BatchedMCTSNode) -> BatchedMCTSNode:
        """
        Select the child with the highest PUCT score.
        
        PUCT formula: Q(s,a) + C * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        """
        if not node.children:
            raise ValueError("Cannot select child from node with no children")
        
        best_score = -float('inf')
        best_child = None

        for move, child_node in list(node.children.items()):
            # Note: child_node.mean_value is from the perspective of the player who made the move.
            # We must negate it to get the value from the current node's (parent's) perspective.
            q_value = -child_node.mean_value 
            
            prior = node.policy_priors[move]
            
            # Apply virtual loss penalty for pending evaluations
            effective_visits = child_node.visits + child_node.virtual_loss
            ucb_component = self.exploration_constant * prior * (math.sqrt(node.visits) / (1 + effective_visits))
            
            puct_score = q_value + ucb_component
            
            if self.verbose >= 4:
                self.logger.debug(
                    f"PUCT scores - Move {move}: Q={q_value:.4f}, prior={prior:.4f}, "
                    f"UCB={ucb_component:.4f}, virtual_loss={child_node.virtual_loss}, total={puct_score:.4f}"
                )
            
            if puct_score > best_score:
                best_score = puct_score
                best_child = child_node
                
        if best_child is None:
            raise ValueError("No child selected - this should not happen")
            
        return best_child

    # TODO (P3): Add brief explanation of the design, e.g. callback deferring evaluation.
    def _expand_and_evaluate(self, node: BatchedMCTSNode) -> None:
        """
        Expand a leaf node and queue it for evaluation.
        
        This method handles the expansion and evaluation phases of MCTS.
        Backpropagation is handled by the evaluation callback when the
        neural network result arrives.
        
        Args:
            node: Leaf node to expand
        """
        if node.is_terminal():
            # For terminal nodes, we need to backpropagate immediately
            value = self._terminal_value(node.state)
            self._backpropagate(node, value)
            return

        # Handle different node states
        if node.node_state == NodeState.UNEXPANDED:
            if self.verbose >= 2:
                self.logger.debug(f"Expanding unexpanded node at depth {node.get_depth()}")
            
            # Request evaluation from batch processor
            node.node_state = NodeState.EVALUATION_PENDING
            node.add_virtual_loss()
            
            # Request evaluation using the batched evaluator
            # Capture values at request time to avoid stale references
            simulation_id = self.stats['total_simulations']
            node_depth = node.get_depth()
            node_id = self._get_node_id(node)
            self.evaluator.request_evaluation(
                state=node.state,
                callback=lambda policy_logits, value: self._handle_evaluation_result(node_id, policy_logits, value),
                metadata={'node_depth': node_depth, 'simulation_id': simulation_id}
            )
            
            # The evaluation request is now handled by the batched evaluator above
            
        elif node.node_state == NodeState.EVALUATION_PENDING:
            # Node is already being evaluated, add virtual loss
            node.add_virtual_loss()
            
        else:  # NodeState.EXPANDED
            # Node is already expanded, this shouldn't happen in normal flow
            raise ValueError("Attempted to expand already expanded node")

    def _handle_evaluation_result(self, node_id: int, policy_logits: np.ndarray, value: float) -> None:
        """
        Handle evaluation results and complete node expansion.
        
        This method is called by the batched evaluator when neural network
        evaluation completes. It processes the results and creates child nodes.
        
        Args:
            node_id: The ID of the node that was evaluated
            policy_logits: Raw policy logits from neural network
            value: Value prediction from neural network
        """
        node = self._get_node_by_id(node_id)
        if node is None:
            self.logger.warning(f"Received evaluation result for unknown node with ID {node_id}")
            return

        if self.verbose >= 2:
            self.logger.debug(f"Evaluation callback received for node at depth {node.get_depth()}, value={value:.4f}")
        
        # Apply softmax and filter for legal moves
        legal_moves = node.state.get_legal_moves()
        if self.verbose >= 2:
            self.logger.debug(f"Legal moves: {len(legal_moves)}")
        
        node.policy_priors = self._get_priors_for_legal_moves(policy_logits, legal_moves)
        
        if self.verbose >= 2:
            self.logger.debug(f"Policy priors: {len(node.policy_priors)} moves")
        
        # Create child nodes for all legal moves using apply/undo pattern
        for move, prior in node.policy_priors.items():
            try:
                # Apply move to current state
                node.state.apply_move(*move)
                
                # Create child node with the modified state
                node.children[move] = self.node_manager.create_child_node(
                    parent=node,
                    move=move,
                    state=node.state.fast_copy()
                )
            finally:
                # Always restore state, even if exception occurs
                node.state.undo_last()
        
        if self.verbose >= 2:
            self.logger.debug(f"Created {len(node.children)} child nodes")
        
        # Mark node as expanded
        node.node_state = NodeState.EXPANDED
        # Remove all virtual loss that was accumulated during evaluation
        node.remove_virtual_loss(node.virtual_loss)
        
        # CRITICAL FIX: Backpropagate the true network value up the tree
        # This ensures that all ancestors get the correct value, not the placeholder
        self._backpropagate(node, value)
        
        if self.verbose >= 2:
            self.logger.debug(f"Expanded node with {len(node.children)} children, value={value:.4f}, visits={node.visits}")
        
        # Clean up the node from registry after processing
        self.node_manager.cleanup_node(node_id)

    def _backpropagate(self, node: BatchedMCTSNode, value: float) -> None:
        """
        Update visit counts and values from a leaf node up to the root.
        
        Args:
            node: Leaf node to start backpropagation from
            value: Value to propagate up the tree
        """
        with PERF.timer("mcts.backprop"):
            current_node = node
            # TODO (P3): Explain that 'None' indictes the root node.
            while current_node is not None:
                # Apply discount factor based on depth (move count penalty)
                depth = current_node.get_depth()
                discounted_value = value * (self.discount_factor ** depth)
                
                current_node.update_statistics(discounted_value)
                
                # IMPORTANT: The value is from the perspective of the player at the current node.
                # For the parent, this outcome has the opposite value.
                # We negate the value BEFORE moving to the parent.
                current_node = current_node.parent
                if current_node is not None:
                    value = -value

    def _terminal_value(self, state: HexGameState) -> float:
        """
        Get the value of a terminal state.
        
        Args:
            state: Terminal game state
            
        Returns:
            Value from the perspective of the player who just moved (win_value for win, -win_value for loss, 0.0 for draw)
        """
        if not state.game_over:
            raise ValueError("Cannot get terminal value for non-terminal state")
        
        # Prefer enum-based winner; fail fast if missing
        winner_enum = state.winner_enum
        if winner_enum is None:
            # Raise an exception and crash. No fallback logic! Find errors fast.
            raise ValueError("CRITICAL: Game ended without a clear winner! This indicates a bug.")
        
        # The player who just moved is the OPPOSITE of current_player (use Enums internally)
        current_player_enum = state.current_player_enum
        just_moved_player = Player.RED if current_player_enum == Player.BLUE else Player.BLUE
        
        # Determine if the player who just moved won using Enums exclusively
        just_moved_won = (player_to_winner(just_moved_player) == winner_enum)
        
        # Return value from perspective of player who just moved
        if just_moved_won:
            return self.win_value  # Player who just moved won
        else:
            return -self.win_value  # Player who just moved lost

    def _get_priors_for_legal_moves(self, policy_logits: np.ndarray, legal_moves: List[Tuple[int, int]]) -> Dict[Tuple[int, int], float]:
        """
        Extract prior probabilities for legal moves from policy logits.
        
        Args:
            policy_logits: Raw policy logits from neural network (1D array, flattened from 2D board)
            legal_moves: List of legal moves
            
        Returns:
            Dictionary mapping moves to prior probabilities
        """
        # Ensure policy_logits is 1D
        if policy_logits.ndim != 1:
            raise ValueError(f"Expected 1D policy_logits, got shape {policy_logits.shape}")
        
        # Convert 1D logits to probabilities using temperature scaling
        # TODO (P1): Don't pass temperature=1.0, this should be a configurable parameter.
        policy_probs = policy_logits_to_probs(policy_logits, temperature=1.0)
        
        # Extract probabilities for legal moves
        move_priors = {}
        total_prior = 0.0
        
        for move in legal_moves:
            row, col = move
            # Convert 2D coordinates to 1D index
            move_index = row * BOARD_SIZE + col
            prior = policy_probs[move_index]
            move_priors[move] = prior
            total_prior += prior
        
        # Normalize to ensure probabilities sum to 1
        if total_prior > 0:
            for move in move_priors:
                move_priors[move] /= total_prior
        else:
            # If all priors are zero, this indicates a serious problem
            error_msg = f"CRITICAL: All policy priors are zero! This indicates a bug."
            error_msg += f"\nPolicy logits min/max: {policy_logits.min():.6f}/{policy_logits.max():.6f}"
            error_msg += f"\nPolicy probs min/max: {policy_probs.min():.6f}/{policy_probs.max():.6f}"
            error_msg += f"\nLegal moves count: {len(legal_moves)}"
            error_msg += f"\nFirst 10 legal moves: {legal_moves[:10]}"
            
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        return move_priors

    def select_move(self, root: BatchedMCTSNode, temperature: float = 1.0) -> Tuple[int, int]:
        """
        Select final move based on visit counts. If all child visits are zero
        (plausible with async batching), fall back to policy priors.
        """
        if not root.children:
            # Optional short readiness wait
            self._wait_for_root_ready(root, max_wait_ms=getattr(self, "selection_wait_ms", 0))
            if not root.children:
                raise ValueError("Cannot select move from root with no children")

        moves = list(root.children.keys())
        visit_counts = [root.children[m].visits for m in moves]

        total_visits = sum(visit_counts)
        if total_visits == 0:
            # Expected race: no traversal reached children yet.
            if not root.policy_priors:
                raise ValueError("Root has children but no policy priors; unexpected in async MCTS.")
            priors = np.array([root.policy_priors.get(m, 0.0) for m in moves], dtype=np.float64)
            s = float(priors.sum())
            if s <= 0.0:
                raise ValueError("CRITICAL: All policy priors are zero! This indicates a bug.")
            if temperature == 0:
                idx = int(np.argmax(priors))
                selected_move = moves[idx]
                self.logger.info(f"Selected {selected_move} (deterministic, priors-only)")
                return selected_move
            scaled = np.power(priors, 1.0 / float(temperature))
            probs = scaled / scaled.sum()
            idx = int(np.random.choice(len(moves), p=probs))
            selected_move = moves[idx]
            self.logger.info(f"Selected {selected_move} (stochastic, priors-only, temp={temperature})")
            return selected_move

        # Normal path: some visits exist
        if temperature == 0:
            idx = int(np.argmax(visit_counts))
            selected_move = moves[idx]
            self.logger.info(f"Selected {selected_move} (deterministic, visits={visit_counts[idx]})")
            return selected_move

        probabilities = self._temperature_scale(visit_counts, temperature)
        idx = int(np.random.choice(len(moves), p=probabilities))
        selected_move = moves[idx]
        self.logger.info(f"Selected {selected_move} (stochastic, visits={visit_counts[idx]}, temp={temperature})")
        return selected_move

    # TODO (P2): For self-play game generation, we should also add Dirichlet noise to the initial policy priors as a secondary form of randomness.
    # TODO (P2): Somewhat reduce temperature as the game progresses. Explore more at the start, less at the end.
    def _temperature_scale(self, visit_counts: List[int], temperature: float) -> List[float]:
        """
        Apply temperature scaling to visit counts to get move probabilities.
        
        Args:
            visit_counts: List of visit counts for each move
            temperature: Temperature parameter (0 = deterministic, 1 = stochastic)
            
        Returns:
            List of probabilities for each move
        """
        if temperature == 0:
            # Deterministic: all probability on most visited
            max_visits = max(visit_counts)
            return [1.0 if count == max_visits else 0.0 for count in visit_counts]
        
        # Apply temperature scaling: p_i = (visits_i)^(1/temperature)
        # TODO (P3): Get further explanation of the logic. Why are we iterating over visit_counts?
        scaled_visits = [count ** (1.0 / temperature) for count in visit_counts]
        total = sum(scaled_visits)
        
        if total == 0:
            # Raise an exception and crash. No fallback logic! Find errors fast.
            raise ValueError(f"CRITICAL: All visit counts are zero! This indicates a bug.")
            # # Fallback to uniform distribution
            # return [1.0 / len(visit_counts)] * len(visit_counts)
        
        return [v / total for v in scaled_visits]

    def get_search_statistics(self) -> Dict[str, Any]:
        """Get statistics about the last search."""
        stats = self.stats.copy()
        if stats['start_time'] and stats['end_time']:
            stats['total_time'] = stats['end_time'] - stats['start_time']
            if stats['total_simulations'] > 0:
                stats['simulations_per_second'] = stats['total_simulations'] / stats['total_time']
        
        # Add evaluator statistics
        evaluator_stats = self.evaluator.get_statistics()
        stats.update(evaluator_stats)
        
        # Add thread information
        if hasattr(self.evaluator.batch_processor, 'get_thread_info'):
            stats['background_thread'] = self.evaluator.batch_processor.get_thread_info()
        
        # Add GPU memory statistics
        from hex_ai.utils.gpu_monitoring import get_gpu_memory_info
        gpu_mem = get_gpu_memory_info()
        if gpu_mem:
            stats['gpu_memory'] = gpu_mem
        
        # Add tree statistics if available
        if hasattr(self, '_last_root') and self._last_root:
            stats['tree_statistics'] = self._get_tree_statistics(self._last_root)
        
        return stats
    
    def _get_tree_statistics(self, root: BatchedMCTSNode) -> Dict[str, Any]:
        """Get detailed statistics about the search tree."""
        if not root.children:
            return {'total_nodes': 0, 'unexpanded': 0, 'pending': 0, 'expanded': 0}
        
        # Count nodes by state
        unexpanded = 0
        pending = 0
        expanded = 0
        total_visits = 0
        
        def count_nodes(node):
            nonlocal unexpanded, pending, expanded, total_visits
            if node.node_state == NodeState.UNEXPANDED:
                unexpanded += 1
            elif node.node_state == NodeState.EVALUATION_PENDING:
                pending += 1
            elif node.node_state == NodeState.EXPANDED:
                expanded += 1
            
            total_visits += node.visits
            
            for child in node.children.values():
                count_nodes(child)
        
        count_nodes(root)
        
        return {
            'total_nodes': unexpanded + pending + expanded,
            'unexpanded': unexpanded,
            'pending': pending,
            'expanded': expanded,
            'total_visits': total_visits,
            'root_visits': root.visits
        }
    
    def reset_search_statistics(self):
        """Reset search statistics for the next search."""
        self.stats['total_simulations'] = 0
        self.stats['total_inferences'] = 0
        self.stats['total_batches_processed'] = 0
        self.stats['start_time'] = None
        self.stats['end_time'] = None
        self.evaluator.reset_statistics()
    
    def cleanup(self):
        """Clean up resources and stop background threads."""
        self.evaluator.cleanup()
        if self.verbose >= 1:
            self.logger.info("MCTS cleanup completed")
    
    def __del__(self):
        """Cleanup when the object is destroyed."""
        self.cleanup()
    
    def get_principal_variation(self, root: BatchedMCTSNode, max_depth: int = 10) -> List[Tuple[int, int]]:
        """
        Extract the principal variation (most likely move sequence) from the MCTS tree.
        
        Args:
            root: Root node of the search tree
            max_depth: Maximum depth to explore
            
        Returns:
            List of moves representing the principal variation
        """
        if not root.children:
            return []
        
        pv = []
        current_node = root
        depth = 0
        
        while current_node.children and depth < max_depth:
            # Find the child with the highest visit count (most likely move)
            best_move = max(current_node.children.keys(), 
                          key=lambda move: current_node.children[move].visits)
            
            pv.append(best_move)
            current_node = current_node.children[best_move]
            depth += 1
            
            # Stop if we reach a terminal state
            if current_node.state.game_over:
                break
        
        return pv
    
    def get_move_sequence_analysis(self, root: BatchedMCTSNode, max_depth: int = 5) -> Dict[str, Any]:
        """
        Get detailed analysis of the most likely move sequences.
        
        Args:
            root: Root node of the search tree
            max_depth: Maximum depth to analyze
            
        Returns:
            Dictionary with move sequence analysis
        """
        if not root.children:
            return {"principal_variation": [], "alternative_lines": []}
        
        # Get principal variation
        pv = self.get_principal_variation(root, max_depth)
        
        # Get alternative lines (second best moves at each level)
        alternative_lines = []
        current_node = root
        depth = 0
        
        while current_node.children and depth < max_depth:
            # Sort children by visit count
            sorted_children = sorted(current_node.children.items(), 
                                   key=lambda x: x[1].visits, reverse=True)
            
            if len(sorted_children) >= 2:
                # Add second best move as alternative
                second_best_move, second_best_node = sorted_children[1]
                total_visits = sum(child.visits for child in current_node.children.values())
                if total_visits > 0:
                    alternative_lines.append({
                        "depth": depth,
                        "move": second_best_move,
                        "visits": second_best_node.visits,
                        "value": second_best_node.mean_value,
                        "probability": second_best_node.visits / total_visits
                    })
            
            # Follow principal variation
            if depth < len(pv):
                current_node = current_node.children[pv[depth]]
            else:
                break
            depth += 1
        
        return {
            "principal_variation": pv,
            "alternative_lines": alternative_lines,
            "pv_length": len(pv)
        }
    
    def _log_detailed_progress(self, sims_completed: int, total_sims: int):
        """Log detailed progress information for debugging."""
        elapsed = time.time() - self.stats['start_time']
        sims_per_sec = sims_completed / elapsed
        
        # Get evaluator statistics
        evaluator_stats = self.evaluator.get_statistics()
        
        self.logger.info(f"=== Detailed Progress Report ===")
        self.logger.info(f"Simulations: {sims_completed}/{total_sims} ({sims_per_sec:.1f} sims/sec)")
        self.logger.info(f"Queue size: {self.evaluator.get_queue_size()}")
        self.logger.info(f"Cache size: {self.evaluator.get_cache_size()}")
        self.logger.info(f"Cache hit rate: {evaluator_stats.get('cache_hit_rate', 0):.1%}")
        self.logger.info(f"Total batches processed: {self.stats['total_batches_processed']}")
        self.logger.info(f"Average batch size: {evaluator_stats.get('average_batch_size', 0):.1f}")
        
        # GPU memory info
        from hex_ai.utils.gpu_monitoring import get_gpu_memory_info
        gpu_mem = get_gpu_memory_info()
        if gpu_mem:
            self.logger.info(f"GPU Memory: {gpu_mem['allocated_mb']:.1f}MB allocated "
                           f"({gpu_mem['utilization_percent']:.1f}% utilization)")
        
        # Tree statistics
        if hasattr(self, '_last_root') and self._last_root:
            self._log_tree_statistics(self._last_root)
        
        self.logger.info(f"================================")
    
    def _log_tree_statistics(self, root: BatchedMCTSNode):
        """Log statistics about the search tree."""
        if not root.children:
            return
        
        # Count nodes by state
        unexpanded = 0
        pending = 0
        expanded = 0
        
        def count_nodes(node):
            nonlocal unexpanded, pending, expanded
            if node.node_state == NodeState.UNEXPANDED:
                unexpanded += 1
            elif node.node_state == NodeState.EVALUATION_PENDING:
                pending += 1
            elif node.node_state == NodeState.EXPANDED:
                expanded += 1
            
            for child in node.children.values():
                count_nodes(child)
        
        count_nodes(root)
        
        self.logger.info(f"Tree nodes: {unexpanded} unexpanded, {pending} pending, {expanded} expanded")
        
        # Show top moves
        moves = list(root.children.keys())
        visit_counts = [root.children[move].visits for move in moves]
        
        if visit_counts:
            sorted_moves = sorted(zip(moves, visit_counts), key=lambda x: x[1], reverse=True)
            self.logger.info(f"Top moves: {sorted_moves[:3]}")
