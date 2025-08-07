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
from hex_ai.value_utils import policy_logits_to_probs, get_top_k_legal_moves
from hex_ai.config import BLUE_PLAYER, RED_PLAYER

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
        self.virtual_loss += amount
    
    def remove_virtual_loss(self, amount: int = 1):
        """Remove virtual loss after evaluation is complete."""
        self.virtual_loss = max(0, self.virtual_loss - amount)


class BatchedNeuralMCTS:
    """Batched MCTS engine guided by a neural network."""
    
    def __init__(self, model: SimpleModelInference, exploration_constant: float = 1.4, 
                 win_value: float = 1.5, discount_factor: float = 0.98, 
                 optimal_batch_size: int = 64, verbose: int = 0):
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
        
        # Initialize batch processor
        self.batch_processor = BatchProcessor(model, optimal_batch_size)
        
        # Statistics tracking
        self.stats = {
            'total_simulations': 0,
            'total_inferences': 0,
            'total_batches_processed': 0,
            'start_time': None,
            'end_time': None
        }
        
        self.logger.info(f"Initialized BatchedNeuralMCTS with exploration_constant={exploration_constant}, "
                        f"win_value={win_value}, discount_factor={discount_factor}, "
                        f"optimal_batch_size={optimal_batch_size}")

    def search(self, root_state_or_node, num_simulations: int) -> BatchedMCTSNode:
        """
        Run MCTS search from a root state or existing node to build up statistics.
        
        Args:
            root_state_or_node: Either a HexGameState (for new search) or BatchedMCTSNode (for continuing search)
            num_simulations: Number of MCTS simulations to run
            
        Returns:
            Root node with populated search tree
        """
        if self.verbose >= 1:
            self.logger.info(f"Starting batched MCTS search with {num_simulations} simulations")
        self.stats['start_time'] = time.time()
        self.stats['total_simulations'] = 0
        
        # Handle both HexGameState and BatchedMCTSNode inputs
        if isinstance(root_state_or_node, HexGameState):
            # Create new root node with deep copy of state
            root = BatchedMCTSNode(state=copy.deepcopy(root_state_or_node))
        elif isinstance(root_state_or_node, BatchedMCTSNode):
            # Continue search from existing node
            root = root_state_or_node
        else:
            raise ValueError(f"Expected HexGameState or BatchedMCTSNode, got {type(root_state_or_node)}")
        
        # Run simulations
        for sim_idx in range(num_simulations):
            self._run_simulation(root)
            self.stats['total_simulations'] += 1
            
            # Process batch periodically or when queue is large enough
            # TODO (P1): Understand the magic number 10, what does "process" mean in this context?
            #            Is this about network inference or some other procesing step?
            if (sim_idx + 1) % 10 == 0 or self.batch_processor.get_queue_size() >= self.optimal_batch_size:
                processed = self.batch_processor.process_batch()
                if processed > 0:
                    self.stats['total_batches_processed'] += 1
            
            # Log progress every 100 simulations
            # TODO (P3): Also log progress after 10 and 50 simulations, to see whether start-up is efficient?
            if self.verbose >= 1 and (sim_idx + 1) % 100 == 0:
                elapsed = time.time() - self.stats['start_time']
                sims_per_sec = (sim_idx + 1) / elapsed
                queue_size = self.batch_processor.get_queue_size()
                self.logger.info(f"Completed {sim_idx + 1}/{num_simulations} simulations "
                               f"({sims_per_sec:.1f} sims/sec, queue: {queue_size})")
        
        # Process any remaining requests
        remaining_processed = self.batch_processor.process_all()
        if remaining_processed > 0:
            self.stats['total_batches_processed'] += 1
        
        self.stats['end_time'] = time.time()
        total_time = self.stats['end_time'] - self.stats['start_time']
        sims_per_sec = num_simulations / total_time
        
        # Get batch processor statistics
        batch_stats = self.batch_processor.get_statistics()
        self.stats['total_inferences'] = batch_stats['total_inferences']
        
        if self.verbose >= 1:
            self.logger.info(f"Batched MCTS search completed: {num_simulations} simulations in {total_time:.2f}s ({sims_per_sec:.1f} sims/sec)")
            self.logger.info(f"Total neural network inferences: {self.stats['total_inferences']}")
            self.logger.info(f"Total batches processed: {self.stats['total_batches_processed']}")
            self.logger.info(f"Cache hit rate: {batch_stats['cache_hit_rate']:.1%}")
            self.logger.info(f"Average batch size: {batch_stats['average_batch_size']:.1f}")
        
        return root

    def _run_simulation(self, root: BatchedMCTSNode) -> None:
        """Run a single MCTS simulation."""
        # 1. Selection: Traverse the tree using PUCT until a leaf node is found.
        leaf_node = self._select(root)
        
        # 2. Expansion & Evaluation: If the game is not over, expand the leaf and get its value from the NN.
        value = self._expand_and_evaluate(leaf_node)
        
        # 3. Backpropagation: Update statistics up the tree from the leaf.
        self._backpropagate(leaf_node, value)

    def _select(self, node: BatchedMCTSNode) -> BatchedMCTSNode:
        """Traverse the tree from the root to a leaf node."""
        # TODO (P3): Crisp explanation: I think it's to descend the tree until we find a leaf node.
        while not node.is_leaf():
            node = self._select_child_with_puct(node)
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

        for move, child_node in node.children.items():
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
    def _expand_and_evaluate(self, node: BatchedMCTSNode) -> float:
        """
        Expand a leaf node, create its children, and return the evaluated value.
        
        Args:
            node: Leaf node to expand
            
        Returns:
            Value of the node from the perspective of the current player
        """
        if node.is_terminal():
            return self._terminal_value(node.state)

        # Handle different node states
        if node.node_state == NodeState.UNEXPANDED:
            # Request evaluation from batch processor
            node.node_state = NodeState.EVALUATION_PENDING
            node.add_virtual_loss()
            
            # TODO (P3): Understand the logic of this method.
            def evaluation_callback(policy_logits: np.ndarray, value: float):
                """Callback to handle evaluation results."""
                # Apply softmax and filter for legal moves
                legal_moves = node.state.get_legal_moves()
                node.policy_priors = self._get_priors_for_legal_moves(policy_logits, legal_moves)
                
                # Create child nodes for all legal moves
                for move, prior in node.policy_priors.items():
                    child_state = node.state.make_move(*move)
                    child_depth = node.get_depth() + 1
                    node.children[move] = BatchedMCTSNode(
                        state=child_state, 
                        parent=node, 
                        move=move, 
                        depth=child_depth
                    )
                
                # Mark node as expanded
                node.node_state = NodeState.EXPANDED
                node.remove_virtual_loss()
                
                if self.verbose >= 2:
                    self.logger.debug(f"Expanded node with {len(node.children)} children, value={value:.4f}")
            
            # Request evaluation
            self.batch_processor.request_evaluation(
                board_state=node.state.get_board_tensor(),
                callback=evaluation_callback,
                metadata={'node_depth': node.get_depth(), 'simulation_id': self.stats['total_simulations']}
            )
            
            # Return a placeholder value - the real value will be used in backpropagation
            # when the evaluation completes
            return 0.0
            
        elif node.node_state == NodeState.EVALUATION_PENDING:
            # Node is already being evaluated, add virtual loss and return placeholder
            node.add_virtual_loss()
            return 0.0
            
        else:  # NodeState.EXPANDED
            # Node is already expanded, this shouldn't happen in normal flow
            raise ValueError("Attempted to expand already expanded node")

    def _backpropagate(self, node: BatchedMCTSNode, value: float) -> None:
        """
        Update visit counts and values from a leaf node up to the root.
        
        Args:
            node: Leaf node to start backpropagation from
            value: Value to propagate up the tree
        """
        current_node = node
        # TODO (P3): Get an explanation of the case where current_node is None.
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
        
        if state.winner is None:
            # Draw (shouldn't happen in Hex, but good to handle)
            # TODO (P2): Raise an exception and crash. No fallback logic! Find errors fast.
            return 0.0
        
        # The player who just moved is the OPPOSITE of current_player
        if not state.current_player in [BLUE_PLAYER, RED_PLAYER]:
            raise ValueError(f"Invalid current player: {state.current_player}")
        just_moved_player = RED_PLAYER if state.current_player == BLUE_PLAYER else BLUE_PLAYER
        
        # Determine if the player who just moved won
        # TODO (P2): Consider replacing "blue" and "red" with the relevant Enums / constants from central utilities.
        just_moved_won = (
            (just_moved_player == BLUE_PLAYER and state.winner == "blue") or
            (just_moved_player == RED_PLAYER and state.winner == "red")
        )
        
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
            # TODO (P2): Replace 13 with BOARD_SIZE from hex_ai.config.
            move_index = row * 13 + col  # Assuming 13x13 board
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
        Select final move based on visit counts.
        
        Args:
            root: Root node of search tree
            temperature: Temperature for move selection (0 = deterministic, 1 = stochastic)
            
        Returns:
            Selected move (row, col)
        """
        if not root.children:
            raise ValueError("Cannot select move from root with no children")
        
        moves = list(root.children.keys())
        visit_counts = [root.children[move].visits for move in moves]
        
        # TODO (P4): Looks like _temperature_scale handles the deterministic case, so this is duplicate code.
        #            Consider removing this. The two implementations are different, which is better?
        if temperature == 0:
            # Deterministic: select most visited
            best_move_idx = np.argmax(visit_counts)
            selected_move = moves[best_move_idx]
            self.logger.info(f"Selected move {selected_move} (deterministic, visits={visit_counts[best_move_idx]})")
        else:
            # Stochastic: sample based on visit counts
            probabilities = self._temperature_scale(visit_counts, temperature)
            selected_move_idx = np.random.choice(len(moves), p=probabilities)
            selected_move = moves[selected_move_idx]
            self.logger.info(f"Selected move {selected_move} (stochastic, visits={visit_counts[selected_move_idx]}, temp={temperature})")
        
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
        # TODO (P1): Understand the logic, why is dividing visits by temperature right?
        scaled_visits = [count ** (1.0 / temperature) for count in visit_counts]
        total = sum(scaled_visits)
        
        if total == 0:
            # TODO (P1): Check that this really does indicate a bug.
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
        
        # Add batch processor statistics
        batch_stats = self.batch_processor.get_statistics()
        stats.update(batch_stats)
        
        return stats
    
    def reset_search_statistics(self):
        """Reset search statistics for the next search."""
        self.stats['total_simulations'] = 0
        self.stats['total_inferences'] = 0
        self.stats['total_batches_processed'] = 0
        self.stats['start_time'] = None
        self.stats['end_time'] = None
        self.batch_processor.reset_statistics()
    
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
