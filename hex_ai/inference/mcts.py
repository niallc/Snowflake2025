"""
Monte Carlo Tree Search (MCTS) implementation for Hex AI.

This module provides a neural network-guided MCTS implementation following the AlphaZero approach.
The design prioritizes correctness first, with optimizations added incrementally.
"""

import copy
import math
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from hex_ai.inference.game_engine import HexGameState
from hex_ai.inference.simple_model_inference import SimpleModelInference
from hex_ai.value_utils import policy_logits_to_probs, get_top_k_legal_moves
from hex_ai.utils.format_conversion import rowcol_to_tensor
from hex_ai.config import BLUE_PLAYER, RED_PLAYER

logger = logging.getLogger(__name__)


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
    
    def update_statistics(self, value: float) -> None:
        """Update node statistics after backpropagation."""
        self.visits += 1
        self.total_value += value
        logger.debug(f"Updated node {self.move}: visits={self.visits}, total_value={self.total_value:.4f}, mean_value={self.mean_value:.4f}")


class NeuralMCTS:
    """MCTS engine guided by a neural network."""
    
    def __init__(self, model: SimpleModelInference, exploration_constant: float = 1.4):
        """
        Initialize the MCTS engine.
        
        Args:
            model: Neural network model for policy and value predictions
            exploration_constant: PUCT exploration constant (default: 1.4)
        """
        self.model = model
        self.exploration_constant = exploration_constant
        self.logger = logging.getLogger(__name__)
        
        # Statistics tracking
        self.stats = {
            'total_simulations': 0,
            'total_inferences': 0,
            'start_time': None,
            'end_time': None
        }
        
        self.logger.info(f"Initialized NeuralMCTS with exploration_constant={exploration_constant}")

    def search(self, root_state: HexGameState, num_simulations: int) -> MCTSNode:
        """
        Run MCTS search from a root state to build up statistics.
        
        Args:
            root_state: Starting game state
            num_simulations: Number of MCTS simulations to run
            
        Returns:
            Root node with populated search tree
        """
        self.logger.info(f"Starting MCTS search with {num_simulations} simulations")
        self.stats['start_time'] = time.time()
        self.stats['total_simulations'] = 0
        
        # Create root node with deep copy of state
        root = MCTSNode(state=copy.deepcopy(root_state))
        
        # Run simulations
        for sim_idx in range(num_simulations):
            self._run_simulation(root)
            self.stats['total_simulations'] += 1
            
            # Log progress every 100 simulations
            if (sim_idx + 1) % 100 == 0:
                elapsed = time.time() - self.stats['start_time']
                sims_per_sec = (sim_idx + 1) / elapsed
                self.logger.info(f"Completed {sim_idx + 1}/{num_simulations} simulations ({sims_per_sec:.1f} sims/sec)")
        
        self.stats['end_time'] = time.time()
        total_time = self.stats['end_time'] - self.stats['start_time']
        sims_per_sec = num_simulations / total_time
        
        self.logger.info(f"MCTS search completed: {num_simulations} simulations in {total_time:.2f}s ({sims_per_sec:.1f} sims/sec)")
        self.logger.info(f"Total neural network inferences: {self.stats['total_inferences']}")
        
        return root

    def _run_simulation(self, root: MCTSNode) -> None:
        """Run a single MCTS simulation."""
        # 1. Selection: Traverse the tree using PUCT until a leaf node is found.
        leaf_node = self._select(root)
        
        # 2. Expansion & Evaluation: If the game is not over, expand the leaf and get its value from the NN.
        value = self._expand_and_evaluate(leaf_node)
        
        # 3. Backpropagation: Update statistics up the tree from the leaf.
        self._backpropagate(leaf_node, value)

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Traverse the tree from the root to a leaf node."""
        while not node.is_leaf():
            node = self._select_child_with_puct(node)
        return node

    def _select_child_with_puct(self, node: MCTSNode) -> MCTSNode:
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
            ucb_component = self.exploration_constant * prior * (math.sqrt(node.visits) / (1 + child_node.visits))
            
            puct_score = q_value + ucb_component
            
            self.logger.debug(
                f"PUCT scores - Move {move}: Q={q_value:.4f}, prior={prior:.4f}, "
                f"UCB={ucb_component:.4f}, total={puct_score:.4f}"
            )
            
            if puct_score > best_score:
                best_score = puct_score
                best_child = child_node
                
        if best_child is None:
            raise ValueError("No child selected - this should not happen")
            
        return best_child

    def _expand_and_evaluate(self, node: MCTSNode) -> float:
        """
        Expand a leaf node, create its children, and return the evaluated value.
        
        Args:
            node: Leaf node to expand
            
        Returns:
            Value of the node from the perspective of the current player
        """
        if node.is_terminal():
            return self._terminal_value(node.state)

        # Get policy and value from the neural network
        self.stats['total_inferences'] += 1
        policy_logits, value = self.model.simple_infer(node.state.get_board_tensor())
        
        # Apply softmax and filter for legal moves
        legal_moves = node.state.get_legal_moves()
        node.policy_priors = self._get_priors_for_legal_moves(policy_logits, legal_moves)

        # Create child nodes for all legal moves
        for move, prior in node.policy_priors.items():
            # IMPORTANT: The new state must be a deep copy.
            child_state = copy.deepcopy(node.state)
            child_state = child_state.make_move(*move) 
            node.children[move] = MCTSNode(state=child_state, parent=node, move=move)
            
        self.logger.debug(f"Expanded node with {len(node.children)} children, value={value:.4f}")
        
        # The value is from the perspective of the current player at 'node'.
        return value

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """
        Update visit counts and values from a leaf node up to the root.
        
        Args:
            node: Leaf node to start backpropagation from
            value: Value to propagate up the tree
        """
        current_node = node
        while current_node is not None:
            current_node.update_statistics(value)
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
            Value from the perspective of the player who just moved (1.0 for win, -1.0 for loss, 0.0 for draw)
        """
        if not state.game_over:
            raise ValueError("Cannot get terminal value for non-terminal state")
        
        if state.winner is None:
            # Draw (shouldn't happen in Hex, but good to handle)
            return 0.0
        
        # The player who just moved is the OPPOSITE of current_player
        just_moved_player = RED_PLAYER if state.current_player == BLUE_PLAYER else BLUE_PLAYER
        
        # Determine if the player who just moved won
        just_moved_won = (
            (just_moved_player == BLUE_PLAYER and state.winner == "blue") or
            (just_moved_player == RED_PLAYER and state.winner == "red")
        )
        
        # Return value from perspective of player who just moved
        if just_moved_won:
            return 1.0  # Player who just moved won
        else:
            return -1.0  # Player who just moved lost

    def _get_priors_for_legal_moves(self, policy_logits: np.ndarray, legal_moves: List[Tuple[int, int]]) -> Dict[Tuple[int, int], float]:
        """
        Extract prior probabilities for legal moves from policy logits.
        
        Args:
            policy_logits: Raw policy logits from neural network
            legal_moves: List of legal moves
            
        Returns:
            Dictionary mapping moves to prior probabilities
        """
        # Convert logits to probabilities
        policy_probs = policy_logits_to_probs(policy_logits)
        
        # Extract probabilities for legal moves
        move_priors = {}
        total_prior = 0.0
        
        for move in legal_moves:
            row, col = move
            # Convert 2D coordinates to 1D index
            index = rowcol_to_tensor(row, col)
            prior = policy_probs[index]
            move_priors[move] = prior
            total_prior += prior
        
        # Normalize to ensure probabilities sum to 1
        if total_prior > 0:
            for move in move_priors:
                move_priors[move] /= total_prior
        else:
            # If all priors are zero, use uniform distribution
            uniform_prior = 1.0 / len(legal_moves)
            for move in legal_moves:
                move_priors[move] = uniform_prior
        
        return move_priors

    def select_move(self, root: MCTSNode, temperature: float = 1.0) -> Tuple[int, int]:
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
        scaled_visits = [count ** (1.0 / temperature) for count in visit_counts]
        total = sum(scaled_visits)
        
        if total == 0:
            # Fallback to uniform distribution
            return [1.0 / len(visit_counts)] * len(visit_counts)
        
        return [v / total for v in scaled_visits]

    def get_search_statistics(self) -> Dict[str, Any]:
        """Get statistics about the last search."""
        stats = self.stats.copy()
        if stats['start_time'] and stats['end_time']:
            stats['total_time'] = stats['end_time'] - stats['start_time']
            if stats['total_simulations'] > 0:
                stats['simulations_per_second'] = stats['total_simulations'] / stats['total_time']
        return stats 