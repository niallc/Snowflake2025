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

from hex_ai.config import BLUE_PLAYER, RED_PLAYER

logger = logging.getLogger(__name__)


@dataclass
class MCTSNode:
    """Represents a node in the MCTS search tree."""
    # Core state
    state: HexGameState  # The game state this node represents.
    parent: Optional['MCTSNode'] = None
    move: Optional[Tuple[int, int]] = None  # The move that led from parent to this node.

    # Search statistics
    visits: int = 0
    total_value: float = 0.0  # Sum of all evaluations from this node's subtree.

    # Neural network priors (cached from the first time this node is expanded)
    policy_priors: Optional[Dict[Tuple[int, int], float]] = None
    
    # Children management
    children: Dict[Tuple[int, int], 'MCTSNode'] = field(default_factory=dict)
    
    # Depth tracking for move count penalty
    depth: int = 0

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
    
    def get_depth(self) -> int:
        """Calculate the depth of this node from the root."""
        if self.parent is None:
            return 0
        return self.parent.get_depth() + 1
    
    def detach_parent(self):
        """Detach this node from its parent, making it a new root."""
        self.parent = None


class NeuralMCTS:
    """MCTS engine guided by a neural network."""
    
    def __init__(self, model: SimpleModelInference, exploration_constant: float = 1.4, win_value: float = 1.5, discount_factor: float = 0.98, verbose: int = 0):
        """
        Initialize the MCTS engine.
        
        Args:
            model: Neural network model for policy and value predictions
            exploration_constant: PUCT exploration constant (default: 1.4)
            win_value: Value assigned to winning terminal states (default: 1.5)
            discount_factor: Discount factor for move count penalty (default: 0.98)
            verbose: Verbosity level (0=quiet, 1=basic, 2=detailed, 3=debug, 4=extreme debug)
        """
        self.model = model
        self.exploration_constant = exploration_constant
        self.win_value = win_value
        self.discount_factor = discount_factor
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        
        # Statistics tracking
        self.stats = {
            'total_simulations': 0,
            'total_inferences': 0,
            'start_time': None,
            'end_time': None
        }
        
        self.logger.info(f"Initialized NeuralMCTS with exploration_constant={exploration_constant}, win_value={win_value}, discount_factor={discount_factor}")

    def search(self, root_state_or_node, num_simulations: int) -> MCTSNode:
        """
        Run MCTS search from a root state or existing node to build up statistics.
        
        Args:
            root_state_or_node: Either a HexGameState (for new search) or MCTSNode (for continuing search)
            num_simulations: Number of MCTS simulations to run
            
        Returns:
            Root node with populated search tree
        """
        if self.verbose >= 1:
            self.logger.info(f"Starting MCTS search with {num_simulations} simulations")
        self.stats['start_time'] = time.time()
        self.stats['total_simulations'] = 0
        
        # Handle both HexGameState and MCTSNode inputs
        if isinstance(root_state_or_node, HexGameState):
            # Create new root node with deep copy of state
            root = MCTSNode(state=copy.deepcopy(root_state_or_node))
        elif isinstance(root_state_or_node, MCTSNode):
            # Continue search from existing node
            root = root_state_or_node
        else:
            raise ValueError(f"Expected HexGameState or MCTSNode, got {type(root_state_or_node)}")
        
        # Run simulations
        for sim_idx in range(num_simulations):
            self._run_simulation(root)
            self.stats['total_simulations'] += 1
            
            # Log progress every 100 simulations
            if self.verbose >= 1 and (sim_idx + 1) % 100 == 0:
                elapsed = time.time() - self.stats['start_time']
                sims_per_sec = (sim_idx + 1) / elapsed
                self.logger.info(f"Completed {sim_idx + 1}/{num_simulations} simulations ({sims_per_sec:.1f} sims/sec)")
        
        self.stats['end_time'] = time.time()
        total_time = self.stats['end_time'] - self.stats['start_time']
        sims_per_sec = num_simulations / total_time
        
        if self.verbose >= 1:
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
        
        # Verbose debug logging (only at extreme debug level)
        if self.verbose >= 4:
            self.logger.debug(f"DEBUG: === SELECTING CHILD ===")
            self.logger.debug(f"DEBUG: Node state - current_player: {node.state.current_player}")
            self.logger.debug(f"DEBUG: Node state - game_over: {node.state.game_over}")
            self.logger.debug(f"DEBUG: Node state - winner: {node.state.winner}")
            self.logger.debug(f"DEBUG: Number of children: {len(node.children)}")
            self.logger.debug(f"DEBUG: Children moves: {list(node.children.keys())[:10]}...")  # Show first 10
        
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
        
        # Apply softmax and filter for legal moves using 2D approach
        legal_moves = node.state.get_legal_moves()
        node.policy_priors = self._get_priors_for_legal_moves(policy_logits, legal_moves)
        
        # Verbose debug logging (only at extreme debug level)
        if self.verbose >= 4:
            self.logger.debug(f"DEBUG: === EXPANDING NODE ===")
            self.logger.debug(f"DEBUG: Node state - current_player: {node.state.current_player}")
            self.logger.debug(f"DEBUG: Node state - game_over: {node.state.game_over}")
            self.logger.debug(f"DEBUG: Node state - winner: {node.state.winner}")
            self.logger.debug(f"DEBUG: Policy logits shape: {policy_logits.shape}")
            self.logger.debug(f"DEBUG: Policy logits min/max: {policy_logits.min():.4f}/{policy_logits.max():.4f}")
            self.logger.debug(f"DEBUG: Legal moves count: {len(legal_moves)}")
            self.logger.debug(f"DEBUG: Top 5 policy priors: {sorted(node.policy_priors.items(), key=lambda x: x[1], reverse=True)[:5]}")

        # Create child nodes for all legal moves
        for move, prior in node.policy_priors.items():
            # make_move returns a new, independent state object.
            child_state = node.state.make_move(*move)
            child_depth = node.get_depth() + 1
            node.children[move] = MCTSNode(state=child_state, parent=node, move=move, depth=child_depth)
            
            # Verbose debug logging for child creation
            if self.verbose >= 4 and len(node.children) <= 5:
                self.logger.debug(f"DEBUG: Created child for move {move} -> child state current_player: {child_state.current_player}, depth: {child_depth}")
            
        if self.verbose >= 2:
            self.logger.info(f"Expanded node with {len(node.children)} children, value={value:.4f}")
        
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
        policy_probs = policy_logits_to_probs(policy_logits, temperature=1.0)
        
        # Extract probabilities for legal moves
        move_priors = {}
        total_prior = 0.0
        
        for move in legal_moves:
            row, col = move
            # Convert 2D coordinates to 1D index
            move_index = row * 13 + col  # Assuming 13x13 board
            prior = policy_probs[move_index]
            move_priors[move] = prior
            total_prior += prior
            
            # Verbose debug logging for policy priors
            if self.verbose >= 4 and len(move_priors) <= 5:
                self.logger.debug(f"DEBUG: Move {move} (row={row}, col={col}) -> prior={prior:.6f}")
        
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
    
    def reset_search_statistics(self):
        """Reset search statistics for the next search."""
        self.stats['total_simulations'] = 0
        self.stats['total_inferences'] = 0
        self.stats['start_time'] = None
        self.stats['end_time'] = None
    
    def get_principal_variation(self, root: MCTSNode, max_depth: int = 10) -> List[Tuple[int, int]]:
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
    
    def get_move_sequence_analysis(self, root: MCTSNode, max_depth: int = 5) -> Dict[str, Any]:
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