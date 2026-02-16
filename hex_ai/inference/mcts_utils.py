"""
MCTS-specific utility functions for tree analysis and data formatting.

This module contains utilities that are specific to MCTS tree operations,
separate from general value processing utilities.
"""

import math
import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Optional

from hex_ai.utils.format_conversion import rowcol_to_trmph

# =============================
# MCTS Tree Analysis Utilities
# =============================

# Threshold for detailed exploration tracking (when simulations <= this value)
DETAILED_EXPLORATION_THRESHOLD = 47


def _get_board_size_from_state(state) -> int:
    """Infer board size from a game state tensor for explicit move conversion validation."""
    return int(state.get_board_tensor().shape[-1])

def compute_win_probability_from_tree_data(tree_data: dict) -> float:
    """
    Compute win probability for the current player based on tree data.
    
    Args:
        tree_data: Dictionary containing MCTS tree analysis data
        
    Returns:
        Win probability for current player (0.0 to 1.0)
    """
    v_ptm_ref_signed_root = tree_data.get("v_ptm_ref_signed_root", 0.0)
    
    # Convert signed value to probability only at the edge (for external API)
    # Root value is already in player-to-move reference frame from backpropagation
    # +1 = current player wins, -1 = current player loses, 0 = neutral
    from hex_ai.value_utils import signed_to_prob
    p_ptm_prob_root = signed_to_prob(v_ptm_ref_signed_root)  # current player win probability
    return p_ptm_prob_root


def extract_principal_variation_from_tree(root_node, max_length: int = 10) -> List[Tuple[int, int]]:
    """
    Extract the principal variation (best move sequence) from the MCTS tree.
    
    Args:
        root_node: Root node of the MCTS tree
        max_length: Maximum length of principal variation to extract
        
    Returns:
        List of (row, col) moves representing the principal variation
    """
    if root_node is None:
        raise ValueError("Root node cannot be None")
    if max_length <= 0:
        raise ValueError(f"max_length must be positive, got {max_length}")
    
    if root_node.is_terminal:
        return []

    pv = []
    current_node = root_node
    
    for _ in range(max_length):
        if current_node.is_terminal or not current_node.is_expanded:
            break
            
        # Find the move with highest visit count
        if len(current_node.N) == 0:
            break
            
        best_move_idx = int(np.argmax(current_node.N))
        best_move = current_node.legal_moves[best_move_idx]
        pv.append(best_move)
        
        # Move to the best child
        child = current_node.children[best_move_idx]
        if child is None:
            break
        current_node = child
        
    return pv


def calculate_tree_statistics(root_node) -> Tuple[int, int]:
    """
    Calculate tree traversal statistics.
    
    Args:
        root_node: Root node of the tree
        
    Returns:
        Tuple of (total_nodes, max_depth)
    """
    if root_node is None:
        return 0, 0
    
    def count_nodes_and_depth(node, current_depth: int) -> Tuple[int, int]:
        if node is None:
            return 0, current_depth - 1
        
        total_nodes = 1
        max_depth = current_depth
        
        for child in node.children:
            if child is not None:
                child_nodes, child_depth = count_nodes_and_depth(child, current_depth + 1)
                total_nodes += child_nodes
                max_depth = max(max_depth, child_depth)
        
        return total_nodes, max_depth
    
    return count_nodes_and_depth(root_node, 0)


def should_enable_detailed_exploration(num_simulations: int) -> bool:
    """
    Determine if detailed exploration tracking should be enabled.
    
    Args:
        num_simulations: Number of simulations to be performed
        
    Returns:
        True if detailed exploration should be enabled (≤10 simulations)
    """
    return num_simulations <= DETAILED_EXPLORATION_THRESHOLD


def create_exploration_step_info(node, action_idx: int, puct_scores: List[float], 
                                selected_action: int, depth: int, simulation_num: int, 
                                path_to_node: List[str] = None) -> Dict[str, Any]:
    """
    Create detailed information about a single exploration step.
    
    Args:
        node: MCTS node being explored
        action_idx: Index of the selected action
        puct_scores: PUCT scores for all actions
        selected_action: Index of the selected action (same as action_idx)
        depth: Current depth in the tree
        simulation_num: Current simulation number
        
    Returns:
        Dictionary containing exploration step information
    """
    # Get move coordinates
    board_size = _get_board_size_from_state(node.state)
    move_coords = node.legal_moves[action_idx]
    # Convert numpy coordinates to Python tuples for JSON serialization
    move_coords_python = (int(move_coords[0]), int(move_coords[1]))
    move_str = rowcol_to_trmph(int(move_coords[0]), int(move_coords[1]), board_size)
    
    # Get top PUCT scores for this node
    top_scores = []
    for i, score in enumerate(puct_scores):
        if i < len(node.legal_moves):
            move = node.legal_moves[i]
            move_name = rowcol_to_trmph(int(move[0]), int(move[1]), board_size)
            
            # Ensure all numeric values are finite for JSON serialization
            safe_score = float(score) if math.isfinite(float(score)) else 0.0
            safe_visits = int(node.N[i]) if node.N[i] >= 0 else 0
            safe_q_value = float(node.Q[i]) if math.isfinite(float(node.Q[i])) else 0.0
            safe_prior = float(node.P[i]) if math.isfinite(float(node.P[i])) else 0.0
            
            top_scores.append({
                'move': move_name,
                'score': safe_score,
                'visits': safe_visits,
                'q_value': safe_q_value,
                'prior': safe_prior
            })
    
    # Sort by PUCT score
    top_scores.sort(key=lambda x: x['score'], reverse=True)
    
    step_info = {
        'simulation': int(simulation_num),
        'depth': int(depth),
        'node_hash': int(node.state_hash),  # Convert to Python int
        'to_play': int(node.to_play.value),  # Convert to Python int
        'legal_moves': [rowcol_to_trmph(int(m[0]), int(m[1]), board_size) for m in node.legal_moves],
        'top_puct_scores': top_scores[:5],  # Top 5 scores
        'selected_action': int(action_idx),
        'selected_move': move_str,
        'selected_move_coords': move_coords_python,
        'is_terminal': bool(node.is_terminal),  # Convert to Python bool
        'winner': int(node.winner.value) if node.winner else None,  # Convert to Python int
        'path_to_node': path_to_node or []
    }
    
    return step_info


def sanitize_numeric_values(obj):
    """
    Recursively sanitize numeric values to ensure they are JSON-serializable.
    Replaces Infinity, -Infinity, and NaN with safe fallback values.
    """
    if isinstance(obj, dict):
        return {k: sanitize_numeric_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_numeric_values(item) for item in obj]
    elif isinstance(obj, float):
        if not math.isfinite(obj):
            return 0.0  # Safe fallback for infinite/NaN values
        return obj
    elif isinstance(obj, int):
        return obj
    else:
        return obj

def add_detailed_exploration_to_tree_data(tree_data: Dict[str, Any], 
                                         detailed_exploration_enabled: bool,
                                         exploration_trace: List[Dict[str, Any]],
                                         simulation_count: int) -> Dict[str, Any]:
    """
    Add detailed exploration data to tree data for API consumption.
    
    Args:
        tree_data: Base tree data dictionary
        detailed_exploration_enabled: Whether detailed exploration was enabled
        exploration_trace: List of exploration steps
        simulation_count: Total number of simulations performed
        
    Returns:
        Tree data with detailed exploration information added
    """
    if detailed_exploration_enabled and exploration_trace:
        # Sanitize the exploration trace to ensure all numeric values are JSON-serializable
        sanitized_trace = sanitize_numeric_values(exploration_trace)
        tree_data['detailed_exploration'] = {
            'enabled': True,
            'simulation_threshold': DETAILED_EXPLORATION_THRESHOLD,
            'total_simulations': simulation_count,
            'trace': sanitized_trace
        }
    else:
        tree_data['detailed_exploration'] = {'enabled': False}
    
    return tree_data


def format_mcts_tree_data_for_api(root_node, cache_misses: int, max_pv_length: int = 10, move_probs: Optional[Dict[str, float]] = None) -> dict:
    """
    Format MCTS tree data for API consumption.
    
    Args:
        root_node: Root node of the MCTS tree
        cache_misses: Number of cache misses (inferences)
        max_pv_length: Maximum length of principal variation
        
    Returns:
        Dictionary containing formatted tree data for API
    """
    if root_node.is_terminal:
        return {
            "visit_counts": {},
            "mcts_probabilities": {},
            "v_ptm_ref_signed_root": 0.0,
            "v_ptm_ref_signed_best_child": 0.0,
            "total_visits": 0,
            "inferences": 0,
            "total_nodes": 0,
            "max_depth": 0,
            "principal_variation": []
        }

    # Get visit counts and convert to TRMPH format
    board_size = _get_board_size_from_state(root_node.state)
    visit_counts = {}
    mcts_probabilities = {}
    total_visits = int(np.sum(root_node.N))
    
    # Check if this is a terminal move shortcut case (no visits but terminal moves detected)
    if total_visits == 0 and hasattr(root_node, 'terminal_moves') and any(root_node.terminal_moves):
        # This is a terminal move shortcut case - provide meaningful data
        for i, (row, col) in enumerate(root_node.legal_moves):
            move_trmph = rowcol_to_trmph(int(row), int(col), board_size)
            if root_node.terminal_moves[i]:
                # Terminal move gets 100% probability and 1 visit
                visit_counts[move_trmph] = 1
                mcts_probabilities[move_trmph] = 1.0
            else:
                # Non-terminal moves get 0 probability and 0 visits
                visit_counts[move_trmph] = 0
                mcts_probabilities[move_trmph] = 0.0
        total_visits = 1  # Set to 1 to indicate one "virtual" visit
    else:
        # Normal MCTS case - use actual visit counts
        for i, (row, col) in enumerate(root_node.legal_moves):
            move_trmph = rowcol_to_trmph(int(row), int(col), board_size)
            visits = int(root_node.N[i])
            visit_counts[move_trmph] = visits
            
            # Calculate MCTS probability (visit count / total visits)
            if total_visits > 0:
                mcts_probabilities[move_trmph] = visits / total_visits
            else:
                mcts_probabilities[move_trmph] = 0.0

    # Get root value (average value of all children) - in player-to-move reference frame
    # root.W contains accumulated values in player-to-move reference frame from backpropagation
    if total_visits > 0:
        v_ptm_ref_signed_root = float(np.sum(root_node.W) / total_visits)
    else:
        v_ptm_ref_signed_root = 0.0

    # Get best child value - in player-to-move reference frame
    if len(root_node.Q) > 0:
        v_ptm_ref_signed_best_child = float(np.max(root_node.Q))
    else:
        v_ptm_ref_signed_best_child = 0.0

    # For terminal move shortcut case, set appropriate values
    if total_visits == 1 and hasattr(root_node, 'terminal_moves') and any(root_node.terminal_moves):
        # Terminal move means guaranteed win for current player
        v_ptm_ref_signed_root = 1.0  # +1 = current player wins
        v_ptm_ref_signed_best_child = 1.0  # +1 = current player wins

    # Calculate total inferences (cache misses)
    total_inferences = cache_misses

    # Calculate tree traversal statistics
    total_nodes, max_depth = calculate_tree_statistics(root_node)

    # Get principal variation
    principal_variation = extract_principal_variation_from_tree(root_node, max_length=max_pv_length)

    result = {
        "visit_counts": visit_counts,
        "mcts_probabilities": mcts_probabilities,
        "v_ptm_ref_signed_root": v_ptm_ref_signed_root,
        "v_ptm_ref_signed_best_child": v_ptm_ref_signed_best_child,
        "total_visits": total_visits,
        "inferences": total_inferences,
        "total_nodes": total_nodes,
        "max_depth": max_depth,
        "principal_variation": principal_variation
    }
    
    # Add move probabilities if provided
    if move_probs is not None:
        result["move_probabilities"] = move_probs
    
    return result


# =============================
# Probability Calculation Utilities
# =============================

def _convert_moves_to_trmph_dict(
    legal_moves: List[Tuple[int, int]], 
    values: np.ndarray,
    board_size: int,
) -> Dict[str, float]:
    """
    Core utility to convert move data to TRMPH format dictionary.
    
    Args:
        legal_moves: List of (row, col) tuples for legal moves
        values: Array of values corresponding to each legal move
        
    Returns:
        Dictionary mapping TRMPH move strings to values
    """
    if len(legal_moves) != len(values):
        raise ValueError(
            f"Length mismatch for move/value conversion: "
            f"{len(legal_moves)} legal moves vs {len(values)} values"
        )

    result = {}
    for i, (row, col) in enumerate(legal_moves):
        move_trmph = rowcol_to_trmph(int(row), int(col), board_size)
        result[move_trmph] = float(values[i])
    
    return result

def calculate_visit_count_probs(root_node, root_state, cfg) -> Dict[str, float]:
    """
    Calculate temperature-scaled probabilities from visit counts (for PUCT mode).
    
    This is the same logic used in MCTS move selection, extracted
    into a reusable utility for debugging and analysis purposes.
    
    Args:
        root_node: MCTS root node containing visit counts
        root_state: Current game state for move count
        cfg: MCTS configuration containing temperature parameters
        
    Returns:
        Dictionary mapping move TRMPH strings to temperature-scaled probabilities
    """
    counts = root_node.N.astype(np.float64)
    total_visits = counts.sum()
    board_size = int(root_state.get_board_tensor().shape[-1])
    
    if total_visits <= 0:
        raise RuntimeError(f"No visits recorded during MCTS search. Need to debug how this happens.")
    
    # Calculate temperature with decay
    move_count = len(root_state.move_history)
    temp = _calculate_root_temperature(move_count, cfg)
    
    if temp <= cfg.temperature_deterministic_cutoff:
        # Deterministic selection - use raw visit counts
        probs = counts / total_visits
    else:
        # Apply temperature scaling using the same logic as move selection
        try:
            # Apply top-k filtering if configured
            if cfg.visit_sampling_top_k > 0 and len(counts) > cfg.visit_sampling_top_k:
                top_k_indices = np.argpartition(counts, -cfg.visit_sampling_top_k)[-cfg.visit_sampling_top_k:]
                filtered_counts = np.zeros_like(counts)
                filtered_counts[top_k_indices] = counts[top_k_indices]
            else:
                filtered_counts = counts
            
            # Apply temperature scaling
            pi = np.power(filtered_counts, 1.0 / temp)
            if np.isfinite(pi).all() and np.sum(pi) > 0:
                probs = pi / np.sum(pi)
            else:
                # Fall back to raw probabilities
                probs = counts / total_visits
        except (OverflowError, ValueError):
            # Fall back to raw probabilities
            probs = counts / total_visits
    
    return _convert_moves_to_trmph_dict(root_node.legal_moves, probs, board_size)

def calculate_policy_probs(root_node, root_state, cfg, mcts_instance) -> Dict[str, float]:
    """
    Calculate raw policy probabilities (for Gumbel mode).
    
    This returns the policy network probabilities that were used for Gumbel selection,
    formatted for tree data output.
    
    Args:
        root_node: MCTS root node containing legal moves and indices
        root_state: Current game state for policy inference
        cfg: MCTS configuration (not used for policy calculation)
        mcts_instance: MCTS instance to access policy methods
        
    Returns:
        Dictionary mapping move TRMPH strings to policy probabilities
    """
    # Get policy logits and legal mask using the same shared utility as Gumbel
    policy_logits_full, legal_mask = mcts_instance._get_policy_logits_and_legal_mask(root_state, root_node.legal_indices)
    board_size = int(root_state.get_board_tensor().shape[-1])
    
    # Convert to probabilities using the same method as Gumbel
    priors_full = mcts_instance._root_priors_from_logits(policy_logits_full, legal_mask, apply_dirichlet=False)
    
    # Extract probabilities for legal moves only
    legal_probs = np.array([priors_full[tensor_idx] for tensor_idx in root_node.legal_indices])
    
    return _convert_moves_to_trmph_dict(root_node.legal_moves, legal_probs, board_size)



def select_move_index(counts: np.ndarray, temp: float, cfg) -> int:
    """
    Select a move index using temperature scaling and top-k filtering.
    
    This is the core move selection logic extracted for reuse.
    
    Args:
        counts: Visit counts for all legal moves
        temp: Current temperature value
        cfg: MCTS configuration containing filtering parameters
        
    Returns:
        Index of the selected move
    """
    if temp <= cfg.temperature_deterministic_cutoff:
        # Use deterministic selection for very low temperatures to avoid numerical issues
        return int(np.argmax(counts))
    else:
        # Apply top-k filtering and temperature scaling with validation
        try:
            # Apply top-k filtering: only consider the top-k most visited moves
            if cfg.visit_sampling_top_k > 0 and len(counts) > cfg.visit_sampling_top_k:
                # Get indices of top-k moves
                top_k_indices = np.argpartition(counts, -cfg.visit_sampling_top_k)[-cfg.visit_sampling_top_k:]
                # Create filtered counts array (set non-top-k moves to 0)
                filtered_counts = np.zeros_like(counts)
                filtered_counts[top_k_indices] = counts[top_k_indices]
            else:
                filtered_counts = counts
            
            # Apply temperature scaling to filtered counts
            pi = np.power(filtered_counts, 1.0 / temp)
            if not np.isfinite(pi).all():
                raise ValueError(f"Temperature scaling produced non-finite values: pi={pi}, counts={filtered_counts}, temp={temp}")
            if np.sum(pi) <= 0:
                raise ValueError(f"Temperature scaling produced non-positive sum: pi={pi}, counts={filtered_counts}, temp={temp}")
            pi /= np.sum(pi)
            return int(np.random.choice(len(pi), p=pi))
        except (OverflowError, ValueError) as e:
            # Fall back to deterministic selection if temperature scaling fails
            print(f"Warning: Temperature scaling failed with temp={temp}, falling back to deterministic selection. Error: {e}")
            return int(np.argmax(counts))


def _calculate_root_temperature(move_count: int, cfg) -> float:
    """
    Calculate the root temperature based on move count and configuration.
    
    This is a helper function that encapsulates the temperature decay logic.
    """
    if cfg.temperature_decay_type == 'exponential':
        # Exponential decay: temp = start * (end/start)^(move_count/decay_moves)
        if cfg.temperature_start <= 0 or cfg.temperature_end <= 0:
            return cfg.temperature_start
        decay_factor = cfg.temperature_end / cfg.temperature_start
        progress = min(move_count / cfg.temperature_decay_moves, 1.0)
        return cfg.temperature_start * (decay_factor ** progress)
    elif cfg.temperature_decay_type == 'step':
        # Step decay: use thresholds and values
        for threshold, value in zip(cfg.temperature_step_thresholds, cfg.temperature_step_values):
            if move_count < threshold:
                return value
        return cfg.temperature_end
    else:
        # Linear decay or unknown type - fall back to linear
        progress = min(move_count / cfg.temperature_decay_moves, 1.0)
        return cfg.temperature_start + (cfg.temperature_end - cfg.temperature_start) * progress
