"""
MCTS-specific utility functions for tree analysis and data formatting.

This module contains utilities that are specific to MCTS tree operations,
separate from general value processing utilities.
"""

import numpy as np
from typing import List, Tuple, Dict, Any

# =============================
# MCTS Tree Analysis Utilities
# =============================

# Threshold for detailed exploration tracking (when simulations <= this value)
DETAILED_EXPLORATION_THRESHOLD = 47

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


def create_exploration_step_info(node, action_idx: int, puct_scores: np.ndarray, 
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
    move_coords = node.legal_moves[action_idx]
    move_str = f"{chr(97 + move_coords[1])}{move_coords[0] + 1}"
    
    # Get top PUCT scores for this node
    top_scores = []
    for i, score in enumerate(puct_scores):
        if i < len(node.legal_moves):
            move = node.legal_moves[i]
            move_name = f"{chr(97 + move[1])}{move[0] + 1}"
            top_scores.append({
                'move': move_name,
                'score': float(score),
                'visits': int(node.N[i]),
                'q_value': float(node.Q[i]),
                'prior': float(node.P[i])
            })
    
    # Sort by PUCT score
    top_scores.sort(key=lambda x: x['score'], reverse=True)
    
    step_info = {
        'simulation': simulation_num,
        'depth': depth,
        'node_hash': node.state_hash,
        'to_play': node.to_play.value,
        'legal_moves': [f"{chr(97 + m[1])}{m[0] + 1}" for m in node.legal_moves],
        'top_puct_scores': top_scores[:5],  # Top 5 scores
        'selected_action': action_idx,
        'selected_move': move_str,
        'selected_move_coords': move_coords,
        'is_terminal': node.is_terminal,
        'winner': node.winner.value if node.winner else None,
        'path_to_node': path_to_node or []
    }
    
    return step_info


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
        tree_data['detailed_exploration'] = {
            'enabled': True,
            'simulation_threshold': DETAILED_EXPLORATION_THRESHOLD,
            'total_simulations': simulation_count,
            'trace': exploration_trace
        }
    else:
        tree_data['detailed_exploration'] = {'enabled': False}
    
    return tree_data


def format_mcts_tree_data_for_api(root_node, cache_misses: int, max_pv_length: int = 10) -> dict:
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
    visit_counts = {}
    mcts_probabilities = {}
    total_visits = int(np.sum(root_node.N))
    
    # Check if this is a terminal move shortcut case (no visits but terminal moves detected)
    if total_visits == 0 and hasattr(root_node, 'terminal_moves') and any(root_node.terminal_moves):
        # This is a terminal move shortcut case - provide meaningful data
        for i, (row, col) in enumerate(root_node.legal_moves):
            move_trmph = f"{chr(ord('a') + col)}{row + 1}"
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
            move_trmph = f"{chr(ord('a') + col)}{row + 1}"
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

    return {
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
