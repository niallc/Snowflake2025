"""
MCTS-specific utility functions for tree analysis and data formatting.

This module contains utilities that are specific to MCTS tree operations,
separate from general value processing utilities.
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from hex_ai.utils.format_conversion import rowcol_to_trmph
from hex_ai.utils.temperature import calculate_mcts_root_temperature

# =============================
# MCTS Tree Analysis Utilities
# =============================

# Threshold for detailed exploration tracking (when simulations <= this value)
DETAILED_EXPLORATION_THRESHOLD = 199


def _validate_board_size_value(board_size, *, source: str) -> int:
    """Validate board-size values used in runtime MCTS conversions/decisions."""
    if isinstance(board_size, bool):
        raise TypeError(f"{source} must be an integer, got bool")
    try:
        size = int(board_size)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{source} must be an integer, got {type(board_size)}") from exc
    if size <= 0:
        raise ValueError(f"{source} must be positive, got {size}")
    return size


def _get_board_size_from_state(state) -> int:
    """Infer board size from a game state tensor for explicit move conversion validation."""
    if state is None:
        raise ValueError("state cannot be None")
    board_tensor = state.get_board_tensor()
    if not hasattr(board_tensor, "shape") or len(board_tensor.shape) < 2:
        raise ValueError(f"Invalid board tensor shape: {getattr(board_tensor, 'shape', None)}")
    board_rows = _validate_board_size_value(board_tensor.shape[-2], source="state board tensor rows")
    board_cols = _validate_board_size_value(board_tensor.shape[-1], source="state board tensor cols")
    if board_rows != board_cols:
        raise ValueError(f"Expected square board tensor, got {board_rows}x{board_cols}")
    return board_cols


def _get_board_size_from_node(node) -> int:
    """Read board size from node runtime metadata and validate it."""
    if not hasattr(node, "board_size"):
        raise AttributeError("MCTS node missing required board_size attribute")
    return _validate_board_size_value(getattr(node, "board_size"), source="node.board_size")


def _assert_matching_board_sizes(state_board_size: int, node_board_size: int) -> None:
    """Fail fast when state-derived and node-derived board sizes disagree."""
    if state_board_size != node_board_size:
        raise ValueError(
            "Board size mismatch between state and node: "
            f"{state_board_size} vs {node_board_size}"
        )


def _normalize_signed_value(value, *, source: str) -> float:
    """Normalize a signed numeric value to a finite float in [-1, 1]."""
    if isinstance(value, bool):
        raise TypeError(f"{source} must be numeric, got bool")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{source} must be numeric, got {type(value)}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{source} must be finite, got {value}")
    if not -1.0 <= value <= 1.0:
        raise ValueError(f"{source} must be in [-1, 1], got {value}")

    return value


def _require_tree_data_signed_value(tree_data: dict, *, key: str) -> float:
    """Require a signed tree-data field and normalize it to float in [-1, 1]."""
    if not isinstance(tree_data, dict):
        raise TypeError(f"tree_data must be dict, got {type(tree_data)}")
    if key not in tree_data:
        raise KeyError(f"tree_data missing required key '{key}'")

    return _normalize_signed_value(
        tree_data[key], source=f"tree_data['{key}']"
    )


def _signed_value_to_checked_probability(v_signed: float, *, source_key: str) -> float:
    """Convert a validated signed value to probability and assert [0, 1] bounds."""
    from hex_ai.value_utils import signed_to_prob

    p_prob = signed_to_prob(v_signed)
    if not 0.0 <= p_prob <= 1.0:
        raise ValueError(
            f"Converted probability from tree_data['{source_key}'] is outside [0, 1], "
            f"got {p_prob} from signed value {v_signed}"
        )
    return p_prob


def validate_required_win_probability_tree_data_fields(tree_data: dict) -> None:
    """Fail fast if tree_data is missing required win-probability signed fields."""
    _require_tree_data_signed_value(tree_data, key="v_ptm_ref_signed_root")
    _require_tree_data_signed_value(tree_data, key="v_ptm_ref_signed_best_child")


def compute_win_probability_from_tree_data(tree_data: dict) -> float:
    """
    Compute root win probability for the current player from required tree-data fields.

    Required contract:
        - tree_data["v_ptm_ref_signed_root"] exists
        - value is numeric, finite, and in [-1, 1]
    """
    v_ptm_ref_signed_root = _require_tree_data_signed_value(
        tree_data, key="v_ptm_ref_signed_root"
    )
    return _signed_value_to_checked_probability(
        v_ptm_ref_signed_root, source_key="v_ptm_ref_signed_root"
    )


def compute_best_child_win_probability_from_tree_data(tree_data: dict) -> float:
    """
    Compute best-child win probability for the current player from required tree-data fields.

    Required contract:
        - tree_data["v_ptm_ref_signed_best_child"] exists
        - value is numeric, finite, and in [-1, 1]
    """
    v_ptm_ref_signed_best_child = _require_tree_data_signed_value(
        tree_data, key="v_ptm_ref_signed_best_child"
    )
    return _signed_value_to_checked_probability(
        v_ptm_ref_signed_best_child, source_key="v_ptm_ref_signed_best_child"
    )


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
        True if detailed exploration should be enabled (<= threshold simulations)
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
    board_size = _get_board_size_from_node(node)
    _assert_matching_board_sizes(_get_board_size_from_state(node.state), board_size)
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


def format_mcts_tree_data_for_api(
    root_node,
    cache_misses: int,
    max_pv_length: int = 10,
    move_probs: Optional[Dict[str, float]] = None,
    root_nn_value_ptm_ref_signed: Optional[float] = None,
) -> dict:
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
            "root_value_source": "terminal_game_state",
            "best_child_value_source": "terminal_game_state",
            "best_child_value_available": False,
            "total_visits": 0,
            "inferences": 0,
            "total_nodes": 0,
            "max_depth": 0,
            "principal_variation": []
        }

    # Get visit counts and convert to TRMPH format
    board_size = _get_board_size_from_node(root_node)
    _assert_matching_board_sizes(_get_board_size_from_state(root_node.state), board_size)
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

    root_value_source = "tree_visits"
    best_child_value_source = "tree_children"
    best_child_value_available = False

    # Get root value (average value of all children) - in player-to-move reference frame.
    # When no visits exist yet but the root has been evaluated, use the cached root
    # network value so confidence-termination callers do not see a synthetic 0.0.
    if total_visits > 0:
        v_ptm_ref_signed_root = float(np.sum(root_node.W) / total_visits)
    elif root_nn_value_ptm_ref_signed is not None:
        v_ptm_ref_signed_root = _normalize_signed_value(
            root_nn_value_ptm_ref_signed,
            source="root_nn_value_ptm_ref_signed",
        )
        root_value_source = "root_network_eval"
    else:
        v_ptm_ref_signed_root = 0.0
        root_value_source = "unavailable_no_visits"

    # Get best child value - in player-to-move reference frame
    if total_visits > 0 and len(root_node.Q) > 0:
        v_ptm_ref_signed_best_child = float(np.max(root_node.Q))
        best_child_value_available = True
    else:
        v_ptm_ref_signed_best_child = 0.0
        best_child_value_source = "unavailable_no_child_search"

    # For terminal move shortcut case, set appropriate values
    if total_visits == 1 and hasattr(root_node, 'terminal_moves') and any(root_node.terminal_moves):
        # Terminal move means guaranteed win for current player
        v_ptm_ref_signed_root = 1.0  # +1 = current player wins
        v_ptm_ref_signed_best_child = 1.0  # +1 = current player wins
        root_value_source = "terminal_move_shortcut"
        best_child_value_source = "terminal_move_shortcut"
        best_child_value_available = True

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
        "root_value_source": root_value_source,
        "best_child_value_source": best_child_value_source,
        "best_child_value_available": best_child_value_available,
        "total_visits": total_visits,
        "inferences": total_inferences,
        "total_nodes": total_nodes,
        "max_depth": max_depth,
        "principal_variation": principal_variation
    }

    # Fail fast if required win-probability fields are missing or malformed.
    validate_required_win_probability_tree_data_fields(result)
    
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
    
    This shares the same root-temperature schedule as MCTS move selection.
    For very low temperatures, this returns normalized visit counts for
    reporting while move selection itself is deterministic.
    
    Args:
        root_node: MCTS root node containing visit counts
        root_state: Current game state for move count
        cfg: MCTS configuration containing temperature parameters
        
    Returns:
        Dictionary mapping move TRMPH strings to temperature-scaled probabilities
    """
    counts = root_node.N.astype(np.float64)
    total_visits = counts.sum()
    board_size = _get_board_size_from_state(root_state)
    node_board_size = _get_board_size_from_node(root_node)
    _assert_matching_board_sizes(board_size, node_board_size)
    
    if total_visits <= 0:
        raise RuntimeError(f"No visits recorded during MCTS search. Need to debug how this happens.")
    
    # Use the canonical non-Gumbel root temperature helper.
    move_count = len(root_state.move_history)
    temp = calculate_mcts_root_temperature(move_count=move_count, cfg=cfg, board_size=board_size)
    
    if temp <= cfg.temperature_deterministic_cutoff:
        # Deterministic selection path - expose normalized visit counts for reporting
        probs = counts / total_visits
    else:
        # Apply the same top-k filtering and temperature transform used in move sampling
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
                # Fall back to raw probabilities for stable reporting output.
                probs = counts / total_visits
        except (OverflowError, ValueError):
            # Fall back to raw probabilities for stable reporting output.
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
    board_size = _get_board_size_from_state(root_state)
    node_board_size = _get_board_size_from_node(root_node)
    _assert_matching_board_sizes(board_size, node_board_size)
    
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
