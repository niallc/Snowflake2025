"""
Compatibility layer for fixed tree search.

This module provides the old function signatures while using the new modernized
implementation internally. This allows for a gradual migration without breaking
existing code.
"""

from typing import List, Tuple, Optional, Dict, Any
import logging

from hex_ai.inference.game_engine import HexGameState
from hex_ai.inference.simple_model_inference import SimpleModelInference
from hex_ai.inference.fixed_tree_search_modern import (
    run_fixed_tree_search,
    create_fixed_tree_config,
    FixedTreeSearchResult
)

logger = logging.getLogger(__name__)


def minimax_policy_value_search(
    state: HexGameState,
    model,
    widths: List[int],
    batch_size: int = 1000,
    use_alpha_beta: bool = True,
    temperature: float = 1.0,
    debug: bool = False,
    return_tree: bool = False,
    verbose: int = 0
) -> Tuple[Tuple[int, int], float, Optional[Any]]:
    """
    Fixed-width, fixed-depth minimax search with alpha-beta pruning and batch evaluation at the leaves.
    
    This is a compatibility wrapper around the new modernized implementation.
    
    Args:
        state: HexGameState (current position)
        model: ModelWrapper (must support batch inference)
        widths: List of ints, e.g. [20, 10, 10, 5] (width at each ply)
        batch_size: Max batch size for evaluation
        use_alpha_beta: Whether to use alpha-beta pruning (not implemented in new version yet)
        temperature: Policy temperature for move selection (default 1.0)
        debug: Whether to enable debug logging
        return_tree: Whether to return the search tree for debugging
        verbose: Verbosity level (0: silent, 1+: show info logs)

    Returns:
        best_move: (row, col) tuple for the best move at the root
        value: estimated value for the root position
        root: search tree root node (if return_tree=True, otherwise None)
    """
    if debug:
        logger.setLevel(logging.DEBUG)
    elif verbose <= 1:
        logger.setLevel(logging.WARNING)  # Suppress INFO logs when not verbose
    else:
        logger.setLevel(logging.INFO)
    
    if verbose >= 2:
        logger.info(f"Starting minimax search with widths {widths}, temperature {temperature}")
        root_player_enum = state.current_player_enum
        logger.info(f"Root state: player {state.current_player} ({'Blue' if root_player_enum == 0 else 'Red'})")
    
    # Create configuration for the new implementation
    config = create_fixed_tree_config(
        search_widths=widths,
        temperature=temperature,
        batch_size=batch_size,
        enable_early_termination=True,
        early_termination_threshold=0.95
    )
    
    # Run the modernized search
    result = run_fixed_tree_search(state, model, config, verbose)
    
    if verbose >= 2:
        logger.info(f"Search complete: best move = {result.move}, value = {result.value}")
    
    if return_tree:
        # For now, return None for the tree since the new implementation doesn't expose it
        # TODO: Add tree extraction to the new implementation if needed
        return result.move, result.value, None
    else:
        return result.move, result.value


def minimax_policy_value_search_with_batching(
    state: HexGameState,
    model,
    widths: List[int],
    batch_size: int = 1000,
    use_alpha_beta: bool = True,
    temperature: float = 1.0,
    debug: bool = False,
    return_tree: bool = False,
    verbose: int = 0
) -> Tuple[Tuple[int, int], float, Optional[Any], Dict[str, Any]]:
    """
    Fixed-width, fixed-depth minimax search with batched inference for all policy calls.
    
    This is a compatibility wrapper around the new modernized implementation.
    The new implementation already uses batching internally, so this function
    is essentially the same as minimax_policy_value_search.
    
    Args:
        state: HexGameState (current position)
        model: ModelWrapper (must support batch inference)
        widths: List of ints, e.g. [20, 10, 10, 5] (width at each ply)
        batch_size: Max batch size for evaluation
        use_alpha_beta: Whether to use alpha-beta pruning (not implemented in new version yet)
        temperature: Policy temperature for move selection (default 1.0)
        debug: Whether to enable debug logging
        return_tree: Whether to return the search tree for debugging
        verbose: Verbosity level (0: silent, 1+: show info logs)

    Returns:
        best_move: (row, col) tuple for the best move at the root
        value: estimated value for the root position
        root: search tree root node (if return_tree=True, otherwise None)
        stats: dictionary with search statistics
    """
    if debug:
        logger.setLevel(logging.DEBUG)
    elif verbose <= 1:
        logger.setLevel(logging.WARNING)  # Suppress INFO logs when not verbose
    else:
        logger.setLevel(logging.INFO)
    
    if verbose >= 2:
        logger.info(f"Starting batched minimax search with widths {widths}, temperature {temperature}")
        root_player_enum = state.current_player_enum
        logger.info(f"Root state: player {state.current_player} ({'Blue' if root_player_enum == 0 else 'Red'})")
    
    # Create configuration for the new implementation
    config = create_fixed_tree_config(
        search_widths=widths,
        temperature=temperature,
        batch_size=batch_size,
        enable_early_termination=True,
        early_termination_threshold=0.95
    )
    
    # Run the modernized search
    result = run_fixed_tree_search(state, model, config, verbose)
    
    if verbose >= 2:
        logger.info(f"Search complete: best move = {result.move}, value = {result.value}")
    
    # Convert result to old format
    stats = {
        'total_positions': result.stats.get('total_positions', 0),
        'total_evaluations': result.stats.get('total_evaluations', 0),
        'elapsed_time': result.stats.get('elapsed_time', 0),
        'early_terminations': result.stats.get('early_terminations', 0),
        'tree_depth': result.stats.get('tree_depth', 0),
        'tree_width': result.stats.get('tree_width', 0),
    }
    
    if return_tree:
        # For now, return None for the tree since the new implementation doesn't expose it
        # TODO: Add tree extraction to the new implementation if needed
        return result.move, result.value, None, stats
    else:
        return result.move, result.value, None, stats


# Import the old PositionCollector for backward compatibility
# This is deprecated and will be removed in a future version
try:
    from hex_ai.inference.fixed_tree_search import PositionCollector
    __all__ = ['minimax_policy_value_search', 'minimax_policy_value_search_with_batching', 'PositionCollector']
except ImportError:
    # If the old module is not available, just export the new functions
    __all__ = ['minimax_policy_value_search', 'minimax_policy_value_search_with_batching']
