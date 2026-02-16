# baseline_mcts.py
# Lean, single-threaded, explicitly-batched AlphaZero-style MCTS for Hex.
# Compatible with flat-file or package imports via shims.
#
# REFERENCE FRAME HANDLING:
# This MCTS implementation manages three distinct reference frames for signed values [-1,1]:
# 1. Neural network outputs: Always in Red's reference frame (red_ref_signed)
#    +1 = Red win, -1 = Blue win
# 2. Terminal nodes: Always in Red's reference frame (red_ref_signed)  
#    +1 = Red win, -1 = Blue win
# 3. MCTS tree values: Stored in player-to-move reference frame (ptm_ref_signed)
#    +1 = current player wins, -1 = current player loses
# 4. Final output: Converted to root player reference frame (root_ref_signed) for external API
#
# Notation: 
# - red_ref = Red's reference frame
# - ptm_ref = Player-to-move reference frame  
# - root_ref = Root player reference frame
# - signed = Values in [-1,1] range
# - prob = Values in [0,1] range (probabilities)
#
# Key variable naming convention:
# - v_red_ref_signed: Always in Red's reference frame (+1 = Red win, -1 = Blue win)
# - v_ptm_ref_signed: In player-to-move reference frame (+1 = current player wins, -1 = current player loses)
# - value_signed: From neural network cache, in Red's reference frame
# - p_red_prob: Red win probability in [0,1] range
# - p_ptm_prob: Player-to-move win probability in [0,1] range
#
# TODO: Future improvements to consider:
# - Add memory pooling for large tree structures to reduce allocation overhead
# - Implement cleanup of old cached evaluations to prevent memory leaks
# - Add support for parallel MCTS with proper synchronization
# - Add tree visualization utilities for debugging
# - Implement different tree policies (UCB1, etc.) as pluggable components
#
# TODO: Code Duplication Cleanup Opportunities (identified 2025-01-27):
# - Move selection logic may be duplicated across mcts.py, move_selection.py, web/app.py
# - Consider creating centralized utilities for these common patterns
# - Current mcts.py file is clean - no internal duplicates found

from __future__ import annotations

import math
import numpy as np
import torch
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Set
from collections import OrderedDict, deque

# ---- Package imports ----
from hex_ai.enums import Player, Winner
from hex_ai.value_utils import red_ref_signed_to_ptm_ref_signed, apply_depth_discount_signed, distance_to_leaf
from hex_ai.inference.mcts_utils import (
    compute_win_probability_from_tree_data,
    extract_principal_variation_from_tree,
    calculate_tree_statistics,
    format_mcts_tree_data_for_api,
    should_enable_detailed_exploration,
    add_detailed_exploration_to_tree_data,
    calculate_visit_count_probs,
    calculate_policy_probs,
    select_move_index
)
from hex_ai.inference.game_engine import HexGameState, HexGameEngine
from hex_ai.inference.model_wrapper import ModelWrapper
from hex_ai.utils.perf import PERF
from hex_ai.utils.math_utils import softmax_np
from hex_ai.utils.format_conversion import (
    rowcol_to_tensor_with_size as move_to_index,
    rowcol_to_trmph,
)
from hex_ai.utils.temperature import calculate_mcts_root_temperature
from hex_ai.utils.state_utils import board_key, validate_move_coordinates
from hex_ai.utils.legal_action_contracts import assert_actions_subset_of_legal
from hex_ai.utils.timing import MCTSTimingTracker
from hex_ai.inference.mcts_config import BaselineMCTSConfig, create_mcts_config
from hex_ai.inference.mcts_gumbel import MCTSGumbelMixin
from hex_ai.inference.mcts_support import (
    AlgorithmTerminationInfo,
    MCTSResult,
    MCTSStatsBuilder,
    TerminalMoveDetector,
    AlgorithmTerminationChecker,
)
from hex_ai.value_utils import winner_to_color

# ---- MCTS Constants ----
# Principal variation extraction limit
PRINCIPAL_VARIATION_MAX_LENGTH = 11

# PUCT calculation threshold for avoiding division by zero
PUCT_CALCULATION_THRESHOLD = 1e-9

# ---- MCTS Invariant Wrappers ----
def q_from_w_n(w: float, n: int) -> float:
    """
    Calculate Q-value (mean value) from accumulated value W and visit count N.
    
    Args:
        w: Accumulated value (W)
        n: Visit count (N), must be >= 1
        
    Returns:
        Mean value (Q = W/N)
        
    Raises:
        AssertionError: If n < 1 (invalid visit count)
    """
    assert n >= 1, f"Visit count must be >= 1, got {n}"
    return w / n

def increment_visit_count(n: int) -> int:
    """
    Increment visit count with validation.
    
    Args:
        n: Current visit count
        
    Returns:
        Incremented visit count (n + 1)
    """
    return n + 1

def add_to_accumulated_value(w: float, v: float) -> float:
    """
    Add value to accumulated value with validation.
    
    Args:
        w: Current accumulated value
        v: Value to add (should be in player-to-move reference frame, signed)
        
    Returns:
        New accumulated value (w + v)
    """
    # Validate that v is in reasonable range for signed values
    assert -1.1 <= v <= 1.1, f"Value should be in [-1,1] range, got {v}"
    return w + v

def safe_puct_denominator(n_sum: float) -> bool:
    """
    Check if PUCT denominator is safe (above threshold).
    
    Args:
        n_sum: Sum of visit counts
        
    Returns:
        True if n_sum > PUCT_CALCULATION_THRESHOLD, False otherwise
    """
    return n_sum > PUCT_CALCULATION_THRESHOLD


def board_size_from_state(state: HexGameState) -> int:
    """Extract and validate board size from a game state tensor."""
    if state is None:
        raise ValueError("State cannot be None")
    board_tensor = state.get_board_tensor()
    if not hasattr(board_tensor, "shape") or len(board_tensor.shape) < 2:
        raise ValueError(f"Invalid board tensor shape: {getattr(board_tensor, 'shape', None)}")

    board_rows = int(board_tensor.shape[-2])
    board_cols = int(board_tensor.shape[-1])
    if board_rows <= 0 or board_cols <= 0:
        raise ValueError(f"Board tensor dimensions must be positive, got {board_rows}x{board_cols}")
    if board_rows != board_cols:
        raise ValueError(f"Expected square board tensor, got {board_rows}x{board_cols}")
    return board_cols


# ------------------ Data structures ------------------

class MCTSNode:
    __slots__ = (
        "state", "to_play", "legal_moves", "legal_indices",
        "children", "N", "W", "Q", "P", "is_expanded",
        "board_size",
        "state_hash", "is_terminal", "winner", "winner_str", "terminal_moves",
        "_terminal_moves_detected", "depth"
    )
    def __init__(self, state: HexGameState, board_size: int):
        if state is None:
            raise ValueError("State cannot be None")
        if isinstance(board_size, bool):
            raise TypeError("Board size must be an integer, got bool")
        if board_size <= 0:
            raise ValueError(f"Board size must be positive, got {board_size}")

        state_board_size = board_size_from_state(state)
        if int(board_size) != state_board_size:
            raise ValueError(
                "Board size mismatch between node argument and state tensor: "
                f"{board_size} vs {state_board_size}"
            )
        board_size = state_board_size
        
        self.state: HexGameState = state
        self.board_size: int = board_size
        self.to_play: Player = state.current_player_enum
        # Legal moves
        self.legal_moves: List[Tuple[int,int]] = state.get_legal_moves()
        
        # Validate legal moves
        for row, col in self.legal_moves:
            validate_move_coordinates(row, col, board_size)
        
        self.legal_indices: List[int] = [move_to_index(r, c, board_size) for (r,c) in self.legal_moves]
        L = len(self.legal_moves)
        # Stats (aligned to legal_moves order)
        self.children: List[Optional[MCTSNode]] = [None] * L
        self.N = np.zeros(L, dtype=np.int32)   # visit counts per action
        self.W = np.zeros(L, dtype=np.float64) # accumulated values in ptm_ref_signed
        self.Q = np.zeros(L, dtype=np.float64) # mean values in ptm_ref_signed
        self.P = np.zeros(L, dtype=np.float64) # prior probability per action (set on expand)
        self.is_expanded: bool = False
        self.state_hash: int = board_key(state)
        self.is_terminal: bool = bool(state.game_over)
        self.winner: Optional[Winner] = state.winner if self.is_terminal else None
        self.winner_str: Optional[str] = winner_to_color(state.winner) if self.is_terminal else None
        self.terminal_moves: List[bool] = [False] * L # New attribute for terminal move detection
        self._terminal_moves_detected: bool = False  # Track if terminal moves have been detected
        self.depth: int = 0  # Track node depth in the tree

# ------------------ Core MCTS ------------------

class BaselineMCTS(MCTSGumbelMixin):
    def __init__(self, engine: HexGameEngine, model: ModelWrapper, cfg: BaselineMCTSConfig):
        if engine is None:
            raise ValueError("Engine cannot be None")
        if model is None:
            raise ValueError("Model cannot be None")
        if cfg is None:
            raise ValueError("Configuration cannot be None")
        
        self.engine = engine
        self.model = model
        self.cfg = cfg
        # Always use global RNG state - randomness should be controlled externally
        # This ensures MCTS instances don't interfere with each other's randomness

        # LRU Cache: board_key -> (policy_logits_np [A], value_signed_float)
        # Uses OrderedDict for O(1) LRU eviction
        # Note: value_signed is the tanh-activated output in [-1,1] range in Red's reference frame (red_ref_signed)
        self.eval_cache: OrderedDict[int, Tuple[np.ndarray, float]] = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        # Metrics
        self._unique_evals_total = 0     # post-dedup, network calls actually done
        self._effective_sims_total = 0   # counts every backprop (incl. duplicates)

        # Gumbel-specific performance counters
        self._gumbel_nn_calls_per_move = 0
        self._gumbel_total_leaves_evaluated = 0
        self._gumbel_distinct_leaves_evaluated = 0
        self._gumbel_candidates_m = 0
        self._gumbel_rounds_R = 0

        # Terminal move detection
        self.terminal_detector = TerminalMoveDetector(
            max_detection_depth=cfg.terminal_detection_max_depth
        )
        
        # Algorithm termination checker
        self.algorithm_termination_checker = AlgorithmTerminationChecker(cfg, self.terminal_detector)
        
        # Stats builder (will be updated with current cache counts when needed)
        self.stats_builder = None
        
        # Detailed exploration tracking for small simulation counts
        self.detailed_exploration_enabled = False
        self.exploration_trace = []
        self.simulation_count = 0

    def _enable_detailed_exploration_if_needed(self, num_simulations: int) -> None:
        """Enable detailed exploration tracking if simulation count is below threshold."""
        verbose_level = int(getattr(self, "verbose", 0))
        self.detailed_exploration_enabled = (
            should_enable_detailed_exploration(num_simulations)
            and verbose_level >= 2
        )
        self.exploration_trace = []
        self.simulation_count = 0
        if self.detailed_exploration_enabled and verbose_level >= 2:
            print(f"🔍 Enabling detailed MCTS exploration tracking for {num_simulations} simulations")

    def _record_descent_start(self, sim: int, root_visits: int, gumbel_forced: bool, pv_hint: Optional[List[str]] = None) -> None:
        """Record the start of a descent."""
        if not self.detailed_exploration_enabled:
            return
        
        event = {
            'type': 'descent_start',
            'sim': int(sim),
            'root_visits': int(root_visits),
            'gumbel_forced': bool(gumbel_forced)
        }
        if pv_hint:
            event['pv_hint'] = pv_hint
        
        self.exploration_trace.append(event)

    def _record_forced_root_action(self, sim: int, tensor_action: int, legal_at_root: bool) -> None:
        """Record when Gumbel forces a root action."""
        if not self.detailed_exploration_enabled:
            return
        
        event = {
            'type': 'forced_root_action',
            'sim': int(sim),
            'tensor_action': int(tensor_action),
            'legal_at_root': bool(legal_at_root)
        }
        
        if not legal_at_root:
            event['note'] = 'masked_illegal'
        
        self.exploration_trace.append(event)

    def _record_select_action(self, depth: int, n_total: float, q: float, p: float, n: int, u: float, 
                            score: float, terminal_flag_for_child: bool = False, note: Optional[str] = None) -> None:
        """Record a PUCT action selection."""
        if not self.detailed_exploration_enabled:
            return
        
        event = {
            'type': 'select_action',
            'depth': int(depth),
            'n_total': float(n_total),
            'q': float(q),
            'p': float(p),
            'n': int(n),
            'u': float(u),
            'score': float(score),
            'terminal_flag_for_child': bool(terminal_flag_for_child)
        }
        
        if note:
            event['note'] = str(note)
        
        self.exploration_trace.append(event)

    def _record_node_realized(self, depth: int, move: str, child_hash: int) -> None:
        """Record when a child node is first created."""
        if not self.detailed_exploration_enabled:
            return
        
        event = {
            'type': 'node_realized',
            'depth': int(depth),
            'move': str(move),
            'child_hash': int(child_hash)
        }
        
        self.exploration_trace.append(event)

    def _record_leaf_selected(self, depth: int, node_hash: int, leaf_reason: str, T: int, U: int, distinct_target: int) -> None:
        """Record when a leaf is selected during descent."""
        if not self.detailed_exploration_enabled:
            return
        
        event = {
            'type': 'leaf_selected',
            'depth': int(depth),
            'node_hash': int(node_hash),
            'leaf_reason': str(leaf_reason),
            'T': int(T),
            'U': int(U),
            'distinct_target': int(distinct_target)
        }
        
        self.exploration_trace.append(event)

    def _record_batch_flush(self, reason: str, T: int, U: int, distinct_target: int, select_budget: int) -> None:
        """Record when a batch is flushed early."""
        if not self.detailed_exploration_enabled:
            return
        
        event = {
            'type': 'batch_flush',
            'reason': str(reason),
            'T': int(T),
            'U': int(U),
            'distinct_target': int(distinct_target),
            'select_budget': int(select_budget)
        }
        
        self.exploration_trace.append(event)

    def _record_nn_eval_start(self, batch_size: int, to_eval: int, distinct: int, cache_hits_in_batch: int) -> None:
        """Record when neural network evaluation starts."""
        if not self.detailed_exploration_enabled:
            return
        
        event = {
            'type': 'nn_eval_start',
            'batch_size': int(batch_size),
            'to_eval': int(to_eval),
            'distinct': int(distinct),
            'cache_hits_in_batch': int(cache_hits_in_batch)
        }
        
        self.exploration_trace.append(event)

    def _record_nn_eval_done(self, effective_batch_size: int, value_range: List[float], 
                            mean_policy_entropy: float, time_ms: float) -> None:
        """Record when neural network evaluation completes."""
        if not self.detailed_exploration_enabled:
            return
        
        event = {
            'type': 'nn_eval_done',
            'effective_batch_size': int(effective_batch_size),
            'value_range': [float(value_range[0]), float(value_range[1])],
            'mean_policy_entropy': float(mean_policy_entropy),
            'time_ms': float(time_ms)
        }
        
        self.exploration_trace.append(event)

    def _record_expand_node(self, depth: int, node_hash: int, children_count: int, 
                           prior_mass_top3: float, value_signed_red_ref: float) -> None:
        """Record when a node is expanded."""
        if not self.detailed_exploration_enabled:
            return
        
        event = {
            'type': 'expand_node',
            'depth': int(depth),
            'node_hash': int(node_hash),
            'children_count': int(children_count),
            'prior_mass_top3': float(prior_mass_top3),
            'value_signed_red_ref': float(value_signed_red_ref)
        }
        
        self.exploration_trace.append(event)

    def _record_backprop_update(self, path_len: int, root_q_before: float, root_q_after: float) -> None:
        """Record when backpropagation updates the root."""
        if not self.detailed_exploration_enabled:
            return
        
        event = {
            'type': 'backprop_update',
            'path_len': int(path_len),
            'root_q_before': float(root_q_before),
            'root_q_after': float(root_q_after)
        }
        
        self.exploration_trace.append(event)

    # ---------- Public API ----------
    
    def run(self, root_state: HexGameState, verbose: int = 0) -> MCTSResult:
        """
        Run MCTS for cfg.sims simulations starting from root_state.
        Returns a complete MCTSResult with move, stats, and analysis data.
        
        Args:
            root_state: The game state to search from
            verbose: Verbosity level for logging (0=quiet, 1=basic, 2=detailed)
            
        Raises:
            ValueError: If root_state is None or invalid
            RuntimeError: If the game state is terminal
        """
        self._validate_run_inputs(root_state, verbose)
        self._reset_run_state(verbose)

        root = self._prepare_root_node(root_state, verbose, expand_root=False)

        maybe_terminated = self._termination_result_if_any(root, verbose)
        if maybe_terminated is not None:
            return maybe_terminated

        self._expand_root_for_search(root, root_state)

        maybe_terminated = self._termination_result_if_any(root, verbose)
        if maybe_terminated is not None:
            return maybe_terminated

        timing_stats = self._run_simulation_loop(root, verbose)
        self._annotate_search_timing_stats(timing_stats)
        return self._build_completed_search_result(root, root_state, timing_stats, verbose)

    # ---------- Data Access (Getters) ----------
    
    def get_tree_data(self, root: MCTSNode, move_probs: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Get formatted tree data for API consumption.
        
        Args:
            root: Root node of the MCTS tree
            move_probs: Move probabilities for all legal moves (policy or temperature-scaled)
        
        Returns:
            Dictionary containing formatted tree data for API consumption
        """
        tree_data = format_mcts_tree_data_for_api(root, self.cache_misses, PRINCIPAL_VARIATION_MAX_LENGTH, move_probs)
        
        # Add detailed exploration data if available
        tree_data = add_detailed_exploration_to_tree_data(
            tree_data, 
            self.detailed_exploration_enabled, 
            self.exploration_trace, 
            self.simulation_count
        )
        
        return tree_data

    def get_win_probability(self, root: MCTSNode, root_state: HexGameState) -> float:
        """Get win probability for the current player."""
        tree_data = self.get_tree_data(root)
        return compute_win_probability_from_tree_data(tree_data)

    def get_principal_variation(self, root: MCTSNode, max_length: int = 12) -> List[Tuple[int, int]]:
        """Get the principal variation (best move sequence) from the MCTS tree."""
        return extract_principal_variation_from_tree(root, max_length)

    def get_tree_statistics(self, root: MCTSNode) -> Tuple[int, int]:
        """Get tree traversal statistics."""
        return calculate_tree_statistics(root)

    def _prepare_root_node(self, root_state: HexGameState, verbose: int, expand_root: bool = True) -> MCTSNode:
        """Prepare and initialize the root node for MCTS search."""
        board_size = board_size_from_state(root_state)
        root = MCTSNode(root_state, board_size)
        
        # Expand root if not terminal
        if expand_root and not root.is_terminal and not root.is_expanded:
            self._expand_root_node(root, board_size)
        
        # Apply root noise if configured (every move for standard AlphaZero behavior)
        if expand_root and self.cfg.add_root_noise and not root.is_terminal and root.is_expanded:
            self._apply_root_noise(root)
        
        return root

    def _expand_root_node(self, root: MCTSNode, board_size: int):
        """Expand the root node using neural network evaluation."""
        action_size = board_size * board_size
        
        # Try cache first
        cached = self._get_from_cache(root.state_hash)
        if cached is not None:
            self.cache_hits += 1
            policy_np, value_signed = cached
            self._expand_node_from_policy(root, policy_np, board_size, action_size)
        else:
            self.cache_misses += 1
            # Evaluate root in micro-batch of size 1
            root_enc = root.state.get_board_tensor().to(dtype=torch.float32)
            batch = torch.stack([root_enc], dim=0)
            
            policy_cpu, value_cpu, _ = self.model.infer_timed(batch)
            policy_np = policy_cpu[0].numpy()
            value_signed = float(value_cpu[0].item())  # tanh-activated output in [-1,1] range
            
            # Cache results
            self._put_in_cache(root.state_hash, policy_np, value_signed)
            self._expand_node_from_policy(root, policy_np, board_size, action_size)

    def _check_algorithm_termination(self, root: MCTSNode, verbose: int) -> Optional[AlgorithmTerminationInfo]:
        """Check if MCTS should terminate early."""
        # Single call with root_is_expanded flag - handles both terminal moves and confidence checking
        termination_info = self.algorithm_termination_checker.should_terminate_early(
            root, verbose, self.eval_cache, root_is_expanded=root.is_expanded
        )
        
        return termination_info

    def _build_algorithm_termination_result(
        self,
        root: MCTSNode,
        termination_info: AlgorithmTerminationInfo,
        verbose: int
    ) -> MCTSResult:
        """Build a complete MCTSResult for algorithm-termination exits."""
        move = self._get_algorithm_termination_move(root, termination_info, verbose)
        tree_data = self.get_tree_data(root)
        win_probability = termination_info.win_probability

        # Attach metrics for algorithm termination cases too.
        stats = self._get_stats_builder().create_algorithm_termination_stats(termination_info)
        stats["unique_evals_total"] = int(self._unique_evals_total)
        stats["effective_sims_total"] = int(self._effective_sims_total)
        stats["unique_evals_per_sec"] = 0.0
        stats["effective_sims_per_sec"] = 0.0

        return MCTSResult(
            move=move,
            stats=stats,
            tree_data=tree_data,
            root_node=root,
            algorithm_termination_info=termination_info,
            win_probability=win_probability
        )

    def _run_simulation_loop(self, root: MCTSNode, verbose: int) -> Dict[str, Any]:
        """Run the main MCTS simulation loop with batching."""
        timing_tracker = MCTSTimingTracker()
        sims_remaining = self.cfg.sims

        if self._should_use_gumbel_root_selection(root, sims_remaining):
            if verbose >= 5:
                print(f"Using Gumbel-AlphaZero root selection for {sims_remaining} simulations")
            return self._run_gumbel_root_selection(root, sims_remaining, timing_tracker, verbose)

        self._run_standard_simulation_loop(root, sims_remaining, timing_tracker)
        return timing_tracker.get_final_stats()

    def _should_use_gumbel_root_selection(self, root: MCTSNode, sims_remaining: int) -> bool:
        """Return whether search should route through Gumbel-AlphaZero root selection."""
        return (
            self.cfg.enable_gumbel_root_selection
            and sims_remaining <= self.cfg.gumbel_sim_threshold
            and root.is_expanded
            and not root.is_terminal
        )

    def _run_standard_simulation_loop(
        self,
        root: MCTSNode,
        sims_remaining: int,
        timing_tracker: MCTSTimingTracker,
    ) -> None:
        """Execute the default batched select/eval/backprop loop."""
        while sims_remaining > 0:
            leaves, paths = self._select_leaves_batch(root, sims_remaining, timing_tracker)
            batch_simulations = self._process_leaves_batch(leaves, paths, timing_tracker, root)
            if batch_simulations <= 0:
                raise ValueError(
                    "Zero-progress standard simulation batch in _run_standard_simulation_loop. "
                    f"sims_remaining={sims_remaining}, batch_simulations={batch_simulations}, "
                    f"selected_leaves={len(leaves)}, selected_paths={len(paths)}"
                )
            if batch_simulations > sims_remaining:
                raise ValueError(
                    "Over-progress standard simulation batch in _run_standard_simulation_loop. "
                    f"sims_remaining={sims_remaining}, batch_simulations={batch_simulations}, "
                    f"selected_leaves={len(leaves)}, selected_paths={len(paths)}"
                )
            sims_remaining -= batch_simulations
            self._effective_sims_total += batch_simulations

    def _validate_run_inputs(self, root_state: HexGameState, verbose: int) -> None:
        """Validate top-level MCTS run arguments."""
        if root_state is None:
            raise ValueError("root_state cannot be None")
        if root_state.game_over:
            raise RuntimeError("Cannot run MCTS on a terminal game state")
        if verbose < 0:
            raise ValueError(f"verbose must be non-negative, got {verbose}")

    def _reset_run_state(self, verbose: int) -> None:
        """Reset per-run state (especially Gumbel diagnostics) before search."""
        # This matters in interactive settings (web UI) where one BaselineMCTS instance may be reused.
        self.verbose = int(verbose)
        self._used_gumbel_root_selection = False
        self._gumbel_selected_action = None
        self._gumbel_selected_tensor_action = None
        self._gumbel_final_rank_top_move_trmph = None
        self._gumbel_final_rank_top5 = None
        self._gumbel_v_pi_01 = None
        self._gumbel_nn_calls_per_move = 0
        self._gumbel_total_leaves_evaluated = 0
        self._gumbel_distinct_leaves_evaluated = 0
        self._gumbel_candidates_m = 0
        self._gumbel_rounds_R = 0
        self._gumbel_timing_breakdown = {}
        self._enable_detailed_exploration_if_needed(self.cfg.sims)

    def _termination_result_if_any(self, root: MCTSNode, verbose: int) -> Optional[MCTSResult]:
        """Run algorithm-termination checks and build a result if search should stop."""
        termination_info = self._check_algorithm_termination(root, verbose)
        if termination_info is None:
            return None
        return self._build_algorithm_termination_result(root, termination_info, verbose)

    def _expand_root_for_search(self, root: MCTSNode, root_state: HexGameState) -> None:
        """Expand root (and apply root noise) for full search when needed."""
        board_size = board_size_from_state(root_state)
        if board_size != root.board_size:
            raise ValueError(
                "Root board size mismatch between root node and root_state: "
                f"{root.board_size} vs {board_size}"
            )
        if not root.is_terminal and not root.is_expanded:
            self._expand_root_node(root, board_size)
        if self.cfg.add_root_noise and not root.is_terminal and root.is_expanded:
            self._apply_root_noise(root)

    def _annotate_search_timing_stats(self, timing_stats: Dict[str, Any]) -> None:
        """Attach aggregate search metrics derived from timing data."""
        total_time = float(timing_stats.get("total_search_time", 0.0)) or 1e-9
        timing_stats["unique_evals_total"] = int(self._unique_evals_total)
        timing_stats["effective_sims_total"] = int(self._effective_sims_total)
        timing_stats["unique_evals_per_sec"] = self._unique_evals_total / total_time
        timing_stats["effective_sims_per_sec"] = self._effective_sims_total / total_time

    def _build_completed_search_result(
        self,
        root: MCTSNode,
        root_state: HexGameState,
        timing_stats: Dict[str, Any],
        verbose: int
    ) -> MCTSResult:
        """Build final result payload for a completed non-terminated MCTS search."""
        move, move_probs = self._compute_move(root, root_state, verbose)
        tree_data = self.get_tree_data(root, move_probs)
        win_probability = compute_win_probability_from_tree_data(tree_data)

        stats = self._get_stats_builder().create_final_stats(
            timing_stats, self.cfg.sims, timing_stats.get("total_search_time", 0.0)
        )

        if getattr(self, "_used_gumbel_root_selection", False):
            # Use actual MCTS metrics for distinct leaves evaluation.
            actual_distinct_leaves = timing_stats.get("unique_evals_total", 0)
            stats.update({
                "gumbel_nn_calls_per_move": self._gumbel_nn_calls_per_move,
                "gumbel_total_leaves_evaluated": self._gumbel_total_leaves_evaluated,
                "gumbel_distinct_leaves_evaluated": actual_distinct_leaves,
                "gumbel_candidates_m": self._gumbel_candidates_m,
                "gumbel_rounds_R": self._gumbel_rounds_R,
                "gumbel_avg_nn_batch_size": self._gumbel_total_leaves_evaluated / max(1, self._gumbel_nn_calls_per_move),
                "gumbel_leaves_distinct_ratio": actual_distinct_leaves / max(1, self._gumbel_total_leaves_evaluated),
                "gumbel_timing_breakdown": getattr(self, "_gumbel_timing_breakdown", {}),
                # Debug/inspection fields (small and safe to serialize).
                "gumbel_selected_tensor_action": self._gumbel_selected_tensor_action,
                "gumbel_final_rank_top_move": self._gumbel_final_rank_top_move_trmph,
                "gumbel_v_pi_01": self._gumbel_v_pi_01,
                "gumbel_final_rank_top5": self._gumbel_final_rank_top5,
            })

        return MCTSResult(
            move=move,
            stats=stats,
            tree_data=tree_data,
            root_node=root,
            algorithm_termination_info=None,
            win_probability=win_probability
        )

    def _root_temperature(self, move_idx: int, board_size: int) -> float:
        """
        Compute root temperature for visit-count move selection.

        This temperature is used when selecting from root visit counts in the
        non-Gumbel path. Gumbel root selection is validated separately and uses
        a fixed temperature contract.
        
        Args:
            move_idx: Current move index (0-based)
            
        Returns:
            Temperature value for this move
        """
        return calculate_mcts_root_temperature(move_count=move_idx, cfg=self.cfg, board_size=board_size)

    def _get_policy_logits_and_legal_mask(self, root_state: HexGameState, legal_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get policy logits and legal mask for a given state.
        
        This is a shared utility used by both Gumbel selection and policy probability calculation.
        
        Args:
            root_state: Game state to get policy for
            legal_indices: List of legal action indices
            
        Returns:
            Tuple of (policy_logits_full, legal_mask)
        """
        board_size = board_size_from_state(root_state)
        action_size = board_size * board_size
        
        # Create legal mask
        legal_mask = np.zeros(action_size, dtype=bool)
        legal_mask[legal_indices] = True
        
        # Get policy logits from cache or re-evaluate
        cached = self._get_from_cache(board_key(root_state))
        if cached is not None:
            policy_logits_full, _ = cached
        else:
            # Re-evaluate to get full tensor logits
            root_enc = root_state.get_board_tensor().to(dtype=torch.float32)
            batch = torch.stack([root_enc], dim=0)
            policy_cpu, _, _ = self.model.infer_timed(batch)
            policy_logits_full = policy_cpu[0].numpy()
        
        return policy_logits_full, legal_mask

    def _root_priors_from_logits(
        self,
        policy_logits_full: np.ndarray,
        legal_mask: np.ndarray,
        apply_dirichlet: bool,
    ) -> np.ndarray:
        """
        Get root priors from logits with optional Dirichlet noise.
        
        Args:
            policy_logits_full: Full policy logits [K] (illegal actions should be -inf)
            legal_mask: Boolean mask over full action space
            apply_dirichlet: Whether to apply Dirichlet noise (True for self-play)
            
        Returns:
            Prior probabilities [K] with optional Dirichlet noise
        """
        # 1) Mask logits -> probabilities
        # Apply softmax to masked logits
        masked_logits = np.where(legal_mask, policy_logits_full, -np.inf)
        probs = softmax_np(masked_logits)
        
        # 2) Optional Dirichlet mix (standard AlphaZero)
        if apply_dirichlet and self.cfg.add_root_noise:
            L = int(legal_mask.sum())
            noise = np.random.dirichlet([self.cfg.dirichlet_alpha] * L)
            probs_legal = probs[legal_mask]
            probs_legal = (1.0 - self.cfg.dirichlet_eps) * probs_legal + self.cfg.dirichlet_eps * noise
            # Replace the legal slice with mixed probabilities
            probs = probs.copy()
            probs[legal_mask] = probs_legal
        
        return probs

    def _compute_leaf_batch_targets(
        self,
        root: MCTSNode,
        sims_remaining: int,
        forced_root_actions: Optional[List[int]]
    ) -> Tuple[int, Deque[int], int, int, bool]:
        """Compute selection budget and batch targets for one leaf-selection pass."""
        board_size = root.board_size
        legal_count = len(root.legal_moves)
        root_total_N = int(np.sum(root.N))

        # Early-phase smaller batches (until a few backprops happen).
        warmup_cap = 16
        is_early = root_total_N < 64
        general_select_budget = min(self.cfg.batch_cap, sims_remaining)
        effective_select_budget = min(general_select_budget, warmup_cap) if is_early else general_select_budget

        # Never try to collect more distinct leaves than legal root moves or sims left.
        effective_distinct_target = min(
            int(self.cfg.distinct_target),
            effective_select_budget,
            legal_count,
            max(1, sims_remaining)
        )

        if forced_root_actions is not None:
            force_q: Deque[int] = deque(forced_root_actions)
            select_budget = min(self.cfg.batch_cap, len(forced_root_actions))
        else:
            force_q = deque()
            select_budget = int(effective_select_budget)

        if self.cfg.adaptive_distinct_target:
            # Encourage earlier backprops at low sims; keep batches tidy.
            guess = max(1, sims_remaining // 8)
            distinct_target = int(max(self.cfg.distinct_target_min,
                                      min(self.cfg.distinct_target_max, guess)))
        else:
            distinct_target = max(1, int(effective_distinct_target))

        use_root_reservation = (root_total_N < 64)
        return board_size, force_q, select_budget, distinct_target, use_root_reservation

    def _should_flush_selected_batch(self, total_leaves: int, distinct_count: int, distinct_target: int) -> Optional[str]:
        """Return flush reason when a batch should be processed immediately."""
        if distinct_count >= distinct_target:
            return "distinct_target_reached"
        if self.cfg.enable_low_distinct_ratio_flush and total_leaves >= 16 and distinct_count / max(1, total_leaves) < 0.5:
            return "low_distinct_ratio"
        return None

    def _resolve_child_index_for_descent(
        self,
        node: MCTSNode,
        root: MCTSNode,
        forced_a_full: Optional[int],
        use_root_reservation: bool,
        used_root_actions: Set[int],
        root_legal_action_to_local_idx: Optional[Dict[int, int]] = None,
        root_legal_set: Optional[Set[int]] = None,
    ) -> int:
        """Select child index for one descent step, handling forced root actions and reservations."""
        if node is root and forced_a_full is not None:
            forced_action = int(forced_a_full)
            if root_legal_action_to_local_idx is not None and root_legal_set is not None:
                if forced_action not in root_legal_set:
                    raise ValueError(
                        "Forced-root legality contract violated at _resolve_child_index_for_descent. "
                        f"Illegal forced action {forced_action}. "
                        f"Current root legal_indices ({len(node.legal_indices)}): {node.legal_indices}"
                    )
                return root_legal_action_to_local_idx[forced_action]

            # Safety fallback for non-forced paths that do not precompute root lookup tables.
            if forced_action not in node.legal_indices:
                raise ValueError(
                    "Forced-root legality contract violated at _resolve_child_index_for_descent. "
                    f"Illegal forced action {forced_action}. "
                    f"Current root legal_indices ({len(node.legal_indices)}): {node.legal_indices}"
                )
            return node.legal_indices.index(forced_action)

        if node is root and use_root_reservation:
            return self._select_child_puct(node, node.depth, used_root_actions)
        return self._select_child_puct(node, node.depth)

    def _realize_child_node_if_needed(
        self,
        node: MCTSNode,
        loc_idx: int,
        board_size: int,
        timing_tracker: MCTSTimingTracker,
    ) -> MCTSNode:
        """Create and attach child node if it does not exist yet."""
        child = node.children[loc_idx]
        if child is not None:
            return child

        timing_tracker.start_timing("state_creation")
        (r, c) = node.legal_moves[loc_idx]
        timing_tracker.start_timing("make_move")
        child_state = node.state.make_move(r, c)
        timing_tracker.end_timing("make_move")
        child = MCTSNode(child_state, board_size)
        child.depth = node.depth + 1
        timing_tracker.end_timing("state_creation")
        node.children[loc_idx] = child

        if self.detailed_exploration_enabled:
            move_str = rowcol_to_trmph(r, c, board_size)
            self._record_node_realized(child.depth, move_str, child.state_hash)
        return child

    def _build_root_pv_hint(self, root: MCTSNode) -> Optional[List[str]]:
        """Build a short principal-variation hint for detailed exploration traces."""
        if not root.is_expanded or len(root.children) == 0:
            return None

        board_size = root.board_size
        pv_moves: List[str] = []
        current = root
        for _ in range(3):
            if not current.is_expanded or len(current.children) == 0:
                break
            best_child_idx = int(np.argmax(current.N))
            if best_child_idx >= len(current.legal_moves):
                break
            r, c = current.legal_moves[best_child_idx]
            pv_moves.append(rowcol_to_trmph(r, c, board_size))
            current = current.children[best_child_idx]
            if current is None:
                break
        return pv_moves if pv_moves else None

    def _record_forced_root_action_if_needed(self, root: MCTSNode, forced_a_full: Optional[int]) -> None:
        """Record forced-root selection diagnostics for detailed exploration traces."""
        if not self.detailed_exploration_enabled or forced_a_full is None:
            return
        legal_at_root = forced_a_full in root.legal_indices
        self._record_forced_root_action(self.simulation_count, forced_a_full, legal_at_root)

    def _try_collect_leaf(
        self,
        node: MCTSNode,
        path: List[Tuple[MCTSNode, int]],
        leaves: List[MCTSNode],
        paths: List[List[Tuple[MCTSNode, int]]],
        distinct_hashes: Set[int],
        distinct_target: int,
    ) -> Tuple[bool, Optional[str]]:
        """
        Try to collect `node` as a leaf.

        Returns:
            Tuple of (leaf_was_collected, flush_reason_if_any).
        """
        if node.is_terminal:
            leaves.append(node)
            paths.append(path.copy())
            if self.detailed_exploration_enabled:
                self._record_leaf_selected(
                    node.depth, node.state_hash, "terminal",
                    len(leaves), len(distinct_hashes), distinct_target
                )
            return True, None

        if not node.is_expanded:
            leaves.append(node)
            paths.append(path)
            if node.state_hash not in self.eval_cache and not node.is_terminal:
                distinct_hashes.add(node.state_hash)

            if self.detailed_exploration_enabled:
                self._record_leaf_selected(
                    node.depth, node.state_hash, "unexpanded",
                    len(leaves), len(distinct_hashes), distinct_target
                )

            flush_reason = self._should_flush_selected_batch(len(leaves), len(distinct_hashes), distinct_target)
            return True, flush_reason

        return False, None

    def _record_descent_start_if_needed(
        self,
        root: MCTSNode,
        forced_root_actions: Optional[List[int]]
    ) -> None:
        """Record per-descent summary diagnostics when detailed exploration is enabled."""
        if not self.detailed_exploration_enabled:
            return
        root_visits = int(np.sum(root.N)) if root.N is not None else 0
        gumbel_forced = forced_root_actions is not None and len(forced_root_actions) > 0
        pv_hint = self._build_root_pv_hint(root)
        self._record_descent_start(self.simulation_count, root_visits, gumbel_forced, pv_hint)

    def _perform_leaf_selection_descent(
        self,
        root: MCTSNode,
        board_size: int,
        distinct_target: int,
        use_root_reservation: bool,
        used_root_actions: Set[int],
        force_q: Deque[int],
        leaves: List[MCTSNode],
        paths: List[List[Tuple[MCTSNode, int]]],
        distinct_hashes: Set[int],
        timing_tracker: MCTSTimingTracker,
        root_legal_action_to_local_idx: Optional[Dict[int, int]],
        root_legal_set: Optional[Set[int]],
    ) -> Optional[str]:
        """
        Execute a single root-to-leaf descent and collect one leaf when found.

        Returns:
            Flush reason when selection should end immediately, otherwise None.
        """
        node = root
        path: List[Tuple[MCTSNode, int]] = []

        # Pop forced root action for this descent when in Gumbel forced mode.
        forced_a_full = force_q.popleft() if force_q else None
        self._record_forced_root_action_if_needed(node, forced_a_full)

        while True:
            leaf_collected, flush_reason = self._try_collect_leaf(
                node, path, leaves, paths, distinct_hashes, distinct_target
            )
            if leaf_collected:
                return flush_reason

            loc_idx = self._resolve_child_index_for_descent(
                node,
                root,
                forced_a_full,
                use_root_reservation,
                used_root_actions,
                root_legal_action_to_local_idx,
                root_legal_set,
            )
            path.append((node, loc_idx))

            if node is root and use_root_reservation:
                used_root_actions.add(loc_idx)

            node = self._realize_child_node_if_needed(node, loc_idx, board_size, timing_tracker)


    def _select_leaves_batch(
        self,
        root: MCTSNode,
        sims_remaining: int,
        timing_tracker: MCTSTimingTracker,
        forced_root_actions: Optional[List[int]] = None
    ) -> Tuple[List[MCTSNode], List[List[Tuple[MCTSNode, int]]]]:
        """
        Select a batch of leaves for expansion with consistent batch flushing behavior.
        
        Batch flushing strategy:
        - Collect leaves until we have distinct_target distinct (uncached) leaves
        - This ensures consistent neural network batch sizes for optimal GPU utilization
        - Low distinct ratio flush is disabled by default to prevent performance drops
        - Fixed distinct_target (32) provides consistent behavior across simulation counts
        
        Root reservation strategy:
        - During early phase (first ~64 root visits), prevent overexploration of top policy moves
        - Use batch-local reservation to spread first batch across top-K root actions
        - This provides earlier feedback and more balanced tree growth
        """
        timing_tracker.start_timing("select")

        leaves: List[MCTSNode] = []
        paths: List[List[Tuple[MCTSNode, int]]] = []

        board_size, force_q, select_budget, distinct_target, use_root_reservation = self._compute_leaf_batch_targets(
            root, sims_remaining, forced_root_actions
        )

        # Track distinct (uncached+unexpanded) leaf hashes this batch
        distinct_hashes: Set[int] = set()

        # Root-only batch-local "reservation" for early phase
        # This prevents overexploration of top policy moves before any backpropagations occur
        used_root_actions: Set[int] = set()  # tracks root actions used in this batch
        root_legal_action_to_local_idx: Optional[Dict[int, int]] = None
        root_legal_set: Optional[Set[int]] = None
        if forced_root_actions is not None:
            root_legal_set = set(root.legal_indices)
            if len(root_legal_set) != len(root.legal_indices):
                raise ValueError(
                    "Forced-root legality contract violated at _select_leaves_batch. "
                    f"Duplicate entries in root legal_indices ({len(root.legal_indices)}): {root.legal_indices}"
                )
            root_legal_action_to_local_idx = {
                int(action): idx for idx, action in enumerate(root.legal_indices)
            }

        # Cheap guardrail on selection work
        max_selection_descents = max(select_budget * 4, 64)  # 4x is a good default
        descents = 0

        while len(leaves) < select_budget and descents < max_selection_descents:
            descents += 1
            flush_reason = self._perform_leaf_selection_descent(
                root=root,
                board_size=board_size,
                distinct_target=distinct_target,
                use_root_reservation=use_root_reservation,
                used_root_actions=used_root_actions,
                force_q=force_q,
                leaves=leaves,
                paths=paths,
                distinct_hashes=distinct_hashes,
                timing_tracker=timing_tracker,
                root_legal_action_to_local_idx=root_legal_action_to_local_idx,
                root_legal_set=root_legal_set,
            )
            # Forced-root execution must honor explicit action budgets exactly.
            # Do not early-flush on distinct-target heuristics in that mode.
            if flush_reason is not None and forced_root_actions is None:
                if self.detailed_exploration_enabled:
                    T = len(leaves)
                    U = len(distinct_hashes)
                    self._record_batch_flush(flush_reason, T, U, distinct_target, select_budget)
                timing_tracker.end_timing("select")
                return leaves, paths

            self._record_descent_start_if_needed(root, forced_root_actions)
            # Outer budget guard (kept from original)
            if len(leaves) >= select_budget:
                break

        # Record budget_full batch flush for detailed exploration
        if self.detailed_exploration_enabled:
            self._record_batch_flush("budget_full", len(leaves), len(distinct_hashes), distinct_target, select_budget)
        
        timing_tracker.end_timing("select")
        return leaves, paths

    def _run_forced_root_batch(self, root: MCTSNode, actions: List[int], timing_tracker: MCTSTimingTracker) -> int:
        """Run exactly len(actions) simulations, forcing each root action once, using the batched pipeline."""
        expected_simulations = len(actions)
        if expected_simulations == 0:
            return 0

        leaves, paths = self._select_leaves_batch(root, sims_remaining=len(actions),
                                                  timing_tracker=timing_tracker,
                                                  forced_root_actions=actions)
        if len(leaves) != expected_simulations:
            raise ValueError(
                "Forced-root simulation contract violated at _run_forced_root_batch. "
                f"Requested {expected_simulations} forced actions but selected {len(leaves)} leaves."
            )

        simulations_completed = self._process_leaves_batch(leaves, paths, timing_tracker, root)
        if simulations_completed != expected_simulations:
            raise ValueError(
                "Forced-root simulation contract violated at _run_forced_root_batch. "
                f"Requested {expected_simulations} forced actions but completed {simulations_completed} simulations."
            )
        return simulations_completed

    def run_forced_root_actions(self, root: MCTSNode, actions: List[int], verbose: int = 0) -> Dict[str, Any]:
        """Public entry-point used by Gumbel root coordinator; respects batch_cap internally."""
        normalized_actions = assert_actions_subset_of_legal(
            actions=actions,
            legal_actions=root.legal_indices,
            context="run_forced_root_actions:entry",
            contract_name="Forced-root legality contract",
            actions_label="Forced actions",
            legal_label="Current root legal_indices",
        )

        timing_tracker = MCTSTimingTracker()
        i = 0
        simulations_completed_total = 0
        while i < len(normalized_actions):
            j = min(i + self.cfg.batch_cap, len(normalized_actions))
            batch_actions = normalized_actions[i:j]
            sims_done = self._run_forced_root_batch(root, batch_actions, timing_tracker)
            expected_batch = len(batch_actions)
            if sims_done != expected_batch:
                raise ValueError(
                    "Forced-root simulation contract violated at run_forced_root_actions. "
                    f"Requested {expected_batch} actions in batch [{i}:{j}] but completed {sims_done} simulations."
                )
            self._effective_sims_total += sims_done
            simulations_completed_total += sims_done
            i = j
        expected_total = len(normalized_actions)
        if simulations_completed_total != expected_total:
            raise ValueError(
                "Forced-root simulation contract violated at run_forced_root_actions. "
                f"Requested {expected_total} total actions but completed {simulations_completed_total} simulations."
            )

        timing_stats = timing_tracker.get_final_stats()
        timing_stats["simulations_completed"] = int(simulations_completed_total)
        timing_stats["simulations_requested"] = int(expected_total)
        return timing_stats

    def _process_leaves_batch(self, leaves: List[MCTSNode], paths: List[List[Tuple[MCTSNode, int]]], 
                            timing_tracker: MCTSTimingTracker, root: MCTSNode) -> int:
        """Process a batch of leaves: expand and backpropagate."""
        if not leaves:
            return 0
        
        # Prepare encodings for neural network evaluation
        encodings, need_eval_idxs, cached_expansions = self._prepare_leaf_evaluations(leaves, timing_tracker)
        
        # Run neural network inference if needed
        if encodings:
            self._run_neural_network_batch(encodings, need_eval_idxs, leaves, timing_tracker)
        
        # Expand cached leaves
        self._expand_cached_leaves(cached_expansions, leaves, timing_tracker)
        
        # Backpropagate values
        prev_root_sum = int(np.sum(root.N))
        simulations_completed = self._backpropagate_batch(leaves, paths, timing_tracker)
        
        # Expect root sum to increase by simulations_completed
        delta = int(np.sum(root.N)) - prev_root_sum
        if not (delta == simulations_completed):
            raise ValueError(f"Expected +{simulations_completed} at root, got +{delta}")

        return simulations_completed

    def _leaf_requires_evaluation(self, leaf: MCTSNode) -> bool:
        """Return whether a leaf still requires NN evaluation for expansion."""
        return not leaf.is_terminal and not leaf.is_expanded

    def _lookup_leaf_cache_timed(
        self,
        leaf: MCTSNode,
        timing_tracker: MCTSTimingTracker
    ) -> Optional[Tuple[np.ndarray, float]]:
        """Lookup leaf evaluation in cache with timing instrumentation."""
        timing_tracker.start_timing("cache_lookup")
        cached = self._get_from_cache(leaf.state_hash)
        timing_tracker.end_timing("cache_lookup")
        return cached

    def _record_nn_eval_start_if_needed(
        self,
        leaves: List[MCTSNode],
        need_eval_idxs: List[int],
        encodings: List[torch.Tensor]
    ) -> None:
        """Record NN evaluation start event for detailed exploration traces."""
        if not self.detailed_exploration_enabled:
            return

        cache_hits_in_batch = 0
        for leaf_idx in need_eval_idxs:
            if leaves[leaf_idx].state_hash in self.eval_cache:
                cache_hits_in_batch += 1

        self._record_nn_eval_start(
            len(leaves), len(need_eval_idxs), len(encodings), cache_hits_in_batch
        )

    def _record_nn_eval_done_if_needed(
        self,
        policy_cpu: torch.Tensor,
        value_cpu: torch.Tensor,
        tm: Dict[str, Any],
        effective_batch_size: int
    ) -> None:
        """Record NN evaluation completion event for detailed exploration traces."""
        if not self.detailed_exploration_enabled:
            return

        values = [float(v.item()) for v in value_cpu]
        value_range = [min(values), max(values)]

        mean_entropy = 0.0
        if len(policy_cpu) > 0:
            entropies = []
            for pol in policy_cpu:
                pol_np = pol.numpy()
                pol_probs = softmax_np(pol_np)
                entropy = -np.sum(pol_probs * np.log(pol_probs + 1e-8))
                entropies.append(entropy)
            mean_entropy = float(np.mean(entropies))

        time_ms = tm.get("forward_ms", 0.0)
        self._record_nn_eval_done(
            effective_batch_size, value_range, mean_entropy, time_ms
        )

    def _cache_and_expand_evaluated_leaves(
        self,
        policy_cpu: torch.Tensor,
        value_cpu: torch.Tensor,
        need_eval_idxs: List[int],
        leaves: List[MCTSNode],
        board_size: int,
        action_size: int
    ) -> None:
        """Write NN outputs to cache and expand corresponding leaves."""
        for j, leaf_idx in enumerate(need_eval_idxs):
            leaf = leaves[leaf_idx]
            policy_np = policy_cpu[j].numpy()
            value_signed = float(value_cpu[j].item())
            self._put_in_cache(leaf.state_hash, policy_np, value_signed)
            self._expand_node_from_policy(leaf, policy_np, board_size, action_size)

    def _prepare_leaf_evaluations(
        self,
        leaves: List[MCTSNode],
        timing_tracker: MCTSTimingTracker
    ) -> Tuple[List[torch.Tensor], List[int], List[Tuple[int, np.ndarray, float]]]:
        """
        Prepare encodings for NN; separate cached from uncached; de-duplicate uncached by board_key.
        Returns:
          encodings          – tensors for unique, uncached leaves (order aligned with need_eval_idxs)
          need_eval_idxs     – indices into `leaves` for those unique encodings
          cached_expansions  – (leaf_idx, policy_np, value_signed) for cache hits
        """
        encodings: List[torch.Tensor] = []
        need_eval_idxs: List[int] = []
        cached_expansions: List[Tuple[int, np.ndarray, float]] = []

        timing_tracker.start_timing("stack")
        timing_tracker.start_timing("encode")

        seen_uncached: Set[int] = set()

        for i, leaf in enumerate(leaves):
            if not self._leaf_requires_evaluation(leaf):
                continue

            cached = self._lookup_leaf_cache_timed(leaf, timing_tracker)

            if cached is not None:
                self.cache_hits += 1
                policy_np, value_signed = cached
                cached_expansions.append((i, policy_np, value_signed))
                continue

            if leaf.state_hash in seen_uncached:
                continue
            seen_uncached.add(leaf.state_hash)

            self.cache_misses += 1
            encodings.append(leaf.state.get_board_tensor().to(dtype=torch.float32))
            need_eval_idxs.append(i)

        timing_tracker.end_timing("encode")
        timing_tracker.end_timing("stack")
        return encodings, need_eval_idxs, cached_expansions

    def _run_neural_network_batch(self, encodings: List[torch.Tensor], need_eval_idxs: List[int], 
                                leaves: List[MCTSNode], timing_tracker: MCTSTimingTracker):
        """Run neural network inference on a batch of leaves."""
        if not encodings:
            return

        self._record_nn_eval_start_if_needed(leaves, need_eval_idxs, encodings)

        batch_tensor = torch.stack(encodings, dim=0)
        board_size = int(batch_tensor.shape[-1])
        action_size = board_size * board_size
        
        policy_cpu, value_cpu, tm = self.model.infer_timed(batch_tensor)
        self._unique_evals_total += int(tm.get("batch_size", len(encodings)))
        
        # Record performance metrics
        self._record_eval_perf(tm, is_first=(timing_tracker.batch_count == 0))
        timing_tracker.record_batch_metrics(tm)

        self._record_nn_eval_done_if_needed(
            policy_cpu=policy_cpu,
            value_cpu=value_cpu,
            tm=tm,
            effective_batch_size=len(need_eval_idxs),
        )

        self._cache_and_expand_evaluated_leaves(
            policy_cpu=policy_cpu,
            value_cpu=value_cpu,
            need_eval_idxs=need_eval_idxs,
            leaves=leaves,
            board_size=board_size,
            action_size=action_size,
        )

    def _expand_cached_leaves(self, cached_expansions: List[Tuple[int, np.ndarray, float]], 
                            leaves: List[MCTSNode], timing_tracker: MCTSTimingTracker):
        """Expand leaves that were found in cache."""
        if not cached_expansions:
            return
        
        timing_tracker.start_timing("expand")
        board_size = leaves[0].board_size
        action_size = board_size * board_size
        
        for (leaf_idx, policy_np, _) in cached_expansions:
            leaf = leaves[leaf_idx]
            if leaf.board_size != board_size:
                raise ValueError(
                    "Mixed board sizes detected in cached leaf expansion batch: "
                    f"{leaf.board_size} vs {board_size}"
                )
            if not leaf.is_expanded and not leaf.is_terminal:
                self._expand_node_from_policy(leaf, policy_np, board_size, action_size)
        
        timing_tracker.end_timing("expand")

    def _backpropagate_batch(self, leaves: List[MCTSNode], paths: List[List[Tuple[MCTSNode, int]]], 
                           timing_tracker: MCTSTimingTracker) -> int:
        """Backpropagate values for a batch of leaves."""
        timing_tracker.start_timing("backprop")
        
        simulations_completed = 0
        for leaf, path in zip(leaves, paths):
            root_q_before = 0.0
            if self.detailed_exploration_enabled:
                self.simulation_count += 1
                root_q_before = self._capture_root_q_snapshot(path)

            v_red_signed = self._leaf_value_signed_red_ref(leaf)
            self._backpropagate_path(path, v_red_signed, leaf.depth)

            self._record_backprop_update_if_needed(path, root_q_before)
            simulations_completed += 1
        
        timing_tracker.end_timing("backprop")
        return simulations_completed

    def _leaf_value_signed_red_ref(self, leaf: MCTSNode) -> float:
        """Return leaf value in Red's reference frame."""
        if leaf.is_terminal:
            return self._get_terminal_value(leaf)
        return self._get_neural_network_value(leaf)

    def _capture_root_q_snapshot(self, path: List[Tuple[MCTSNode, int]]) -> float:
        """Capture root max-Q for detailed backprop traces."""
        if not path:
            return 0.0

        root = path[0][0]
        if root.is_expanded and len(root.Q) > 0:
            return float(np.max(root.Q))
        return 0.0

    def _record_backprop_update_if_needed(
        self,
        path: List[Tuple[MCTSNode, int]],
        root_q_before: float,
    ) -> None:
        """Record detailed backprop diagnostics when enabled."""
        if not self.detailed_exploration_enabled:
            return

        root_q_after = self._capture_root_q_snapshot(path)
        self._record_backprop_update(len(path), root_q_before, root_q_after)

    def _get_terminal_value(self, leaf: MCTSNode) -> float:
        """Get signed value for a terminal leaf in Red's reference frame: +1 = Red win, -1 = Blue win."""
        if leaf.winner == Winner.RED:
            return 1.0  # +1 = certain Red win
        elif leaf.winner == Winner.BLUE:
            return -1.0  # -1 = certain Blue win
        else:
            raise ValueError(f"Invalid winner enum for terminal Hex node: {leaf.winner!r} (draws are not possible in Hex)")

    def _get_neural_network_value(self, leaf: MCTSNode) -> float:
        """Get signed value for a non-terminal leaf from neural network in Red's reference frame: +1 = Red win, -1 = Blue win."""
        cached = self._get_from_cache(leaf.state_hash)
        if cached is None:
            raise RuntimeError(f"Leaf state not found in cache: {leaf.state_hash}")
        _, value_signed = cached
        # Return signed value directly - tanh activation gives values in [-1,1] range in Red's reference frame
        return float(value_signed)

    def _backpropagate_path(self, path: List[Tuple[MCTSNode, int]], v_red_ref_signed: float, leaf_depth: int):
        """
        Backpropagate signed value along a path from leaf to root.
        
        Args:
            path: List of (node, action_index) pairs from root to leaf
            v_red_ref_signed: Signed value in Red's reference frame (+1 = Red win, -1 = Blue win)
            leaf_depth: Depth of the leaf node for distance-to-leaf calculation
        """
        for (node, a_idx) in reversed(path):
            # Convert Red's signed value to player-to-move reference frame
            # If RED to move: keep v_red_ref_signed; if BLUE to move: flip to -v_red_ref_signed
            # This converts the leaf evaluation to "how good is this for the player at this node"
            v_ptm_ref_signed = red_ref_signed_to_ptm_ref_signed(v_red_ref_signed, node.to_play)
            
            # Apply depth discounting in signed space (shrink toward 0)
            if self.cfg.enable_depth_discounting and node.depth > 0:
                # Use distance-to-leaf for more intuitive discounting: prefer shorter wins
                distance = distance_to_leaf(node.depth, leaf_depth)
                v_ptm_ref_signed = apply_depth_discount_signed(v_ptm_ref_signed, self.cfg.depth_discount_factor, distance)
            
            node.N[a_idx] = increment_visit_count(node.N[a_idx])
            node.W[a_idx] = add_to_accumulated_value(node.W[a_idx], v_ptm_ref_signed)
            node.Q[a_idx] = q_from_w_n(node.W[a_idx], node.N[a_idx])

    # ---------- New Result Computation Methods ----------
    
    def _get_algorithm_termination_move(self, root: MCTSNode, termination_info: AlgorithmTerminationInfo, verbose: int) -> Tuple[int, int]:
        """Get the move for algorithm termination cases."""
        if termination_info.reason == "terminal_move":
            return termination_info.move
        elif termination_info.reason == "neural_network_confidence":
            # Use top policy move
            best_move_idx = int(np.argmax(root.P))
            best_move = root.legal_moves[best_move_idx]
            if verbose >= 2:
                print(
                    "🎮 MCTS: Using top policy move "
                    f"(confidence-based termination, win probability: {termination_info.win_probability:.3f}): "
                    f"{best_move}"
                )
            return best_move
        else:
            raise ValueError(f"Unknown algorithm termination reason: {termination_info.reason}")

    def _compute_move_from_gumbel_if_used(
        self, root: MCTSNode, root_state: HexGameState, verbose: int
    ) -> Optional[Tuple[Tuple[int, int], Dict[str, float]]]:
        """Return move/probs if Gumbel root selection already picked the move."""
        if not getattr(self, "_used_gumbel_root_selection", False):
            return None

        selected_action = self._gumbel_selected_action
        if selected_action is None:
            raise RuntimeError("Gumbel root selection marked as used, but no selected action was recorded.")

        selected_move = root.legal_moves[selected_action]
        if verbose >= 2:
            print(f"🎮 MCTS: Using Gumbel-selected move: {selected_move}")

        move_probs = calculate_policy_probs(root, root_state, self.cfg, self)
        return selected_move, move_probs

    def _compute_move_from_terminal_detection_if_any(
        self, root: MCTSNode, root_state: HexGameState, verbose: int
    ) -> Optional[Tuple[Tuple[int, int], Dict[str, float]]]:
        """Return move/probs when a terminal root move has been pre-detected."""
        if not self.cfg.enable_terminal_move_detection:
            return None
        if not any(root.terminal_moves):
            return None

        terminal_indices = [i for i, is_terminal in enumerate(root.terminal_moves) if is_terminal]
        if not terminal_indices:
            return None

        terminal_move = root.legal_moves[terminal_indices[0]]
        if verbose >= 2:
            print(f"🎮 MCTS: Using pre-detected terminal move: {terminal_move}")
        move_probs = calculate_visit_count_probs(root, root_state, self.cfg)
        return terminal_move, move_probs

    def _compute_move_from_visit_counts(
        self, root: MCTSNode, root_state: HexGameState, verbose: int
    ) -> Tuple[Tuple[int, int], Dict[str, float]]:
        """Select move from visit counts with configured temperature sampling."""
        counts = root.N.astype(np.float64)
        if counts.sum() <= 0:
            raise RuntimeError("No visits recorded during MCTS search. This indicates a bug in the search algorithm.")

        move_count = len(root_state.move_history)
        state_board_size = board_size_from_state(root_state)
        if state_board_size != root.board_size:
            raise ValueError(
                "Root board size mismatch between root node and root_state during move selection: "
                f"{root.board_size} vs {state_board_size}"
            )
        temp = self._root_temperature(move_count, root.board_size)

        if verbose >= 4:
            top_k_info = f", top-k={self.cfg.visit_sampling_top_k}" if self.cfg.visit_sampling_top_k > 0 else ""
            print(f"🎮 MCTS: Move {move_count}, effective temperature: {temp:.3f}{top_k_info}")

        move_probs = calculate_visit_count_probs(root, root_state, self.cfg)
        a_idx = select_move_index(counts, temp, self.cfg)
        return root.legal_moves[a_idx], move_probs

    def _compute_move(self, root: MCTSNode, root_state: HexGameState, verbose: int) -> Tuple[Tuple[int, int], Dict[str, float]]:
        """Compute the selected move from the root node."""
        gumbel_result = self._compute_move_from_gumbel_if_used(root, root_state, verbose)
        if gumbel_result is not None:
            return gumbel_result

        terminal_result = self._compute_move_from_terminal_detection_if_any(root, root_state, verbose)
        if terminal_result is not None:
            return terminal_result

        return self._compute_move_from_visit_counts(root, root_state, verbose)


    # ---------- Internal Implementation ----------
    
    def _get_stats_builder(self) -> MCTSStatsBuilder:
        """Get stats builder with current cache counts."""
        return MCTSStatsBuilder(self.cache_hits, self.cache_misses)

    # ---------- Internal ----------

    def _record_eval_perf(self, tm: Dict[str, Any], is_first: bool):
        """Record per-batch timing samples to PERF and set meta (first batch only)."""
        try:
            PERF.add_sample("eval_h2d_ms", float(tm.get("h2d_ms", 0.0)))
            PERF.add_sample("eval_forward_ms", float(tm.get("forward_ms", 0.0)))
            PERF.add_sample("eval_d2h_ms", float(tm.get("d2h_ms", 0.0)))
            PERF.add_sample("eval_batch_size", float(tm.get("batch_size", 0.0)))
            if is_first:
                PERF.set_meta("eval_device", str(tm.get("device", "")))
                PERF.set_meta("eval_param_dtype", str(tm.get("param_dtype", "")))
        except Exception:
            # PERF is best-effort
            pass

    def _apply_root_noise(self, root: MCTSNode):
        """Apply Dirichlet noise to root priors once."""
        if not root.is_expanded or not root.legal_moves:
            return
        L = len(root.legal_moves)
        noise = np.random.dirichlet([self.cfg.dirichlet_alpha] * L)
        root.P = (1 - self.cfg.dirichlet_eps) * root.P + self.cfg.dirichlet_eps * noise

    def _expand_node_from_policy(self, node: MCTSNode, policy_logits_np: np.ndarray, board_size: int, action_size: int):
        """Set node.P over legal actions using softmax of legal logits; mark expanded."""
        if node.is_terminal:
            node.is_expanded = True
            return
        # Validate policy logits shape
        if policy_logits_np.shape[0] != action_size:
            raise ValueError(f"Policy logits shape mismatch: expected {action_size}, got {policy_logits_np.shape[0]}")
        logits = policy_logits_np.astype(np.float64, copy=False)

        legal_logits = logits[node.legal_indices] if len(node.legal_indices) > 0 else np.array([0.0], dtype=np.float64)
        node.P = softmax_np(legal_logits)
        node.is_expanded = True
        
        # Record expand node for detailed exploration
        if self.detailed_exploration_enabled:
            # Calculate top3 prior mass
            if len(node.P) >= 3:
                top3_indices = np.argsort(node.P)[-3:]
                prior_mass_top3 = float(np.sum(node.P[top3_indices]))
            else:
                prior_mass_top3 = float(np.sum(node.P))
            
            # Get value from cache (should be available since we just put it there)
            value_signed_red_ref = 0.0
            cached = self._get_from_cache(node.state_hash)
            if cached:
                _, value_signed_red_ref = cached
            
            self._record_expand_node(
                node.depth, node.state_hash, len(node.legal_moves),
                prior_mass_top3, value_signed_red_ref
            )

    def _detect_terminal_moves_if_enabled(self, node: MCTSNode) -> None:
        """Populate node terminal-move flags when terminal detection is enabled."""
        if self.cfg.enable_terminal_move_detection:
            self.terminal_detector.detect_terminal_moves(node)

    def _forced_terminal_child_index(self, node: MCTSNode) -> Optional[int]:
        """Return forced terminal child index when immediate-terminal preference is active."""
        if not self.cfg.enable_terminal_move_detection:
            return None
        if not self.cfg.prefer_immediate_terminal:
            return None
        if not any(node.terminal_moves):
            return None
        terminal_idxs = [i for i, is_terminal in enumerate(node.terminal_moves) if is_terminal]
        return int(max(terminal_idxs, key=lambda i: float(node.P[i])))

    def _compute_puct_u_values(self, node: MCTSNode, n_sum_adjusted: float) -> np.ndarray:
        """Compute PUCT exploration term U with optional terminal-move boosting."""
        u_values = self.cfg.c_puct * node.P * math.sqrt(n_sum_adjusted) / (1.0 + node.N)
        if self.cfg.enable_terminal_move_detection:
            for i, is_terminal in enumerate(node.terminal_moves):
                if is_terminal:
                    u_values[i] += self.cfg.terminal_move_boost
        return u_values

    def _apply_root_reservation_mask(self, scores: np.ndarray, used_root_actions: Optional[Set[int]]) -> np.ndarray:
        """Mask already-reserved root actions so current batch explores distinct root choices."""
        if used_root_actions is None or len(used_root_actions) == 0:
            return scores
        mask = np.zeros_like(scores, dtype=bool)
        for action_idx in used_root_actions:
            if 0 <= action_idx < len(scores):
                mask[action_idx] = True
        return np.where(mask, -np.inf, scores)

    def _record_forced_terminal_selection_if_needed(self, current_depth: int) -> None:
        """Record detailed trace event for forced terminal selection."""
        if not self.detailed_exploration_enabled:
            return
        self._record_select_action(
            current_depth, 0.0, 0.0, 0.0, 0, 0.0, 0.0,
            terminal_flag_for_child=True, note="forced_terminal_win"
        )

    def _record_selected_puct_action_if_needed(
        self,
        node: MCTSNode,
        selected_idx: int,
        u_values: np.ndarray,
        score_values: np.ndarray,
        n_sum_adjusted: float,
        current_depth: int
    ) -> None:
        """Record selected PUCT action details for detailed exploration traces."""
        if not self.detailed_exploration_enabled:
            return
        q = float(node.Q[selected_idx])
        p = float(node.P[selected_idx])
        n = int(node.N[selected_idx])
        u = float(u_values[selected_idx])
        score_val = float(score_values[selected_idx])
        terminal_flag = bool(selected_idx < len(node.terminal_moves) and node.terminal_moves[selected_idx])
        self._record_select_action(
            current_depth, n_sum_adjusted, q, p, n, u, score_val, terminal_flag
        )

    def _select_child_puct(self, node: MCTSNode, current_depth: int = 0, used_root_actions: Optional[Set[int]] = None) -> int:
        """Return index into node.legal_moves of the action maximizing PUCT score."""
        # PUCT: U = c_puct * P * sqrt(sum(N)) / (1 + N)
        # score = Q + U

        self._detect_terminal_moves_if_enabled(node)
        forced_terminal_idx = self._forced_terminal_child_index(node)
        if forced_terminal_idx is not None:
            self._record_forced_terminal_selection_if_needed(current_depth)
            return forced_terminal_idx

        N_sum_adjusted = 1.0 + np.sum(node.N, dtype=np.float64)
        if not safe_puct_denominator(N_sum_adjusted):
            raise RuntimeError("N_sum is 0, which should never happen. Need to debug how this happens.")

        # Record select action for detailed exploration (degenerate case)
        if self.detailed_exploration_enabled:
            self._record_select_action(
                current_depth, 0.0, 0.0, 0.0, 0, 0.0, 0.0,
                note="standard_puct_selection"
            )

        U = self._compute_puct_u_values(node, N_sum_adjusted)
        score = node.Q + U
        score = self._apply_root_reservation_mask(score, used_root_actions)
        result = int(np.argmax(score))
        self._record_selected_puct_action_if_needed(node, result, U, score, N_sum_adjusted, current_depth)
        return result

    def clear_cache(self) -> None:
        """
        Clear the evaluation cache to free memory.
        
        This method can be called periodically to prevent memory leaks
        in long-running processes.
        """
        self.eval_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        # Reset metrics when clearing cache
        self._unique_evals_total = 0
        self._effective_sims_total = 0

    def _get_from_cache(self, board_key: int) -> Optional[Tuple[np.ndarray, float]]:
        """
        Get a value from the LRU cache, updating access order.
        
        Args:
            board_key: The board state key to look up
            
        Returns:
            The cached (policy, value) tuple if found, None otherwise
        """
        if board_key in self.eval_cache:
            # Move to end (most recently used)
            value = self.eval_cache.pop(board_key)
            self.eval_cache[board_key] = value
            return value
        return None

    def _put_in_cache(self, board_key: int, policy: np.ndarray, value: float) -> None:
        """
        Put a value in the LRU cache, evicting least recently used if needed.
        
        Args:
            board_key: The board state key
            policy: The policy logits
            value: The signed value in Red's reference frame (tanh-activated output in [-1,1] range)
        """
        # If key already exists, remove it first (will be re-added at end)
        if board_key in self.eval_cache:
            self.eval_cache.pop(board_key)
        
        # If cache is full, evict least recently used (first item)
        if len(self.eval_cache) >= self.cfg.cache_size:
            self.eval_cache.popitem(last=False)  # Remove first (least recently used)
        
        # Add new item at end (most recently used)
        self.eval_cache[board_key] = (policy, value)

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics for monitoring.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "cache_size": len(self.eval_cache),
            "max_cache_size": self.cfg.cache_size,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hits / max(1, self.cache_hits + self.cache_misses),
            "cache_utilization": len(self.eval_cache) / self.cfg.cache_size
        }

def run_mcts_move(engine: HexGameEngine, model: ModelWrapper, state: HexGameState, cfg: BaselineMCTSConfig, verbose: int = 0) -> Tuple[Tuple[int,int], Dict[str, Any], Dict[str, Any], Optional[AlgorithmTerminationInfo]]:
    """Run MCTS for one move and return (row,col), stats, tree_data, algorithm_termination_info."""
    if cfg is None:
        raise ValueError("cfg must be provided")
    mcts = BaselineMCTS(engine, model, cfg)
    result = mcts.run(state, verbose=verbose)
    return result.move, result.stats, result.tree_data, result.algorithm_termination_info
