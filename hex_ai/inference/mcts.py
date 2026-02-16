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
import random
import numpy as np
import torch
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Set
from collections import OrderedDict, deque

# ---- Package imports ----
from hex_ai.enums import Player, Winner
from hex_ai.value_utils import player_to_winner, red_ref_signed_to_ptm_ref_signed, apply_depth_discount_signed, signed_to_prob, distance_to_leaf
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
    tensor_to_trmph,
)
from hex_ai.utils.temperature import calculate_temperature_decay
from hex_ai.utils.state_utils import board_key, validate_move_coordinates
from hex_ai.utils.timing import MCTSTimingTracker
from hex_ai.utils.gumbel_utils import gumbel_alpha_zero_root_batched
from hex_ai.config import BOARD_SIZE as CFG_BOARD_SIZE
from hex_ai.inference.mcts_config import BaselineMCTSConfig, create_mcts_config
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


# Default terminal detection parameters
DEFAULT_MIN_MOVES_FOR_TERMINAL_DETECTION = 2  # Multiplier for board size

# ------------------ Terminal Move Detector ------------------
class TerminalMoveDetector:
    """Centralized terminal move detection with consistent behavior."""
    
    def __init__(self, max_detection_depth: int = 3):
        self.max_detection_depth = max_detection_depth
    
    def should_detect_terminal_moves(self, node: MCTSNode) -> bool:
        """Determine if terminal moves should be detected for this node."""
        # Only detect at shallow depths
        if node.depth > self.max_detection_depth:
            return False
        
        # Only detect after minimum move count (impossible to win before BS * 2 - 1, unlikely before BS * 3)
        min_move_count = CFG_BOARD_SIZE * DEFAULT_MIN_MOVES_FOR_TERMINAL_DETECTION - 2
        if len(node.state.move_history) < min_move_count:
            return False
        
        # Don't detect if already done
        if node._terminal_moves_detected:
            return False
        
        return True
    
    def detect_terminal_moves(self, node: MCTSNode) -> bool:
        """
        Detect terminal moves for a given node.

        This method uses the underlying game logic to check, for each legal move,
        whether it results in an immediate win for the current player. It provides
        a definitive proof of a win, rather than relying on heuristics or neural
        network evaluations.

        Returns:
            bool: True if any terminal (winning) moves are found, False otherwise.
        """
        if not self.should_detect_terminal_moves(node):
            return False
        
        # Reset terminal moves
        node.terminal_moves = [False] * len(node.legal_moves)
        
        # Check each legal move
        for i, (row, col) in enumerate(node.legal_moves):
            new_state = node.state.make_move(row, col)
            if new_state.game_over and new_state.winner == player_to_winner(node.to_play):
                node.terminal_moves[i] = True
        
        node._terminal_moves_detected = True
        return any(node.terminal_moves)
    
    def get_terminal_move(self, node: MCTSNode) -> Optional[Tuple[int, int]]:
        """Get the first terminal move if any exist."""
        if not node._terminal_moves_detected:
            return None
        
        for i, is_terminal in enumerate(node.terminal_moves):
            if is_terminal:
                return node.legal_moves[i]
        return None

# ------------------ Algorithm Termination Info ------------------
@dataclass
class AlgorithmTerminationInfo:
    """Simple info about algorithm termination."""
    reason: str  # "terminal_move" or "neural_network_confidence"
    move: Optional[Tuple[int, int]]  # The move to play (None for NN confidence)
    win_prob: float  # Win probability

# ------------------ MCTS Result ------------------
@dataclass(frozen=True)
class MCTSResult:
    """Complete result of an MCTS search."""
    move: Tuple[int, int]  # The selected move
    stats: Dict[str, Any]  # Performance statistics
    tree_data: Dict[str, Any]  # Tree information for analysis
    root_node: MCTSNode  # The search tree root (for advanced use cases)
    algorithm_termination_info: Optional[AlgorithmTerminationInfo]  # Algorithm termination details
    win_probability: float  # Win probability for current player

# ------------------ Stats Builder ------------------
class MCTSStatsBuilder:
    """Centralized stats creation with consistent structure."""
    
    def __init__(self, cache_hits: int, cache_misses: int):
        self.cache_hits = cache_hits
        self.cache_misses = cache_misses
    
    def create_base_stats(self) -> Dict[str, Any]:
        """Create base stats structure with all common fields."""
        return {
            "encode_ms": 0.0, "stack_ms": 0.0, "h2d_ms": 0.0, "forward_ms": 0.0,
            "pure_forward_ms": 0.0, "sync_ms": 0.0, "d2h_ms": 0.0, "expand_ms": 0.0,
            "backprop_ms": 0.0, "select_ms": 0.0, "cache_lookup_ms": 0.0, "state_creation_ms": 0.0,
            "batch_count": 0, "batch_sizes": [], "forward_ms_list": [],
            "select_times": [], "cache_hit_times": [], "cache_miss_times": [],
            "median_forward_ms_ex_warm": 0.0, "p90_forward_ms_ex_warm": 0.0,
            "median_select_ms": 0.0, "median_cache_hit_ms": 0.0, "median_cache_miss_ms": 0.0,
            "cache_hits": self.cache_hits, "cache_misses": self.cache_misses,
        }
    
    def create_algorithm_termination_stats(self, termination_info: Optional[AlgorithmTerminationInfo] = None) -> Dict[str, Any]:
        """Create stats for algorithm termination cases."""
        stats = self.create_base_stats()
        stats.update({
            "total_simulations": 0, "simulations_per_second": 0.0,
            "algorithm_termination_occurred": True,
            "algorithm_termination_reason": termination_info.reason if termination_info else "unknown"
        })
        return stats
    
    def create_final_stats(self, timing_stats: Dict[str, Any], total_simulations: int, 
                          total_search_time: float) -> Dict[str, Any]:
        """Create final stats for completed MCTS runs."""
        stats = self.create_base_stats()
        stats.update(timing_stats)
        stats.update({
            "total_simulations": total_simulations,
            "simulations_per_second": total_simulations / total_search_time if total_search_time > 0 else 0.0,
            "algorithm_termination_occurred": False,
            "algorithm_termination_reason": "none"
        })
        return stats

# ------------------ Algorithm Termination Checker ------------------
class AlgorithmTerminationChecker:
    """Centralized algorithm termination checking with simple priority order."""
    
    def __init__(self, cfg: BaselineMCTSConfig, terminal_detector: TerminalMoveDetector):
        self.cfg = cfg
        self.terminal_detector = terminal_detector
    
    def should_terminate_early(self, root: MCTSNode, verbose: int, eval_cache: Dict[int, Tuple[np.ndarray, float]], root_is_expanded: bool = False) -> Optional[AlgorithmTerminationInfo]:
        """
        Check if we should terminate early. Returns None if we should continue with MCTS.
        Returns EarlyTerminationInfo if we should terminate.
        
        Args:
            root: The root node to check
            verbose: Verbosity level
            eval_cache: Evaluation cache for neural network confidence
            root_is_expanded: Whether the root node has been expanded (affects confidence checking)
        """
        # 1. Check for terminal moves (highest priority) - works regardless of expansion
        if self.cfg.enable_terminal_move_detection:
            if self.terminal_detector.detect_terminal_moves(root):
                terminal_move = self.terminal_detector.get_terminal_move(root)
                if verbose >= 2:
                    print(f"🎮 MCTS: Found terminal move: {terminal_move}")
                return AlgorithmTerminationInfo(
                    reason="terminal_move",
                    move=terminal_move,
                    win_prob=1.0  # Guaranteed win
                )
        
        # 2. Check neural network confidence (requires root expansion)
        if self.cfg.enable_confidence_termination and root_is_expanded and not root.is_terminal:
            signed_value = self._get_root_signed_value(root, eval_cache)
            if self._is_position_clearly_decided(signed_value):
                # In self-play, we sometimes keep running MCTS even for clearly decided positions.
                # This reduces drift by ensuring a small fraction of training targets remain
                # MCTS-improved instead of pure-policy fallbacks.
                if random.random() >= self.cfg.confidence_termination_probability:
                    if verbose >= 3:
                        print(
                            f"🎮 MCTS: Confidence termination suppressed (prob={self.cfg.confidence_termination_probability:.3f}, "
                            f"signed value: {signed_value:.3f})"
                        )
                    return None
                if verbose >= 2:
                    print(f"🎮 MCTS: Confidence-based termination (signed value: {signed_value:.3f})")
                return AlgorithmTerminationInfo(
                    reason="neural_network_confidence",
                    move=None,  # Will use top policy move
                    win_prob=signed_value
                )
        
        return None  # Continue with MCTS
    
    def _get_root_signed_value(self, root: MCTSNode, eval_cache: OrderedDict[int, Tuple[np.ndarray, float]]) -> float:
        """Get signed value for current player from neural network (edge conversion)."""
        cached = eval_cache.get(root.state_hash)
        if cached is None:
            raise RuntimeError(f"Root state not found in cache: {root.state_hash}")
        _, value_signed = cached
        
        # Validate that the cached value is in the expected signed range
        if not -1.1 <= value_signed <= 1.1:
            raise ValueError(f"Neural network output {value_signed} is outside expected signed range [-1, 1]. "
                           f"This suggests a mismatch between probability and signed value semantics.")
        
        # Convert signed value to player-to-move reference frame for confidence termination
        # value_signed is the tanh-activated output in [-1,1] range in Red's reference frame
        v_red_ref_signed = float(value_signed)
        # Flip to player-to-move reference frame: if RED to move keep v_red_ref_signed, if BLUE to move flip to -v_red_ref_signed
        v_ptm_ref_signed = red_ref_signed_to_ptm_ref_signed(v_red_ref_signed, root.to_play)
        # Return signed value directly (no conversion to probability needed)
        return v_ptm_ref_signed
    
    def _is_position_clearly_decided(self, signed_value: float) -> bool:
        """Check if position is clearly won or lost using signed values."""
        # signed_value is in [-1, 1] range in player-to-move reference frame
        # Check if position is clearly won (> threshold) or clearly lost (< -threshold)
        return (signed_value >= self.cfg.confidence_termination_threshold or 
                signed_value <= -self.cfg.confidence_termination_threshold)

# ------------------ Data structures ------------------

class MCTSNode:
    __slots__ = (
        "state", "to_play", "legal_moves", "legal_indices",
        "children", "N", "W", "Q", "P", "is_expanded",
        "state_hash", "is_terminal", "winner", "winner_str", "terminal_moves",
        "_terminal_moves_detected", "depth"
    )
    def __init__(self, state: HexGameState, board_size: int):
        if state is None:
            raise ValueError("State cannot be None")
        if board_size <= 0:
            raise ValueError(f"Board size must be positive, got {board_size}")
        
        self.state: HexGameState = state
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

class BaselineMCTS:
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
        self.detailed_exploration_enabled = should_enable_detailed_exploration(num_simulations)
        self.exploration_trace = []
        self.simulation_count = 0
        if self.detailed_exploration_enabled and hasattr(self, 'verbose') and self.verbose >= 2:
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

    def _decorate_gumbel_score_rows_with_moves(
        self, rows: List[Dict[str, Any]], board_size: int
    ) -> List[Dict[str, Any]]:
        """Attach TRMPH move labels to rows that contain tensor action indices."""
        decorated_rows: List[Dict[str, Any]] = []
        for row in rows:
            row_copy = dict(row)
            action = row_copy.get("tensor_action", None)
            if action is not None:
                try:
                    row_copy["move"] = tensor_to_trmph(int(action), board_size)
                except Exception:
                    row_copy["move"] = None
            decorated_rows.append(row_copy)
        return decorated_rows

    def _record_gumbel_trace_event(self, event: Dict[str, Any], board_size: int) -> None:
        """Record a Gumbel-specific detailed trace event with human-readable move labels."""
        if not self.detailed_exploration_enabled:
            return

        event_copy = dict(event)

        # Decorate single-action fields
        if event_copy.get("selected_action", None) is not None:
            try:
                event_copy["selected_move"] = tensor_to_trmph(
                    int(event_copy["selected_action"]), board_size
                )
            except Exception:
                event_copy["selected_move"] = None
        if event_copy.get("tensor_action", None) is not None:
            try:
                event_copy["move"] = tensor_to_trmph(
                    int(event_copy["tensor_action"]), board_size
                )
            except Exception:
                event_copy["move"] = None

        # Decorate action-list fields
        for actions_key, moves_key in (
            ("kept_actions", "kept_moves"),
            ("dropped_actions", "dropped_moves"),
        ):
            if actions_key in event_copy and isinstance(event_copy[actions_key], list):
                moves: List[str] = []
                for action in event_copy[actions_key]:
                    try:
                        moves.append(tensor_to_trmph(int(action), board_size))
                    except Exception:
                        continue
                event_copy[moves_key] = moves

        # Decorate score row collections
        for rows_key in ("candidate_rows", "selected_rows", "excluded_rows", "final_rank_rows"):
            if rows_key in event_copy and isinstance(event_copy[rows_key], list):
                event_copy[rows_key] = self._decorate_gumbel_score_rows_with_moves(
                    event_copy[rows_key], board_size
                )

        self.exploration_trace.append(event_copy)

    @staticmethod
    def _player_to_color_label(player: Player) -> str:
        """Convert Player enum to a simple lowercase color label."""
        if player == Player.RED:
            return "red"
        if player == Player.BLUE:
            return "blue"
        return str(player)

    @staticmethod
    def _red_ref_signed_to_root_ref_signed(value_signed_red_ref: float, root_player: Player) -> float:
        """Convert a red-reference signed value into root-player reference frame."""
        return float(value_signed_red_ref if root_player == Player.RED else -value_signed_red_ref)

    def _debug_value_summary_for_state(self, state: HexGameState, root_player: Player) -> Dict[str, Any]:
        """
        Get value-head summary for an arbitrary state.

        Uses cache when available; otherwise performs a direct model eval without mutating cache.
        """
        cached = self._get_from_cache(board_key(state))
        from_cache = cached is not None

        if cached is not None:
            _, value_signed_red_ref = cached
        else:
            enc = state.get_board_tensor().to(dtype=torch.float32)
            batch = torch.stack([enc], dim=0)
            _, value_cpu, _ = self.model.infer_timed(batch)
            value_signed_red_ref = float(value_cpu[0].item())

        value_signed_red_ref = float(value_signed_red_ref)
        value_signed_root_ref = self._red_ref_signed_to_root_ref_signed(value_signed_red_ref, root_player)
        value_signed_ptm_ref = float(red_ref_signed_to_ptm_ref_signed(value_signed_red_ref, state.current_player_enum))

        return {
            "red_ref_signed": value_signed_red_ref,
            "root_ref_signed": value_signed_root_ref,
            "root_win_prob": float(signed_to_prob(value_signed_root_ref)),
            "ptm_ref_signed": value_signed_ptm_ref,
            "ptm_win_prob": float(signed_to_prob(value_signed_ptm_ref)),
            "to_play": self._player_to_color_label(state.current_player_enum),
            "from_cache": bool(from_cache),
        }

    def _build_gumbel_action_dive_event(
        self, root: MCTSNode, tensor_action: int, board_size: int, top_replies: int = 6
    ) -> Optional[Dict[str, Any]]:
        """
        Build a deep-dive debug event for one root action:
        - Root child tree Q/N
        - Value-head eval after root move
        - Opponent reply diagnostics by visits and by prior
        """
        if tensor_action not in root.legal_indices:
            return None

        root_player = root.to_play
        action = int(tensor_action)
        child_idx = root.legal_indices.index(action)
        move = tensor_to_trmph(action, board_size)
        root_child_visits = int(root.N[child_idx])
        root_child_q_ptm_signed = float(root.Q[child_idx]) if root_child_visits > 0 else 0.0
        root_child_q_01 = float((root_child_q_ptm_signed + 1.0) / 2.0) if root_child_visits > 0 else 0.5

        # Construct state after root move and evaluate value head.
        move_row, move_col = root.legal_moves[child_idx]
        state_after_root = root.state.make_move(move_row, move_col)
        value_after_root = self._debug_value_summary_for_state(state_after_root, root_player)

        event: Dict[str, Any] = {
            "type": "gumbel_action_dive",
            "tensor_action": action,
            "move": move,
            "root_player": self._player_to_color_label(root_player),
            "opponent_to_play": self._player_to_color_label(state_after_root.current_player_enum),
            "root_child_visits": root_child_visits,
            "root_child_q_ptm_signed": root_child_q_ptm_signed,
            "root_child_q_01": root_child_q_01,
            "value_after_root": value_after_root,
        }

        child_node = root.children[child_idx]
        if child_node is None:
            event["note"] = "Child node was not realized during search; no opponent reply tree stats available."
            event["reply_count_total"] = int(len(state_after_root.get_legal_moves()))
            event["reply_count_visited"] = 0
            event["top_replies_by_visits"] = []
            event["top_replies_by_policy"] = []
            return event

        reply_count_total = len(child_node.legal_moves)
        visited_indices = [i for i, n in enumerate(child_node.N) if int(n) > 0]
        visited_indices.sort(key=lambda i: int(child_node.N[i]), reverse=True)
        visited_top = visited_indices[:top_replies]

        policy_top: List[int] = []
        if child_node.is_expanded and len(child_node.P) == reply_count_total and reply_count_total > 0:
            policy_top = list(np.argsort(child_node.P)[::-1][:top_replies].astype(int))

        # Evaluate value head after replies for the union of indices shown in the report.
        eval_indices = sorted(set(visited_top + policy_top))
        value_after_reply_by_idx: Dict[int, Dict[str, Any]] = {}
        for idx in eval_indices:
            reply_row, reply_col = child_node.legal_moves[idx]
            reply_state = child_node.state.make_move(reply_row, reply_col)
            value_after_reply_by_idx[idx] = self._debug_value_summary_for_state(reply_state, root_player)

        def build_reply_row(idx: int) -> Dict[str, Any]:
            reply_row, reply_col = child_node.legal_moves[idx]
            reply_move = rowcol_to_trmph(reply_row, reply_col, board_size)
            visits = int(child_node.N[idx])
            prior = float(child_node.P[idx]) if child_node.is_expanded and len(child_node.P) == reply_count_total else None

            if visits > 0:
                q_opp_ptm_signed = float(child_node.Q[idx])  # Child node is opponent-to-play.
                q_root_ref_signed = float(-q_opp_ptm_signed)  # Opponent perspective -> root perspective.
                q_root_win_prob = float(signed_to_prob(q_root_ref_signed))
                q_source = "tree"
            else:
                q_opp_ptm_signed = None
                q_root_ref_signed = None
                q_root_win_prob = None
                q_source = "unvisited"

            return {
                "move": reply_move,
                "visits": visits,
                "prior": prior,
                "q_source": q_source,
                "q_opp_ptm_signed": q_opp_ptm_signed,
                "q_root_ref_signed": q_root_ref_signed,
                "q_root_win_prob": q_root_win_prob,
                "value_after_reply": value_after_reply_by_idx.get(idx, None),
            }

        event["reply_count_total"] = int(reply_count_total)
        event["reply_count_visited"] = int(len(visited_indices))
        top_replies_by_visits = [build_reply_row(idx) for idx in visited_top]
        top_replies_by_policy = [build_reply_row(idx) for idx in policy_top]
        event["top_replies_by_visits"] = top_replies_by_visits
        event["top_replies_by_policy"] = top_replies_by_policy

        # Debug-only stress-check on the top policy replies:
        # summarizes whether potentially dangerous unvisited responses exist.
        policy_rows_for_stats: List[Dict[str, Any]] = []
        for row in top_replies_by_policy:
            prior_val = row.get("prior", None)
            value_summary = row.get("value_after_reply", None)
            root_win_prob = value_summary.get("root_win_prob", None) if isinstance(value_summary, dict) else None
            if prior_val is None:
                continue
            try:
                prior = float(prior_val)
                if not np.isfinite(prior) or prior <= 0.0:
                    continue
            except Exception:
                continue

            if root_win_prob is not None:
                try:
                    root_win_prob = float(root_win_prob)
                except Exception:
                    root_win_prob = None

            policy_rows_for_stats.append(
                {
                    "move": row.get("move", None),
                    "visits": int(row.get("visits", 0)),
                    "prior": prior,
                    "root_win_prob": root_win_prob,
                }
            )

        total_policy_mass = float(sum(item["prior"] for item in policy_rows_for_stats))
        visited_policy_mass = float(sum(item["prior"] for item in policy_rows_for_stats if item["visits"] > 0))
        policy_rows_with_value = [item for item in policy_rows_for_stats if item["root_win_prob"] is not None]

        weighted_value_head_root_win = None
        if policy_rows_with_value:
            denom = float(sum(item["prior"] for item in policy_rows_with_value))
            if denom > 1e-12:
                weighted_value_head_root_win = float(
                    sum(item["prior"] * float(item["root_win_prob"]) for item in policy_rows_with_value) / denom
                )

        worst_policy_reply = None
        if policy_rows_with_value:
            worst = min(policy_rows_with_value, key=lambda item: float(item["root_win_prob"]))
            worst_policy_reply = {
                "move": worst["move"],
                "visits": int(worst["visits"]),
                "prior": float(worst["prior"]),
                "root_win_prob": float(worst["root_win_prob"]),
            }

        event["policy_reply_stress"] = {
            "top_policy_count": int(len(policy_rows_for_stats)),
            "top_policy_count_visited": int(sum(1 for item in policy_rows_for_stats if item["visits"] > 0)),
            "top_policy_prior_mass": total_policy_mass,
            "top_policy_prior_mass_visited": visited_policy_mass,
            "top_policy_prior_mass_visited_ratio": (
                float(visited_policy_mass / total_policy_mass) if total_policy_mass > 1e-12 else None
            ),
            "weighted_value_head_root_win": weighted_value_head_root_win,
            "worst_policy_reply": worst_policy_reply,
        }
        event["note"] = (
            "Reply Q values are from opponent perspective at depth-1 child (q_opp_ptm_signed). "
            "q_root_ref_signed flips sign to root perspective. "
            "value_after_reply is direct value-head eval of the resulting position."
        )
        return event

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
        board_tensor = root_state.get_board_tensor()
        board_size = int(board_tensor.shape[-1])
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
        win_probability = termination_info.win_prob

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
        self.verbose = verbose

    def _termination_result_if_any(self, root: MCTSNode, verbose: int) -> Optional[MCTSResult]:
        """Run algorithm-termination checks and build a result if search should stop."""
        termination_info = self._check_algorithm_termination(root, verbose)
        if termination_info is None:
            return None
        return self._build_algorithm_termination_result(root, termination_info, verbose)

    def _expand_root_for_search(self, root: MCTSNode, root_state: HexGameState) -> None:
        """Expand root (and apply root noise) for full search when needed."""
        board_size = int(root_state.get_board_tensor().shape[-1])
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
        win_probability = self.get_win_probability(root, root_state)

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

    def _load_gumbel_policy_inputs(
        self,
        root: MCTSNode,
        timing_tracker: MCTSTimingTracker
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        """Load policy logits and legal mask for Gumbel root selection."""
        timing_tracker.start_timing("gumbel_policy_retrieval")
        board_size = int(root.state.get_board_tensor().shape[-1])
        policy_logits_full, legal_mask = self._get_policy_logits_and_legal_mask(root.state, root.legal_indices)
        timing_tracker.end_timing("gumbel_policy_retrieval")
        return board_size, policy_logits_full, legal_mask

    def _compute_gumbel_context(
        self,
        root: MCTSNode,
        policy_logits_full: np.ndarray,
        legal_mask: np.ndarray,
    ) -> Tuple[int, float, np.ndarray]:
        """Compute move index, root temperature, and legal priors for Gumbel root selection."""
        move_idx = len(root.state.move_history)
        tau = self._root_temperature(move_idx) if self.cfg.gumbel_temperature_enabled else 1.0
        priors_full = self._root_priors_from_logits(policy_logits_full, legal_mask, apply_dirichlet=False)
        return move_idx, tau, priors_full

    def _set_gumbel_selected_action(self, selected_action: int, selected_tensor_action: int) -> None:
        """Persist selected Gumbel root action in both local and tensor-index forms."""
        self._gumbel_selected_action = int(selected_action)
        self._gumbel_selected_tensor_action = int(selected_tensor_action)
        self._used_gumbel_root_selection = True

    def _maybe_finish_gumbel_deterministic_cutoff(
        self,
        root: MCTSNode,
        legal_mask: np.ndarray,
        priors_full: np.ndarray,
        move_idx: int,
        tau: float,
        timing_tracker: MCTSTimingTracker,
        verbose: int,
    ) -> Optional[Dict[str, Any]]:
        """Handle deterministic-cutoff fast path for Gumbel root selection."""
        if tau > self.cfg.gumbel_temperature_deterministic_cutoff:
            return None

        selected_tensor_action = int(np.argmax(np.where(legal_mask, priors_full, -np.inf)))
        selected_action = root.legal_indices.index(selected_tensor_action)
        self._set_gumbel_selected_action(selected_action, selected_tensor_action)
        if verbose >= 4:
            print(f"Gumbel root: move={move_idx}, tau={tau:.3f}, deterministic wrt Dirichlet noise")
        timing_tracker.end_timing("gumbel_algorithm")
        timing_tracker.end_timing("gumbel_selection")
        return timing_tracker.get_final_stats()

    def _build_gumbel_child_accessors(
        self,
        root: MCTSNode
    ) -> Tuple[Callable[[int], float], Callable[[int], int]]:
        """Build Q and N accessors expected by gumbel_alpha_zero_root_batched."""
        action_to_legal_idx = {int(action): idx for idx, action in enumerate(root.legal_indices)}

        def q_of_child(action: int) -> float:
            legal_move_idx = action_to_legal_idx[int(action)]
            if root.N[legal_move_idx] == 0:
                return 0.5
            q_raw = root.Q[legal_move_idx]
            return (q_raw + 1.0) / 2.0

        def n_of_child(action: int) -> int:
            legal_move_idx = action_to_legal_idx[int(action)]
            return int(root.N[legal_move_idx])

        return q_of_child, n_of_child

    def _build_gumbel_trace_callback(self, board_size: int) -> Optional[Callable[[Dict[str, Any]], None]]:
        """Build optional trace callback for detailed Gumbel diagnostics."""
        if not self.detailed_exploration_enabled:
            return None
        return lambda event: self._record_gumbel_trace_event(event, board_size)

    def _run_batched_gumbel_algorithm(
        self,
        root: MCTSNode,
        total_sims: int,
        board_size: int,
        tau: float,
        priors_full: np.ndarray,
        q_of_child: Callable[[int], float],
        n_of_child: Callable[[int], int],
        verbose: int,
    ) -> Tuple[int, Dict[str, Any]]:
        """Run the batched Gumbel root-selection algorithm and return selection + metrics."""
        legal_actions = root.legal_indices.copy()
        logits_for_gumbel = np.log(np.clip(priors_full, 1e-12, 1.0))

        if tau <= 0.1 and verbose >= 5:
            print("MCTS GUMBEL CALL DEBUG:")
            print(f"  Temperature: {tau}")
            print(f"  Total sims: {total_sims}")
            print(f"  Legal actions: {len(legal_actions)}")
            print(f"  Logits range: [{np.min(logits_for_gumbel):.3f}, {np.max(logits_for_gumbel):.3f}]")
            print(f"  Top policy action: {int(np.argmax(logits_for_gumbel))}")

        trace_event_cb = self._build_gumbel_trace_callback(board_size)
        return gumbel_alpha_zero_root_batched(
            mcts=self,
            root=root,
            policy_logits=logits_for_gumbel,
            total_sims=total_sims,
            legal_actions=legal_actions,
            q_of_child=q_of_child,
            n_of_child=n_of_child,
            m=self.cfg.gumbel_m_candidates,
            c_visit=self.cfg.gumbel_c_visit,
            c_scale=self.cfg.gumbel_c_scale,
            temperature=tau,
            verbose=verbose,
            candidate_power_scale=self.cfg.gumbel_candidate_power_scale,
            candidate_power_rate=self.cfg.gumbel_candidate_power_rate,
            candidate_power_offset=self.cfg.gumbel_candidate_power_offset,
            candidate_min=self.cfg.gumbel_candidate_min,
            candidate_max=self.cfg.gumbel_candidate_max,
            use_gumbel_in_final_eval=self.cfg.gumbel_use_gumbel_in_final_eval,
            eval_mode=False,
            trace_event=trace_event_cb,
        )

    def _record_gumbel_metrics(self, gumbel_metrics: Dict[str, Any]) -> None:
        """Record Gumbel performance metrics for inclusion in final stats."""
        self._gumbel_nn_calls_per_move = gumbel_metrics["nn_calls_per_move"]
        self._gumbel_total_leaves_evaluated = gumbel_metrics["total_leaves_evaluated"]
        # For distinct leaves, we'll use final MCTS metrics since per-batch stats don't include this.
        self._gumbel_distinct_leaves_evaluated = 0
        self._gumbel_candidates_m = gumbel_metrics["candidates_m"]
        self._gumbel_rounds_R = gumbel_metrics["rounds_R"]
        self._gumbel_timing_breakdown = gumbel_metrics.get("timing_breakdown", {})

    def _finalize_gumbel_selection(
        self,
        root: MCTSNode,
        selected_tensor_action: int,
        timing_tracker: MCTSTimingTracker,
        verbose: int
    ) -> int:
        """Finalize selected Gumbel action and close timing scopes."""
        timing_tracker.start_timing("gumbel_final_conversion")
        selected_action = root.legal_indices.index(int(selected_tensor_action))
        timing_tracker.end_timing("gumbel_final_conversion")
        timing_tracker.end_timing("gumbel_selection")

        if verbose >= 2:
            print(f"Gumbel selection completed. Selected tensor action {selected_tensor_action} "
                  f"-> legal action {selected_action} ({root.legal_moves[selected_action]})")

        self._set_gumbel_selected_action(selected_action, int(selected_tensor_action))
        return selected_action

    def _capture_gumbel_debug_fields(
        self,
        root: MCTSNode,
        selected_tensor_action: int,
        gumbel_metrics: Dict[str, Any],
        board_size: int
    ) -> None:
        """Best-effort capture of extra Gumbel debug fields and deep-dive events."""
        try:
            self._gumbel_v_pi_01 = gumbel_metrics.get("v_pi_01", None)
            final_rows_raw = gumbel_metrics.get("final_rank_rows", []) or []
            final_rows = self._decorate_gumbel_score_rows_with_moves(final_rows_raw, board_size)
            self._gumbel_final_rank_top5 = final_rows[:5] if final_rows else None
            self._gumbel_final_rank_top_move_trmph = final_rows[0]["move"] if final_rows else None

            if self.detailed_exploration_enabled:
                dive_actions: List[int] = []

                for row in (gumbel_metrics.get("last_round_rows", []) or []):
                    action = row.get("tensor_action", None)
                    if action is None:
                        continue
                    action_int = int(action)
                    if action_int not in dive_actions:
                        dive_actions.append(action_int)

                if not dive_actions:
                    for row in final_rows_raw:
                        action = row.get("tensor_action", None)
                        if action is None:
                            continue
                        action_int = int(action)
                        if action_int not in dive_actions:
                            dive_actions.append(action_int)

                selected_action_int = int(selected_tensor_action)
                if selected_action_int in dive_actions:
                    dive_actions = [selected_action_int] + [a for a in dive_actions if a != selected_action_int]
                else:
                    dive_actions = [selected_action_int] + dive_actions

                for action in dive_actions[:3]:
                    dive_event = self._build_gumbel_action_dive_event(root, action, board_size, top_replies=6)
                    if dive_event is not None:
                        self.exploration_trace.append(dive_event)
        except Exception:
            # Best-effort debug only: do not risk crashing inference due to debug formatting.
            self._gumbel_final_rank_top5 = None
            self._gumbel_final_rank_top_move_trmph = None
            self._gumbel_v_pi_01 = None

    def _run_gumbel_root_selection(self, root: MCTSNode, total_sims: int,
                                 timing_tracker: MCTSTimingTracker, verbose: int) -> Dict[str, Any]:
        """
        Run Gumbel-AlphaZero root selection for small simulation budgets.
        
        This method uses the batched Gumbel implementation that reuses the existing
        MCTS batching infrastructure for maximum efficiency.
        """
        timing_tracker.start_timing("gumbel_selection")
        board_size, policy_logits_full, legal_mask = self._load_gumbel_policy_inputs(root, timing_tracker)

        timing_tracker.start_timing("gumbel_algorithm")
        move_idx, tau, priors_full = self._compute_gumbel_context(root, policy_logits_full, legal_mask)

        maybe_fast_stats = self._maybe_finish_gumbel_deterministic_cutoff(
            root, legal_mask, priors_full, move_idx, tau, timing_tracker, verbose
        )
        if maybe_fast_stats is not None:
            return maybe_fast_stats

        q_of_child, n_of_child = self._build_gumbel_child_accessors(root)
        selected_tensor_action, gumbel_metrics = self._run_batched_gumbel_algorithm(
            root=root,
            total_sims=total_sims,
            board_size=board_size,
            tau=tau,
            priors_full=priors_full,
            q_of_child=q_of_child,
            n_of_child=n_of_child,
            verbose=verbose,
        )

        self._record_gumbel_metrics(gumbel_metrics)
        if verbose >= 4:
            print(f"Gumbel root: move={move_idx}, tau={tau:.3f}")
        timing_tracker.end_timing("gumbel_algorithm")

        self._finalize_gumbel_selection(root, selected_tensor_action, timing_tracker, verbose)
        self._capture_gumbel_debug_fields(root, selected_tensor_action, gumbel_metrics, board_size)
        return timing_tracker.get_final_stats()

    def _root_temperature(self, move_idx: int) -> float:
        """
        Compute temperature for root node based on move index and configuration.
        
        Args:
            move_idx: Current move index (0-based)
            
        Returns:
            Temperature value for this move
        """
        return calculate_temperature_decay(
            temperature_start=self.cfg.temperature_start,
            temperature_end=self.cfg.temperature_end,
            temperature_decay_type=self.cfg.temperature_decay_type,
            temperature_decay_moves=self.cfg.temperature_decay_moves,
            temperature_step_thresholds=self.cfg.temperature_step_thresholds,
            temperature_step_values=self.cfg.temperature_step_values,
            move_count=move_idx,
        )

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
        board_size = int(root_state.get_board_tensor().shape[-1])
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
        board_size = int(root.state.get_board_tensor().shape[-1])
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
    ) -> int:
        """Select child index for one descent step, handling forced root actions and reservations."""
        if node is root and forced_a_full is not None:
            if forced_a_full not in node.legal_indices:
                raise ValueError(
                    f"Gumbel forced illegal root action {forced_a_full}; "
                    f"legal actions count={len(node.legal_indices)}"
                )
            return node.legal_indices.index(forced_a_full)

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

        board_size = int(root.state.get_board_tensor().shape[-1])
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
                node, root, forced_a_full, use_root_reservation, used_root_actions
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
            )
            if flush_reason is not None:
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
        leaves, paths = self._select_leaves_batch(root, sims_remaining=len(actions),
                                                  timing_tracker=timing_tracker,
                                                  forced_root_actions=actions)
        return self._process_leaves_batch(leaves, paths, timing_tracker, root)

    def run_forced_root_actions(self, root: MCTSNode, actions: List[int], verbose: int = 0) -> Dict[str, Any]:
        """Public entry-point used by Gumbel root coordinator; respects batch_cap internally."""
        timing_tracker = MCTSTimingTracker()
        i = 0
        while i < len(actions):
            j = min(i + self.cfg.batch_cap, len(actions))
            sims_done = self._run_forced_root_batch(root, actions[i:j], timing_tracker)
            self._effective_sims_total += sims_done
            i = j
        return timing_tracker.get_final_stats()

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
        board_size = int(leaves[0].state.get_board_tensor().shape[-1])
        action_size = board_size * board_size
        
        for (leaf_idx, policy_np, _) in cached_expansions:
            leaf = leaves[leaf_idx]
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
                print(f"🎮 MCTS: Using top policy move (confidence-based termination, win prob: {termination_info.win_prob:.3f}): {best_move}")
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
        temp = self._root_temperature(move_count)

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
