"""
Support types and helpers for MCTS runtime.

This module isolates non-search orchestration concerns (termination checks, result
containers, and stats shaping) from the core tree search flow in `mcts.py`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING
from collections import OrderedDict

import numpy as np

from hex_ai.config import BOARD_SIZE as CFG_BOARD_SIZE
from hex_ai.value_utils import player_to_winner, red_ref_signed_to_ptm_ref_signed
from hex_ai.inference.mcts_config import BaselineMCTSConfig

if TYPE_CHECKING:
    from hex_ai.inference.mcts import MCTSNode


# Default terminal detection parameters
DEFAULT_MIN_MOVES_FOR_TERMINAL_DETECTION = 2  # Multiplier for board size


class TerminalMoveDetector:
    """Centralized terminal move detection with consistent behavior."""

    def __init__(self, max_detection_depth: int = 3):
        self.max_detection_depth = max_detection_depth

    def should_detect_terminal_moves(self, node: MCTSNode) -> bool:
        """Determine if terminal moves should be detected for this node."""
        if node.depth > self.max_detection_depth:
            return False

        # Impossible to win before BS*2-1 and still unlikely before about BS*3.
        min_move_count = CFG_BOARD_SIZE * DEFAULT_MIN_MOVES_FOR_TERMINAL_DETECTION - 2
        if len(node.state.move_history) < min_move_count:
            return False

        if node._terminal_moves_detected:
            return False

        return True

    def detect_terminal_moves(self, node: MCTSNode) -> bool:
        """
        Detect terminal (immediate winning) moves for a node.

        Returns:
            True if any immediate winning move exists, else False.
        """
        if not self.should_detect_terminal_moves(node):
            return False

        node.terminal_moves = [False] * len(node.legal_moves)
        for i, (row, col) in enumerate(node.legal_moves):
            new_state = node.state.make_move(row, col)
            if new_state.game_over and new_state.winner == player_to_winner(node.to_play):
                node.terminal_moves[i] = True

        node._terminal_moves_detected = True
        return any(node.terminal_moves)

    def get_terminal_move(self, node: MCTSNode) -> Optional[Tuple[int, int]]:
        """Get the first detected terminal move, if any."""
        if not node._terminal_moves_detected:
            return None

        for i, is_terminal in enumerate(node.terminal_moves):
            if is_terminal:
                return node.legal_moves[i]
        return None


@dataclass
class AlgorithmTerminationInfo:
    """Simple info about algorithm termination."""

    reason: str  # "terminal_move" or "neural_network_confidence"
    move: Optional[Tuple[int, int]]  # Move to play (None for NN confidence)
    win_prob: float  # Win probability


@dataclass(frozen=True)
class MCTSResult:
    """Complete result of an MCTS search."""

    move: Tuple[int, int]
    stats: Dict[str, Any]
    tree_data: Dict[str, Any]
    root_node: MCTSNode
    algorithm_termination_info: Optional[AlgorithmTerminationInfo]
    win_probability: float


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
            "algorithm_termination_reason": termination_info.reason if termination_info else "unknown",
        })
        return stats

    def create_final_stats(self, timing_stats: Dict[str, Any], total_simulations: int, total_search_time: float) -> Dict[str, Any]:
        """Create final stats for completed MCTS runs."""
        stats = self.create_base_stats()
        stats.update(timing_stats)
        stats.update({
            "total_simulations": total_simulations,
            "simulations_per_second": total_simulations / total_search_time if total_search_time > 0 else 0.0,
            "algorithm_termination_occurred": False,
            "algorithm_termination_reason": "none",
        })
        return stats


class AlgorithmTerminationChecker:
    """Centralized algorithm termination checking with simple priority order."""

    def __init__(self, cfg: BaselineMCTSConfig, terminal_detector: TerminalMoveDetector):
        self.cfg = cfg
        self.terminal_detector = terminal_detector

    def should_terminate_early(
        self,
        root: MCTSNode,
        verbose: int,
        eval_cache: Dict[int, Tuple[np.ndarray, float]],
        root_is_expanded: bool = False,
    ) -> Optional[AlgorithmTerminationInfo]:
        """
        Check if we should terminate early.

        Returns None if search should continue, otherwise termination info.
        """
        if self.cfg.enable_terminal_move_detection:
            if self.terminal_detector.detect_terminal_moves(root):
                terminal_move = self.terminal_detector.get_terminal_move(root)
                if verbose >= 2:
                    print(f"🎮 MCTS: Found terminal move: {terminal_move}")
                return AlgorithmTerminationInfo(
                    reason="terminal_move",
                    move=terminal_move,
                    win_prob=1.0,
                )

        if self.cfg.enable_confidence_termination and root_is_expanded and not root.is_terminal:
            signed_value = self._get_root_signed_value(root, eval_cache)
            if self._is_position_clearly_decided(signed_value):
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
                    move=None,
                    win_prob=signed_value,
                )

        return None

    def _get_root_signed_value(self, root: MCTSNode, eval_cache: OrderedDict[int, Tuple[np.ndarray, float]]) -> float:
        """Get signed value for current player from neural network cache."""
        cached = eval_cache.get(root.state_hash)
        if cached is None:
            raise RuntimeError(f"Root state not found in cache: {root.state_hash}")
        _, value_signed = cached

        if not -1.1 <= value_signed <= 1.1:
            raise ValueError(
                f"Neural network output {value_signed} is outside expected signed range [-1, 1]. "
                "This suggests a mismatch between probability and signed value semantics."
            )

        v_red_ref_signed = float(value_signed)
        return red_ref_signed_to_ptm_ref_signed(v_red_ref_signed, root.to_play)

    def _is_position_clearly_decided(self, signed_value: float) -> bool:
        """Check if position is clearly won or lost using signed values."""
        return (
            signed_value >= self.cfg.confidence_termination_threshold
            or signed_value <= -self.cfg.confidence_termination_threshold
        )

