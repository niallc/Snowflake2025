# baseline_mcts.py
# Lean, single-threaded, explicitly-batched AlphaZero-style MCTS for Hex.
# Compatible with flat-file or package imports via shims.
#
# WINNER PERSPECTIVE HANDLING:
# This MCTS implementation uses a consistent approach to handle winner perspectives:
# 1. Neural network outputs values in Red's perspective: +1 = Red win, -1 = Blue win
# 2. Terminal nodes return values in Red's perspective: +1 = Red win, -1 = Blue win  
# 3. During backpropagation, values are flipped to player-to-move perspective using signed_for_player_to_move()
# 4. Final output is converted to probability using to_prob() for external API
# 
# Key variable naming convention:
# - v_red_signed: Always in Red's perspective (+1 = Red win, -1 = Blue win)
# - v_player_to_move_signed: In player-to-move perspective (+1 = current player wins, -1 = current player loses)
# - value_signed: From neural network cache, in Red's perspective
#
# TODO: Future improvements to consider:
# - Add memory pooling for large tree structures to reduce allocation overhead
# - Implement cleanup of old cached evaluations to prevent memory leaks
# - Add support for parallel MCTS with proper synchronization
# - Create extensible early termination strategy framework
# - Add tree visualization utilities for debugging
# - Implement different tree policies (UCB1, etc.) as pluggable components
#
# TODO: Code Duplication Cleanup Opportunities (identified 2025-01-27):
# - Move selection logic may be duplicated across mcts.py, move_selection.py, web/app.py
# - Consider creating centralized utilities for these common patterns
# - Current mcts.py file is clean - no internal duplicates found

from __future__ import annotations

import math
import time
import random
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set
from collections import OrderedDict

# ---- Package imports ----
from hex_ai.enums import Player, Winner
from hex_ai.value_utils import player_to_winner, prob_for_player_to_move, apply_depth_discount_toward_neutral_prob, signed_for_player_to_move, apply_depth_discount_signed, distance_to_leaf, to_prob
from hex_ai.inference.game_engine import HexGameState, HexGameEngine
from hex_ai.inference.model_wrapper import ModelWrapper
from hex_ai.utils.perf import PERF
from hex_ai.utils.math_utils import softmax_np
from hex_ai.utils.format_conversion import rowcol_to_tensor_with_size as move_to_index, tensor_to_rowcol as index_to_move
from hex_ai.utils.temperature import calculate_temperature_decay
from hex_ai.utils.state_utils import board_key, validate_move_coordinates, is_valid_move_coordinates
from hex_ai.utils.timing import MCTSTimingTracker
from hex_ai.config import BOARD_SIZE as CFG_BOARD_SIZE, POLICY_OUTPUT_SIZE as CFG_POLICY_OUTPUT_SIZE, DEFAULT_BATCH_CAP, DEFAULT_C_PUCT
from hex_ai.value_utils import ValuePredictor, winner_to_color

# ---- MCTS Constants ----
# Temperature comparison threshold for move selection
TEMPERATURE_COMPARISON_THRESHOLD = 0.02  # Use deterministic selection for very low temperatures

# Principal variation extraction limit
PRINCIPAL_VARIATION_MAX_LENGTH = 10

# PUCT calculation threshold for avoiding division by zero
PUCT_CALCULATION_THRESHOLD = 1e-9

# Default confidence-based termination threshold
DEFAULT_CONFIDENCE_TERMINATION_THRESHOLD = 0.9

# Tournament-specific confidence-based termination threshold (higher confidence for tournament play)
TOURNAMENT_CONFIDENCE_TERMINATION_THRESHOLD = 0.95

# Default terminal move boost factor
DEFAULT_TERMINAL_MOVE_BOOST = 4.0

# Default virtual loss for non-terminal moves
DEFAULT_VIRTUAL_LOSS_FOR_NON_TERMINAL = 0.01

# Default depth discount factor
DEFAULT_DEPTH_DISCOUNT_FACTOR = 0.95



# Default batch cap for neural network evaluation (imported from hex_ai.config)
# DEFAULT_BATCH_CAP = 64

# Default PUCT exploration constant (imported from hex_ai.config)
# DEFAULT_C_PUCT = 1.5

# Default cache size for LRU eviction
DEFAULT_CACHE_SIZE = 100000  # 100k entries

# Default Dirichlet noise parameters
DEFAULT_DIRICHLET_ALPHA = 0.3
DEFAULT_DIRICHLET_EPS = 0.25

# Default temperature parameters
DEFAULT_TEMPERATURE_START = 1.0
DEFAULT_TEMPERATURE_END = 0.1
DEFAULT_TEMPERATURE_DECAY_TYPE = "exponential"
DEFAULT_TEMPERATURE_DECAY_MOVES = 50

# Default terminal detection parameters
DEFAULT_TERMINAL_DETECTION_MAX_DEPTH = 3
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
        min_move_count = CFG_BOARD_SIZE * DEFAULT_MIN_MOVES_FOR_TERMINAL_DETECTION - 1
        if len(node.state.move_history) < min_move_count:
            return False
        
        # Don't detect if already done
        if node._terminal_moves_detected:
            return False
        
        return True
    
    def detect_terminal_moves(self, node: MCTSNode, board_size: int) -> bool:
        """Detect terminal moves for a node. Returns True if any found."""
        if not self.should_detect_terminal_moves(node):
            return False
        
        # Reset terminal moves
        node.terminal_moves = [False] * len(node.legal_moves)
        
        # Check each legal move
        for i, (row, col) in enumerate(node.legal_moves):
            try:
                new_state = node.state.make_move(row, col)
                if new_state.game_over and new_state.winner == player_to_winner(node.to_play):
                    node.terminal_moves[i] = True
            except Exception:
                pass
        
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
    
    def should_terminate_early(self, root: MCTSNode, board_size: int, verbose: int, eval_cache: Dict[int, Tuple[np.ndarray, float]], root_is_expanded: bool = False) -> Optional[AlgorithmTerminationInfo]:
        """
        Check if we should terminate early. Returns None if we should continue with MCTS.
        Returns EarlyTerminationInfo if we should terminate.
        
        Args:
            root: The root node to check
            board_size: Board size for terminal move detection
            verbose: Verbosity level
            eval_cache: Evaluation cache for neural network confidence
            root_is_expanded: Whether the root node has been expanded (affects confidence checking)
        """
        # 1. Check for terminal moves (highest priority) - works regardless of expansion
        if self.cfg.enable_terminal_move_detection:
            if self.terminal_detector.detect_terminal_moves(root, board_size):
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
            win_prob = self._get_root_win_probability(root, eval_cache)
            if self._is_position_clearly_decided(win_prob):
                if verbose >= 2:
                    print(f"🎮 MCTS: Confidence-based termination (win prob: {win_prob:.3f})")
                return AlgorithmTerminationInfo(
                    reason="neural_network_confidence",
                    move=None,  # Will use top policy move
                    win_prob=win_prob
                )
        
        return None  # Continue with MCTS
    
    def _get_root_win_probability(self, root: MCTSNode, eval_cache: OrderedDict[int, Tuple[np.ndarray, float]]) -> float:
        """Get win probability for current player from neural network (edge conversion)."""
        cached = eval_cache.get(root.state_hash)
        if cached is None:
            raise RuntimeError(f"Root state not found in cache: {root.state_hash}")
        _, value_signed = cached
        # Convert signed value to probability only at the edge (for confidence termination)
        # value_signed is the tanh-activated output in [-1,1] range in Red's perspective
        v_red_signed = float(value_signed)
        # Flip to player-to-move perspective: if RED to move keep v_red_signed, if BLUE to move flip to -v_red_signed
        v_player_to_move_signed = signed_for_player_to_move(v_red_signed, root.to_play)
        # Convert to probability for confidence checking
        return to_prob(v_player_to_move_signed)
    
    def _is_position_clearly_decided(self, win_prob: float) -> bool:
        """Check if position is clearly won or lost."""
        return (win_prob >= self.cfg.confidence_termination_threshold or 
                win_prob <= (1.0 - self.cfg.confidence_termination_threshold))

# ------------------ Config ------------------
@dataclass
class BaselineMCTSConfig:
    sims: int = 200
    batch_cap: int = DEFAULT_BATCH_CAP
    c_puct: float = DEFAULT_C_PUCT
    cache_size: int = DEFAULT_CACHE_SIZE
    dirichlet_alpha: float = DEFAULT_DIRICHLET_ALPHA
    dirichlet_eps: float = DEFAULT_DIRICHLET_EPS
    add_root_noise: bool = False
    # Temperature scaling parameters (always used)
    temperature_start: float = DEFAULT_TEMPERATURE_START  # Starting temperature
    temperature_end: float = DEFAULT_TEMPERATURE_END  # Final temperature (minimum)
    temperature_decay_type: str = DEFAULT_TEMPERATURE_DECAY_TYPE  # "linear", "exponential", "step", "game_progress"
    temperature_decay_moves: int = DEFAULT_TEMPERATURE_DECAY_MOVES  # Number of moves for decay (for linear/exponential)
    temperature_step_thresholds: List[int] = field(default_factory=lambda: [10, 25, 50])  # Move thresholds for step decay
    temperature_step_values: List[float] = field(default_factory=lambda: [0.8, 0.5, 0.2])  # Temperature values for step decay
    # Terminal move detection parameters
    enable_terminal_move_detection: bool = True  # Enable immediate terminal move detection
    terminal_move_boost: float = DEFAULT_TERMINAL_MOVE_BOOST  # Boost factor for terminal moves in PUCT calculation
    virtual_loss_for_non_terminal: float = DEFAULT_VIRTUAL_LOSS_FOR_NON_TERMINAL  # Small penalty for non-terminal moves
    terminal_detection_max_depth: int = DEFAULT_TERMINAL_DETECTION_MAX_DEPTH  # Maximum depth for terminal move detection
    # Note: Pre-check only happens after move BOARD_SIZE * 3 (minimum moves needed for a win)
    # Removed seed parameter - randomness should be controlled externally

    # Confidence-based termination parameters
    enable_confidence_termination: bool = False
    confidence_termination_threshold: float = DEFAULT_CONFIDENCE_TERMINATION_THRESHOLD
    
    # Depth-based discounting parameters
    enable_depth_discounting: bool = True
    depth_discount_factor: float = DEFAULT_DEPTH_DISCOUNT_FACTOR  # Discount wins by this factor per depth level
    # When enabled, wins found deeper in the search tree are discounted to encourage
    # the algorithm to prefer shorter winning sequences. This helps avoid meandering
    # when the position is already won. Only applies during MCTS search, not during
    # early termination when using the policy network.
    

    # This makes actual terminal wins (immediate wins) even more attractive than
    # neural network evaluations, encouraging the algorithm to find and prefer them.
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        if self.sims <= 0:
            raise ValueError(f"sims must be positive, got {self.sims}")
        if self.batch_cap <= 0:
            raise ValueError(f"batch_cap must be positive, got {self.batch_cap}")
        if self.c_puct <= 0:
            raise ValueError(f"c_puct must be positive, got {self.c_puct}")
        if self.cache_size <= 0:
            raise ValueError(f"cache_size must be positive, got {self.cache_size}")
        if self.dirichlet_alpha <= 0:
            raise ValueError(f"dirichlet_alpha must be positive, got {self.dirichlet_alpha}")
        if not 0 <= self.dirichlet_eps <= 1:
            raise ValueError(f"dirichlet_eps must be between 0 and 1, got {self.dirichlet_eps}")
        if self.temperature_start <= 0:
            raise ValueError(f"temperature_start must be positive, got {self.temperature_start}")
        if self.temperature_end <= 0:
            raise ValueError(f"temperature_end must be positive, got {self.temperature_end}")
        if self.temperature_start < self.temperature_end:
            raise ValueError(f"temperature_start ({self.temperature_start}) must be >= temperature_end ({self.temperature_end})")
        if self.temperature_decay_moves <= 0:
            raise ValueError(f"temperature_decay_moves must be positive, got {self.temperature_decay_moves}")
        if self.terminal_move_boost < 0:
            raise ValueError(f"terminal_move_boost must be non-negative, got {self.terminal_move_boost}")
        if self.virtual_loss_for_non_terminal < 0:
            raise ValueError(f"virtual_loss_for_non_terminal must be non-negative, got {self.virtual_loss_for_non_terminal}")
        if self.terminal_detection_max_depth < 0:
            raise ValueError(f"terminal_detection_max_depth must be non-negative, got {self.terminal_detection_max_depth}")
        if not 0 <= self.confidence_termination_threshold <= 1:
            raise ValueError(f"confidence_termination_threshold must be between 0 and 1, got {self.confidence_termination_threshold}")
        if not 0 < self.depth_discount_factor <= 1:
            raise ValueError(f"depth_discount_factor must be between 0 and 1, got {self.depth_discount_factor}")

        
        # Validate temperature step parameters if using step decay
        if self.temperature_decay_type == "step":
            if len(self.temperature_step_thresholds) != len(self.temperature_step_values):
                raise ValueError("temperature_step_thresholds and temperature_step_values must have the same length")
            if not all(t >= 0 for t in self.temperature_step_thresholds):
                raise ValueError("All temperature_step_thresholds must be non-negative")
            if not all(0 < v <= 1 for v in self.temperature_step_values):
                raise ValueError("All temperature_step_values must be between 0 and 1")

# ------------------ Data structures ------------------

class MCTSNode:
    __slots__ = (
        "state", "to_play", "legal_moves", "legal_indices",
        "children", "N", "W", "Q", "P", "is_expanded",
        "state_hash", "is_terminal", "winner_str", "terminal_moves",
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
        self.W = np.zeros(L, dtype=np.float64) # total value per action
        self.Q = np.zeros(L, dtype=np.float64) # mean value per action
        self.P = np.zeros(L, dtype=np.float64) # prior probability per action (set on expand)
        self.is_expanded: bool = False
        self.state_hash: int = board_key(state)
        self.is_terminal: bool = bool(state.game_over)
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
        # Note: value_signed is the tanh-activated output in [-1,1] range in Red's perspective (not a logit)
        self.eval_cache: OrderedDict[int, Tuple[np.ndarray, float]] = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        # Metrics
        self._unique_evals_total = 0     # post-dedup, network calls actually done
        self._effective_sims_total = 0   # counts every backprop (incl. duplicates)

        # Terminal move detection
        self.terminal_detector = TerminalMoveDetector(
            max_detection_depth=cfg.terminal_detection_max_depth
        )
        
        # Algorithm termination checker
        self.algorithm_termination_checker = AlgorithmTerminationChecker(cfg, self.terminal_detector)
        
        # Stats builder (will be updated with current cache counts when needed)
        self.stats_builder = None

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
        if root_state is None:
            raise ValueError("root_state cannot be None")
        if root_state.game_over:
            raise RuntimeError("Cannot run MCTS on a terminal game state")
        if verbose < 0:
            raise ValueError(f"verbose must be non-negative, got {verbose}")
        # Prepare root node
        root = self._prepare_root_node(root_state, verbose)
        
        # Check for algorithm termination opportunities
        termination_info = self._check_algorithm_termination(root, verbose)
        if termination_info:
            # Handle algorithm termination
            move = self._get_algorithm_termination_move(root, termination_info, verbose)
            tree_data = self._compute_tree_data(root)
            win_probability = termination_info.win_prob
            
            # Attach metrics for algorithm termination cases too
            stats = self._get_stats_builder().create_algorithm_termination_stats(termination_info)
            total_time = 0.0  # Algorithm termination has minimal time
            stats["unique_evals_total"] = int(self._unique_evals_total)
            stats["effective_sims_total"] = int(self._effective_sims_total)
            stats["unique_evals_per_sec"] = 0.0  # No meaningful time for algorithm termination
            stats["effective_sims_per_sec"] = 0.0
            
            return MCTSResult(
                move=move,
                stats=stats,
                tree_data=tree_data,
                root_node=root,
                algorithm_termination_info=termination_info,
                win_probability=win_probability
            )
        
        # Run main simulation loop
        timing_stats = self._run_simulation_loop(root, verbose)
        # Attach metrics (don't rely on TimingTracker internals)
        total_time = float(timing_stats.get("total_search_time", 0.0)) or 1e-9
        timing_stats["unique_evals_total"] = int(self._unique_evals_total)
        timing_stats["effective_sims_total"] = int(self._effective_sims_total)
        timing_stats["unique_evals_per_sec"] = self._unique_evals_total / total_time
        timing_stats["effective_sims_per_sec"] = self._effective_sims_total / total_time
        
        # Compute results directly
        move = self._compute_move(root, root_state, verbose)
        tree_data = self._compute_tree_data(root)
        win_probability = self._compute_win_probability(root, root_state)
        
        return MCTSResult(
            move=move,
            stats=self._get_stats_builder().create_final_stats(
                timing_stats, self.cfg.sims, timing_stats.get("total_search_time", 0.0)
            ),
            tree_data=tree_data,
            root_node=root,
            algorithm_termination_info=None,
            win_probability=win_probability
        )

    def _prepare_root_node(self, root_state: HexGameState, verbose: int) -> MCTSNode:
        """Prepare and initialize the root node for MCTS search."""
        board_tensor = root_state.get_board_tensor()
        board_size = int(board_tensor.shape[-1])
        root = MCTSNode(root_state, board_size)
        
        # Expand root if not terminal
        if not root.is_terminal and not root.is_expanded:
            self._expand_root_node(root, board_size)
        
        # Apply root noise if configured (only on first move for consistency)
        if self.cfg.add_root_noise and not root.is_terminal and root.is_expanded and len(root_state.move_history) == 0:
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
        board_size = int(root.state.get_board_tensor().shape[-1])
        
        # Single call with root_is_expanded flag - handles both terminal moves and confidence checking
        termination_info = self.algorithm_termination_checker.should_terminate_early(
            root, board_size, verbose, self.eval_cache, root_is_expanded=root.is_expanded
        )
        
        return termination_info

    def _run_simulation_loop(self, root: MCTSNode, verbose: int) -> Dict[str, Any]:
        """Run the main MCTS simulation loop with batching."""
        timing_tracker = MCTSTimingTracker()
        sims_remaining = self.cfg.sims
        
        while sims_remaining > 0:
            # Select leaves for this batch
            leaves, paths = self._select_leaves_batch(root, sims_remaining, timing_tracker)
            
            # Process leaves (expand and backpropagate)
            batch_simulations = self._process_leaves_batch(leaves, paths, timing_tracker)
            sims_remaining -= batch_simulations
            self._effective_sims_total += batch_simulations
        
        return timing_tracker.get_final_stats()

    def _select_leaves_batch(
        self,
        root: MCTSNode,
        sims_remaining: int,
        timing_tracker: MCTSTimingTracker
    ) -> Tuple[List[MCTSNode], List[List[Tuple[MCTSNode, int]]]]:
        """Select a batch of leaves for expansion with early flush triggers."""
        timing_tracker.start_timing("select")

        leaves: List[MCTSNode] = []
        paths: List[List[Tuple[MCTSNode, int]]] = []

        board_size = int(root.state.get_board_tensor().shape[-1])
        select_budget = min(self.cfg.batch_cap, sims_remaining)

        # Distinct-leaf target: ~50% of budget, clamped to [16, budget]
        distinct_target = max(16, int(round(select_budget * 0.5)))
        distinct_target = min(distinct_target, select_budget)

        # Track distinct (uncached+unexpanded) leaf hashes this batch
        distinct_hashes: Set[int] = set()

        # Cheap guardrail on selection work
        max_selection_descents = 4 * select_budget
        descents = 0

        while len(leaves) < select_budget and descents < max_selection_descents:
            descents += 1

            node = root
            path: List[Tuple[MCTSNode, int]] = []

            while True:
                if node.is_terminal:
                    leaves.append(node)
                    paths.append(path.copy())
                    break
                if not node.is_expanded:
                    leaves.append(node)
                    paths.append(path)

                    # Only count toward distinct if this state actually needs eval
                    if node.state_hash not in self.eval_cache and not node.is_terminal:
                        distinct_hashes.add(node.state_hash)

                    # Flush triggers
                    U = len(distinct_hashes)
                    T = len(leaves)
                    if U >= distinct_target:
                        timing_tracker.end_timing("select")
                        return leaves, paths
                    if T >= 16 and U / max(1, T) < 0.5:
                        timing_tracker.end_timing("select")
                        return leaves, paths
                    break

                # PUCT descent
                child_idx = self._select_child_puct(node)
                path.append((node, child_idx))
                child = node.children[child_idx]

                if child is None:
                    timing_tracker.start_timing("state_creation")
                    (r, c) = node.legal_moves[child_idx]
                    timing_tracker.start_timing("make_move")
                    child_state = node.state.make_move(r, c)
                    timing_tracker.end_timing("make_move")
                    child = MCTSNode(child_state, board_size)
                    child.depth = node.depth + 1
                    timing_tracker.end_timing("state_creation")
                    node.children[child_idx] = child

                node = child

                # Outer budget guard (kept from original)
                if len(leaves) >= select_budget:
                    break

        timing_tracker.end_timing("select")
        return leaves, paths

    def _process_leaves_batch(self, leaves: List[MCTSNode], paths: List[List[Tuple[MCTSNode, int]]], 
                            timing_tracker: MCTSTimingTracker) -> int:
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
        simulations_completed = self._backpropagate_batch(leaves, paths, timing_tracker)
        
        return simulations_completed

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
            # Skip leaves that don't need eval
            if leaf.is_terminal or leaf.is_expanded:
                continue

            # Cached?
            timing_tracker.start_timing("cache_lookup")
            cached = self._get_from_cache(leaf.state_hash)
            timing_tracker.end_timing("cache_lookup")

            if cached is not None:
                self.cache_hits += 1
                policy_np, value_signed = cached
                cached_expansions.append((i, policy_np, value_signed))
                continue

            # Uncached → only encode the first occurrence of this state in the batch
            if leaf.state_hash in seen_uncached:
                # Another copy will piggy-back on the first result via eval_cache.
                continue
            seen_uncached.add(leaf.state_hash)

            self.cache_misses += 1
            enc = leaf.state.get_board_tensor().to(dtype=torch.float32)
            encodings.append(enc)
            need_eval_idxs.append(i)

        timing_tracker.end_timing("encode")
        timing_tracker.end_timing("stack")
        return encodings, need_eval_idxs, cached_expansions

    def _run_neural_network_batch(self, encodings: List[torch.Tensor], need_eval_idxs: List[int], 
                                leaves: List[MCTSNode], timing_tracker: MCTSTimingTracker):
        """Run neural network inference on a batch of leaves."""
        if not encodings:
            return
        
        batch_tensor = torch.stack(encodings, dim=0)
        board_size = int(batch_tensor.shape[-1])
        action_size = board_size * board_size
        
        policy_cpu, value_cpu, tm = self.model.infer_timed(batch_tensor)
        self._unique_evals_total += int(tm.get("batch_size", len(encodings)))
        
        # Record performance metrics
        self._record_eval_perf(tm, is_first=(timing_tracker.batch_count == 0))
        timing_tracker.record_batch_metrics(tm)
        
        # Cache results and expand nodes
        for j, leaf_idx in enumerate(need_eval_idxs):
            leaf = leaves[leaf_idx]
            pol = policy_cpu[j].numpy()
            val_signed = float(value_cpu[j].item())  # tanh-activated output in [-1,1] range
            
            # Cache CPU-native results
            self._put_in_cache(leaf.state_hash, pol, val_signed)
            self._expand_node_from_policy(leaf, pol, board_size, action_size)

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
            # Determine signed value for this leaf (always in Red's perspective: +1 = Red win, -1 = Blue win)
            if leaf.is_terminal:
                v_red_signed = self._get_terminal_value(leaf)
            else:
                v_red_signed = self._get_neural_network_value(leaf)
            
            # Backpropagate Red's signed value along path (will be flipped to player-to-move perspective during backprop)
            self._backpropagate_path(path, v_red_signed)
            simulations_completed += 1
        
        timing_tracker.end_timing("backprop")
        return simulations_completed

    def _get_terminal_value(self, leaf: MCTSNode) -> float:
        """Get signed value for a terminal leaf in Red's perspective: +1 = Red win, -1 = Blue win."""
        if leaf.winner_str == "red":
            return 1.0  # +1 = certain Red win
        elif leaf.winner_str == "blue":
            return -1.0  # -1 = certain Blue win
        else:
            raise ValueError(f"Invalid winner_str for terminal Hex node: {leaf.winner_str!r} (draws are not possible in Hex)")

    def _get_neural_network_value(self, leaf: MCTSNode) -> float:
        """Get signed value for a non-terminal leaf from neural network in Red's perspective: +1 = Red win, -1 = Blue win."""
        cached = self._get_from_cache(leaf.state_hash)
        if cached is None:
            raise RuntimeError(f"Leaf state not found in cache: {leaf.state_hash}")
        _, value_signed = cached
        # Return signed value directly - tanh activation gives values in [-1,1] range in Red's perspective
        return float(value_signed)

    def _backpropagate_path(self, path: List[Tuple[MCTSNode, int]], v_red_signed: float):
        """
        Backpropagate signed value along a path from leaf to root.
        
        Args:
            path: List of (node, action_index) pairs from root to leaf
            v_red_signed: Signed value in Red's perspective (+1 = Red win, -1 = Blue win)
        """
        for (node, a_idx) in reversed(path):
            # Convert Red's signed value to player-to-move perspective
            # If RED to move: keep v_red_signed; if BLUE to move: flip to -v_red_signed
            v_player_to_move_signed = signed_for_player_to_move(v_red_signed, node.to_play)
            
            # Apply depth discounting in signed space (shrink toward 0)
            if self.cfg.enable_depth_discounting and node.depth > 0:
                # TODO(step: tune): consider using distance_to_leaf instead of absolute depth
                # TODO(step: tune): retune depth_discount_factor for signed space
                v_player_to_move_signed = apply_depth_discount_signed(v_player_to_move_signed, self.cfg.depth_discount_factor, node.depth)
            
            node.N[a_idx] += 1
            node.W[a_idx] += v_player_to_move_signed
            # TODO: The max(1, node.N[a_idx]) looks redundant, surely node.N[a_idx] >= 1 from above.
            node.Q[a_idx] = node.W[a_idx] / max(1, node.N[a_idx])

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

    def _compute_move(self, root: MCTSNode, root_state: HexGameState, verbose: int) -> Tuple[int, int]:
        """Compute the selected move from the root node."""
        # Check if a terminal move was found during pre-check
        if self.cfg.enable_terminal_move_detection and any(root.terminal_moves):
            terminal_indices = [i for i, is_terminal in enumerate(root.terminal_moves) if is_terminal]
            if terminal_indices:
                terminal_move = root.legal_moves[terminal_indices[0]]
                if verbose >= 2:
                    print(f"🎮 MCTS: Using pre-detected terminal move: {terminal_move}")
                return terminal_move

        # Check terminal moves (no fallback needed - already detected if appropriate)
        if self.cfg.enable_terminal_move_detection:
            terminal_move = self.terminal_detector.get_terminal_move(root)
            if terminal_move:
                if verbose >= 2:
                    print(f"🎮 MCTS: Using terminal move: {terminal_move}")
                return terminal_move

        # Use visit counts accumulated during run()
        counts = root.N.astype(np.float64)
        if counts.sum() <= 0:
            raise RuntimeError(f"No visits recorded during MCTS search. This indicates a bug in the search algorithm.")

        # Calculate temperature with decay
        move_count = len(root_state.move_history)
        start_temp = self.cfg.temperature_start
        temp = calculate_temperature_decay(
            temperature_start=self.cfg.temperature_start,
            temperature_end=self.cfg.temperature_end,
            temperature_decay_type=self.cfg.temperature_decay_type,
            temperature_decay_moves=self.cfg.temperature_decay_moves,
            temperature_step_thresholds=self.cfg.temperature_step_thresholds,
            temperature_step_values=self.cfg.temperature_step_values,
            move_count=move_count,
            start_temp_override=start_temp
        )
        
        # Log effective temperature if verbose
        if verbose >= 4:
            print(f"🎮 MCTS: Move {move_count}, effective temperature: {temp:.3f}")
        
        if temp <= TEMPERATURE_COMPARISON_THRESHOLD:
            # Use deterministic selection for very low temperatures to avoid numerical issues
            a_idx = int(np.argmax(counts))
        else:
            # Apply temperature scaling with validation
            try:
                pi = np.power(counts, 1.0 / temp)
                if not np.isfinite(pi).all():
                    raise ValueError(f"Temperature scaling produced non-finite values: pi={pi}, counts={counts}, temp={temp}")
                if np.sum(pi) <= 0:
                    raise ValueError(f"Temperature scaling produced non-positive sum: pi={pi}, counts={counts}, temp={temp}")
                pi /= np.sum(pi)
                a_idx = int(np.random.choice(len(pi), p=pi))
            except (OverflowError, ValueError) as e:
                # Fall back to deterministic selection if temperature scaling fails
                print(f"Warning: Temperature scaling failed with temp={temp}, falling back to deterministic selection. Error: {e}")
                a_idx = int(np.argmax(counts))
        return root.legal_moves[a_idx]

    def _compute_tree_data(self, root: MCTSNode) -> Dict[str, Any]:
        """Compute tree data for analysis."""
        # TODO: Consider caching this computation if called multiple times
        if root.is_terminal:
            return {
                "visit_counts": {},
                "mcts_probabilities": {},
                "root_value": 0.0,
                "best_child_value": 0.0,
                "total_visits": 0,
                "inferences": 0,
                "total_nodes": 0,
                "max_depth": 0
            }

        # Get visit counts and convert to TRMPH format
        visit_counts = {}
        mcts_probabilities = {}
        total_visits = int(np.sum(root.N))
        
        for i, (row, col) in enumerate(root.legal_moves):
            move_trmph = f"{chr(ord('a') + col)}{row + 1}"
            visits = int(root.N[i])
            visit_counts[move_trmph] = visits
            
            # Calculate MCTS probability (visit count / total visits)
            if total_visits > 0:
                mcts_probabilities[move_trmph] = visits / total_visits
            else:
                mcts_probabilities[move_trmph] = 0.0

        # Get root value (average value of all children)
        if total_visits > 0:
            root_value = float(np.sum(root.W) / total_visits)
        else:
            root_value = 0.0

        # Get best child value
        if len(root.Q) > 0:
            best_child_value = float(np.max(root.Q))
        else:
            best_child_value = 0.0

        # Calculate total inferences (cache misses)
        total_inferences = self.cache_misses

        # Calculate tree traversal statistics
        total_nodes, max_depth = self._calculate_tree_statistics(root)

        # Get principal variation
        principal_variation = self._extract_principal_variation(root, max_length=PRINCIPAL_VARIATION_MAX_LENGTH)

        return {
            "visit_counts": visit_counts,
            "mcts_probabilities": mcts_probabilities,
            "root_value": root_value,
            "best_child_value": best_child_value,
            "total_visits": total_visits,
            "inferences": total_inferences,
            "total_nodes": total_nodes,
            "max_depth": max_depth,
            "principal_variation": principal_variation
        }

    def _compute_win_probability(self, root: MCTSNode, root_state: HexGameState) -> float:
        """Compute win probability for the current player based on root value (edge conversion)."""
        tree_data = self._compute_tree_data(root)
        root_value = tree_data["root_value"]
        
        # Convert signed value to probability only at the edge (for external API)
        # Root value is already in player-to-move perspective from backpropagation
        # +1 = current player wins, -1 = current player loses, 0 = neutral
        return to_prob(root_value)

    def _extract_principal_variation(self, root: MCTSNode, max_length: int = 10) -> List[Tuple[int, int]]:
        """Extract the principal variation (best move sequence) from the MCTS tree."""
        if root is None:
            raise ValueError("Root node cannot be None")
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")
        
        if root.is_terminal:
            return []

        pv = []
        current_node = root
        
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

    def _calculate_tree_statistics(self, root: MCTSNode) -> Tuple[int, int]:
        """
        Calculate tree traversal statistics.
        
        Args:
            root: Root node of the tree
            
        Returns:
            Tuple of (total_nodes, max_depth)
        """
        if root is None:
            return 0, 0
        
        def count_nodes_and_depth(node: MCTSNode, current_depth: int) -> Tuple[int, int]:
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
        
        return count_nodes_and_depth(root, 0)


    # ---------- Internal ----------

    # Removed old early termination methods - replaced by EarlyTerminationChecker

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

    def _select_child_puct(self, node: MCTSNode) -> int:
        """Return index into node.legal_moves of the action maximizing PUCT score."""
        # PUCT: U = c_puct * P * sqrt(sum(N)) / (1 + N)
        # score = Q + U
        # TODO(step: tune): retune c_puct for signed Q values in [-1,1] range
        
        # Detect terminal moves if enabled and appropriate
        if self.cfg.enable_terminal_move_detection:
            self.terminal_detector.detect_terminal_moves(node, int(node.state.get_board_tensor().shape[-1]))
        
        # Start timing PUCT calculation
        t_puct_start = time.perf_counter()
        
        N_sum = np.sum(node.N, dtype=np.float64)
        if N_sum <= PUCT_CALCULATION_THRESHOLD:
            # All U terms reduce to c*P; just pick argmax P
            # But prioritize terminal moves
            if self.cfg.enable_terminal_move_detection and any(node.terminal_moves):
                terminal_indices = [i for i, is_terminal in enumerate(node.terminal_moves) if is_terminal]
                return terminal_indices[0]  # Return first terminal move
            return int(np.argmax(node.P))
        
        U = self.cfg.c_puct * node.P * math.sqrt(N_sum) / (1.0 + node.N)
        
        # Apply terminal move detection modifications
        if self.cfg.enable_terminal_move_detection:
            for i, is_terminal in enumerate(node.terminal_moves):
                if is_terminal:
                    # Boost terminal moves
                    U[i] += self.cfg.terminal_move_boost
                else:
                    # Apply small penalty to non-terminal moves
                    U[i] -= self.cfg.virtual_loss_for_non_terminal
        
        score = node.Q + U
        result = int(np.argmax(score))
        
        # End timing PUCT calculation
        t_puct_end = time.perf_counter()
        puct_time_ms = (t_puct_end - t_puct_start) * 1000.0
        
        # Store timing info for later reporting
        if not hasattr(self, '_puct_calc_times'):
            self._puct_calc_times = []
        self._puct_calc_times.append(puct_time_ms)
        
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
            value: The signed value in Red's perspective (tanh-activated output in [-1,1] range)
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


def create_mcts_config(
    config_type: str = "tournament",
    sims: Optional[int] = None,
    confidence_termination_threshold: Optional[float] = None,
    cache_size: Optional[int] = None,
    **kwargs
) -> BaselineMCTSConfig:
    """
    Create an MCTS configuration with preset defaults for different use cases.
    
    Args:
        config_type: Type of configuration ("tournament", "selfplay", "fast_selfplay")
        sims: Number of simulations (overrides preset default)
        confidence_termination_threshold: Win probability threshold for confidence termination (overrides preset default)
        cache_size: Cache size for MCTS evaluation cache (overrides preset default)
        **kwargs: Additional parameters to override in the configuration
        
    Returns:
        BaselineMCTSConfig with appropriate settings for the specified use case
    """
    # Define preset configurations
    presets = {
        "tournament": {
            "sims": 200,
            "confidence_termination_threshold": TOURNAMENT_CONFIDENCE_TERMINATION_THRESHOLD,
            "temperature_start": 1.0,
            "temperature_end": 0.1,
            "add_root_noise": False,
        },
        "selfplay": {
            "sims": 500,
            "confidence_termination_threshold": 0.85,
            "temperature_start": 0.5,
            "temperature_end": 0.01,
            "add_root_noise": False,  # Disable for self-play consistency
        },
        "fast_selfplay": {
            "sims": 200,
            "confidence_termination_threshold": 0.8,
            "temperature_start": 0.5,
            "temperature_end": 0.01,
            "add_root_noise": False,
        }
    }
    
    if config_type not in presets:
        raise ValueError(f"Unknown config_type: {config_type}. Must be one of {list(presets.keys())}")
    
    # Start with preset configuration
    config_params = presets[config_type].copy()
    
    # Override with provided parameters
    if sims is not None:
        config_params["sims"] = sims
    if confidence_termination_threshold is not None:
        config_params["confidence_termination_threshold"] = confidence_termination_threshold
    if cache_size is not None:
        config_params["cache_size"] = cache_size
    
    # Override with any additional kwargs
    config_params.update(kwargs)
    
    # Add common parameters that are the same across all presets
    config_params.update({
        "batch_cap": DEFAULT_BATCH_CAP,
        "c_puct": DEFAULT_C_PUCT,
        "dirichlet_alpha": DEFAULT_DIRICHLET_ALPHA,
        "dirichlet_eps": DEFAULT_DIRICHLET_EPS,
        "temperature_decay_type": DEFAULT_TEMPERATURE_DECAY_TYPE,
        "temperature_decay_moves": DEFAULT_TEMPERATURE_DECAY_MOVES,
        # Terminal move detection (always enabled)
        "enable_terminal_move_detection": True,
        "terminal_move_boost": DEFAULT_TERMINAL_MOVE_BOOST,
        "virtual_loss_for_non_terminal": DEFAULT_VIRTUAL_LOSS_FOR_NON_TERMINAL,
        "terminal_detection_max_depth": DEFAULT_TERMINAL_DETECTION_MAX_DEPTH,
        # Confidence-based termination (always enabled)
        "enable_confidence_termination": True,
        # Depth-based discounting to encourage shorter wins
        "enable_depth_discounting": True,
        "depth_discount_factor": DEFAULT_DEPTH_DISCOUNT_FACTOR,

    })
    
    return BaselineMCTSConfig(**config_params)




def run_mcts_move(engine: HexGameEngine, model: ModelWrapper, state: HexGameState, cfg: Optional[BaselineMCTSConfig] = None, verbose: int = 0) -> Tuple[Tuple[int,int], Dict[str, Any], Dict[str, Any], Optional[AlgorithmTerminationInfo]]:
    """Run MCTS for one move and return (row,col), stats, tree_data, algorithm_termination_info."""
    if cfg is None:
        cfg = BaselineMCTSConfig()
    mcts = BaselineMCTS(engine, model, cfg)
    result = mcts.run(state, verbose=verbose)
    return result.move, result.stats, result.tree_data, result.algorithm_termination_info
