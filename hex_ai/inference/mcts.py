# baseline_mcts.py
# Lean, single-threaded, explicitly-batched AlphaZero-style MCTS for Hex.
# Compatible with flat-file or package imports via shims.

from __future__ import annotations

import math
import time
import random
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---- Package imports ----
from hex_ai.enums import Player, Winner
from hex_ai.inference.game_engine import HexGameState, HexGameEngine
from hex_ai.inference.model_wrapper import ModelWrapper
from hex_ai.utils.perf import PERF
from hex_ai.utils.math_utils import softmax_np
from hex_ai.utils.format_conversion import rowcol_to_tensor as move_to_index, tensor_to_rowcol as index_to_move
from hex_ai.utils.temperature import calculate_temperature_decay
from hex_ai.config import BOARD_SIZE as CFG_BOARD_SIZE, POLICY_OUTPUT_SIZE as CFG_POLICY_OUTPUT_SIZE


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
        min_move_count = CFG_BOARD_SIZE * 3
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
                if new_state.game_over and new_state.winner == node.to_play:
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

# ------------------ Early Termination Info ------------------
@dataclass
class EarlyTerminationInfo:
    """Simple info about early termination."""
    reason: str  # "terminal_move" or "neural_network_confidence"
    move: Optional[Tuple[int, int]]  # The move to play (None for NN confidence)
    win_prob: float  # Win probability

# ------------------ MCTS Result ------------------
@dataclass
class MCTSResult:
    """Complete result of an MCTS search."""
    move: Tuple[int, int]  # The selected move
    stats: Dict[str, Any]  # Performance statistics
    tree_data: Dict[str, Any]  # Tree information for analysis
    root_node: MCTSNode  # The search tree root (for advanced use cases)
    early_termination_info: Optional[EarlyTerminationInfo]  # Early termination details
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
    
    def create_early_termination_stats(self, early_info: Optional[EarlyTerminationInfo] = None) -> Dict[str, Any]:
        """Create stats for early termination cases."""
        stats = self.create_base_stats()
        stats.update({
            "total_simulations": 0, "simulations_per_second": 0.0,
            "early_termination_occurred": True,
            "early_termination_reason": early_info.reason if early_info else "unknown"
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
            "early_termination_occurred": False,
            "early_termination_reason": "none"
        })
        return stats

# ------------------ Early Termination Checker ------------------
class EarlyTerminationChecker:
    """Centralized early termination checking with simple priority order."""
    
    def __init__(self, cfg: BaselineMCTSConfig, terminal_detector: TerminalMoveDetector):
        self.cfg = cfg
        self.terminal_detector = terminal_detector
    
    def should_terminate_early(self, root: MCTSNode, board_size: int, verbose: int, eval_cache: Dict[int, Tuple[np.ndarray, float]]) -> Optional[EarlyTerminationInfo]:
        """
        Check if we should terminate early. Returns None if we should continue with MCTS.
        Returns EarlyTerminationInfo if we should terminate.
        """
        # 1. Check for terminal moves (highest priority)
        if self.cfg.enable_terminal_move_detection:
            if self.terminal_detector.detect_terminal_moves(root, board_size):
                terminal_move = self.terminal_detector.get_terminal_move(root)
                if verbose >= 2:
                    print(f"🎮 MCTS: Found terminal move: {terminal_move}")
                return EarlyTerminationInfo(
                    reason="terminal_move",
                    move=terminal_move,
                    win_prob=1.0  # Guaranteed win
                )
        
        # 2. Check neural network confidence (requires root expansion)
        if self.cfg.enable_early_termination and root.is_expanded and not root.is_terminal:
            win_prob = self._get_root_win_probability(root, eval_cache)
            if self._is_position_clearly_decided(win_prob):
                if verbose >= 2:
                    print(f"🎮 MCTS: Early termination (win prob: {win_prob:.3f})")
                return EarlyTerminationInfo(
                    reason="neural_network_confidence",
                    move=None,  # Will use top policy move
                    win_prob=win_prob
                )
        
        return None  # Continue with MCTS
    
    def _get_root_win_probability(self, root: MCTSNode, eval_cache: Dict[int, Tuple[np.ndarray, float]]) -> float:
        """Get win probability for current player from neural network."""
        _, value_logit = eval_cache[root.state_hash]
        nn_win_prob = float(torch.sigmoid(torch.tensor(value_logit)).item())
        
        if root.to_play == Player.RED:
            return nn_win_prob
        else:
            return 1.0 - nn_win_prob
    
    def _is_position_clearly_decided(self, win_prob: float) -> bool:
        """Check if position is clearly won or lost."""
        return (win_prob >= self.cfg.early_termination_threshold or 
                win_prob <= (1.0 - self.cfg.early_termination_threshold))

# ------------------ Config ------------------
@dataclass
class BaselineMCTSConfig:
    sims: int = 200
    batch_cap: int = 64
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    add_root_noise: bool = False
    # Temperature scaling parameters (always used)
    temperature_start: float = 1.0  # Starting temperature
    temperature_end: float = 0.1  # Final temperature (minimum)
    temperature_decay_type: str = "exponential"  # "linear", "exponential", "step", "game_progress"
    temperature_decay_moves: int = 50  # Number of moves for decay (for linear/exponential)
    temperature_step_thresholds: List[int] = field(default_factory=lambda: [10, 25, 50])  # Move thresholds for step decay
    temperature_step_values: List[float] = field(default_factory=lambda: [0.8, 0.5, 0.2])  # Temperature values for step decay
    # Terminal move detection parameters
    enable_terminal_move_detection: bool = True  # Enable immediate terminal move detection
    terminal_move_boost: float = 10.0  # Boost factor for terminal moves in PUCT calculation
    virtual_loss_for_non_terminal: float = 0.01  # Small penalty for non-terminal moves
    terminal_detection_max_depth: int = 3  # Maximum depth for terminal move detection
    # Note: Pre-check only happens after move BOARD_SIZE * 3 (minimum moves needed for a win)
    # Removed seed parameter - randomness should be controlled externally

    # Early termination parameters
    enable_early_termination: bool = False
    early_termination_threshold: float = 0.9
    
    # Depth-based discounting parameters
    enable_depth_discounting: bool = True
    depth_discount_factor: float = 0.95  # Discount wins by this factor per depth level
    # When enabled, wins found deeper in the search tree are discounted to encourage
    # the algorithm to prefer shorter winning sequences. This helps avoid meandering
    # when the position is already won. Only applies during MCTS search, not during
    # early termination when using the policy network.
    
    # Terminal move value boosting
    terminal_win_value_boost: float = 1.5  # Multiply terminal win values by this factor
    # This makes actual terminal wins (immediate wins) even more attractive than
    # neural network evaluations, encouraging the algorithm to find and prefer them.

# ------------------ Helpers ------------------

def state_hash_from(state: HexGameState) -> int:
    """Hash only immutable, CPU-native parts (no tensor bytes)."""
    # Use move history and current player enum value.
    # Ensure stable, bounded integer (mask to 63 bits to avoid Python hash randomization effects).
    key = (tuple(state.move_history), int(state.current_player_enum.value))
    h = hash(key) & ((1 << 63) - 1)
    return h

# ------------------ Data structures ------------------

class MCTSNode:
    __slots__ = (
        "state", "to_play", "legal_moves", "legal_indices",
        "children", "N", "W", "Q", "P", "is_expanded",
        "state_hash", "is_terminal", "winner_str", "terminal_moves",
        "_terminal_moves_detected", "depth"
    )
    def __init__(self, state: HexGameState, board_size: int):
        self.state: HexGameState = state
        self.to_play: Player = state.current_player_enum
        # Legal moves
        self.legal_moves: List[Tuple[int,int]] = state.get_legal_moves()
        self.legal_indices: List[int] = [move_to_index(r, c, board_size) for (r,c) in self.legal_moves]
        L = len(self.legal_moves)
        # Stats (aligned to legal_moves order)
        self.children: List[Optional[MCTSNode]] = [None] * L
        self.N = np.zeros(L, dtype=np.int32)   # visit counts per action
        self.W = np.zeros(L, dtype=np.float64) # total value per action
        self.Q = np.zeros(L, dtype=np.float64) # mean value per action
        self.P = np.zeros(L, dtype=np.float64) # prior probability per action (set on expand)
        self.is_expanded: bool = False
        self.state_hash: int = state_hash_from(state)
        self.is_terminal: bool = bool(state.game_over)
        self.winner_str: Optional[str] = state.winner if self.is_terminal else None
        self.terminal_moves: List[bool] = [False] * L # New attribute for terminal move detection
        self._terminal_moves_detected: bool = False  # Track if terminal moves have been detected
        self.depth: int = 0  # Track node depth in the tree

# ------------------ Core MCTS ------------------

class BaselineMCTS:
    def __init__(self, engine: HexGameEngine, model: ModelWrapper, cfg: BaselineMCTSConfig):
        self.engine = engine
        self.model = model
        self.cfg = cfg
        # Always use global RNG state - randomness should be controlled externally
        # This ensures MCTS instances don't interfere with each other's randomness

        # Cache: state_hash -> (policy_logits_np [A], value_logit_float)
        self.eval_cache: Dict[int, Tuple[np.ndarray, float]] = {}
        self.cache_hits = 0
        self.cache_misses = 0


        
        # Terminal move detection
        self.terminal_detector = TerminalMoveDetector(
            max_detection_depth=cfg.terminal_detection_max_depth
        )
        
        # Early termination checker
        self.early_termination_checker = EarlyTerminationChecker(cfg, self.terminal_detector)
        
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
        """
        # Prepare root node
        root = self._prepare_root_node(root_state, verbose)
        
        # Check for early termination opportunities
        early_info = self._check_early_termination(root, verbose)
        if early_info:
            # Handle early termination
            move = self._get_early_termination_move(root, early_info, verbose)
            tree_data = self._compute_tree_data(root)
            win_probability = early_info.win_prob
            
            return MCTSResult(
                move=move,
                stats=self._get_stats_builder().create_early_termination_stats(early_info),
                tree_data=tree_data,
                root_node=root,
                early_termination_info=early_info,
                win_probability=win_probability
            )
        
        # Run main simulation loop
        timing_stats = self._run_simulation_loop(root, verbose)
        
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
            early_termination_info=None,
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
        cached = self.eval_cache.get(root.state_hash, None)
        if cached is not None:
            self.cache_hits += 1
            policy_np, value_logit = cached
            self._expand_node_from_policy(root, policy_np, board_size, action_size)
        else:
            self.cache_misses += 1
            # Evaluate root in micro-batch of size 1
            root_enc = root.state.get_board_tensor().to(dtype=torch.float32)
            batch = torch.stack([root_enc], dim=0)
            
            policy_cpu, value_cpu, _ = self.model.infer_timed(batch)
            policy_np = policy_cpu[0].numpy()
            value_logit = float(value_cpu[0].item())
            
            # Cache results
            self.eval_cache[root.state_hash] = (policy_np, value_logit)
            self._expand_node_from_policy(root, policy_np, board_size, action_size)

    def _check_early_termination(self, root: MCTSNode, verbose: int) -> Optional[EarlyTerminationInfo]:
        """Check if MCTS should terminate early."""
        board_size = int(root.state.get_board_tensor().shape[-1])
        
        # Check before root expansion
        early_info = self.early_termination_checker.should_terminate_early(
            root, board_size, verbose, self.eval_cache
        )
        if early_info:
            return early_info
        
        # Check after root expansion (for neural network confidence)
        if self.cfg.enable_early_termination and root.is_expanded and not root.is_terminal:
            early_info = self.early_termination_checker.should_terminate_early(
                root, board_size, verbose, self.eval_cache
            )
            if early_info:
                return early_info
        
        return None

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
        
        return timing_tracker.get_final_stats()

    def _select_leaves_batch(self, root: MCTSNode, sims_remaining: int, 
                           timing_tracker: MCTSTimingTracker) -> Tuple[List[MCTSNode], List[List[Tuple[MCTSNode, int]]]]:
        """Select a batch of leaves for expansion."""
        timing_tracker.start_timing("select")
        
        leaves: List[MCTSNode] = []
        paths: List[List[Tuple[MCTSNode, int]]] = []
        board_size = int(root.state.get_board_tensor().shape[-1])
        
        select_budget = min(self.cfg.batch_cap, sims_remaining)
        
        while len(leaves) < select_budget:
            node = root
            path: List[Tuple[MCTSNode, int]] = []
            
            # Descend until reaching a leaf or terminal
            while True:
                if node.is_terminal:
                    leaves.append(node)
                    paths.append(path.copy())
                    break
                if not node.is_expanded:
                    leaves.append(node)
                    paths.append(path)
                    break
                
                # Select child via PUCT
                child_idx = self._select_child_puct(node)
                path.append((node, child_idx))
                child = node.children[child_idx]
                
                if child is None:
                    # Materialize child state on demand
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

    def _prepare_leaf_evaluations(self, leaves: List[MCTSNode], 
                                timing_tracker: MCTSTimingTracker) -> Tuple[List[torch.Tensor], List[int], List[Tuple[int, np.ndarray, float]]]:
        """Prepare leaf evaluations, separating cached from uncached."""
        encodings: List[torch.Tensor] = []
        need_eval_idxs: List[int] = []
        cached_expansions: List[Tuple[int, np.ndarray, float]] = []
        
        timing_tracker.start_timing("stack")
        timing_tracker.start_timing("encode")
        
        for i, leaf in enumerate(leaves):
            if leaf.is_terminal or leaf.is_expanded:
                continue
            
            # Check cache
            timing_tracker.start_timing("cache_lookup")
            cached = self.eval_cache.get(leaf.state_hash, None)
            timing_tracker.end_timing("cache_lookup")
            
            if cached is not None:
                self.cache_hits += 1
                cached_expansions.append((i, cached[0], cached[1]))
            else:
                self.cache_misses += 1
                # Prepare encoding for neural network
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
        
        # Record performance metrics
        self._record_eval_perf(tm, is_first=(timing_tracker.batch_count == 0))
        timing_tracker.record_batch_metrics(tm)
        
        # Cache results and expand nodes
        for j, leaf_idx in enumerate(need_eval_idxs):
            leaf = leaves[leaf_idx]
            pol = policy_cpu[j].numpy()
            val_logit = float(value_cpu[j].item())
            
            # Cache CPU-native results
            self.eval_cache[leaf.state_hash] = (pol, val_logit)
            self._expand_node_from_policy(leaf, pol, board_size, action_size)

    def _expand_cached_leaves(self, cached_expansions: List[Tuple[int, np.ndarray, float]], 
                            leaves: List[MCTSNode], timing_tracker: MCTSTimingTracker):
        """Expand leaves that were found in cache."""
        if not cached_expansions:
            return
        
        timing_tracker.start_timing("expand")
        board_size = int(leaves[0].state.get_board_tensor().shape[-1])
        action_size = board_size * board_size
        
        for (leaf_idx, policy_np, value_logit) in cached_expansions:
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
            # Determine value for this leaf
            if leaf.is_terminal:
                p_red = self._get_terminal_value(leaf)
            else:
                p_red = self._get_neural_network_value(leaf)
            
            # Backpropagate along path
            self._backpropagate_path(path, p_red)
            simulations_completed += 1
        
        timing_tracker.end_timing("backprop")
        return simulations_completed

    def _get_terminal_value(self, leaf: MCTSNode) -> float:
        """Get value for a terminal leaf."""
        if leaf.winner_str is None:
            return 0.5  # Draw
        
        p_red = 1.0 if leaf.winner_str == "red" else 0.0
        
        # Apply terminal win value boost
        if self.cfg.terminal_win_value_boost != 1.0:
            if (leaf.winner_str == "red" and leaf.to_play == Player.RED) or \
               (leaf.winner_str == "blue" and leaf.to_play == Player.BLUE):
                p_red = min(1.0, p_red * self.cfg.terminal_win_value_boost)
        
        return p_red

    def _get_neural_network_value(self, leaf: MCTSNode) -> float:
        """Get value for a non-terminal leaf from neural network."""
        _, value_logit = self.eval_cache[leaf.state_hash]
        return float(torch.sigmoid(torch.tensor(value_logit)).item())

    def _backpropagate_path(self, path: List[Tuple[MCTSNode, int]], p_red: float):
        """Backpropagate value along a path from leaf to root."""
        for (node, a_idx) in reversed(path):
            v_node = p_red if node.to_play == Player.RED else (1.0 - p_red)
            
            # Apply depth discounting
            if self.cfg.enable_depth_discounting and node.depth > 0:
                discount = self.cfg.depth_discount_factor ** node.depth
                v_node *= discount
            
            node.N[a_idx] += 1
            node.W[a_idx] += v_node
            node.Q[a_idx] = node.W[a_idx] / max(1, node.N[a_idx])

    # ---------- New Result Computation Methods ----------
    
    def _get_early_termination_move(self, root: MCTSNode, early_info: EarlyTerminationInfo, verbose: int) -> Tuple[int, int]:
        """Get the move for early termination cases."""
        if early_info.reason == "terminal_move":
            return early_info.move
        elif early_info.reason == "neural_network_confidence":
            # Use top policy move
            best_move_idx = int(np.argmax(root.P))
            best_move = root.legal_moves[best_move_idx]
            if verbose >= 2:
                print(f"🎮 MCTS: Using top policy move (early termination, win prob: {early_info.win_prob:.3f}): {best_move}")
            return best_move
        else:
            raise ValueError(f"Unknown early termination reason: {early_info.reason}")

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
        
        if temp <= 1e-6:
            a_idx = int(np.argmax(counts))
        else:
            pi = np.power(counts, 1.0 / temp)
            if not np.isfinite(pi).all() or np.sum(pi) <= 0:
                raise ValueError(f"Invalid temperature scaling result: pi={pi}, counts={counts}, temp={temp}")
            pi /= np.sum(pi)
            a_idx = int(np.random.choice(len(pi), p=pi))
        return root.legal_moves[a_idx]

    def _compute_tree_data(self, root: MCTSNode) -> Dict[str, Any]:
        """Compute tree data for analysis."""
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
        principal_variation = self._extract_principal_variation(root, max_length=10)

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
        """Compute win probability for the current player based on root value."""
        tree_data = self._compute_tree_data(root)
        root_value = tree_data["root_value"]
        
        # Convert root value to win probability
        # Root value is from the perspective of the player to move
        # For RED player: root_value is probability RED wins
        # For BLUE player: root_value is probability BLUE wins
        if root_state.current_player_enum == Player.RED:
            return root_value
        else:
            return 1.0 - root_value

    def _extract_principal_variation(self, root: MCTSNode, max_length: int = 10) -> List[Tuple[int, int]]:
        """Extract the principal variation (best move sequence) from the MCTS tree."""
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


class MCTSTimingTracker:
    """Tracks timing information for MCTS operations."""
    
    def __init__(self):
        self.timings = {}
        self.batch_count = 0
        self.batch_sizes = []
        self.forward_ms_list = []
        self.select_times = []
        self.cache_hit_times = []
        self.cache_miss_times = []
        self.puct_calc_times = []
        self.make_move_times = []
        
        # Cumulative totals
        self.h2d_ms_total = 0.0
        self.forward_ms_total = 0.0
        self.pure_forward_ms_total = 0.0
        self.sync_ms_total = 0.0
        self.d2h_ms_total = 0.0
        
        # Current timing
        self.current_timing = None
        self.current_timing_start = None
    
    def start_timing(self, operation: str):
        """Start timing an operation."""
        self.current_timing = operation
        self.current_timing_start = time.perf_counter()
    
    def end_timing(self, operation: str):
        """End timing an operation."""
        if self.current_timing == operation and self.current_timing_start is not None:
            duration_ms = (time.perf_counter() - self.current_timing_start) * 1000.0
            if operation not in self.timings:
                self.timings[operation] = 0.0
            self.timings[operation] += duration_ms
            self.current_timing = None
            self.current_timing_start = None
    
    def record_batch_metrics(self, tm: Dict[str, Any]):
        """Record metrics from a neural network batch."""
        self.batch_count += 1
        self.batch_sizes.append(int(tm["batch_size"]))
        self.forward_ms_list.append(float(tm["forward_ms"]))
        self.h2d_ms_total += float(tm["h2d_ms"])
        self.forward_ms_total += float(tm["forward_ms"])
        self.pure_forward_ms_total += float(tm.get("pure_forward_ms", tm["forward_ms"]))
        self.sync_ms_total += float(tm.get("sync_ms", 0.0))
        self.d2h_ms_total += float(tm["d2h_ms"])
    
    def get_final_stats(self) -> Dict[str, Any]:
        """Get final timing statistics."""
        total_search_time = (
            self.timings.get("encode", 0.0) + self.timings.get("stack", 0.0) + 
            self.h2d_ms_total + self.forward_ms_total + self.d2h_ms_total + 
            self.timings.get("expand", 0.0) + self.timings.get("backprop", 0.0) + 
            self.timings.get("select", 0.0) + self.timings.get("cache_lookup", 0.0) + 
            self.timings.get("state_creation", 0.0)
        ) / 1000.0
        
        return {
            "encode_ms": self.timings.get("encode", 0.0),
            "stack_ms": self.timings.get("stack", 0.0),
            "h2d_ms": self.h2d_ms_total,
            "forward_ms": self.forward_ms_total,
            "pure_forward_ms": self.pure_forward_ms_total,
            "sync_ms": self.sync_ms_total,
            "d2h_ms": self.d2h_ms_total,
            "expand_ms": self.timings.get("expand", 0.0),
            "backprop_ms": self.timings.get("backprop", 0.0),
            "select_ms": self.timings.get("select", 0.0),
            "cache_lookup_ms": self.timings.get("cache_lookup", 0.0),
            "state_creation_ms": self.timings.get("state_creation", 0.0),
            "puct_calc_ms": self.timings.get("puct_calc", 0.0),
            "make_move_ms": self.timings.get("make_move", 0.0),
            "batch_count": self.batch_count,
            "batch_sizes": self.batch_sizes,
            "forward_ms_list": self.forward_ms_list,
            "select_times": self.select_times,
            "cache_hit_times": self.cache_hit_times,
            "cache_miss_times": self.cache_miss_times,
            "puct_calc_times": self.puct_calc_times,
            "make_move_times": self.make_move_times,
            "median_forward_ms_ex_warm": _median_excluding_first(self.forward_ms_list),
            "p90_forward_ms_ex_warm": _p90_excluding_first(self.forward_ms_list),
            "median_select_ms": _median_excluding_first(self.select_times) if self.select_times else 0.0,
            "median_cache_hit_ms": _median_excluding_first(self.cache_hit_times) if self.cache_hit_times else 0.0,
            "median_cache_miss_ms": _median_excluding_first(self.cache_miss_times) if self.cache_miss_times else 0.0,
            "median_puct_calc_ms": _median_excluding_first(self.puct_calc_times) if self.puct_calc_times else 0.0,
            "median_make_move_ms": _median_excluding_first(self.make_move_times) if self.make_move_times else 0.0,
            "total_search_time": total_search_time,
        }

    


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
        
        # Detect terminal moves if enabled and appropriate
        if self.cfg.enable_terminal_move_detection:
            self.terminal_detector.detect_terminal_moves(node, int(node.state.get_board_tensor().shape[-1]))
        
        # Start timing PUCT calculation
        t_puct_start = time.perf_counter()
        
        N_sum = np.sum(node.N, dtype=np.float64)
        if N_sum <= 1e-9:
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


def create_mcts_config(
    config_type: str = "tournament",
    sims: Optional[int] = None,
    early_termination_threshold: Optional[float] = None,
    **kwargs
) -> BaselineMCTSConfig:
    """
    Create an MCTS configuration with preset defaults for different use cases.
    
    Args:
        config_type: Type of configuration ("tournament", "selfplay", "fast_selfplay")
        sims: Number of simulations (overrides preset default)
        early_termination_threshold: Win probability threshold for early termination (overrides preset default)
        **kwargs: Additional parameters to override in the configuration
        
    Returns:
        BaselineMCTSConfig with appropriate settings for the specified use case
    """
    # Define preset configurations
    presets = {
        "tournament": {
            "sims": 200,
            "early_termination_threshold": 0.95,
            "temperature_start": 1.0,
            "temperature_end": 0.1,
            "add_root_noise": False,
        },
        "selfplay": {
            "sims": 500,
            "early_termination_threshold": 0.85,
            "temperature_start": 0.5,
            "temperature_end": 0.01,
            "add_root_noise": False,  # Disable for self-play consistency
        },
        "fast_selfplay": {
            "sims": 200,
            "early_termination_threshold": 0.8,
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
    if early_termination_threshold is not None:
        config_params["early_termination_threshold"] = early_termination_threshold
    
    # Override with any additional kwargs
    config_params.update(kwargs)
    
    # Add common parameters that are the same across all presets
    config_params.update({
        "batch_cap": 64,
        "c_puct": 1.5,
        "dirichlet_alpha": 0.3,
        "dirichlet_eps": 0.25,
        "temperature_decay_type": "exponential",
        "temperature_decay_moves": 50,
        # Terminal move detection (always enabled)
        "enable_terminal_move_detection": True,
        "terminal_move_boost": 10.0,
        "virtual_loss_for_non_terminal": 0.01,
        "terminal_detection_max_depth": 3,
        # Early termination (always enabled)
        "enable_early_termination": True,
        # Depth-based discounting to encourage shorter wins
        "enable_depth_discounting": True,
        "depth_discount_factor": 0.95,
        # Terminal win value boosting
        "terminal_win_value_boost": 1.5,
    })
    
    return BaselineMCTSConfig(**config_params)




def run_mcts_move(engine: HexGameEngine, model: ModelWrapper, state: HexGameState, cfg: Optional[BaselineMCTSConfig] = None, verbose: int = 0) -> Tuple[Tuple[int,int], Dict[str, Any], Dict[str, Any]]:
    """Run MCTS for one move and return (row,col), stats, tree_data."""
    if cfg is None:
        cfg = BaselineMCTSConfig()
    mcts = BaselineMCTS(engine, model, cfg)
    result = mcts.run(state, verbose=verbose)
    return result.move, result.stats, result.tree_data


# --------- Stats helpers ---------

def _median_excluding_first(xs: List[float]) -> float:
    if not xs:
        return 0.0
    if len(xs) == 1:
        return xs[0]
    arr = np.array(xs[1:], dtype=np.float64)
    return float(np.median(arr))

def _p90_excluding_first(xs: List[float]) -> float:
    if not xs:
        return 0.0
    if len(xs) == 1:
        return xs[0]
    arr = np.array(xs[1:], dtype=np.float64)
    k = max(0, int(math.ceil(0.9 * len(arr)) - 1))
    arr.sort()
    return float(arr[k])
