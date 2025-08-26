# baseline_mcts.py
# Lean, single-threaded, explicitly-batched AlphaZero-style MCTS for Hex.
# Compatible with flat-file or package imports via shims.

# ============================== MCTS BASELINE — TODOs ==============================

# ===================================================================================


from __future__ import annotations

import math
import time
import json
import random
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---- Package imports ----
from hex_ai.enums import Player, Winner
from hex_ai.inference.game_engine import HexGameState, HexGameEngine
from hex_ai.inference.model_wrapper import ModelWrapper
from hex_ai.value_utils import get_win_prob_from_model_output
from hex_ai.utils.perf import PERF
from hex_ai.config import BOARD_SIZE as CFG_BOARD_SIZE, POLICY_OUTPUT_SIZE as CFG_POLICY_OUTPUT_SIZE


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
    # Note: Pre-check only happens after move 25 (minimum moves needed for a win)
    # Removed seed parameter - randomness should be controlled externally

    # Early termination parameters
    enable_early_termination: bool = False
    early_termination_threshold: float = 0.9

# ------------------ Helpers ------------------

def softmax_np(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over 1D logits -> probs that sum to 1."""
    m = np.max(logits)
    exps = np.exp(logits - m)
    s = np.sum(exps)
    if s <= 0 or not np.isfinite(s):
        raise ValueError(f"Invalid softmax input: sum={s}, logits={logits}")
    return exps / s

def move_to_index(row: int, col: int, board_size: int) -> int:
    return row * board_size + col

def index_to_move(a: int, board_size: int) -> Tuple[int, int]:
    return divmod(a, board_size)

def state_hash_from(state: HexGameState) -> int:
    """Hash only immutable, CPU-native parts (no tensor bytes)."""
    # Use move history and current player enum value.
    # Ensure stable, bounded integer (mask to 63 bits to avoid Python hash randomization effects).
    key = (tuple(state.move_history), int(state.current_player_enum.value))
    h = hash(key) & ((1 << 63) - 1)
    return h

def calculate_temperature_decay(cfg: BaselineMCTSConfig, move_count: int, start_temp_override: Optional[float] = None) -> float:
    """
    Calculate temperature based on decay configuration and current move count.
    
    Args:
        cfg: MCTS configuration with temperature decay parameters
        move_count: Number of moves played so far (0-based)
        start_temp_override: Optional override for starting temperature
    
    Returns:
        Current temperature value
    """
    # Determine the starting temperature
    # Use override if provided, otherwise use temperature_start
    if start_temp_override is not None:
        start_temp = start_temp_override
    else:
        start_temp = cfg.temperature_start
    
    if cfg.temperature_decay_type == "linear":
        # Linear decay from temperature_start to temperature_end over temperature_decay_moves
        progress = min(move_count / max(1, cfg.temperature_decay_moves), 1.0)
        return start_temp + (cfg.temperature_end - start_temp) * progress
    
    elif cfg.temperature_decay_type == "exponential":
        # Exponential decay: T = T_start * (T_end/T_start)^(move_count/decay_moves)
        if start_temp <= 0 or cfg.temperature_end <= 0:
            return cfg.temperature_end  # Safety fallback
        progress = min(move_count / max(1, cfg.temperature_decay_moves), 1.0)
        decay_factor = (cfg.temperature_end / start_temp) ** progress
        return start_temp * decay_factor
    
    elif cfg.temperature_decay_type == "step":
        # Step decay: temperature drops at specific move thresholds
        if not cfg.temperature_step_thresholds or not cfg.temperature_step_values:
            return start_temp
        
        # Find the appropriate temperature for current move count
        for i, threshold in enumerate(cfg.temperature_step_thresholds):
            if move_count < threshold:
                return cfg.temperature_step_values[i] if i < len(cfg.temperature_step_values) else cfg.temperature_end
        
        # If we've passed all thresholds, use the final temperature
        return cfg.temperature_end
    
    elif cfg.temperature_decay_type == "game_progress":
        # Temperature based on percentage of game completed
        # Estimate total game length as board_size^2 (full board)
        board_size = CFG_BOARD_SIZE
        estimated_total_moves = board_size * board_size
        progress = min(move_count / max(1, estimated_total_moves), 1.0)
        return start_temp + (cfg.temperature_end - start_temp) * progress
    
    else:
        # Unknown decay type, return starting temperature
        return start_temp

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

        self._root_noise_applied = False

        # Root node from the last run() (used by pick_move)
        self._root_node = None
        
        # Early termination tracking
        self._early_termination_occurred = False
        self._early_termination_win_prob = 0.0

    # ---------- Public API ----------
    def run(self, root_state: HexGameState, verbose: int = 0) -> Dict[str, Any]:
        """
        Run MCTS for cfg.sims simulations starting from root_state.
        Returns a dict of timing and batching stats for this move.
        
        Args:
            root_state: The game state to search from
            verbose: Verbosity level for logging (0=quiet, 1=basic, 2=detailed)
        """
        # Prepare root
        board_tensor = root_state.get_board_tensor()
        board_size = int(board_tensor.shape[-1])
        action_size = board_size * board_size
        root = MCTSNode(root_state, board_size)

        # Pre-check for immediate terminal moves (before any MCTS work)
        if self.cfg.enable_terminal_move_detection:
            # Only check after move 25 (impossible to win before then)
            if len(root_state.move_history) >= 25:
                self._detect_terminal_moves(root, board_size)
                if any(root.terminal_moves):
                    # Found a terminal move - return early with minimal stats
                    terminal_indices = [i for i, is_terminal in enumerate(root.terminal_moves) if is_terminal]
                    terminal_move = root.legal_moves[terminal_indices[0]]
                    
                    # Store the terminal move for pick_move to use
                    root.terminal_moves = [False] * len(root.legal_moves)  # Reset
                    root.terminal_moves[terminal_indices[0]] = True  # Mark the found move
                    
                    # Return minimal stats since we didn't do any MCTS work
                    return {
                        "encode_ms": 0.0,
                        "stack_ms": 0.0,
                        "h2d_ms": 0.0,
                        "forward_ms": 0.0,
                        "pure_forward_ms": 0.0,
                        "sync_ms": 0.0,
                        "d2h_ms": 0.0,
                        "expand_ms": 0.0,
                        "backprop_ms": 0.0,
                        "select_ms": 0.0,
                        "cache_lookup_ms": 0.0,
                        "state_creation_ms": 0.0,
                        "batch_count": 0,
                        "batch_sizes": [],
                        "forward_ms_list": [],
                        "select_times": [],
                        "cache_hit_times": [],
                        "cache_miss_times": [],
                        "median_forward_ms_ex_warm": 0.0,
                        "p90_forward_ms_ex_warm": 0.0,
                        "median_select_ms": 0.0,
                        "median_cache_hit_ms": 0.0,
                        "median_cache_miss_ms": 0.0,
                        "cache_hits": self.cache_hits,
                        "cache_misses": self.cache_misses,
                        "total_simulations": 0,  # No simulations needed
                        "simulations_per_second": 0.0,
                        "terminal_move_found": True,  # Flag that we found a terminal move
                        "terminal_move": terminal_move,
                    }

        # Initialize timing variables
        encode_ms = 0.0
        stack_ms = 0.0
        expand_ms = 0.0
        backprop_ms = 0.0
        select_ms = 0.0
        cache_lookup_ms = 0.0
        state_creation_ms = 0.0
        terminal_detect_ms = 0.0  # New: terminal move detection time
        puct_calc_ms = 0.0  # New: PUCT calculation time
        make_move_ms = 0.0  # New: make_move operation time
        h2d_ms_total = 0.0
        forward_ms_total = 0.0
        pure_forward_ms_total = 0.0
        sync_ms_total = 0.0
        d2h_ms_total = 0.0
        batch_sizes: List[int] = []
        forward_ms_list: List[float] = []
        select_times: List[float] = []
        cache_hit_times: List[float] = []
        cache_miss_times: List[float] = []
        terminal_detect_times: List[float] = []  # New: terminal detection times
        puct_calc_times: List[float] = []  # New: PUCT calculation times
        make_move_times: List[float] = []  # New: make_move operation times

        sims_remaining = self.cfg.sims

        # One-time root expansion if not expanded and not terminal
        if not root.is_terminal and not root.is_expanded:
            # Try cache
            cached = self.eval_cache.get(root.state_hash, None)
            if cached is not None:
                self.cache_hits += 1
                t0 = time.perf_counter()
                policy_np, value_logit = cached
                p_red = float(torch.sigmoid(torch.tensor(value_logit)).item())
                self._expand_node_from_policy(root, policy_np, board_size, action_size)
                expand_ms += (time.perf_counter() - t0) * 1000.0
                # No backprop for root on initial expansion
            else:
                self.cache_misses += 1
                # Evaluate just the root in a micro-batch of size 1
                t_enc0 = time.perf_counter()
                root_enc = root_state.get_board_tensor().to(dtype=torch.float32)  # CPU
                encode_ms += (time.perf_counter() - t_enc0) * 1000.0

                t_stack0 = time.perf_counter()
                batch = torch.stack([root_enc], dim=0)  # [1,3,N,N]
                stack_ms += (time.perf_counter() - t_stack0) * 1000.0

                policy_cpu, value_cpu, tm = self.model.infer_timed(batch)
                # PERF per-batch
                self._record_eval_perf(tm, is_first=(len(batch_sizes) == 0))
                batch_sizes.append(int(tm["batch_size"]))
                forward_ms_list.append(float(tm["forward_ms"]))
                h2d_ms_total += float(tm["h2d_ms"])
                forward_ms_total += float(tm["forward_ms"])
                pure_forward_ms_total += float(tm.get("pure_forward_ms", tm["forward_ms"]))
                sync_ms_total += float(tm.get("sync_ms", 0.0))
                d2h_ms_total += float(tm["d2h_ms"])

                policy_np = policy_cpu[0].numpy()
                value_logit = float(value_cpu[0].item())
                # Cache CPU-native
                self.eval_cache[root.state_hash] = (policy_np, value_logit)

                t0 = time.perf_counter()
                self._expand_node_from_policy(root, policy_np, board_size, action_size)
                expand_ms += (time.perf_counter() - t0) * 1000.0

        # Optionally add root Dirichlet noise (once)
        if self.cfg.add_root_noise and not root.is_terminal and root.is_expanded and not self._root_noise_applied:
            self._apply_root_noise(root)
            self._root_noise_applied = True

        # Simple confidence check after root expansion (using neural network value)
        if self.cfg.enable_early_termination and root.is_expanded and not root.is_terminal:
            # Get the neural network value directly
            _, value_logit = self.eval_cache[root.state_hash]
            nn_win_prob = float(torch.sigmoid(torch.tensor(value_logit)).item())
            
            # Convert to win probability for current player
            if root.to_play == Player.RED:
                win_prob = nn_win_prob
            else:
                win_prob = 1.0 - nn_win_prob
            
            # Check if position is clearly won or lost
            if (win_prob >= self.cfg.early_termination_threshold or 
                win_prob <= (1.0 - self.cfg.early_termination_threshold)):
                if verbose >= 2:
                    print(f"🎮 MCTS: Simple confidence-based termination (NN win prob: {win_prob:.3f})")
                # Store early termination info for pick_move()
                self._early_termination_occurred = True
                self._early_termination_win_prob = win_prob
                self._root_node = root  # Store root node for pick_move()
                # Return early with minimal stats
                return {
                    "encode_ms": encode_ms,
                    "stack_ms": stack_ms,
                    "h2d_ms": h2d_ms_total,
                    "forward_ms": forward_ms_total,
                    "pure_forward_ms": pure_forward_ms_total,
                    "sync_ms": sync_ms_total,
                    "d2h_ms": d2h_ms_total,
                    "expand_ms": expand_ms,
                    "backprop_ms": 0.0,
                    "select_ms": 0.0,
                    "cache_lookup_ms": cache_lookup_ms,
                    "state_creation_ms": state_creation_ms,
                    "batch_count": len(batch_sizes),
                    "batch_sizes": batch_sizes,
                    "forward_ms_list": forward_ms_list,
                    "select_times": [],
                    "cache_hit_times": cache_hit_times,
                    "cache_miss_times": cache_miss_times,
                    "median_forward_ms_ex_warm": _median_excluding_first(forward_ms_list),
                    "p90_forward_ms_ex_warm": _p90_excluding_first(forward_ms_list),
                    "median_select_ms": 0.0,
                    "median_cache_hit_ms": _median_excluding_first(cache_hit_times) if cache_hit_times else 0.0,
                    "median_cache_miss_ms": _median_excluding_first(cache_miss_times) if cache_miss_times else 0.0,
                    "cache_hits": self.cache_hits,
                    "cache_misses": self.cache_misses,
                    "total_simulations": 0,
                    "simulations_per_second": 0.0,
                    "early_termination_occurred": True,
                }

        # ---- Simulation loop with explicit batched leaves ----
        while sims_remaining > 0:
            # 1) SELECT up to K leaves
            t_select_start = time.perf_counter()
            leaves: List[MCTSNode] = []
            paths: List[List[Tuple[MCTSNode, int]]] = []  # per-leaf path of (node, child_idx chosen at that node)
            # We may expand some via cache immediately; we only batch NN-evals for uncached
            encodings: List[torch.Tensor] = []

            select_budget = min(self.cfg.batch_cap, sims_remaining)
            while len(leaves) < select_budget:
                node = root
                path: List[Tuple[MCTSNode, int]] = []
                # Descend until reaching a leaf (not expanded) or terminal
                while True:
                    if node.is_terminal:
                        # terminal leaf: no expansion; we will backprop a value directly
                        path_terminal = path.copy()
                        leaves.append(node)
                        paths.append(path_terminal)
                        break
                    if not node.is_expanded:
                        # unexpanded leaf
                        leaves.append(node)
                        paths.append(path)
                        break
                    # select child via PUCT among legal actions
                    child_idx = self._select_child_puct(node)
                    path.append((node, child_idx))
                    child = node.children[child_idx]
                    if child is None:
                        # Materialize child state on demand
                        t_state_start = time.perf_counter()
                        (r, c) = node.legal_moves[child_idx]
                        
                        # Time the make_move operation specifically
                        t_make_move_start = time.perf_counter()
                        child_state = node.state.make_move(r, c)
                        t_make_move_end = time.perf_counter()
                        make_move_time_ms = (t_make_move_end - t_make_move_start) * 1000.0
                        make_move_ms += make_move_time_ms
                        make_move_times.append(make_move_time_ms)
                        
                        child = MCTSNode(child_state, board_size)
                        child.depth = node.depth + 1  # Set child depth
                        state_creation_ms += (time.perf_counter() - t_state_start) * 1000.0
                        node.children[child_idx] = child
                    node = child

                # If we reached select_budget, stop; else continue to select next leaf
                if len(leaves) >= select_budget:
                    break

            select_ms += (time.perf_counter() - t_select_start) * 1000.0
            select_times.append((time.perf_counter() - t_select_start) * 1000.0)
            
            # Collect terminal detection times from this batch
            if hasattr(self, '_terminal_detect_times') and self._terminal_detect_times:
                terminal_detect_times.extend(self._terminal_detect_times)
                terminal_detect_ms += sum(self._terminal_detect_times)
                self._terminal_detect_times = []  # Reset for next batch
            
            # Collect PUCT calculation times from this batch
            if hasattr(self, '_puct_calc_times') and self._puct_calc_times:
                puct_calc_times.extend(self._puct_calc_times)
                puct_calc_ms += sum(self._puct_calc_times)
                self._puct_calc_times = []  # Reset for next batch

            # 2) STACK encodings for NN where needed and expand cached/terminal immediately
            t_stack0 = time.perf_counter()
            t_enc_sum = 0.0
            need_eval_idxs: List[int] = []
            cached_expansions: List[Tuple[int, np.ndarray, float]] = []  # (leaf_idx, policy_np, value_logit)

            for i, leaf in enumerate(leaves):
                if leaf.is_terminal:
                    # immediate value from winner
                    continue
                if leaf.is_expanded:
                    # shouldn't happen: selection stops at unexpanded
                    continue
                
                # Time cache lookup
                t_cache_start = time.perf_counter()
                cached = self.eval_cache.get(leaf.state_hash, None)
                cache_lookup_ms += (time.perf_counter() - t_cache_start) * 1000.0
                
                if cached is not None:
                    self.cache_hits += 1
                    cache_hit_times.append((time.perf_counter() - t_cache_start) * 1000.0)
                    cached_expansions.append((i, cached[0], cached[1]))
                else:
                    self.cache_misses += 1
                    cache_miss_times.append((time.perf_counter() - t_cache_start) * 1000.0)
                    t_enc0 = time.perf_counter()
                    enc = leaf.state.get_board_tensor().to(dtype=torch.float32)  # CPU
                    t_enc_sum += (time.perf_counter() - t_enc0) * 1000.0
                    encodings.append(enc)
                    need_eval_idxs.append(i)
            encode_ms += t_enc_sum
            batch_tensor: Optional[torch.Tensor] = None
            if encodings:
                batch_tensor = torch.stack(encodings, dim=0)  # [B,3,N,N]
            stack_ms += (time.perf_counter() - t_stack0) * 1000.0

            # 3) INFER once on the batch
            if batch_tensor is not None:
                policy_cpu, value_cpu, tm = self.model.infer_timed(batch_tensor)
                # PERF per-batch
                self._record_eval_perf(tm, is_first=(len(batch_sizes) == 0))
                batch_sizes.append(int(tm["batch_size"]))
                forward_ms_list.append(float(tm["forward_ms"]))
                h2d_ms_total += float(tm["h2d_ms"])
                forward_ms_total += float(tm["forward_ms"])
                pure_forward_ms_total += float(tm.get("pure_forward_ms", tm["forward_ms"]))
                sync_ms_total += float(tm.get("sync_ms", 0.0))
                d2h_ms_total += float(tm["d2h_ms"])

            # 4) EXPAND nodes with policy; BACKPROP values
            # First expand any cached ones
            t0 = time.perf_counter()
            for (leaf_idx, policy_np, value_logit) in cached_expansions:
                leaf = leaves[leaf_idx]
                if not leaf.is_expanded and not leaf.is_terminal:
                    self._expand_node_from_policy(leaf, policy_np, board_size, action_size)
                    # cache already contains this state, nothing to update
            # Then expand evaluated ones
            if batch_tensor is not None:
                # iterate over need_eval_idxs in order
                for j, leaf_idx in enumerate(need_eval_idxs):
                    leaf = leaves[leaf_idx]
                    pol = policy_cpu[j].numpy()
                    val_logit = float(value_cpu[j].item())
                    # cache CPU-native results
                    self.eval_cache[leaf.state_hash] = (pol, val_logit)
                    self._expand_node_from_policy(leaf, pol, board_size, action_size)
            expand_ms += (time.perf_counter() - t0) * 1000.0

            # Backprop for all leaves (terminal or just expanded)
            t1 = time.perf_counter()
            for leaf, path in zip(leaves, paths):
                # Determine p_red for this leaf
                if leaf.is_terminal:
                    # winner_str is "blue" | "red"
                    if leaf.winner_str is None:
                        # Shouldn't happen; treat as draw ~0.5 for safety
                        p_red = 0.5
                    else:
                        p_red = 1.0 if leaf.winner_str == "red" else 0.0
                else:
                    # Use cached value_logit
                    _, val_logit = self.eval_cache[leaf.state_hash]
                    p_red = float(torch.sigmoid(torch.tensor(val_logit)).item())
                # Backprop along path: for each (node, action idx) pair
                for (node, a_idx) in reversed(path):
                    v_node = p_red if node.to_play == Player.RED else (1.0 - p_red)
                    node.N[a_idx] += 1
                    node.W[a_idx] += v_node
                    node.Q[a_idx] = node.W[a_idx] / max(1, node.N[a_idx])
                sims_remaining -= 1
                if sims_remaining <= 0:
                    break
            backprop_ms += (time.perf_counter() - t1) * 1000.0

        # Calculate total search time
        total_search_time = (encode_ms + stack_ms + h2d_ms_total + forward_ms_total + d2h_ms_total + 
                           expand_ms + backprop_ms + select_ms + cache_lookup_ms + state_creation_ms) / 1000.0
        
        # Build stats
        _store_root = True
        self._root_node = root
        stats = {
            "encode_ms": encode_ms,
            "stack_ms": stack_ms,
            "h2d_ms": h2d_ms_total,
            "forward_ms": forward_ms_total,
            "pure_forward_ms": pure_forward_ms_total,
            "sync_ms": sync_ms_total,
            "d2h_ms": d2h_ms_total,
            "expand_ms": expand_ms,
            "backprop_ms": backprop_ms,
            "select_ms": select_ms,
            "cache_lookup_ms": cache_lookup_ms,
            "state_creation_ms": state_creation_ms,
            "terminal_detect_ms": terminal_detect_ms,  # New: terminal move detection time
            "puct_calc_ms": puct_calc_ms,  # New: PUCT calculation time
            "make_move_ms": make_move_ms,  # New: make_move operation time
            "batch_count": len(batch_sizes),
            "batch_sizes": batch_sizes,
            "forward_ms_list": forward_ms_list,
            "select_times": select_times,
            "cache_hit_times": cache_hit_times,
            "cache_miss_times": cache_miss_times,
            "terminal_detect_times": terminal_detect_times,  # New: terminal detection times
            "puct_calc_times": puct_calc_times,  # New: PUCT calculation times
            "make_move_times": make_move_times,  # New: make_move operation times
            "median_forward_ms_ex_warm": _median_excluding_first(forward_ms_list),
            "p90_forward_ms_ex_warm": _p90_excluding_first(forward_ms_list),
            "median_select_ms": _median_excluding_first(select_times) if select_times else 0.0,
            "median_cache_hit_ms": _median_excluding_first(cache_hit_times) if cache_hit_times else 0.0,
            "median_cache_miss_ms": _median_excluding_first(cache_miss_times) if cache_miss_times else 0.0,
            "median_terminal_detect_ms": _median_excluding_first(terminal_detect_times) if terminal_detect_times else 0.0,
            "median_puct_calc_ms": _median_excluding_first(puct_calc_times) if puct_calc_times else 0.0,
            "median_make_move_ms": _median_excluding_first(make_move_times) if make_move_times else 0.0,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_simulations": self.cfg.sims,
            "simulations_per_second": self.cfg.sims / total_search_time if total_search_time > 0 else 0.0,
            "early_termination_occurred": False,  # Will be set to True in early termination case
        }
        return stats

    
    def pick_move(self, root_state: HexGameState, temperature: Optional[float] = None, verbose: int = 0) -> Tuple[int,int]:
        """
        Choose a move from the root based on visit counts and temperature.
        Requires that run() has been executed on the SAME root_state.
        """
        if self._root_node is None:
            raise RuntimeError("BaselineMCTS.pick_move() called before run(); run() must be called first.")
        # Verify we are selecting for the same state that was searched
        if self._root_node.state_hash != state_hash_from(root_state):
            raise ValueError("BaselineMCTS.pick_move() called with a different state than the last run() root.")
        root = self._root_node
        
        # Check if early termination occurred - use top policy move
        if self._early_termination_occurred:
            # Use the top policy move from the neural network
            policy_probs = root.P
            best_move_idx = int(np.argmax(policy_probs))
            best_move = root.legal_moves[best_move_idx]
            if verbose >= 2:
                print(f"🎮 MCTS: Using top policy move due to early termination (win prob: {self._early_termination_win_prob:.3f}): {best_move}")
            return best_move

        if root.is_terminal:
            if root.legal_moves:
                # Terminal shouldn't have legal moves, but just in case
                return random.choice(root.legal_moves)
            raise RuntimeError("No legal moves at root (terminal state).")

        # Check if a terminal move was found during pre-check
        if self.cfg.enable_terminal_move_detection and any(root.terminal_moves):
            terminal_indices = [i for i, is_terminal in enumerate(root.terminal_moves) if is_terminal]
            if terminal_indices:
                terminal_move = root.legal_moves[terminal_indices[0]]
                if verbose >= 2:
                    print(f"🎮 MCTS: Using pre-detected terminal move: {terminal_move}")
                return terminal_move

        # Check for immediate terminal moves if enabled (fallback for cases not caught by pre-check)
        if self.cfg.enable_terminal_move_detection:
            board_size = int(root_state.get_board_tensor().shape[-1])
            self._detect_terminal_moves(root, board_size)
            if any(root.terminal_moves):
                # Return the first terminal move found
                terminal_indices = [i for i, is_terminal in enumerate(root.terminal_moves) if is_terminal]
                if verbose >= 2:
                    print(f"🎮 MCTS: Found terminal move, immediately selecting: {root.legal_moves[terminal_indices[0]]}")
                return root.legal_moves[terminal_indices[0]]



        # Use visit counts accumulated during run()
        counts = root.N.astype(np.float64)
        if counts.sum() <= 0:
            raise RuntimeError(f"No visits recorded during MCTS search. This indicates a bug in the search algorithm.")

        # Calculate temperature with decay
        move_count = len(root_state.move_history)
        # Use passed temperature as starting point if provided, otherwise use config
        start_temp = temperature if temperature is not None else self.cfg.temperature_start
        temp = calculate_temperature_decay(self.cfg, move_count, start_temp_override=start_temp)
        
        # Log effective temperature if verbose
        if verbose >= 4:
            move_count = len(root_state.move_history)
            print(f"🎮 MCTS: Move {move_count}, effective temperature: {temp:.3f}")
        
        # Detailed temperature calculation logging at verbose 5+
        if verbose >= 5:
            move_count = len(root_state.move_history)
            print(f"🎮 MCTS DEBUG: move_count={move_count}, decay_type={self.cfg.temperature_decay_type}")
            # Use same logic as the actual calculation
            start_temp = temperature if temperature is not None else self.cfg.temperature_start
            print(f"🎮 MCTS DEBUG: start_temp={start_temp:.3f}, end_temp={self.cfg.temperature_end:.3f}, decay_moves={self.cfg.temperature_decay_moves}")
            if self.cfg.temperature_decay_type == "exponential":
                progress = min(move_count / max(1, self.cfg.temperature_decay_moves), 1.0)
                decay_factor = (self.cfg.temperature_end / start_temp) ** progress
                calculated_temp = start_temp * decay_factor
                print(f"🎮 MCTS DEBUG: progress={progress:.3f}, decay_factor={decay_factor:.3f}, calculated_temp={calculated_temp:.3f}")
            print(f"🎮 MCTS DEBUG: Final temperature used: {temp:.3f}")
            
        if temp <= 1e-6:
            a_idx = int(np.argmax(counts))
        else:
            pi = np.power(counts, 1.0 / temp)
            if not np.isfinite(pi).all() or np.sum(pi) <= 0:
                raise ValueError(f"Invalid temperature scaling result: pi={pi}, counts={counts}, temp={temp}")
            pi /= np.sum(pi)
            a_idx = int(np.random.choice(len(pi), p=pi))
        return root.legal_moves[a_idx]

    def get_principal_variation(self, root_state: HexGameState, max_length: int = 10) -> List[Tuple[int, int]]:
        """
        Extract the principal variation (best move sequence) from the MCTS tree.
        Requires that run() has been executed on the SAME root_state.
        """
        if self._root_node is None:
            raise RuntimeError("BaselineMCTS.get_principal_variation() called before run(); run() must be called first.")
        # Verify we are selecting for the same state that was searched
        if self._root_node.state_hash != state_hash_from(root_state):
            raise ValueError("BaselineMCTS.get_principal_variation() called with a different state than the last run() root.")
        root = self._root_node

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

    def get_tree_data(self, root_state: HexGameState) -> Dict[str, Any]:
        """
        Extract tree data for web interface display.
        Requires that run() has been executed on the SAME root_state.
        """
        if self._root_node is None:
            raise RuntimeError("BaselineMCTS.get_tree_data() called before run(); run() must be called first.")
        # Verify we are selecting for the same state that was searched
        if self._root_node.state_hash != state_hash_from(root_state):
            raise ValueError("BaselineMCTS.get_tree_data() called with a different state than the last run() root.")
        root = self._root_node

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
        principal_variation = self.get_principal_variation(root_state, max_length=10)

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

    def _calculate_tree_statistics(self, node: MCTSNode) -> Tuple[int, int]:
        """
        Recursively calculate total nodes and max depth of the MCTS tree.
        This is a lightweight traversal that doesn't affect performance significantly.
        """
        if node is None:
            return 0, 0
        
        # Count this node
        total_nodes = 1
        max_depth = 0
        
        # Recursively count children
        for child in node.children:
            if child is not None:
                child_nodes, child_depth = self._calculate_tree_statistics(child)
                total_nodes += child_nodes
                max_depth = max(max_depth, child_depth + 1)
        
        return total_nodes, max_depth

    def get_win_probability(self, root_state: HexGameState) -> float:
        """
        Get the win probability for the current player based on root value.
        Requires that run() has been executed on the SAME root_state.
        """
        tree_data = self.get_tree_data(root_state)
        root_value = tree_data["root_value"]
        
        # Convert root value to win probability
        # Root value is from the perspective of the player to move
        # For RED player: root_value is probability RED wins
        # For BLUE player: root_value is probability BLUE wins
        if root_state.current_player_enum == Player.RED:
            return root_value
        else:
            return 1.0 - root_value

    # ---------- Internal ----------

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

    def _detect_terminal_moves(self, node: MCTSNode, board_size: int) -> None:
        """Detect which legal moves immediately win the game."""
        if node.is_terminal:
            return  # Already terminal state
        
        # Only detect terminal moves if we haven't done so already
        # This prevents repeated expensive state creation during selection
        if hasattr(node, '_terminal_moves_detected') and node._terminal_moves_detected:
            return
        
        # SHALLOW DETECTION: Only detect terminal moves at shallow depths
        # This targets the specific problem of missing 1-move wins while avoiding
        # expensive detection on deep nodes that are unlikely to have terminal moves
        node_depth = self._get_node_depth(node)
        if node_depth > 3:  # Only detect at top 3 levels
            return
        
        # Start timing for this detection
        t_detect_start = time.perf_counter()
        
        for i, (row, col) in enumerate(node.legal_moves):
            # Try the move and check if it wins
            try:
                new_state = node.state.make_move(row, col)
                if new_state.game_over and new_state.winner == node.to_play:
                    node.terminal_moves[i] = True
            except Exception:
                # If move fails, it's not terminal
                pass
        
        # End timing and record
        t_detect_end = time.perf_counter()
        detect_time_ms = (t_detect_end - t_detect_start) * 1000.0
        
        # Store timing info for later reporting
        if not hasattr(self, '_terminal_detect_times'):
            self._terminal_detect_times = []
        self._terminal_detect_times.append(detect_time_ms)
        
        # Mark that we've detected terminal moves for this node
        node._terminal_moves_detected = True

    def _select_child_puct(self, node: MCTSNode) -> int:
        """Return index into node.legal_moves of the action maximizing PUCT score."""
        # PUCT: U = c_puct * P * sqrt(sum(N)) / (1 + N)
        # score = Q + U
        
        # Detect terminal moves if enabled
        if self.cfg.enable_terminal_move_detection:
            self._detect_terminal_moves(node, int(node.state.get_board_tensor().shape[-1]))
        
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

    def _get_node_depth(self, node: MCTSNode) -> int:
        """
        Get the depth of a node in the MCTS tree.
        
        Args:
            node: The MCTS node to get depth for
            
        Returns:
            The depth of the node (0 for root, 1 for root's children, etc.)
        """
        return node.depth


# -------- Convenience ----------

def create_selfplay_mcts_config(sims: int = 500, early_termination_threshold: float = 0.85) -> BaselineMCTSConfig:
    """
    Create an MCTS configuration optimized for self-play game generation.
    
    Args:
        sims: Number of simulations (default: 500)
        early_termination_threshold: Win probability threshold for simple termination (default: 0.85)
        
    Returns:
        BaselineMCTSConfig optimized for self-play
    """
    return BaselineMCTSConfig(
        sims=sims,
        batch_cap=64,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_eps=0.25,
        add_root_noise=False,  # Disable for self-play consistency
        temperature_start=0.5,
        temperature_end=0.01,
        temperature_decay_type="exponential",
        temperature_decay_moves=50,
        # Terminal move detection (always enabled)
        enable_terminal_move_detection=True,
        terminal_move_boost=10.0,
        virtual_loss_for_non_terminal=0.01,
        # Simple confidence-based termination for speed
        enable_early_termination=True,
        early_termination_threshold=early_termination_threshold,
    )

def create_tournament_mcts_config(sims: int = 200, early_termination_threshold: float = 0.95) -> BaselineMCTSConfig:
    """
    Create an MCTS configuration optimized for tournament play.
    
    Args:
        sims: Number of simulations (default: 200)
        early_termination_threshold: Win probability threshold for simple termination (default: 0.95)
        
    Returns:
        BaselineMCTSConfig optimized for tournament play
    """
    return BaselineMCTSConfig(
        sims=sims,
        batch_cap=64,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_eps=0.25,
        add_root_noise=False,
        temperature_start=1.0,
        temperature_end=0.1,
        temperature_decay_type="exponential",
        temperature_decay_moves=50,
        # Terminal move detection (always enabled)
        enable_terminal_move_detection=True,
        terminal_move_boost=10.0,
        virtual_loss_for_non_terminal=0.01,
        # Conservative simple termination for quality
        enable_early_termination=True,
        early_termination_threshold=early_termination_threshold,
    )

def create_fast_selfplay_mcts_config(sims: int = 200, early_termination_threshold: float = 0.8) -> BaselineMCTSConfig:
    """
    Create an MCTS configuration for very fast self-play game generation.
    
    Args:
        sims: Number of simulations (default: 200)
        early_termination_threshold: Win probability threshold for simple termination (default: 0.8)
        
    Returns:
        BaselineMCTSConfig optimized for very fast self-play
    """
    return BaselineMCTSConfig(
        sims=sims,
        batch_cap=64,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_eps=0.25,
        add_root_noise=False,
        temperature_start=0.5,
        temperature_end=0.01,
        temperature_decay_type="exponential",
        temperature_decay_moves=50,
        # Terminal move detection (always enabled)
        enable_terminal_move_detection=True,
        terminal_move_boost=10.0,
        virtual_loss_for_non_terminal=0.01,
        # Very aggressive simple termination for maximum speed
        enable_early_termination=True,
        early_termination_threshold=early_termination_threshold,
    )

def run_mcts_move(engine: HexGameEngine, model: ModelWrapper, state: HexGameState, cfg: Optional[BaselineMCTSConfig] = None, verbose: int = 0) -> Tuple[Tuple[int,int], Dict[str, Any], Dict[str, Any]]:
    """Run MCTS for one move and return (row,col), stats, tree_data."""
    if cfg is None:
        cfg = BaselineMCTSConfig()
    mcts = BaselineMCTS(engine, model, cfg)
    stats = mcts.run(state, verbose=verbose)
    move = mcts.pick_move(state, temperature=cfg.temperature_start, verbose=verbose)
    tree_data = mcts.get_tree_data(state)
    return move, stats, tree_data


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
