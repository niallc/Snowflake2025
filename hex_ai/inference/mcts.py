# baseline_mcts.py
# Lean, single-threaded, explicitly-batched AlphaZero-style MCTS for Hex.
# Compatible with flat-file or package imports via shims.

# ============================== MCTS BASELINE — TODOs ==============================
# Strictness & Error Handling
# - [ ] Add `cfg.strict_shapes=True` and replace defensive pad/truncate with asserts:
#       assert policy_logits.shape[-1] == action_size, "Policy size mismatch"
# - [ ] Add `cfg.strict_probs=True` to validate priors after masking/softmax:
#       assert np.isfinite(P).all() and abs(P.sum() - 1.0) < 1e-6 and (P >= 0).all()
# - [ ] Validate action_size: assert action_size == board_size * board_size
# - [ ] Validate legal index bounds: all(i in [0, action_size)) for i in `legal_indices`
# - [ ] Fail-fast root invariants (post expansion): len(root.P) == len(root.legal_moves) and sum≈1

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
    temperature: float = 1.0  # for pick_move
    seed: int = 1234

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

# ------------------ Data structures ------------------

class MCTSNode:
    __slots__ = (
        "state", "to_play", "legal_moves", "legal_indices",
        "children", "N", "W", "Q", "P", "is_expanded",
        "state_hash", "is_terminal", "winner_str"
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

# ------------------ Core MCTS ------------------

class BaselineMCTS:
    def __init__(self, engine: HexGameEngine, model: ModelWrapper, cfg: BaselineMCTSConfig):
        self.engine = engine
        self.model = model
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        random.seed(cfg.seed)

        # Cache: state_hash -> (policy_logits_np [A], value_logit_float)
        self.eval_cache: Dict[int, Tuple[np.ndarray, float]] = {}
        self.cache_hits = 0
        self.cache_misses = 0

        self._root_noise_applied = False

        # Root node from the last run() (used by pick_move)
        self._root_node = None

    # ---------- Public API ----------
    def run(self, root_state: HexGameState) -> Dict[str, Any]:
        """
        Run MCTS for cfg.sims simulations starting from root_state.
        Returns a dict of timing and batching stats for this move.
        """
        # Prepare root
        board_tensor = root_state.get_board_tensor()
        board_size = int(board_tensor.shape[-1])
        action_size = board_size * board_size
        root = MCTSNode(root_state, board_size)

        # One-time root expansion if not expanded and not terminal
        encode_ms = 0.0
        stack_ms = 0.0
        expand_ms = 0.0
        backprop_ms = 0.0
        h2d_ms_total = 0.0
        forward_ms_total = 0.0
        d2h_ms_total = 0.0
        batch_sizes: List[int] = []
        forward_ms_list: List[float] = []

        sims_remaining = self.cfg.sims

        # Ensure root expanded (through cache or model) unless terminal
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

        # ---- Simulation loop with explicit batched leaves ----
        while sims_remaining > 0:
            # 1) SELECT up to K leaves
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
                        (r, c) = node.legal_moves[child_idx]
                        child_state = node.state.make_move(r, c)
                        child = MCTSNode(child_state, board_size)
                        node.children[child_idx] = child
                    node = child

                # If we reached select_budget, stop; else continue to select next leaf
                if len(leaves) >= select_budget:
                    break

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
                cached = self.eval_cache.get(leaf.state_hash, None)
                if cached is not None:
                    self.cache_hits += 1
                    cached_expansions.append((i, cached[0], cached[1]))
                else:
                    self.cache_misses += 1
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

        # Build stats
        _store_root = True
        self._root_node = root
        stats = {
            "encode_ms": encode_ms,
            "stack_ms": stack_ms,
            "h2d_ms": h2d_ms_total,
            "forward_ms": forward_ms_total,
            "d2h_ms": d2h_ms_total,
            "expand_ms": expand_ms,
            "backprop_ms": backprop_ms,
            "batch_count": len(batch_sizes),
            "batch_sizes": batch_sizes,
            "forward_ms_list": forward_ms_list,
            "median_forward_ms_ex_warm": _median_excluding_first(forward_ms_list),
            "p90_forward_ms_ex_warm": _p90_excluding_first(forward_ms_list),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
        }
        return stats

    
    def pick_move(self, root_state: HexGameState, temperature: Optional[float] = None) -> Tuple[int,int]:
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

        if root.is_terminal:
            if root.legal_moves:
                # Terminal shouldn't have legal moves, but just in case
                return random.choice(root.legal_moves)
            raise RuntimeError("No legal moves at root (terminal state).")

        # Use visit counts accumulated during run()
        counts = root.N.astype(np.float64)
        if counts.sum() <= 0:
            raise RuntimeError(f"No visits recorded during MCTS search. This indicates a bug in the search algorithm.")

        temp = self.cfg.temperature if temperature is None else temperature
        if temp <= 1e-6:
            a_idx = int(np.argmax(counts))
        else:
            pi = np.power(counts, 1.0 / temp)
            if not np.isfinite(pi).all() or np.sum(pi) <= 0:
                raise ValueError(f"Invalid temperature scaling result: pi={pi}, counts={counts}, temp={temp}")
            pi /= np.sum(pi)
            a_idx = int(self.rng.choice(len(pi), p=pi))
        return root.legal_moves[a_idx]

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
        noise = self.rng.dirichlet([self.cfg.dirichlet_alpha] * L)
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
        N_sum = np.sum(node.N, dtype=np.float64)
        if N_sum <= 1e-9:
            # All U terms reduce to c*P; just pick argmax P
            return int(np.argmax(node.P))
        U = self.cfg.c_puct * node.P * math.sqrt(N_sum) / (1.0 + node.N)
        score = node.Q + U
        return int(np.argmax(score))


# -------- Convenience ----------

def run_mcts_move(engine: HexGameEngine, model: ModelWrapper, state: HexGameState, cfg: Optional[BaselineMCTSConfig] = None) -> Tuple[Tuple[int,int], Dict[str, Any]]:
    """Run MCTS for one move and return (row,col), stats."""
    if cfg is None:
        cfg = BaselineMCTSConfig()
    mcts = BaselineMCTS(engine, model, cfg)
    stats = mcts.run(state)
    move = mcts.pick_move(state, temperature=cfg.temperature)
    return move, stats


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
