"""
Gumbel-AlphaZero utilities for efficient root selection with small simulation budgets.

This module implements the Gumbel-AlphaZero root selection algorithm, which is designed
to work efficiently with small numbers of simulations (50-500). It uses Gumbel-Top-k
sampling and Sequential Halving to select actions at the root node.

Reference: "Gumbel AlphaZero" by Danihelka et al. (2022)
"""

import math
import numpy as np
import time
from typing import Callable, List, Optional, Tuple, Dict, Any


def sample_gumbel(shape: Tuple[int, ...], eps: float = 1e-20, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """
    Sample from Gumbel(0,1) distribution using inverse CDF method.
    
    Args:
        shape: Shape of the output array
        eps: Small epsilon to avoid log(0) or log(1)
        rng: Random number generator (uses numpy.random if None)
        
    Returns:
        Gumbel(0,1) samples
    """
    if rng is None:
        rng = np.random
    
    # Gumbel(0,1) via inverse CDF: G = -log(-log(U)) where U ~ Uniform(0,1)
    u = rng.uniform(0.0 + eps, 1.0 - eps, size=shape)
    return -np.log(-np.log(u))


def gumbel_alpha_zero_root_select(
    policy_logits: np.ndarray,
    total_sims: int,
    run_one_sim: Callable[[int], None],
    q_of_child: Callable[[int], float],
    n_of_child: Callable[[int], int],
    legal_actions: List[int],
    m: Optional[int] = None,
    c_visit: float = 50.0,
    c_scale: float = 1.0,
    rng: Optional[np.random.RandomState] = None
) -> int:
    """
    Gumbel-AlphaZero root selection algorithm for small simulation budgets.
    
    This function implements the root-only Gumbel-AlphaZero selection procedure.
    It uses Gumbel-Top-k sampling to select candidate actions and Sequential Halving
    to allocate simulations efficiently among candidates.
    
    Args:
        policy_logits: Policy logits [K] BEFORE softmax; illegal actions should be -inf
        total_sims: Total number of simulations to allocate (n)
        run_one_sim: Function that runs one simulation with forced root action
        q_of_child: Function that returns current empirical mean value for root child a in [0,1]
        n_of_child: Function that returns current visit count for root child a
        legal_actions: List of legal action indices at root
        m: Number of actions to consider via Top-m (None for auto)
        c_visit: Gumbel-AlphaZero parameter (default: 50.0)
        c_scale: Gumbel-AlphaZero parameter (default: 1.0)
        rng: Random number generator (uses numpy.random if None)
        
    Returns:
        Selected action index
        
    Raises:
        ValueError: If parameters are invalid
        RuntimeError: If no legal actions available
    """
    if rng is None:
        rng = np.random
    
    if not legal_actions:
        raise RuntimeError("No legal actions available")
    
    if total_sims <= 0:
        raise ValueError(f"total_sims must be positive, got {total_sims}")
    
    K = len(policy_logits)
    
    # Create mask for illegal actions
    mask = np.full(K, -np.inf)
    mask[legal_actions] = 0.0
    
    # Use same Gumbel vector 'g' for both Top-m and final scoring (avoids double-counting bias)
    g = sample_gumbel(K, rng=rng)
    
    # Apply mask to logits
    logits = policy_logits + mask
    
    # Choose candidate set via Gumbel Top-m on (g + logits)
    if m is None:
        # Safe default: don't consider more actions than sims
        m = min(len(legal_actions), total_sims)
    
    top_scores = g + logits
    
    # Get indices of top-m actions (unordered)
    top_idx = np.argpartition(-top_scores, range(m))[:m]
    
    # Sequential Halving over the candidate set
    cand = list(top_idx)
    R = max(1, math.ceil(math.log2(len(cand))))  # number of rounds
    sims_used = 0
    
    def score(a: int) -> float:
        """Score function for action a: g[a] + logits[a] + σ(q̂[a])"""
        # σ(q) = (c_visit + max_b N(b))^c_scale * q
        maxN = max(1, max(n_of_child(b) for b in cand) if cand else 1)
        sigma = (c_visit + maxN) ** c_scale
        q_val = q_of_child(a)
        score_val = g[a] + logits[a] + sigma * q_val
        return score_val
    
    for r in range(R):
        if not cand:
            break
        
        rounds_left = R - r
        
        # Evenly spread remaining simulations across remaining candidates and rounds
        per_arm = max(1, (total_sims - sims_used) // (len(cand) * rounds_left))
        
        # Ensure at least one new visit per arm per round (paper uses this safeguard)
        for a in list(cand):
            for _ in range(per_arm):
                if sims_used >= total_sims:
                    break
                run_one_sim(a)
                sims_used += 1
        
        if len(cand) <= 1 or sims_used >= total_sims:
            break
        
        # Rank by g + logits + σ(q̂) and keep top half
        cand.sort(key=score, reverse=True)
        keep = max(1, (len(cand) + 1) // 2)
        cand = cand[:keep]
    
    # Final pick: argmax of g + logits + σ(q̂) among remaining
    if len(cand) > 1:
        cand.sort(key=score, reverse=True)
    
    return cand[0]


def normalize_q_values(q_values: np.ndarray, min_val: float = -1.0, max_val: float = 1.0) -> np.ndarray:
    """
    Normalize Q-values to [0,1] range for Gumbel-AlphaZero.
    
    Args:
        q_values: Q-values in [min_val, max_val] range
        min_val: Minimum possible Q-value
        max_val: Maximum possible Q-value
        
    Returns:
        Normalized Q-values in [0,1] range
    """
    if max_val <= min_val:
        raise ValueError(f"max_val ({max_val}) must be greater than min_val ({min_val})")
    
    # Clip to valid range first
    q_clipped = np.clip(q_values, min_val, max_val)
    
    # Normalize to [0,1]
    return (q_clipped - min_val) / (max_val - min_val)


def gumbel_alpha_zero_root_batched(
    *,
    mcts,                    # your BaselineMCTS instance
    root,                    # root node
    policy_logits,           # np.array [K], BEFORE softmax; illegal set to -inf
    total_sims: int,         # 50..500
    legal_actions: List[int],
    q_of_child: Callable[[int], float],  # returns Q in [0,1]
    n_of_child: Callable[[int], int],
    m: Optional[int] = None,
    c_visit: float = 50.0,
    c_scale: float = 1.0,
    temperature: float = 1.0,  # Temperature for Gumbel sampling
    rng=np.random,
):
    """
    Batched Gumbel-AlphaZero root selection that reuses existing MCTS batching infrastructure.
    
    This function replaces the per-sim loop with batched forced action execution,
    making it much more efficient by leveraging the existing neural network batching.
    
    Args:
        mcts: BaselineMCTS instance with run_forced_root_actions method
        root: Root MCTS node
        policy_logits: Policy logits [K] BEFORE softmax; illegal actions should be -inf
        total_sims: Total number of simulations to allocate
        legal_actions: List of legal action indices at root
        q_of_child: Function that returns current empirical mean value for root child a in [0,1]
        n_of_child: Function that returns current visit count for root child a
        m: Number of actions to consider via Top-m (None for auto)
        c_visit: Gumbel-AlphaZero parameter (default: 50.0)
        c_scale: Gumbel-AlphaZero parameter (default: 1.0)
        temperature: Temperature for Gumbel sampling (default: 1.0)
        rng: Random number generator (uses numpy.random if None)
        
    Returns:
        Tuple of (selected_action_index, performance_metrics_dict)
        
    Raises:
        ValueError: If parameters are invalid
        RuntimeError: If no legal actions available
    """
    # Detailed timing instrumentation
    timing_data = {
        'setup_time': 0.0,
        'gumbel_sampling_time': 0.0,
        'top_m_selection_time': 0.0,
        'round_allocation_time': 0.0,
        'mcts_execution_time': 0.0,
        'ranking_time': 0.0,
        'total_time': 0.0
    }
    
    total_start = time.perf_counter()
    
    if rng is None:
        rng = np.random
    
    if not legal_actions:
        raise RuntimeError("No legal actions available")
    
    if total_sims <= 0:
        raise ValueError(f"total_sims must be positive, got {total_sims}")
    
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    
    K = policy_logits.shape[0]
    
    # Setup phase timing
    setup_start = time.perf_counter()
    
    # CRITICAL FIX: Validate that legal_actions are actually legal at current root state
    # This prevents Gumbel from selecting actions that became illegal due to state changes
    current_legal_indices = set(root.legal_indices)
    validated_legal_actions = [a for a in legal_actions if a in current_legal_indices]
    
    if len(validated_legal_actions) != len(legal_actions):
        # Log the mismatch for debugging
        illegal_actions = [a for a in legal_actions if a not in current_legal_indices]
        print(f"WARNING: Gumbel received {len(illegal_actions)} illegal actions: {illegal_actions}")
        print(f"Current root legal indices: {sorted(current_legal_indices)}")
        print(f"Original legal_actions: {sorted(legal_actions)}")
        
        # If no actions remain valid, this is a critical error
        if not validated_legal_actions:
            raise RuntimeError(f"All Gumbel legal actions became illegal. Root state may have changed unexpectedly.")
        
        # Update legal_actions to only include valid ones
        legal_actions = validated_legal_actions
    
    # Create mask for illegal actions
    mask = np.full(K, -np.inf)
    mask[legal_actions] = 0.0
    
    # Apply mask to logits
    logits = policy_logits + mask
    
    # Choose candidate set via Gumbel Top-m on (g + logits)
    if m is None:
        # IMPROVEMENT: Cap candidates to prevent explosion in mid-game positions
        # Good default: don't consider more actions than we can reasonably evaluate
        m = min(len(legal_actions), total_sims, 32)  # Cap at 32 for k≈200-500
    
    timing_data['setup_time'] = time.perf_counter() - setup_start
    
    # Gumbel sampling timing
    gumbel_start = time.perf_counter()
    
    # Use same Gumbel vector 'g' for both Top-m and final scoring (avoids double-counting bias)
    # Apply temperature scaling to Gumbel samples
    g = sample_gumbel(K, rng=rng) * temperature
    
    timing_data['gumbel_sampling_time'] = time.perf_counter() - gumbel_start
    
    # Top-m selection timing
    top_m_start = time.perf_counter()
    
    top_scores = g + logits
    
    # Get indices of top-m actions (unordered)
    top_idx = np.argpartition(-top_scores, range(m))[:m]
    
    timing_data['top_m_selection_time'] = time.perf_counter() - top_m_start
    
    # Sequential Halving over the candidate set
    cand = list(top_idx)
    R = max(1, math.ceil(math.log2(len(cand))))  # number of rounds
    sims_used = 0
    
    # Performance tracking
    nn_calls_per_move = 0
    total_leaves_evaluated = 0
    distinct_leaves_evaluated = 0
    
    def rank_key(a):
        """Score function for action a: g[a] + logits[a] + σ(q̂[a])"""
        # σ(q) = (c_visit + max_b N(b))^c_scale * q
        maxN = max(1, max(n_of_child(b) for b in cand) if cand else 1)
        sigma = (c_visit + maxN) ** c_scale
        q_val = q_of_child(a)
        score_val = g[a] + logits[a] + sigma * q_val
        return score_val
    
    def schedule_round(arms_list, sims_left, rounds_left, batch_cap):
        """
        IMPROVEMENT: Allocate per round to fill batches, not per-sim.
        This ensures we get full batches instead of tiny 1-off NN calls.
        """
        # At least one full batch, try to split budget evenly across remaining rounds
        per_round = max(batch_cap, sims_left // rounds_left)
        per_round = min(per_round, sims_left)
        
        # Distribute across arms as evenly as possible
        A = len(arms_list)
        base = per_round // max(1, A)
        extra = per_round - base * A
        
        counts = {a: base for a in arms_list}
        for a in rng.permutation(arms_list)[:extra]:
            counts[a] += 1
        
        # Flatten into one list for this round and shuffle
        actions = [a for a in arms_list for _ in range(counts[a])]
        rng.shuffle(actions)
        return actions
    
    # Round allocation and MCTS execution timing
    round_start = time.perf_counter()
    mcts_execution_start = time.perf_counter()
    
    for r in range(R):
        if not cand or sims_used >= total_sims:
            break
        rounds_left = R - r
        arms = len(cand)
        
        # IMPROVEMENT: Use round-based allocation instead of per-arm
        actions_this_round = schedule_round(
            cand, 
            total_sims - sims_used, 
            rounds_left, 
            mcts.cfg.batch_cap
        )
        
        if actions_this_round:
            # Track performance metrics from this round
            stats = mcts.run_forced_root_actions(root, actions_this_round, verbose=0)
            # Track batch metrics more accurately
            nn_calls_per_move += stats.get("batch_count", 0)
            total_leaves_evaluated += len(actions_this_round)  # Each action = one simulation
            # Note: unique_evals_total is not available in individual batch stats
            # We'll track this separately by looking at the final MCTS metrics
            sims_used += len(actions_this_round)
        
        if arms <= 1 or sims_used >= total_sims:
            break
        
        # Halve: keep the top half by the current score
        cand.sort(key=rank_key, reverse=True)
        keep = max(1, (arms + 1) // 2)
        cand = cand[:keep]
    
    timing_data['mcts_execution_time'] = time.perf_counter() - mcts_execution_start
    timing_data['round_allocation_time'] = time.perf_counter() - round_start
    
    # Final ranking timing
    ranking_start = time.perf_counter()
    
    # Final pick
    if len(cand) > 1:
        cand.sort(key=rank_key, reverse=True)
    
    timing_data['ranking_time'] = time.perf_counter() - ranking_start
    timing_data['total_time'] = time.perf_counter() - total_start
    
    # Return both the selected action and performance metrics
    performance_metrics = {
        "nn_calls_per_move": nn_calls_per_move,
        "total_leaves_evaluated": total_leaves_evaluated,
        "distinct_leaves_evaluated": distinct_leaves_evaluated,
        "candidates_m": m,
        "rounds_R": R,
        "avg_nn_batch_size": total_leaves_evaluated / max(1, nn_calls_per_move),
        "leaves_distinct_ratio": distinct_leaves_evaluated / max(1, total_leaves_evaluated),
        "timing_breakdown": timing_data
    }
    
    return cand[0], performance_metrics
