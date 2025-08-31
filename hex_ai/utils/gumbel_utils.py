"""
Gumbel-AlphaZero utilities for efficient root selection with small simulation budgets.

This module implements the Gumbel-AlphaZero root selection algorithm, which is designed
to work efficiently with small numbers of simulations (50-500). It uses Gumbel-Top-k
sampling and Sequential Halving to select actions at the root node.

Reference: "Gumbel AlphaZero" by Danihelka et al. (2022)
"""

import math
import numpy as np
from typing import List, Callable, Optional, Tuple


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
        return g[a] + logits[a] + sigma * q_of_child(a)
    
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
