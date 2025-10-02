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

from hex_ai.config import (
    DEFAULT_GUMBEL_SIM_THRESHOLD,
    DEFAULT_GUMBEL_CANDIDATE_LOG_BASE,
    DEFAULT_GUMBEL_CANDIDATE_LOG_OFFSET,
    DEFAULT_GUMBEL_CANDIDATE_MIN,
    DEFAULT_GUMBEL_CANDIDATE_MAX
)


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


def gumbel_alpha_zero_root_batched(
    *,
    mcts,                    # your BaselineMCTS instance
    root,                    # root node
    policy_logits,           # np.array [K], log-probabilities (log of softmax output); illegal set to -inf
    total_sims: int,         # 50..500
    legal_actions: List[int],
    q_of_child: Callable[[int], float],  # returns Q in [0,1]
    n_of_child: Callable[[int], int],
    m: Optional[int] = None,
    c_visit: float = 50.0,
    c_scale: float = 1.0,
    temperature: float = 1.0,  # Noise scale for Gumbel sampling (beta)
    verbose: int = 0,        # Verbosity level for debug output
    rng=np.random,
    # Configurable candidate scaling parameters (use config defaults)
    candidate_log_base: float = DEFAULT_GUMBEL_CANDIDATE_LOG_BASE,
    candidate_log_offset: float = DEFAULT_GUMBEL_CANDIDATE_LOG_OFFSET,
    candidate_min: int = DEFAULT_GUMBEL_CANDIDATE_MIN,
    candidate_max: int = DEFAULT_GUMBEL_CANDIDATE_MAX,
):
    """
    Batched Gumbel-AlphaZero root selection that reuses existing MCTS batching infrastructure.
    
    This function replaces the per-sim loop with batched forced action execution,
    making it much more efficient by leveraging the existing neural network batching.
    
    Args:
        mcts: BaselineMCTS instance with run_forced_root_actions method
        root: Root MCTS node
        policy_logits: Log-probabilities [K] (log of softmax output); illegal actions should be -inf
        total_sims: Total number of simulations to allocate
        legal_actions: List of legal action indices at root
        q_of_child: Function that returns current empirical mean value for root child a in [0,1]
        n_of_child: Function that returns current visit count for root child a
        m: Number of actions to consider via Top-m (None for auto)
        c_visit: Gumbel-AlphaZero parameter (default: 50.0)
        c_scale: Gumbel-AlphaZero parameter (default: 1.0)
        temperature: Noise scale for Gumbel sampling (beta, default: 1.0)
        verbose: Verbosity level for debug output (default: 0)
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
    
    if temperature < 0:
        raise ValueError(f"temperature must be non-negative, got {temperature}")
    
    # NOTE: Temperature scaling with Gumbel root selection is currently broken.
    # The game uses automatic temperature scaling throughout the game (decreasing from 1.0),
    # but the Gumbel implementation doesn't handle this properly. Using beta (temperature)
    # as noise scale causes later moves to have lower performance due to reduced randomness.
    # For now, we disable temperature scaling in Gumbel to maintain consistent performance.
    # TODO: Implement proper temperature handling for Gumbel root selection.
    
    # Interpret temperature as noise scale beta (no scaling of logits or value terms)
    # beta = float(temperature)
    # temp_tol = 0.15
    # if beta <= 1.0 - temp_tol or beta >= 1.0 + temp_tol:
    #     message = f"Adjusting randomness in Gumbel by adjusting temperature is not yet supported.\n"
    #     message += f"For now, temperature must be between {1.0 - temp_tol} and {1.0 + temp_tol}, got {temperature}"
    #     raise ValueError(message)
    
    # DEBUG: Log noise scaling effects
    # DISABLED: beta-based debug logging due to temperature scaling issues
    # if temperature <= 0.1 and verbose >= 5:  # Only log for low temperatures to avoid spam
    #     print(f"GUMBEL NOISE SCALE DEBUG: beta={beta}")
    #     print(f"  Original logits range: [{np.min(policy_logits):.3f}, {np.max(policy_logits):.3f}]")
    #     print(f"  Noise scale: {beta}")
    
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
        
        # If no actions remain valid, this is a critical error
        if not validated_legal_actions:
            raise RuntimeError(f"All Gumbel legal actions became illegal. Root state may have changed unexpectedly.")
        
        # Update legal_actions to only include valid ones
        legal_actions = validated_legal_actions
    
    # Create mask for illegal actions
    mask = np.full(K, -np.inf)
    mask[legal_actions] = 0.0
    
    # Apply mask to logits (no temperature scaling)
    logits = policy_logits + mask
    
    # Choose candidate set via Gumbel Top-m on (g + logits)
    if m is None:
        # Configurable logarithmic candidate scaling: grows slowly with simulation count
        # Formula: m = min(max, max(min, log_base(total_sims) + offset))
        m_auto = int(min(candidate_max, max(candidate_min, math.log(total_sims, candidate_log_base) + candidate_log_offset)))
        m = min(len(legal_actions), total_sims, m_auto)
    
    timing_data['setup_time'] = time.perf_counter() - setup_start
    
    # Gumbel sampling timing
    gumbel_start = time.perf_counter()
    
    # Use same Gumbel vector 'g' for both Top-m and final scoring (avoids double-counting bias)
    # DISABLED: Scale Gumbel noise by beta (temperature as noise scale)
    # NOTE: Temperature scaling disabled due to automatic game temperature scaling issues
    g = sample_gumbel(K, rng=rng)
    # if beta <= 0.0:
    #     g.fill(0.0)  # deterministic, but keep Top-m + halving pipeline
    # else:
    #     g *= beta  # DISABLED: causes performance issues with automatic temperature scaling
    
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
    
    if verbose >= 5:
        print(f"GUMBEL DEBUG: Starting with {len(cand)} candidates, {R} rounds, {total_sims} total sims")
    
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
        n_val = n_of_child(a)
        score_val = g[a] + logits[a] + sigma * q_val
        
        # DEBUG: Print Q-values and visit counts for low temperatures
        # DISABLED: beta-based debug logging due to temperature scaling issues
        # if beta <= 0.1 and verbose >= 5:
        #     print(f"  Action {a}: g={g[a]:.3f}, logits={logits[a]:.3f}, q={q_val:.3f}, n={n_val}, sigma={sigma:.3f}, score={score_val:.3f}")
        #     print(f"    Components: gumbel={g[a]:.3f}, prior={logits[a]:.3f}, value={sigma * q_val:.3f}")
        
        return score_val
    
    def per_arm_allocation(total_left, rounds_left, num_arms):
        """
        Calculate per-arm allocation for equal budgeting per round.
        This restores the behavior from 5760a837: each surviving arm gets
        exactly per_arm targeted root simulations during the current round.
        
        Handles cases where we can't allocate even 1 simulation per arm.
        """
        if num_arms == 0:
            return 0
        
        # Calculate the theoretical per-arm allocation
        theoretical_per_arm = total_left // max(1, rounds_left * num_arms)
        
        # But ensure we don't exceed the available simulations
        max_per_arm = total_left // num_arms
        
        # If we can't allocate even 1 simulation per arm, return 0
        # This will cause the algorithm to terminate early
        if max_per_arm == 0:
            return 0
        
        # Return the minimum of theoretical allocation and budget constraint
        return max(1, min(theoretical_per_arm, max_per_arm))
    
    def schedule_round(arms_list, sims_left, rounds_left):
        """
        REVERTED: Use per-arm equal allocation per round (restore behavior from 5760a837).
        This removes early-round asymmetries that were introduced by the batch-fill strategy.
        
        Each surviving arm gets exactly per_arm targeted root simulations during this round.
        Batching is handled internally by MCTS and does not affect per-arm counts.
        """
        # Calculate exactly per_arm sims per arm in this round
        per_arm = per_arm_allocation(sims_left, rounds_left, len(arms_list))
        
        # Create exactly per_arm simulations for each arm
        actions = [a for a in arms_list for _ in range(per_arm)]
        return actions
    
    # Guards for degenerate cases
    if not cand:
        raise RuntimeError("No candidates available for Gumbel selection")
    if R <= 0:
        # Single candidate case - just return it
        return cand[0], {"nn_calls_per_move": 0, "total_leaves_evaluated": 0, "distinct_leaves_evaluated": 0, "candidates_m": m, "rounds_R": R, "avg_nn_batch_size": 0, "leaves_distinct_ratio": 0, "timing_breakdown": timing_data}
    
    # Round allocation and MCTS execution timing
    round_start = time.perf_counter()
    mcts_execution_start = time.perf_counter()
    
    for r in range(R):
        if not cand or sims_used >= total_sims:
            break
        rounds_left = R - r
        arms = len(cand)
        
        # Allocate exactly per_arm per arm for this round (equal budgeting)
        actions_this_round = schedule_round(
            cand, 
            total_sims - sims_used, 
            rounds_left
        )
        
        # Assertion-based test for equal allocation
        if actions_this_round:
            from collections import Counter
            action_counts = Counter(actions_this_round)
            per_arm = per_arm_allocation(total_sims - sims_used, rounds_left, arms)
            for a in cand:
                assert action_counts[a] == per_arm, f"Arm {a} got {action_counts[a]} sims, expected {per_arm}"
            assert len(actions_this_round) <= total_sims - sims_used, f"Round used {len(actions_this_round)} sims, only {total_sims - sims_used} left"
        
        if actions_this_round:
            # Track performance metrics from this round
            stats = mcts.run_forced_root_actions(root, actions_this_round, verbose=0)
            # Track batch metrics more accurately
            nn_calls_per_move += stats.get("batch_count", 0)
            total_leaves_evaluated += len(actions_this_round)  # Each action = one simulation
            # Note: unique_evals_total is not available in individual batch stats
            # We'll track this separately by looking at the final MCTS metrics
            sims_used += len(actions_this_round)
            
            # Log per_arm and len(cand) per round at verbose>=2
            if verbose >= 4:
                per_arm = per_arm_allocation(total_sims - sims_used + len(actions_this_round), rounds_left, arms)
                print(f"GUMBEL Round {r+1}: {arms} candidates, {per_arm} sims/arm, {len(actions_this_round)} total sims")
        
        if arms <= 1 or sims_used >= total_sims:
            break
        
        # Halve: keep the top half by the current score
        cand.sort(key=rank_key, reverse=True)
        keep = max(1, (arms + 1) // 2)
        cand = cand[:keep]
    
    timing_data['mcts_execution_time'] = time.perf_counter() - mcts_execution_start
    timing_data['round_allocation_time'] = time.perf_counter() - round_start
    
    # Final assertion: ensure we didn't exceed total_sims
    assert sims_used <= total_sims, f"Used {sims_used} sims, but only {total_sims} were allocated"
    
    # Final ranking timing
    ranking_start = time.perf_counter()
    
    # Final pick
    if len(cand) > 1:
        cand.sort(key=rank_key, reverse=True)
    
    # DEBUG: Compare final selection with top policy move
    # DISABLED: beta-based debug logging due to temperature scaling issues
    # if beta <= 0.1 and verbose >= 5:
    #     selected_action = cand[0]
    #     top_policy_action = int(np.argmax(logits))
    #     print(f"GUMBEL FINAL SELECTION DEBUG:")
    #     print(f"  Selected action: {selected_action} (score: {rank_key(selected_action):.3f})")
    #     print(f"  Top policy action: {top_policy_action} (score: {rank_key(top_policy_action):.3f})")
    #     print(f"  Same as top policy: {selected_action == top_policy_action}")
    #     if selected_action != top_policy_action:
    #         print(f"  Difference in scores: {rank_key(selected_action) - rank_key(top_policy_action):.3f}")
    
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


# Configuration display utilities

def generate_gumbel_summary_from_configs(strategy_configs: List[Any]) -> str:
    """
    Generate a concise summary of Gumbel configuration for strategy configurations.
    
    This function works with StrategyConfig objects from the tournament system.
    
    Args:
        strategy_configs: List of strategy configurations with config dictionaries
        
    Returns:
        String summary of Gumbel settings, or empty string if no MCTS strategies
    """
    mcts_configs = [c for c in strategy_configs if c.strategy_type == "mcts"]
    
    if not mcts_configs:
        return ""
    
    # Get Gumbel settings from the first MCTS config (they should all be the same)
    gumbel_enabled = any(c.config.get("enable_gumbel_root_selection", False) for c in mcts_configs)
    gumbel_sim_threshold = mcts_configs[0].config.get("gumbel_sim_threshold", DEFAULT_GUMBEL_SIM_THRESHOLD)
    
    # Build summary
    summary_parts = [
        f"Gumbel: Flag = {'on' if gumbel_enabled else 'off'}",
        f"sim threshold = {gumbel_sim_threshold}"
    ]
    
    # Add per-participant status
    participant_status = []
    for i, config in enumerate(strategy_configs):
        if config.strategy_type == "mcts":
            sims = config.config.get("mcts_sims", 0)
            gumbel_enabled_for_this = config.config.get("enable_gumbel_root_selection", False)
            will_use_gumbel = gumbel_enabled_for_this and sims <= gumbel_sim_threshold
            status = "on" if will_use_gumbel else "off"
            participant_status.append(f"s{i+1}: {status}")
        else:
            participant_status.append(f"s{i+1}: N/A")
    
    summary_parts.append(", ".join(participant_status))
    
    return ", ".join(summary_parts)


def generate_gumbel_summary_from_params(
    mcts_sims: int,
    enable_gumbel: bool,
    gumbel_sim_threshold: int = DEFAULT_GUMBEL_SIM_THRESHOLD,
    strategy_name: str = "selfplay"
) -> str:
    """
    Generate a concise summary of Gumbel configuration from individual parameters.
    
    This function works with individual parameters, suitable for selfplay scenarios.
    
    Args:
        mcts_sims: Number of MCTS simulations
        enable_gumbel: Whether Gumbel root selection is enabled
        gumbel_sim_threshold: Simulation threshold for Gumbel (default: from config)
        strategy_name: Name of the strategy (default: "selfplay")
        
    Returns:
        String summary of Gumbel settings
    """
    will_use_gumbel = enable_gumbel and mcts_sims <= gumbel_sim_threshold
    status = "on" if will_use_gumbel else "off"
    
    return f"Gumbel: Flag = {'on' if enable_gumbel else 'off'}, sim threshold = {gumbel_sim_threshold}, {strategy_name}: {status}"


def generate_gumbel_summary_from_mcts_config(mcts_config: Any) -> str:
    """
    Generate a concise summary of Gumbel configuration from an MCTS config object.
    
    Args:
        mcts_config: MCTS configuration object with Gumbel settings
        
    Returns:
        String summary of Gumbel settings
    """
    # Extract Gumbel settings from MCTS config
    enable_gumbel = getattr(mcts_config, 'enable_gumbel_root_selection', False)
    gumbel_sim_threshold = getattr(mcts_config, 'gumbel_sim_threshold', DEFAULT_GUMBEL_SIM_THRESHOLD)
    mcts_sims = getattr(mcts_config, 'sims', 0)
    
    return generate_gumbel_summary_from_params(
        mcts_sims=mcts_sims,
        enable_gumbel=enable_gumbel,
        gumbel_sim_threshold=gumbel_sim_threshold,
        strategy_name="selfplay"
    )


