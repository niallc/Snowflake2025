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
    DEFAULT_GUMBEL_CANDIDATE_POWER_SCALE,
    DEFAULT_GUMBEL_CANDIDATE_POWER_RATE,
    DEFAULT_GUMBEL_CANDIDATE_POWER_OFFSET,
    DEFAULT_GUMBEL_CANDIDATE_MIN,
    DEFAULT_GUMBEL_CANDIDATE_MAX,
    DEFAULT_GUMBEL_USE_GUMBEL_IN_FINAL_EVAL
)
from hex_ai.utils.legal_action_contracts import assert_exact_legal_action_match


def calculate_power_law_candidates(
    total_sims: int,
    power_scale: float,
    power_rate: float,
    power_offset: float,
    candidate_min: int,
    candidate_max: int,
    num_legal_actions: int
) -> int:
    """
    Calculate the number of Gumbel candidates using power-law scaling.
    
    Formula: m = (total_sims * power_scale)^power_rate + power_offset
    
    Args:
        total_sims: Total number of simulations
        power_scale: Scale factor for power-law scaling
        power_rate: Rate (exponent) for power-law scaling
        power_offset: Offset for power-law scaling
        candidate_min: Minimum number of candidates
        candidate_max: Maximum number of candidates
        num_legal_actions: Number of legal actions available
        
    Returns:
        Number of candidates to consider
    """
    m_auto = int(min(candidate_max, max(candidate_min, (total_sims * power_scale) ** power_rate + power_offset)))
    m = min(num_legal_actions, total_sims, m_auto)
    return m


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


def compute_completed_baseline_v_pi(
    pi: np.ndarray,
    legal_actions: List[int],
    q_of_child: Callable[[int], float],
    n_of_child: Callable[[int], int],
) -> float:
    """
    Compute completed baseline v_pi in [0,1].

    Uses policy-weighted mean over visited children only, then normalizes by visited mass.
    Falls back to 0.5 when no visited mass exists.
    """
    pi_unvisited = 0.0
    num = 0.0
    for action in legal_actions:
        if n_of_child(action) > 0:
            num += float(pi[action]) * float(q_of_child(action))
        else:
            pi_unvisited += float(pi[action])

    denom = 1.0 - pi_unvisited
    if denom <= 1e-12:
        return 0.5
    return num / denom


def build_gumbel_score_rows(
    *,
    actions: List[int],
    pi: np.ndarray,
    logits: np.ndarray,
    gumbel_noise: np.ndarray,
    c_scale: float,
    v_pi_01: float,
    q_of_child: Callable[[int], float],
    n_of_child: Callable[[int], int],
    include_gumbel_term: bool,
) -> List[Dict[str, Any]]:
    """
    Build sorted per-action score rows using the same terms as Gumbel root ranking.

    score_without_gumbel = log_prior + c_scale * (q_01 - v_pi)
    score_with_gumbel = score_without_gumbel + gumbel_noise
    score = score_with_gumbel if include_gumbel_term else score_without_gumbel
    """
    rows: List[Dict[str, Any]] = []
    sigma = float(c_scale)
    baseline = float(v_pi_01)

    for action in actions:
        action_int = int(action)
        visits = int(n_of_child(action_int))

        if visits > 0:
            q_01 = float(q_of_child(action_int))
            q_source = "tree"
        else:
            q_01 = baseline
            q_source = "v_pi_completion"

        adv = q_01 - baseline
        value_term = sigma * adv
        log_prior = float(logits[action_int])
        raw_gumbel = float(gumbel_noise[action_int])
        gumbel_term = raw_gumbel if include_gumbel_term else 0.0

        score_without_gumbel = log_prior + value_term
        score_with_gumbel = score_without_gumbel + raw_gumbel
        score = score_with_gumbel if include_gumbel_term else score_without_gumbel

        rows.append(
            {
                "tensor_action": action_int,
                "visits": visits,
                "prior": float(pi[action_int]),
                "log_prior": log_prior,
                "gumbel": raw_gumbel,
                "gumbel_term": gumbel_term,
                "q_01": q_01,
                "q_ptm_signed": (2.0 * q_01) - 1.0,
                "q_source": q_source,
                "v_pi_01": baseline,
                "adv_01": adv,
                "value_term": value_term,
                "score_without_gumbel": score_without_gumbel,
                "score_with_gumbel": score_with_gumbel,
                "score": score,
            }
        )

    rows.sort(key=lambda row: float(row["score"]), reverse=True)
    return rows


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
    temperature: float = 1.0,  # Fixed contract for Gumbel root selection (must be 1.0)
    verbose: int = 0,        # Verbosity level for debug output
    rng=np.random,
    # Power-law candidate scaling parameters
    candidate_power_scale: float = DEFAULT_GUMBEL_CANDIDATE_POWER_SCALE,
    candidate_power_rate: float = DEFAULT_GUMBEL_CANDIDATE_POWER_RATE,
    candidate_power_offset: float = DEFAULT_GUMBEL_CANDIDATE_POWER_OFFSET,
    candidate_min: int = DEFAULT_GUMBEL_CANDIDATE_MIN,
    candidate_max: int = DEFAULT_GUMBEL_CANDIDATE_MAX,
    # Gumbel ranking stabilization parameters
    use_gumbel_in_final_eval: bool = DEFAULT_GUMBEL_USE_GUMBEL_IN_FINAL_EVAL,  # Remove Gumbel noise in final evaluation
    eval_mode: bool = False,  # Whether this is evaluation mode (affects Gumbel noise usage)
    trace_event: Optional[Callable[[Dict[str, Any]], None]] = None,
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
        c_visit: Gumbel-AlphaZero parameter
        c_scale: Gumbel-AlphaZero parameter
        temperature: Fixed Gumbel temperature contract value (must be 1.0)
        verbose: Verbosity level for debug output
        rng: Random number generator (uses numpy.random if None)
        use_gumbel_in_final_eval: Whether to use Gumbel noise in final evaluation
        eval_mode: Whether this is evaluation mode
        
    Returns:
        Tuple of (selected_action_index, performance_metrics_dict)
        
    Raises:
        ValueError: If parameters are invalid or legality contracts are violated
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

    if not np.isfinite(temperature):
        raise ValueError(f"temperature must be finite, got {temperature}")

    if not math.isclose(float(temperature), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "Gumbel root temperature scaling is unsupported: "
            f"temperature must be 1.0, got {temperature}."
        )
    
    K = policy_logits.shape[0]
    
    # Setup phase timing
    setup_start = time.perf_counter()
    
    def assert_root_legality(context: str) -> None:
        assert_exact_legal_action_match(
            expected_actions=legal_actions,
            observed_actions=root.legal_indices,
            context=context,
            contract_name="Gumbel legality contract",
            expected_label="Provided legal_actions",
            observed_label="Current root legal_indices",
        )

    assert_root_legality("gumbel_alpha_zero_root_batched:entry")
    legal_actions = [int(action) for action in legal_actions]
    
    # Create mask for illegal actions
    mask = np.full(K, -np.inf)
    mask[legal_actions] = 0.0
    
    # Apply mask to logits (no temperature scaling)
    logits = policy_logits + mask
    
    # Softmax over legal actions
    pi = np.zeros(K, dtype=np.float64)
    logits_legal = logits[legal_actions]  # logits already has -inf for illegal
    # stable softmax for legal slice
    shift = np.max(logits_legal)
    exp_legal = np.exp(logits_legal - shift)
    Z = np.sum(exp_legal)
    if Z == 0.0:
        # All -inf or degenerate; assign uniform over legal to avoid NaNs
        uniform = 1.0 / max(1, len(legal_actions))
        for a in legal_actions:
            pi[a] = uniform
    else:
        pi_legal = exp_legal / Z
        for a, p in zip(legal_actions, pi_legal):
            pi[a] = p

    # Choose candidate set via Gumbel Top-m on (g + logits)
    if m is None:
        # Power-law candidate scaling: grows faster than logarithmic
        # Formula: m = (total_sims * power_scale)^power_rate + power_offset
        m_auto = calculate_power_law_candidates(
            total_sims, candidate_power_scale, candidate_power_rate, candidate_power_offset,
            candidate_min, candidate_max, len(legal_actions)
        )
        m = min(len(legal_actions), total_sims, m_auto)
    
    timing_data['setup_time'] = time.perf_counter() - setup_start
    
    # Gumbel sampling timing
    gumbel_start = time.perf_counter()
    
    # Use same Gumbel vector 'g' for both Top-m and final scoring (avoids double-counting bias)
    # Sample Gumbel noise only for legal actions (optimization: skip illegal actions)
    g = np.zeros(K, dtype=np.float64)
    g_legal = sample_gumbel(len(legal_actions), rng=rng)
    for i, a in enumerate(legal_actions):
        g[a] = g_legal[i]
    
    timing_data['gumbel_sampling_time'] = time.perf_counter() - gumbel_start
    
    # Top-m selection timing
    top_m_start = time.perf_counter()
    
    # Micro-optimization: compute top-m on legal slice only (avoids argpartition over illegal entries)
    top_scores_legal = g[legal_actions] + logits[legal_actions]
    idx_local = np.argpartition(top_scores_legal, -m)[-m:]  # unordered
    top_idx = np.array([legal_actions[i] for i in idx_local], dtype=int)
    
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
    last_round_rows: List[Dict[str, Any]] = []
    forced_stats_totals: Dict[str, Any] = {
        "batch_count": 0,
        "batch_sizes": [],
        "h2d_ms": 0.0,
        "forward_ms": 0.0,
        "pure_forward_ms": 0.0,
        "sync_ms": 0.0,
        "d2h_ms": 0.0,
        "select_ms": 0.0,
        "encode_ms": 0.0,
        "stack_ms": 0.0,
        "expand_ms": 0.0,
        "backprop_ms": 0.0,
        "cache_lookup_ms": 0.0,
        "state_creation_ms": 0.0,
        "make_move_ms": 0.0,
    }

    # Compute v_pi once per root and use it consistently for all ranking/debug rows.
    v_pi = compute_completed_baseline_v_pi(pi, legal_actions, q_of_child, n_of_child)
    round_uses_gumbel = not (eval_mode and not use_gumbel_in_final_eval)

    def emit_trace(event: Dict[str, Any]) -> None:
        """Best-effort trace callback; must never affect search behavior."""
        if trace_event is None:
            return
        try:
            trace_event(event)
        except Exception:
            pass

    def build_top_m_rows(actions: List[int]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for action in actions:
            action_int = int(action)
            log_prior = float(logits[action_int])
            gumbel_val = float(g[action_int])
            rows.append(
                {
                    "tensor_action": action_int,
                    "prior": float(pi[action_int]),
                    "log_prior": log_prior,
                    "gumbel": gumbel_val,
                    "top_m_score": log_prior + gumbel_val,
                }
            )
        rows.sort(key=lambda row: row["top_m_score"], reverse=True)
        return rows
    
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
    
    # Guards for degenerate cases
    if not cand:
        raise RuntimeError("No candidates available for Gumbel selection")

    emit_trace(
        {
            "type": "gumbel_root_setup",
            "total_sims": int(total_sims),
            "legal_action_count": int(len(legal_actions)),
            "candidate_count": int(len(cand)),
            "rounds_R": int(R),
            "c_visit": float(c_visit),
            "c_scale": float(c_scale),
            "temperature": float(temperature),
            "v_pi_01": float(v_pi),
            "round_uses_gumbel": bool(round_uses_gumbel),
            "score_formula_round": "g + log_prior + c_scale*(q_01 - v_pi)",
            "score_formula_final": "log_prior + c_scale*(q_01 - v_pi)",
        }
    )

    top_m_rows = build_top_m_rows(legal_actions)
    top_m_selected_set = {int(action) for action in cand}
    top_m_selected_rows = [row for row in top_m_rows if row["tensor_action"] in top_m_selected_set]
    top_m_excluded_rows = [row for row in top_m_rows if row["tensor_action"] not in top_m_selected_set][:5]
    emit_trace(
        {
            "type": "gumbel_top_m_selection",
            "candidate_count": int(len(top_m_selected_rows)),
            "selected_rows": top_m_selected_rows,
            "excluded_rows": top_m_excluded_rows,
        }
    )
    
    # Round allocation and MCTS execution timing
    round_start = time.perf_counter()
    mcts_execution_start = time.perf_counter()
    
    for r in range(R):
        if not cand or sims_used >= total_sims:
            break
        rounds_left = R - r
        arms = len(cand)

        # Compute this stage's per-arm allocation (equal budgeting)
        per_arm = per_arm_allocation(total_sims - sims_used, rounds_left, arms)

        pre_round_rows = build_gumbel_score_rows(
            actions=cand,
            pi=pi,
            logits=logits,
            gumbel_noise=g,
            c_scale=c_scale,
            v_pi_01=v_pi,
            q_of_child=q_of_child,
            n_of_child=n_of_child,
            include_gumbel_term=round_uses_gumbel,
        )
        emit_trace(
            {
                "type": "gumbel_round_start",
                "round_index": int(r + 1),
                "rounds_total": int(R),
                "arms": int(arms),
                "per_arm": int(per_arm),
                "sims_used_before": int(sims_used),
                "sims_left_before": int(total_sims - sims_used),
                "candidate_rows": pre_round_rows,
            }
        )

        # NEW: if we cannot afford even 1 sim per arm, do not prune on stale evidence
        if per_arm == 0:
            last_round_rows = pre_round_rows
            emit_trace(
                {
                    "type": "gumbel_round_end",
                    "round_index": int(r + 1),
                    "rounds_total": int(R),
                    "sims_used_after": int(sims_used),
                    "sims_left_after": int(total_sims - sims_used),
                    "candidate_rows": pre_round_rows,
                    "keep_count": int(len(cand)),
                    "kept_actions": [int(a) for a in cand],
                    "dropped_actions": [],
                    "reason": "insufficient_budget",
                }
            )
            break  # exit SH loop; proceed to final ranking over 'cand' as-is

        assert_root_legality(
            f"gumbel_alpha_zero_root_batched:round_{r + 1}_pre_forced_actions"
        )

        # Create exactly per_arm simulations for each arm
        actions_this_round = [a for a in cand for _ in range(per_arm)]

        # (Optional assertion) each arm gets per_arm sims
        if actions_this_round:
            from collections import Counter
            counts = Counter(actions_this_round)
            for a in cand:
                assert counts[a] == per_arm

        # Run the forced actions
        stats = mcts.run_forced_root_actions(root, actions_this_round, verbose=0)
        batch_count = int(stats.get("batch_count", 0))
        nn_calls_per_move += batch_count
        forced_stats_totals["batch_count"] += batch_count
        batch_sizes = stats.get("batch_sizes", []) or []
        if isinstance(batch_sizes, list):
            forced_stats_totals["batch_sizes"].extend(batch_sizes)
        for key in (
            "h2d_ms",
            "forward_ms",
            "pure_forward_ms",
            "sync_ms",
            "d2h_ms",
            "select_ms",
            "encode_ms",
            "stack_ms",
            "expand_ms",
            "backprop_ms",
            "cache_lookup_ms",
            "state_creation_ms",
            "make_move_ms",
        ):
            forced_stats_totals[key] += float(stats.get(key, 0.0))
        if "simulations_completed" not in stats:
            raise ValueError(
                "Forced-root simulation contract violated in gumbel_alpha_zero_root_batched. "
                "run_forced_root_actions must report simulations_completed."
            )
        simulations_completed = int(stats["simulations_completed"])
        requested_simulations = len(actions_this_round)
        if simulations_completed != requested_simulations:
            raise ValueError(
                "Forced-root simulation contract violated in gumbel_alpha_zero_root_batched. "
                f"Requested {requested_simulations} forced simulations but completed {simulations_completed}."
            )
        total_leaves_evaluated += simulations_completed
        sims_used += simulations_completed

        # Log per_arm and len(cand) per round at verbose>=4
        if verbose >= 4:
            print(f"GUMBEL Round {r+1}: {arms} candidates, {per_arm} sims/arm, {len(actions_this_round)} total sims")

        post_round_rows = build_gumbel_score_rows(
            actions=cand,
            pi=pi,
            logits=logits,
            gumbel_noise=g,
            c_scale=c_scale,
            v_pi_01=v_pi,
            q_of_child=q_of_child,
            n_of_child=n_of_child,
            include_gumbel_term=round_uses_gumbel,
        )
        last_round_rows = post_round_rows

        if arms <= 1 or sims_used >= total_sims:
            keep = len(post_round_rows)
            kept_actions = [int(row["tensor_action"]) for row in post_round_rows[:keep]]
            dropped_actions: List[int] = []
            emit_trace(
                {
                    "type": "gumbel_round_end",
                    "round_index": int(r + 1),
                    "rounds_total": int(R),
                    "sims_used_after": int(sims_used),
                    "sims_left_after": int(total_sims - sims_used),
                    "candidate_rows": post_round_rows,
                    "keep_count": int(keep),
                    "kept_actions": kept_actions,
                    "dropped_actions": dropped_actions,
                    "reason": "budget_exhausted_or_single_candidate",
                }
            )
            cand = kept_actions
            break

        # Halve after NEW evidence using the same score rows used for debug trace.
        keep = max(1, (arms + 1) // 2)
        ranked_actions = [int(row["tensor_action"]) for row in post_round_rows]
        kept_actions = ranked_actions[:keep]
        dropped_actions = ranked_actions[keep:]
        emit_trace(
            {
                "type": "gumbel_round_end",
                "round_index": int(r + 1),
                "rounds_total": int(R),
                "sims_used_after": int(sims_used),
                "sims_left_after": int(total_sims - sims_used),
                "candidate_rows": post_round_rows,
                "keep_count": int(keep),
                "kept_actions": kept_actions,
                "dropped_actions": dropped_actions,
            }
        )
        cand = kept_actions
    
    timing_data['mcts_execution_time'] = time.perf_counter() - mcts_execution_start
    timing_data['round_allocation_time'] = time.perf_counter() - round_start
    
    # Final assertion: ensure we didn't exceed total_sims
    assert sims_used <= total_sims, f"Used {sims_used} sims, but only {total_sims} were allocated"
    
    # Final ranking timing
    ranking_start = time.perf_counter()

    # Final pick - deterministic ranking without Gumbel noise.
    final_rank_rows = build_gumbel_score_rows(
        actions=cand,
        pi=pi,
        logits=logits,
        gumbel_noise=g,
        c_scale=c_scale,
        v_pi_01=v_pi,
        q_of_child=q_of_child,
        n_of_child=n_of_child,
        include_gumbel_term=False,
    )
    if not final_rank_rows:
        raise RuntimeError("Gumbel final ranking produced no candidates")

    cand = [int(row["tensor_action"]) for row in final_rank_rows]
    selected_action = int(cand[0])
    emit_trace(
        {
            "type": "gumbel_final_selection",
            "selected_action": selected_action,
            "final_rank_rows": final_rank_rows,
        }
    )
    
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
        "timing_breakdown": timing_data,
        "forced_stats_totals": forced_stats_totals,
        "v_pi_01": float(v_pi),
        "round_uses_gumbel": bool(round_uses_gumbel),
        "selected_action": selected_action,
        "final_rank_rows": final_rank_rows,
        "last_round_rows": last_round_rows,
        "top_m_selected_rows": top_m_selected_rows,
        "top_m_excluded_rows": top_m_excluded_rows,
    }
    
    return selected_action, performance_metrics


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
