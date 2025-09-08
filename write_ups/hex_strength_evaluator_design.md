
# Hex Strength Evaluator — Design Specification (Implementation Scaffold)

GOAL
-----
Build a reproducible “strength evaluator” that consumes a complete Hex game and produces, for each player and for each game phase (opening, middlegame, endgame), two aggregate scores:
(1) a POLICY-BASED score and (2) a VALUE-BASED score. Optionally, also produce coarse error buckets per move (-2 / -1 / 0) and summary statistics. The design must plug into the existing AlphaZero-style engine (policy/value nets + MCTS) and be robust to reference-frame subtleties and noisy evaluations.

HIGH-LEVEL IDEA
---------------
For every ply (move) in the game:
• Reconstruct the position BEFORE the move.
• Run an EVALUATION PROCEDURE that yields:
  - a policy distribution over legal moves (from policy net, or MCTS root visit distribution/priors),
  - a value estimate of the position under a chosen policy (value net or MCTS-derived),
  - a policy/value view for the move actually played,
  - a policy/value view for the evaluator’s preferred move.
• Convert raw outputs into DELTA METRICS and a COARSE BUCKET (-2, -1, 0).
• Assign the ply to a GAME PHASE (opening/middlegame/endgame).
• Aggregate per player × phase × metric.

KEY OUTPUTS
-----------
For each game, produce:
• Per-player × phase (opening, middle, end):
  - policy_score_mean (or median/trimmed mean),
  - value_score_mean,
  - policy_bucket_rates: fraction of moves in {-2, -1, 0},
  - value_bucket_rates: fraction of moves in {-2, -1, 0},
  - counts (number of plies by that player in phase).
• Global summary (optionally): weighted combination across phases.
• Per-move table (optional artifact for debugging): evaluator details for every ply.

DEFINITIONS & REFERENCE FRAMES
------------------------------
• Value head in your code: red_ref_signed ∈ [-1, 1] where +1 = Red win, -1 = Blue win (see mcts.py comments). Internally MCTS stores values in player-to-move (PTM) frame.
• To compare choices for the actor who moves at ply t:
  1) Convert all value outputs to the ACTOR’S reference frame (actor_ref_signed). If the actor is Red, actor_ref_signed = red_ref_signed; if Blue, actor_ref_signed = -red_ref_signed.
  2) When reporting probabilities (policy), use legal-move probabilities normalized over legal actions only.
• Win-probability form: p = (signed + 1) / 2. Deltas can be measured either in signed space or probability space; probability space is more interpretable.

CONFIGURATION (make everything pluggable)
----------------------------------------
struct EvaluatorConfig:
  opening_plies: int = 12                           # first 12 plies (6 each) define “opening”
  endgame_value_thresh: float = 0.90                # |actor_ref_signed| >= 0.90 defines “decided”
  endgame_streak: int = 3                           # need ≥3 consecutive decided positions to enter endgame
  use_value_prob_space: bool = True                 # compute deltas in probability space (recommended)
  policy_source: enum { POLICY_NET, MCTS_PRIORS }   # how to get the policy vector
  value_source: enum { VALUE_NET, MCTS_Q, MCTS_VALUE_AT_ROOT } # how to get value
  mcts_sims: int = 200
  mcts_c_puct: float = 1.5
  mcts_batch_cap: Optional[int] = None
  enable_gumbel_root: bool = False
  gumbel_params: {...}                              # c_visit, c_scale, m, threshold
  aggregation: enum { MEAN, MEDIAN, TRIMMED_MEAN } = MEAN
  trimmed_fraction: float = 0.1                     # for TRIMMED_MEAN
  bucket_policy_thresholds: (float small, float big) = (0.10, 0.30)  # Δpolicy prob cutpoints
  bucket_value_thresholds: (float small, float big) = (0.10, 0.30)   # Δwin-prob cutpoints
  phase_weighting: dict = {"opening":1, "middle":1, "end":1}         # for global summaries
  rng_seed: Optional[int] = None                    # enforce determinism
  batch_nn: bool = True                             # batch evaluations across plies for speed
  ignore_early_noise_until: int = 2*board_size      # optional: downweight or ignore very early plies
  downweight_function: Optional[callable(move_idx)->float] = None    # optional per-ply weight

INPUT/OUTPUT INTERFACES
-----------------------
Input:
• GameRecord:
  - board_size: int
  - moves: List[(row:int, col:int, player:enum{RED, BLUE})] in chronological order.
  - starting_player: enum
  - metadata (optional): game_id, players, date, etc.

Output (machine-readable):
• EvaluatorReport:
  - per_phase_per_player: map[(phase, player)] -> { policy_score, value_score, policy_bucket_counts, value_bucket_counts, n }
  - per_move_details (optional): List of MoveEval entries
  - combined_summary: { per_player: {...}, overall: {...} }

MoveEval entry (optional but useful for debugging):
  - ply_idx, actor, phase
  - chosen_move: (r,c)
  - policy_prob_chosen, policy_prob_best, delta_policy := policy_prob_best - policy_prob_chosen
  - value_prob_after_chosen, value_prob_after_best, delta_value := value_prob_after_best - value_prob_after_chosen
  - bucket_policy ∈ {-2,-1,0}, bucket_value ∈ {-2,-1,0}
  - evaluator_metadata: { mcts_sims, c_puct, timing, cache_hits, etc. }

PIPELINE OVERVIEW
-----------------
1) Reconstruct Positions
   - Start from an empty board with known starting_player.
   - For each ply t in [0..T-1], snapshot the PREMOVE state S_t (player to move = actor_t).
   - You’ll also need the POSTMOVE state S_t+1 to evaluate “value after chosen move”.

2) Phase Assignment Pass (two-pass or single pass with a small buffer)
   - Opening: plies [0 .. opening_plies-1].
   - Endgame: find the earliest ply k where there is a streak of endgame_streak plies with |value_actor_ref(S_i)| >= endgame_value_thresh. Once triggered, all subsequent plies are “endgame.”
     • IMPORTANT: The “value” here should be the evaluator’s value of the position BEFORE the move (S_i), in the actor’s frame. Use value_source for consistency. Because this definition depends on values, you may want to compute these quick evaluations in a first pass (e.g., value net only) to determine the boundary, then reuse or recompute with the full chosen settings.
   - Middle: everything else between opening and endgame.
   - Store phase(S_t).

3) Per-Ply Evaluation Pass
   For each S_t with actor A and chosen move m_chosen:
   3.1) POLICY VIEW
        - Obtain a probability vector over LEGAL actions:
          a) If policy_source == POLICY_NET:
             • Run the policy head once on S_t. Mask illegal actions; renormalize.
          b) If policy_source == MCTS_PRIORS:
             • Expand MCTS root (no rollouts needed beyond expansion) to get priors at root
               OR use root visit counts after a configured number of simulations.
             • Use the prior distribution (or normalized root visits) restricted to legal moves.
        - Identify the evaluator’s preferred move m_best_policy = argmax policy_prob.
        - Extract p_chosen = policy_prob(m_chosen), p_best = policy_prob(m_best_policy).
        - Define Δpolicy := p_best − p_chosen (≥ 0). This is the policy-based error size.
        - Bucket Δpolicy into {-2,-1,0} using bucket_policy_thresholds:
             if Δpolicy ≥ big -> -2 (big loss)
             elif Δpolicy ≥ small -> -1 (small loss)
             else -> 0 (too close)

   3.2) VALUE VIEW
        Choose a consistent value_source (must reflect “value after move,” not just at S_t):
        - Option A: VALUE_NET
          • Evaluate the value head on POSTMOVE states:
              v_chosen_signed_red = V(S_t after m_chosen) in red_ref_signed
              v_best_signed_red   = V(S_t after m_best_value) where m_best_value = argmax V(S_t after a) in actor_ref
          • Convert to actor’s frame: v*_actor = v*_signed_red if actor=Red else -v*_signed_red
          • Convert to probability if use_value_prob_space: p*_val = (v*_actor + 1)/2
          • Define Δvalue := p_val_best − p_val_chosen (≥ 0). (If working in signed space, use v_actor values directly.)
        - Option B: MCTS_Q (recommended if you already run MCTS)
          • Run MCTS from S_t for mcts_sims.
          • For each legal child a, read child Q in PTM frame (already actor frame at root), convert to probability if desired.
          • Let m_best_value = argmax Q(a). Define p_val_best = Q(m_best_value), p_val_chosen = Q(m_chosen).
          • Δvalue := p_val_best − p_val_chosen.
        - Option C: MCTS_VALUE_AT_ROOT (value estimate of S_t or of child after chosen/best through one-step rollout) — keep consistent across chosen/best.

        Bucket Δvalue into {-2,-1,0} using bucket_value_thresholds similarly.

   3.3) RECORD
        - Store Δpolicy, bucket_policy, Δvalue, bucket_value, actor, phase, timings.

4) Aggregation
   - For each (player, phase):
       • Compute aggregate policy_score via aggregation (mean/median/trimmed mean) over Δpolicy of that player’s plies in phase.
       • Compute value_score similarly over Δvalue.
       • Compute bucket histograms (counts or rates) for policy and value buckets.
       • Keep n (count of plies) for context.
   - Optionally compute a per-player global summary by combining phase scores with phase_weighting. (E.g., weighted average of the per-phase means.)

ROBUSTNESS & CALIBRATION
------------------------
• Use cached NN evaluations (LRU) keyed by state hash (board_key) to avoid recomputation; your mcts.py already exposes/effects a cache. For net-only evaluation, build a simple OrderedDict cache with capacity limit.
• Consider MEDIAN or TRIMMED mean to reduce sensitivity to a few extreme plies.
• You may optionally cap Δs at, say, 0.80 to avoid outliers dominating averages.
• Early-move noise: optionally downweight or ignore plies before ignore_early_noise_until.
• Determinism: set RNG seeds for numpy/torch and MCTS Dirichlet noise off for evaluation runs (mcts_config.add_root_noise = False). Prefer fixed temperature at root for evaluation.
• Make policy_source and value_source consistent across the entire run of a game for comparability.

PERFORMANCE
-----------
• Batch policy/value net calls: collect to-be-evaluated states across plies, run in batches with torch.stack, then scatter results back. For MCTS-based sources, stick with your existing batched leaf evaluation.
• If value_source == MCTS_Q and policy_source == MCTS_PRIORS, a single MCTS run per ply can feed both policy (root priors / visits) and value (child Qs). Reuse that run’s outputs. For speed, allow a low-sims mode (e.g., 64–128 sims) for evaluator purposes, configurable.

MAPPING TO EXISTING CODE
------------------------
• mcts.py (BaselineMCTS):
  - Use _expand_root_node or run() to obtain priors/visits and child Qs. Since run() selects a move, you still can read root_node after search to access N, Q, P arrays per child (legal move order). Also MCTSResult contains tree_data and win_probability; prefer direct arrays from root_node for evaluator.
  - Ensure reference-frame correctness: Q at root is in player-to-move frame already (good). Value head is in red_ref; convert to actor frame when you use VALUE_NET.
• move_selection.py:
  - Configuration examples for tournament play; shows how to build BaselineMCTSConfig via create_mcts_config("tournament", ...). Mirror those switches (no root noise, confidence termination, temp settings) for consistent evaluator runs.

DATA STRUCTURES
---------------
class StrengthEvaluator:
  def __init__(self, engine, model_wrapper, cfg: EvaluatorConfig)
  def evaluate_game(self, game: GameRecord) -> EvaluatorReport

Internal helpers:
  _reconstruct_state_prefix(board_size, starting_player, moves[:t]) -> HexGameState
  _phase_scan(values_at_S_t) -> List[phase]  # determines opening/middle/end boundaries
  _evaluate_policy(S_t) -> (policy_dict: {(r,c)->prob}, m_best_policy)
  _evaluate_value(S_t) -> (per_child_value_dict, m_best_value)
  _delta_policy(policy_dict, m_chosen) -> float
  _delta_value(value_dict, m_chosen) -> float
  _bucket(delta, (small,big)) -> int in {-2,-1,0}
  _aggregate(per_move: List[MoveEval]) -> per_phase_per_player + summaries

PSEUDOCODE (value_source=MCTS_Q, policy_source=MCTS_PRIORS)
----------------------------------------------------------
evaluate_game(game):
  states = []
  for t in 0..T-1:
    S_t = reconstruct_state(game, t)   # premove position
    states.append(S_t)

  # Phase pass (quick value net or cheap MCTS):
  values_actor = []
  for S_t in states:
    v_red = value_net(S_t)                          # red_ref_signed
    v_actor = v_red if actor(S_t)==RED else -v_red  # actor_ref_signed
    p_actor = (v_actor+1)/2
    values_actor.append(p_actor)
  phases = assign_phases(values_actor, opening_plies, endgame_value_thresh, endgame_streak)

  # Eval pass:
  results = []
  for t, S_t in enumerate(states):
    actor = player_to_move(S_t)
    m_chosen = game.moves[t].move

    if value_source == MCTS_Q or policy_source == MCTS_PRIORS:
      root = run_mcts_for_analysis(S_t, sims=mcts_sims, ...)
      policy = priors_or_root_visits(root)  # dict (r,c)->prob
      values = child_Q_probs(root)          # dict (r,c)->prob in actor frame
    else:
      policy = policy_net_distribution(S_t)
      values = value_net_over_children(S_t) # evaluate S_t after each legal child

    m_best_policy = argmax(policy)
    m_best_value  = argmax(values)

    d_policy = max(0, policy[m_best_policy] - policy[m_chosen])
    d_value  = max(0, values[m_best_value]  - values[m_chosen])

    b_policy = bucket(d_policy, bucket_policy_thresholds)
    b_value  = bucket(d_value,  bucket_value_thresholds)

    results.append(MoveEval(ply=t, actor=actor, phase=phases[t], chosen=m_chosen,
                            delta_policy=d_policy, delta_value=d_value,
                            bucket_policy=b_policy, bucket_value=b_value))

  report = aggregate(results, aggregation, trimmed_fraction, phase_weighting, downweight_function)
  return report

BUCKETING DETAILS
-----------------
• For policy deltas, Δpolicy is already a probability difference in [0,1]. Choose small=0.10, big=0.30 as initial defaults; tune by looking at distributions in your data.
• For value deltas, if using probability space: small=0.10, big=0.30 means a 10% or 30% swing in win probability between chosen vs best line is treated as small or big loss.
• If you stay in signed space, thresholds are the same numerically but centered at 0 (because Δ is nonnegative it’s the same).

EDGE CASES
----------
• Terminal states: if S_t is terminal (shouldn’t happen if you stop at last legal move), mark phase/buckets as N/A.
• Illegal moves in record: fail fast with a clear error.
• Missing model outputs (NaN/inf): skip ply and mark invalid; record counts.
• If evaluator cannot compute values for some plies (e.g., timeouts), aggregate over the plies that succeeded and report coverage.
• If opening_plies exceeds game length, opening ends at the game’s last ply; middle/end may be empty sets.

TESTING STRATEGY
----------------
1) Unit tests:
   - Reference-frame conversions: red_ref → actor frame correctness.
   - Phase boundary logic with synthetic sequences that cross thresholds.
   - Bucket function with crafted deltas.
2) Integration tests:
   - Evaluate short toy games where the best move is forced; check Δs are near 0 for optimal plays and positive for intentionally bad plays.
   - Determinism checks with fixed seeds and no Dirichlet noise.
3) Performance tests:
   - Ensure batching reduces wall clock time for 100–200 plies.
4) Sanity dashboards (optional):
   - Histograms of Δpolicy/Δvalue by phase and by player.
   - Scatter of Δpolicy vs Δvalue to ensure they’re correlated but not identical.

IMPLEMENTATION NOTES
--------------------
• Reuse your existing board_key/state hashing to index caches. Keep separate caches for:
  - policy/value net on S_t (premove),
  - value net on S_t after child a,
  - MCTS root expansions keyed by S_t with cfg tuple.
• For MCTS-based evaluation:
  - Use BaselineMCTS with add_root_noise=False, confidence termination disabled (or conservative), fixed temperature.
  - After search, read root.N (visits), root.Q (actor-frame), root.P (priors). Normalize N to probabilities: N / sum(N). Pick which one (P or N) to call “policy.”
• API ergonomics: expose a simple function evaluate_game(game, cfg) returning EvaluatorReport plus (optional) per-move CSV/JSON for offline analysis.

EXAMPLE OUTPUT SHAPE (JSON-ish)
--------------------------------
{
  "per_phase_per_player": {
    ("opening", "RED"): { "policy_score": 0.08, "value_score": 0.05,
                          "policy_bucket_rates": {"-2":0.10,"-1":0.20,"0":0.70},
                          "value_bucket_rates":  {"-2":0.06,"-1":0.18,"0":0.76},
                          "n": 14 },
    ("opening", "BLUE"): { ... },
    ("middle",  "RED"):  { ... },
    ("end",     "RED"):  { ... }
  },
  "combined_summary": {
    "RED":  { "policy_score": 0.07, "value_score": 0.06 },
    "BLUE": { "policy_score": 0.09, "value_score": 0.08 }
  },
  "coverage": { "evaluated_plies": 90, "total_plies": 92 }
}

TUNING & EXTENSIONS
-------------------
• Swap VALUE_NET vs MCTS_Q to study robustness; you may prefer MCTS_Q for noisier nets.
• Try Δlogit of win-prob instead of Δprobability for more “Elo-like” sensitivity: logit(p)=log(p/(1-p)).
• Add per-importance weighting: weight Δs by (1 - |value_actor(S_t)|) to emphasize uncertain positions.
• Output principal variation snippets for big (-2) mistakes to aid qualitative review.
• Learn bucket thresholds data-driven via mixture modeling on Δ distributions.

DELIVERABLES FOR CODING AGENT
-----------------------------
1) A new module hex_ai/eval/strength_evaluator.py implementing StrengthEvaluator and EvaluatorConfig.
2) CLI script tools/evaluate_game_strength.py that loads a game JSON/SGF-equivalent and prints JSON report + writes optional CSV of per-move details.
3) Unit tests under tests/eval/test_strength_evaluator.py covering core logic.
4) Optional: a small notebook or script that plots histograms of Δpolicy/Δvalue by phase.
