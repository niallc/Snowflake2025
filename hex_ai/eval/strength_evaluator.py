"""
Hex Strength Evaluator - Implementation

This module implements a comprehensive strength evaluator that analyzes Hex games
to produce policy-based and value-based scores for each player across different
game phases (opening, middlegame, endgame).

The evaluator integrates with the existing AlphaZero-style engine (policy/value nets + MCTS)
and handles reference frame conversions, phase detection, and robust evaluation metrics.
"""

import time
import random
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from enum import Enum
from collections import OrderedDict
import numpy as np
import torch

from hex_ai.enums import Player, Winner, Piece
from hex_ai.inference.game_engine import HexGameState, HexGameEngine, make_empty_hex_state
from hex_ai.inference.mcts import BaselineMCTS, BaselineMCTSConfig
from hex_ai.inference.model_wrapper import ModelWrapper
from hex_ai.inference.simple_model_inference import SimpleModelInference
from hex_ai.value_utils import ValuePredictor, red_ref_signed_to_ptm_ref_signed, policy_logits_to_probs
from hex_ai.utils.format_conversion import trmph_to_moves, rowcol_to_trmph
from hex_ai.utils.state_utils import board_key
from hex_ai.config import BOARD_SIZE

logger = logging.getLogger(__name__)


class AggregationMethod(Enum):
    """Method for aggregating scores across moves."""
    MEAN = "mean"
    MEDIAN = "median"
    TRIMMED_MEAN = "trimmed_mean"


class GamePhase(Enum):
    """Game phases for analysis."""
    OPENING = "opening"
    MIDDLE = "middle"
    END = "end"


@dataclass
class EvaluatorConfig:
    """Configuration for the strength evaluator."""
    # Phase detection parameters
    opening_plies: int = 12
    endgame_value_thresh: float = 0.90
    endgame_streak: int = 3
    
    # Evaluation parameters
    use_mcts: bool = True  # True = MCTS-based evaluation, False = neural network only
    
    # MCTS parameters
    mcts_sims: int = 200
    mcts_c_puct: float = 1.5
    mcts_batch_cap: Optional[int] = None
    enable_gumbel_root: bool = False
    
    # Aggregation parameters
    aggregation: AggregationMethod = AggregationMethod.MEAN
    trimmed_fraction: float = 0.1
    
    # Bucketing thresholds (small, big)
    bucket_policy_thresholds: Tuple[float, float] = (0.10, 0.30)
    bucket_value_thresholds: Tuple[float, float] = (0.10, 0.30)
    
    # Phase weighting
    phase_weighting: Dict[GamePhase, float] = field(default_factory=lambda: {
        GamePhase.OPENING: 1.0,
        GamePhase.MIDDLE: 1.0,
        GamePhase.END: 1.0
    })
    
    # Performance and robustness
    rng_seed: Optional[int] = None
    batch_nn: bool = True
    ignore_early_noise_until: int = 2 * BOARD_SIZE
    downweight_function: Optional[Callable[[int], float]] = None
    
    # Caching
    cache_size: int = 10000


@dataclass
class GameRecord:
    """Represents a complete Hex game for evaluation."""
    board_size: int
    moves: List[Tuple[int, int, Player]]  # (row, col, player) in chronological order
    starting_player: Player
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class MoveEval:
    """Evaluation results for a single move."""
    ply_idx: int
    actor: Player
    phase: GamePhase
    chosen_move: Tuple[int, int]
    
    # Policy evaluation
    policy_prob_chosen: float
    policy_prob_best: float
    delta_policy: float
    bucket_policy: int  # -2, -1, or 0
    
    # Value evaluation
    value_prob_after_chosen: float
    value_prob_after_best: float
    delta_value: float
    bucket_value: int  # -2, -1, or 0
    
    # Metadata
    evaluator_metadata: Dict[str, Any]


@dataclass
class PhaseResults:
    """Results for a specific player and phase."""
    policy_score: float
    value_score: float
    policy_bucket_counts: Dict[int, int]  # {-2: count, -1: count, 0: count}
    value_bucket_counts: Dict[int, int]
    n: int  # number of moves in this phase


@dataclass
class EvaluatorReport:
    """Complete evaluation report for a game."""
    per_phase_per_player: Dict[Tuple[GamePhase, Player], PhaseResults]
    per_move_details: Optional[List[MoveEval]] = None
    combined_summary: Optional[Dict[str, Any]] = None
    coverage: Optional[Dict[str, int]] = None


class StrengthEvaluator:
    """
    Main strength evaluator class.
    
    Evaluates Hex games to produce policy-based and value-based scores
    for each player across different game phases.
    """
    
    def __init__(self, engine: HexGameEngine, model_wrapper: ModelWrapper, cfg: EvaluatorConfig):
        """
        Initialize the strength evaluator.
        
        Args:
            engine: Hex game engine for state management
            model_wrapper: Model wrapper for neural network inference
            cfg: Configuration for the evaluator
        """
        self.engine = engine
        self.model_wrapper = model_wrapper
        self.cfg = cfg
        
        # Set random seed for reproducibility
        if cfg.rng_seed is not None:
            random.seed(cfg.rng_seed)
            np.random.seed(cfg.rng_seed)
        
        # Initialize caches
        self.policy_cache: OrderedDict[int, Dict[Tuple[int, int], float]] = OrderedDict()
        self.value_cache: OrderedDict[int, Dict[Tuple[int, int], float]] = OrderedDict()
        self.mcts_cache: OrderedDict[int, Any] = OrderedDict()
        
        # Performance tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self.evaluation_time = 0.0
        
        # Create MCTS configuration for analysis
        self.mcts_config = self._create_mcts_config()
        
        logger.info(f"StrengthEvaluator initialized with config: {cfg}")
    
    def _create_mcts_config(self) -> BaselineMCTSConfig:
        """Create MCTS configuration for analysis."""
        return BaselineMCTSConfig(
            sims=self.cfg.mcts_sims,
            c_puct=self.cfg.mcts_c_puct,
            batch_cap=self.cfg.mcts_batch_cap if self.cfg.mcts_batch_cap is not None else 100,
            add_root_noise=False,  # Deterministic for analysis
            temperature_start=0.01,  # Very low temperature for deterministic play
            temperature_end=0.01,
            enable_gumbel_root_selection=self.cfg.enable_gumbel_root,
            confidence_termination_threshold=0.95,  # Conservative for quality
            enable_depth_discounting=False  # Disable for analysis
        )
    
    def evaluate_game(self, game: GameRecord) -> EvaluatorReport:
        """
        Evaluate a complete game and produce a strength report.
        
        Args:
            game: Game record to evaluate
            
        Returns:
            Complete evaluation report
        """
        start_time = time.time()
        
        try:
            # Step 1: Reconstruct positions
            logger.debug("Reconstructing positions...")
            states = self._reconstruct_positions(game)
            logger.debug(f"Reconstructed {len(states)} states")
            
            # Step 2: Phase assignment
            logger.debug("Assigning phases...")
            phases = self._assign_phases(states)
            logger.debug(f"Assigned {len(phases)} phases")
            
            # Step 3: Per-ply evaluation
            logger.debug("Evaluating moves...")
            move_evals = self._evaluate_moves(states, phases, game.moves)
            logger.debug(f"Evaluated {len(move_evals)} moves")
            
            # Step 4: Aggregation
            logger.debug("Aggregating results...")
            report = self._aggregate_results(move_evals)
            
            # Add coverage information
            report.coverage = {
                "evaluated_plies": len(move_evals),
                "total_plies": len(game.moves)
            }
            
            self.evaluation_time = time.time() - start_time
            logger.info(f"Game evaluation completed in {self.evaluation_time:.2f}s")
            
            return report
            
        except Exception as e:
            logger.error(f"Error evaluating game: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def _reconstruct_positions(self, game: GameRecord) -> List[HexGameState]:
        """Reconstruct game positions from move sequence."""
        states = []
        
        # Start with empty board
        state = make_empty_hex_state()
        states.append(state)
        
        # Apply moves one by one
        for i, (row, col, player) in enumerate(game.moves):
            if not state.is_valid_move(row, col):
                raise ValueError(f"Invalid move at ply {i}: ({row}, {col})")
            
            state = state.make_move(row, col)
            states.append(state)
        
        return states
    
    def _assign_phases(self, states: List[HexGameState]) -> List[GamePhase]:
        """Assign game phases to each position."""
        phases = []
        
        # Quick value estimates for phase detection
        values = []
        for state in states[:-1]:  # Exclude final state
            if state.game_over:
                # Terminal state - use actual winner
                if state.winner == Winner.RED:
                    values.append(1.0)
                elif state.winner == Winner.BLUE:
                    values.append(-1.0)
                else:
                    values.append(0.0)
            else:
                # Use value net for quick estimate
                try:
                    _, value_signed = self.model_wrapper.predict(
                        state.get_board_tensor()
                    )
                    # Convert to actor reference frame
                    actor = state.current_player_enum
                    value_actor = red_ref_signed_to_ptm_ref_signed(value_signed.item(), actor)
                    values.append(value_actor)
                except Exception as e:
                    logger.warning(f"Failed to get value estimate: {e}")
                    values.append(0.0)
        
        # Assign phases
        for i, value in enumerate(values):
            if i < self.cfg.opening_plies:
                phases.append(GamePhase.OPENING)
            else:
                # Check for endgame streak
                abs_value = abs(value) if value is not None else 0.0
                if abs_value >= self.cfg.endgame_value_thresh:
                    # Check if we have a streak
                    streak_count = 1
                    for j in range(i - 1, max(0, i - self.cfg.endgame_streak), -1):
                        if j < len(values) and values[j] is not None and abs(values[j]) >= self.cfg.endgame_value_thresh:
                            streak_count += 1
                        else:
                            break
                    
                    if streak_count >= self.cfg.endgame_streak:
                        phases.append(GamePhase.END)
                    else:
                        phases.append(GamePhase.MIDDLE)
                else:
                    phases.append(GamePhase.MIDDLE)
        
        return phases
    
    def _evaluate_moves(self, states: List[HexGameState], phases: List[GamePhase], 
                       moves: List[Tuple[int, int, Player]]) -> List[MoveEval]:
        """Evaluate each move in the game."""
        move_evals = []
        
        for i, (state, phase, (row, col, player)) in enumerate(zip(states[:-1], phases, moves)):
            try:
                # Skip early noise if configured
                if hasattr(self.cfg, 'ignore_early_noise_until') and i < self.cfg.ignore_early_noise_until:
                    continue
                
                # Evaluate policy and value
                policy_dict, m_best_policy = self._evaluate_policy(state)
                value_dict, m_best_value = self._evaluate_value(state)
                
                # Calculate deltas
                delta_policy = self._calculate_delta_policy(policy_dict, (row, col))
                delta_value = self._calculate_delta_value(value_dict, (row, col))
                
                # Calculate buckets
                bucket_policy = self._calculate_bucket(delta_policy, self.cfg.bucket_policy_thresholds)
                bucket_value = self._calculate_bucket(delta_value, self.cfg.bucket_value_thresholds)
                
                # Create move evaluation
                move_eval = MoveEval(
                    ply_idx=i,
                    actor=player,
                    phase=phase,
                    chosen_move=(row, col),
                    policy_prob_chosen=policy_dict.get((row, col), 0.0),
                    policy_prob_best=policy_dict.get(m_best_policy, 0.0),
                    delta_policy=delta_policy,
                    bucket_policy=bucket_policy,
                    value_prob_after_chosen=value_dict.get((row, col), 0.0),
                    value_prob_after_best=value_dict.get(m_best_value, 0.0),
                    delta_value=delta_value,
                    bucket_value=bucket_value,
                    evaluator_metadata={
                        "mcts_sims": self.cfg.mcts_sims,
                        "c_puct": self.cfg.mcts_c_puct,
                        "use_mcts": self.cfg.use_mcts
                    }
                )
                
                move_evals.append(move_eval)
                
            except Exception as e:
                logger.warning(f"Failed to evaluate move {i}: {e}")
                continue
        
        return move_evals
    
    def _evaluate_policy(self, state: HexGameState) -> Tuple[Dict[Tuple[int, int], float], Tuple[int, int]]:
        """Evaluate policy for a position."""
        state_hash = board_key(state)
        
        # Check cache
        if state_hash in self.policy_cache:
            self.cache_hits += 1
            policy_dict = self.policy_cache[state_hash]
            m_best = max(policy_dict.keys(), key=lambda k: policy_dict[k])
            return policy_dict, m_best
        
        self.cache_misses += 1
        
        if self.cfg.use_mcts:
            policy_dict, _ = self._run_mcts_shared(state)
        else:
            policy_dict = self._evaluate_policy_neural_net(state)
        
        # Cache result
        if len(self.policy_cache) >= self.cfg.cache_size:
            self.policy_cache.popitem(last=False)
        self.policy_cache[state_hash] = policy_dict
        
        m_best = max(policy_dict.keys(), key=lambda k: policy_dict[k])
        return policy_dict, m_best
    
    def _evaluate_policy_neural_net(self, state: HexGameState) -> Dict[Tuple[int, int], float]:
        """Evaluate policy using neural network only."""
        policy_logits, _ = self.model_wrapper.predict(state.get_board_tensor())
        policy_logits_np = policy_logits.numpy()
        
        legal_moves = state.get_legal_moves()
        
        # Use utility function to properly mask illegal moves and get legal probabilities
        from hex_ai.utils.math_utils import policy_logits_to_legal_probs
        legal_probs = policy_logits_to_legal_probs(policy_logits_np, legal_moves, BOARD_SIZE)
        
        # Create dictionary mapping moves to probabilities
        policy_dict = {}
        for (row, col), prob in zip(legal_moves, legal_probs):
            policy_dict[(row, col)] = float(prob)
        
        return policy_dict
    
    def _run_mcts_shared(self, state: HexGameState) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]]:
        """
        Run MCTS once and return both policy (visit counts) and value (Q-values) data.
        
        This avoids duplicate MCTS runs when both policy_source=MCTS_PRIORS and value_source=MCTS_Q.
        
        Returns:
            Tuple of (policy_dict, value_dict) where both map moves to their respective scores
        """
        state_hash = board_key(state)
        
        # Check MCTS cache first
        if state_hash in self.mcts_cache:
            self.cache_hits += 1
            return self.mcts_cache[state_hash]
        
        self.cache_misses += 1
        
        # Run MCTS once
        mcts = BaselineMCTS(self.engine, self.model_wrapper, self.mcts_config)
        result = mcts.run(state)
        root = result.root_node
        
        # Extract policy data (visit counts normalized to probabilities)
        policy_dict = {}
        total_visits = sum(root.N)
        if total_visits > 0:
            for i, (row, col) in enumerate(root.legal_moves):
                policy_dict[(row, col)] = float(root.N[i]) / total_visits
        else:
            # Fallback to uniform distribution
            uniform_prob = 1.0 / len(root.legal_moves)
            for row, col in root.legal_moves:
                policy_dict[(row, col)] = uniform_prob
        
        # Extract value data (Q-values in [-1, 1] signed space)
        value_dict = {}
        for i, (row, col) in enumerate(root.legal_moves):
            q_value = root.Q[i]  # Already in actor reference frame from MCTS
            
            # Always use [-1, 1] signed space for consistency
            value_dict[(row, col)] = q_value
        
        # Cache the result
        if len(self.mcts_cache) >= self.cfg.cache_size:
            self.mcts_cache.popitem(last=False)
        self.mcts_cache[state_hash] = (policy_dict, value_dict)
        
        return policy_dict, value_dict
    
# Removed _evaluate_policy_mcts - now handled by _run_mcts_shared
    
    def _evaluate_value(self, state: HexGameState) -> Tuple[Dict[Tuple[int, int], float], Tuple[int, int]]:
        """Evaluate value for a position."""
        state_hash = board_key(state)
        
        # Check cache
        if state_hash in self.value_cache:
            self.cache_hits += 1
            value_dict = self.value_cache[state_hash]
            m_best = max(value_dict.keys(), key=lambda k: value_dict[k])
            return value_dict, m_best
        
        self.cache_misses += 1
        
        if self.cfg.use_mcts:
            _, value_dict = self._run_mcts_shared(state)
        else:
            value_dict = self._evaluate_value_neural_net(state)
        
        # Cache result
        if len(self.value_cache) >= self.cfg.cache_size:
            self.value_cache.popitem(last=False)
        self.value_cache[state_hash] = value_dict
        
        m_best = max(value_dict.keys(), key=lambda k: value_dict[k])
        return value_dict, m_best
    
    def _evaluate_value_neural_net(self, state: HexGameState) -> Dict[Tuple[int, int], float]:
        """Evaluate value using neural network for all legal moves with batching."""
        legal_moves = state.get_legal_moves()
        
        # Prepare batch of board tensors for all legal moves
        board_tensors = []
        for row, col in legal_moves:
            new_state = state.make_move(row, col)
            board_tensors.append(new_state.get_board_tensor())
        
        # Use existing batched inference (much more efficient than individual calls)
        batch_tensor = torch.stack(board_tensors, dim=0)
        _, values_signed = self.model_wrapper.batch_predict(batch_tensor)
        
        # Convert from RED reference frame to actor reference frame and build result dict
        actor = state.current_player_enum
        value_dict = {}
        for (row, col), value_signed in zip(legal_moves, values_signed):
            value_actor = red_ref_signed_to_ptm_ref_signed(value_signed.item(), actor)
            value_dict[(row, col)] = value_actor
        
        return value_dict
    
# Removed _evaluate_value_mcts - now handled by _run_mcts_shared
    
    
    def _calculate_delta_policy(self, policy_dict: Dict[Tuple[int, int], float], 
                               chosen_move: Tuple[int, int]) -> float:
        """Calculate policy delta for chosen move."""
        if chosen_move not in policy_dict:
            return 0.0
        
        chosen_prob = policy_dict[chosen_move]
        best_prob = max(policy_dict.values())
        
        return max(0.0, best_prob - chosen_prob)
    
    def _calculate_delta_value(self, value_dict: Dict[Tuple[int, int], float], 
                              chosen_move: Tuple[int, int]) -> float:
        """
        Calculate value delta for chosen move.
        
        Note: This operates on [-1, 1] signed space values for consistency with MCTS.
        """
        if chosen_move not in value_dict:
            return 0.0
        
        chosen_value = value_dict[chosen_move]
        best_value = max(value_dict.values())
        
        return max(0.0, best_value - chosen_value)
    
    def _calculate_bucket(self, delta: float, thresholds: Tuple[float, float]) -> int:
        """Calculate bucket for delta value."""
        small_thresh, big_thresh = thresholds
        
        if delta >= big_thresh:
            return -2
        elif delta >= small_thresh:
            return -1
        else:
            return 0
    
    def _aggregate_results(self, move_evals: List[MoveEval]) -> EvaluatorReport:
        """Aggregate move evaluations into final report."""
        # Group by phase and player
        phase_player_data = {}
        
        for move_eval in move_evals:
            key = (move_eval.phase, move_eval.actor)
            if key not in phase_player_data:
                phase_player_data[key] = {
                    'policy_deltas': [],
                    'value_deltas': [],
                    'policy_buckets': {0: 0, -1: 0, -2: 0},
                    'value_buckets': {0: 0, -1: 0, -2: 0}
                }
            
            data = phase_player_data[key]
            data['policy_deltas'].append(move_eval.delta_policy)
            data['value_deltas'].append(move_eval.delta_value)
            data['policy_buckets'][move_eval.bucket_policy] += 1
            data['value_buckets'][move_eval.bucket_value] += 1
        
        # Calculate aggregated results
        per_phase_per_player = {}
        
        for (phase, player), data in phase_player_data.items():
            n = len(data['policy_deltas'])
            
            # Aggregate policy scores
            policy_score = self._aggregate_values(data['policy_deltas'])
            
            # Aggregate value scores
            value_score = self._aggregate_values(data['value_deltas'])
            
            # Convert bucket counts to rates
            policy_bucket_rates = {
                bucket: count / n for bucket, count in data['policy_buckets'].items()
            }
            value_bucket_rates = {
                bucket: count / n for bucket, count in data['value_buckets'].items()
            }
            
            per_phase_per_player[(phase, player)] = PhaseResults(
                policy_score=policy_score,
                value_score=value_score,
                policy_bucket_counts=data['policy_buckets'],
                value_bucket_counts=data['value_buckets'],
                n=n
            )
        
        # Create combined summary
        combined_summary = self._create_combined_summary(per_phase_per_player)
        
        return EvaluatorReport(
            per_phase_per_player=per_phase_per_player,
            per_move_details=move_evals,  # Always include per-move details for now
            combined_summary=combined_summary
        )
    
    def _aggregate_values(self, values: List[float]) -> float:
        """Aggregate values using configured method."""
        if not values:
            return 0.0
        
        if self.cfg.aggregation == AggregationMethod.MEAN:
            return np.mean(values)
        elif self.cfg.aggregation == AggregationMethod.MEDIAN:
            return np.median(values)
        elif self.cfg.aggregation == AggregationMethod.TRIMMED_MEAN:
            return self._trimmed_mean(values)
        else:
            raise ValueError(f"Unknown aggregation method: {self.cfg.aggregation}")
    
    def _trimmed_mean(self, values: List[float]) -> float:
        """Calculate trimmed mean."""
        if len(values) <= 2:
            return np.mean(values)
        
        sorted_values = sorted(values)
        trim_count = int(len(values) * self.cfg.trimmed_fraction)
        
        if trim_count > 0:
            trimmed_values = sorted_values[trim_count:-trim_count]
        else:
            trimmed_values = sorted_values
        
        return np.mean(trimmed_values) if trimmed_values else 0.0
    
    def _create_combined_summary(self, per_phase_per_player: Dict[Tuple[GamePhase, Player], PhaseResults]) -> Dict[str, Any]:
        """Create combined summary across phases."""
        player_scores = {}
        
        for player in [Player.BLUE, Player.RED]:
            policy_scores = []
            value_scores = []
            total_weight = 0.0
            
            for phase in [GamePhase.OPENING, GamePhase.MIDDLE, GamePhase.END]:
                key = (phase, player)
                if key in per_phase_per_player:
                    results = per_phase_per_player[key]
                    weight = self.cfg.phase_weighting.get(phase, 1.0) * results.n
                    
                    policy_scores.append(results.policy_score * weight)
                    value_scores.append(results.value_score * weight)
                    total_weight += weight
            
            if total_weight > 0:
                player_scores[player.value] = {
                    "policy_score": sum(policy_scores) / total_weight,
                    "value_score": sum(value_scores) / total_weight
                }
            else:
                player_scores[player.value] = {
                    "policy_score": 0.0,
                    "value_score": 0.0
                }
        
        return {
            "per_player": player_scores,
            "overall": {
                "avg_policy_score": np.mean([s["policy_score"] for s in player_scores.values()]),
                "avg_value_score": np.mean([s["value_score"] for s in player_scores.values()])
            }
        }
    
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": self.cache_hits / max(1, self.cache_hits + self.cache_misses),
            "policy_cache_size": len(self.policy_cache),
            "value_cache_size": len(self.value_cache),
            "mcts_cache_size": len(self.mcts_cache)
        }
