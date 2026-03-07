from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hex_ai.config import BOARD_SIZE
from hex_ai.data_processing import parse_trmph_to_gamerecord
from hex_ai.enums import Player, Winner
from hex_ai.eval.strength_evaluator import GameRecord
from hex_ai.inference.game_engine import HexGameState, apply_move_to_state_trmph, make_empty_hex_state
from hex_ai.inference.simple_model_inference import SimpleModelInference
from hex_ai.utils.format_conversion import (
    normalize_game_input,
    rowcol_to_trmph,
    trmph_move_to_rowcol,
)
from hex_ai.web.move_heatmap import build_policy_value_heatmap
from hex_ai.value_utils import ValuePredictor, get_legal_policy_probs, policy_logits_to_probs, select_top_k_moves

logger = logging.getLogger(__name__)

WIN_PROBABILITY_LOSS_THRESHOLDS = {
    "minor": 0.03,
    "moderate": 0.08,
    "major": 0.15,
}

WINNING_PROBABILITY_THRESHOLD = 0.55
DEFAULT_CANDIDATE_POLICY_TOP_K = 8
DEFAULT_SUGGESTION_COUNT = 3
DEFAULT_POLICY_TEMPERATURE = 1.0

GAME_PHASE_THRESHOLDS = {
    "opening_end": 12,
    "endgame_start_offset": 20,
}


@dataclass(frozen=True)
class MoveCandidate:
    row: int
    col: int
    move_trmph: str
    win_probability_for_player: float
    blue_win_probability: float
    policy_probability: float
    policy_rank: int
    is_played_move: bool
    review_score: float
    distance_to_even: Optional[float]


@dataclass(frozen=True)
class MoveAnalysis:
    ply: int
    move_number: int
    player: str
    game_phase: str
    position_trmph: str
    board_before: List[List[str]]
    move_played: Tuple[int, int]
    move_played_trmph: str
    position_win_probability_for_player: float
    move_played_win_probability_for_player: float
    best_move: Tuple[int, int]
    best_move_trmph: str
    best_move_win_probability_for_player: float
    review_metric: str
    review_metric_label: str
    review_score_played: float
    best_review_score: float
    review_score_loss: float
    win_probability_loss: float
    move_played_policy_probability: float
    move_played_policy_rank: int
    legal_move_count: int
    candidate_move_count: int
    suggestions: List[MoveCandidate]
    move_played_distance_to_even: Optional[float]
    best_move_distance_to_even: Optional[float]
    is_mistake: bool
    mistake_severity: str
    mistake_reason: str
    is_losing_move: bool
    was_winning_before: bool
    is_winning_after: bool
    blue_win_probability_before: float
    blue_win_probability_after_played: float
    blue_win_probability_after_best: float


@dataclass(frozen=True)
class GameReview:
    game_metadata: Dict[str, Any]
    analysis_metadata: Dict[str, Any]
    total_moves: int
    move_analyses: List[MoveAnalysis]
    total_mistakes: int
    major_mistakes: int
    losing_moves: int
    mistake_by_player: Dict[str, int]
    mistake_by_phase: Dict[str, int]
    win_probability_trajectory: List[Dict[str, Any]]
    critical_moments: List[Dict[str, Any]]


def _board_to_display_slice(board, display_board_size: int) -> List[List[str]]:
    return [
        [str(cell) for cell in row[:display_board_size]]
        for row in board[:display_board_size]
    ]


def _determine_game_phase(ply: int, total_moves: int) -> str:
    if ply < GAME_PHASE_THRESHOLDS["opening_end"]:
        return "opening"
    if ply < max(0, total_moves - GAME_PHASE_THRESHOLDS["endgame_start_offset"]):
        return "middle"
    return "endgame"


def _normalize_mistake(win_probability_loss: float) -> Tuple[bool, str, str]:
    if win_probability_loss < WIN_PROBABILITY_LOSS_THRESHOLDS["minor"]:
        return False, "none", "Move stayed within the model's acceptable range"
    if win_probability_loss < WIN_PROBABILITY_LOSS_THRESHOLDS["moderate"]:
        return True, "minor", f"Small drop of {win_probability_loss * 100:.1f} win-probability points"
    if win_probability_loss < WIN_PROBABILITY_LOSS_THRESHOLDS["major"]:
        return True, "moderate", f"Notable drop of {win_probability_loss * 100:.1f} win-probability points"
    return True, "major", f"Large drop of {win_probability_loss * 100:.1f} win-probability points"


def _summarize_player(player: Player) -> str:
    return "blue" if player == Player.BLUE else "red"


class GameReviewer:
    """
    Review move sequences using policy-ranked candidate moves and value-head scoring.

    The reviewer evaluates the played move plus a policy top-k candidate set for every
    position. That keeps the analysis responsive enough for the web UI while still
    surfacing the strongest alternatives the current model is likely to care about.
    """

    def __init__(
        self,
        model: SimpleModelInference,
        *,
        model_label: str,
        display_board_size: int = BOARD_SIZE,
        candidate_policy_top_k: int = DEFAULT_CANDIDATE_POLICY_TOP_K,
        suggestion_count: int = DEFAULT_SUGGESTION_COUNT,
        policy_temperature: float = DEFAULT_POLICY_TEMPERATURE,
        swap_opening_scores: Optional[Mapping[str, float]] = None,
    ):
        if candidate_policy_top_k < 1:
            raise ValueError("candidate_policy_top_k must be >= 1")
        if suggestion_count < 1:
            raise ValueError("suggestion_count must be >= 1")
        if policy_temperature <= 0:
            raise ValueError("policy_temperature must be > 0")
        if not (1 <= display_board_size <= BOARD_SIZE):
            raise ValueError(f"display_board_size must be between 1 and {BOARD_SIZE}")

        self.model = model
        self.model_label = model_label
        self.display_board_size = display_board_size
        self.candidate_policy_top_k = candidate_policy_top_k
        self.suggestion_count = suggestion_count
        self.policy_temperature = policy_temperature
        self.swap_opening_scores = (
            {str(move).lower(): float(score) for move, score in swap_opening_scores.items()}
            if swap_opening_scores is not None
            else None
        )

    def review_game_record(self, game: GameRecord) -> GameReview:
        if game.board_size != BOARD_SIZE:
            raise ValueError(
                f"Only {BOARD_SIZE}x{BOARD_SIZE} reviews are currently supported by the CLI reviewer"
            )

        initial_player = game.starting_player
        state = build_initial_state_for_player(initial_player)
        moves_trmph: List[str] = []

        for ply, (row, col, player) in enumerate(game.moves):
            if player != state.current_player_enum:
                raise ValueError(
                    f"GameRecord player mismatch at ply {ply + 1}: expected "
                    f"{state.current_player_enum.name}, got {player.name}"
                )
            move_trmph = rowcol_to_trmph(row, col, board_size=game.board_size)
            moves_trmph.append(move_trmph)
            state = apply_move_to_state_trmph(state, move_trmph)

        initial_state = build_initial_state_for_player(initial_player)
        metadata = dict(game.metadata or {})
        metadata.setdefault("board_size", game.board_size)
        return self.review_move_sequence(initial_state, moves_trmph, metadata=metadata)

    def review_move_sequence(
        self,
        initial_state: HexGameState,
        moves_trmph: Sequence[str],
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> GameReview:
        state = initial_state
        move_analyses: List[MoveAnalysis] = []
        trajectory: List[Dict[str, Any]] = []
        total_moves = len(moves_trmph)

        initial_blue_win_probability = self._get_blue_win_probability(state)
        trajectory.append(
            {
                "ply": 0,
                "move_number": 0,
                "label": "start",
                "blue_win_probability": initial_blue_win_probability,
            }
        )

        for ply, move_trmph in enumerate(moves_trmph):
            analysis = self._analyze_move(state, move_trmph, ply, total_moves)
            move_analyses.append(analysis)
            trajectory.append(
                {
                    "ply": ply + 1,
                    "move_number": analysis.move_number,
                    "label": analysis.move_played_trmph,
                    "player": analysis.player,
                    "blue_win_probability": analysis.blue_win_probability_after_played,
                    "best_blue_win_probability": analysis.blue_win_probability_after_best,
                    "loss": analysis.review_score_loss,
                }
            )
            state = apply_move_to_state_trmph(state, move_trmph)

        summary = self._create_summary(move_analyses)
        critical_moments = self._identify_critical_moments(move_analyses)

        return GameReview(
            game_metadata=dict(metadata or {}),
            analysis_metadata={
                "model": self.model_label,
                "analysis_mode": "policy_top_k_plus_played",
                "candidate_policy_top_k": self.candidate_policy_top_k,
                "suggestion_count": self.suggestion_count,
                "policy_temperature": self.policy_temperature,
                "display_board_size": self.display_board_size,
                "network_board_size": BOARD_SIZE,
            },
            total_moves=total_moves,
            move_analyses=move_analyses,
            total_mistakes=summary["total_mistakes"],
            major_mistakes=summary["major_mistakes"],
            losing_moves=summary["losing_moves"],
            mistake_by_player=summary["mistake_by_player"],
            mistake_by_phase=summary["mistake_by_phase"],
            win_probability_trajectory=trajectory,
            critical_moments=critical_moments,
        )

    def _analyze_move(
        self,
        state: HexGameState,
        move_played_trmph: str,
        ply: int,
        total_moves: int,
    ) -> MoveAnalysis:
        current_player = state.current_player_enum
        player_name = _summarize_player(current_player)
        legal_moves = state.get_legal_moves()
        move_played = trmph_move_to_rowcol(move_played_trmph, board_size=BOARD_SIZE)

        if move_played not in legal_moves:
            raise ValueError(f"Illegal move at ply {ply + 1}: {move_played_trmph}")

        policy_logits, position_value_signed = self.model.simple_infer(state.to_trmph())
        position_win_probability_for_player = self._signed_value_to_player_probability(
            position_value_signed,
            current_player,
        )
        blue_win_probability_before = ValuePredictor.get_win_probability(
            float(position_value_signed),
            Player.BLUE,
        )

        if self._is_swap_aware_opening_position(ply, current_player):
            return self._analyze_swap_aware_opening(
                state=state,
                move_played=move_played,
                move_played_trmph=move_played_trmph,
                ply=ply,
                total_moves=total_moves,
                legal_moves=legal_moves,
                position_win_probability_for_player=position_win_probability_for_player,
                blue_win_probability_before=blue_win_probability_before,
                policy_logits=policy_logits,
            )

        ranked_legal_moves, policy_probability_by_move = self._rank_legal_moves_by_policy(
            policy_logits,
            legal_moves,
        )
        policy_rank_by_move = {
            rowcol_to_trmph(row, col, board_size=BOARD_SIZE): index + 1
            for index, (row, col) in enumerate(ranked_legal_moves)
        }

        selected_moves = self._select_candidate_moves(
            ranked_legal_moves,
            policy_probability_by_move,
            move_played,
        )
        candidate_by_move = self._evaluate_candidate_moves(
            state,
            current_player,
            selected_moves,
            policy_probability_by_move,
            policy_rank_by_move,
            move_played,
        )

        played_candidate = candidate_by_move[move_played_trmph]
        ranked_candidates = sorted(
            candidate_by_move.values(),
            key=lambda item: (
                item.win_probability_for_player,
                item.policy_probability,
                -item.policy_rank,
            ),
            reverse=True,
        )
        best_candidate = ranked_candidates[0]
        suggestions = [
            candidate
            for candidate in ranked_candidates
            if not candidate.is_played_move
        ][: self.suggestion_count]

        win_probability_loss = max(
            0.0,
            best_candidate.win_probability_for_player - played_candidate.win_probability_for_player,
        )
        is_mistake, mistake_severity, mistake_reason = _normalize_mistake(win_probability_loss)

        was_winning_before = position_win_probability_for_player >= WINNING_PROBABILITY_THRESHOLD
        is_winning_after = played_candidate.win_probability_for_player >= WINNING_PROBABILITY_THRESHOLD
        alternative_winning_options = any(
            candidate.win_probability_for_player >= WINNING_PROBABILITY_THRESHOLD
            for candidate in suggestions
        )
        is_losing_move = was_winning_before and not is_winning_after and alternative_winning_options

        return MoveAnalysis(
            ply=ply,
            move_number=ply + 1,
            player=player_name,
            game_phase=_determine_game_phase(ply, total_moves),
            position_trmph=state.to_trmph(),
            board_before=_board_to_display_slice(state.board, self.display_board_size),
            move_played=move_played,
            move_played_trmph=move_played_trmph,
            position_win_probability_for_player=position_win_probability_for_player,
            move_played_win_probability_for_player=played_candidate.win_probability_for_player,
            best_move=(best_candidate.row, best_candidate.col),
            best_move_trmph=best_candidate.move_trmph,
            best_move_win_probability_for_player=best_candidate.win_probability_for_player,
            review_metric="win_probability",
            review_metric_label="Win probability",
            review_score_played=played_candidate.review_score,
            best_review_score=best_candidate.review_score,
            review_score_loss=win_probability_loss,
            win_probability_loss=win_probability_loss,
            move_played_policy_probability=played_candidate.policy_probability,
            move_played_policy_rank=played_candidate.policy_rank,
            legal_move_count=len(legal_moves),
            candidate_move_count=len(candidate_by_move),
            suggestions=suggestions,
            move_played_distance_to_even=played_candidate.distance_to_even,
            best_move_distance_to_even=best_candidate.distance_to_even,
            is_mistake=is_mistake,
            mistake_severity=mistake_severity,
            mistake_reason=mistake_reason,
            is_losing_move=is_losing_move,
            was_winning_before=was_winning_before,
            is_winning_after=is_winning_after,
            blue_win_probability_before=blue_win_probability_before,
            blue_win_probability_after_played=played_candidate.blue_win_probability,
            blue_win_probability_after_best=best_candidate.blue_win_probability,
        )

    def _rank_legal_moves_by_policy(
        self,
        policy_logits,
        legal_moves: Sequence[Tuple[int, int]],
    ) -> Tuple[List[Tuple[int, int]], Dict[str, float]]:
        policy_probs = policy_logits_to_probs(policy_logits, self.policy_temperature)
        legal_policy_probs = get_legal_policy_probs(policy_probs, legal_moves, BOARD_SIZE)
        policy_probability_by_move = {
            rowcol_to_trmph(row, col, board_size=BOARD_SIZE): float(probability)
            for (row, col), probability in zip(legal_moves, legal_policy_probs)
        }
        ranked_legal_moves = sorted(
            legal_moves,
            key=lambda move: policy_probability_by_move[rowcol_to_trmph(*move, board_size=BOARD_SIZE)],
            reverse=True,
        )
        return ranked_legal_moves, policy_probability_by_move

    @staticmethod
    def _is_swap_aware_opening_position(ply: int, current_player: Player) -> bool:
        return ply == 0 and current_player == Player.BLUE

    @staticmethod
    def _distance_to_even(opening_win_probability: float) -> float:
        return abs(float(opening_win_probability) - 0.5)

    @classmethod
    def _swap_evenness_score(cls, opening_win_probability: float) -> float:
        return 1.0 - 2.0 * cls._distance_to_even(opening_win_probability)

    def _load_swap_opening_scores(self, state: HexGameState) -> Dict[str, float]:
        if self.swap_opening_scores is not None:
            return dict(self.swap_opening_scores)

        heatmap = build_policy_value_heatmap(
            state=state,
            model=self.model,
            selection_mode="all_legal",
            top_k=None,
            policy_temperature=1.0,
        )
        return {move.lower(): float(score) for move, score in heatmap.scores.items()}

    def _analyze_swap_aware_opening(
        self,
        *,
        state: HexGameState,
        move_played: Tuple[int, int],
        move_played_trmph: str,
        ply: int,
        total_moves: int,
        legal_moves: Sequence[Tuple[int, int]],
        position_win_probability_for_player: float,
        blue_win_probability_before: float,
        policy_logits,
    ) -> MoveAnalysis:
        ranked_legal_moves, policy_probability_by_move = self._rank_legal_moves_by_policy(
            policy_logits,
            legal_moves,
        )
        policy_rank_by_move = {
            rowcol_to_trmph(row, col, board_size=BOARD_SIZE): index + 1
            for index, (row, col) in enumerate(ranked_legal_moves)
        }
        opening_scores = self._load_swap_opening_scores(state)

        candidate_by_move: Dict[str, MoveCandidate] = {}
        for row, col in legal_moves:
            move_trmph = rowcol_to_trmph(row, col, board_size=BOARD_SIZE)
            opening_win_probability = opening_scores.get(move_trmph.lower())
            if opening_win_probability is None:
                raise ValueError(f"Missing swap-aware opening score for move {move_trmph}")

            distance_to_even = self._distance_to_even(opening_win_probability)
            review_score = self._swap_evenness_score(opening_win_probability)
            candidate_by_move[move_trmph] = MoveCandidate(
                row=row,
                col=col,
                move_trmph=move_trmph,
                win_probability_for_player=float(opening_win_probability),
                blue_win_probability=float(opening_win_probability),
                policy_probability=float(policy_probability_by_move[move_trmph]),
                policy_rank=int(policy_rank_by_move[move_trmph]),
                is_played_move=(row, col) == move_played,
                review_score=review_score,
                distance_to_even=distance_to_even,
            )

        played_candidate = candidate_by_move[move_played_trmph]
        ranked_candidates = sorted(
            candidate_by_move.values(),
            key=lambda item: (
                item.review_score,
                item.policy_probability,
                -item.policy_rank,
            ),
            reverse=True,
        )
        best_candidate = ranked_candidates[0]
        suggestions = [
            candidate
            for candidate in ranked_candidates
            if not candidate.is_played_move
        ][: self.suggestion_count]

        played_distance = played_candidate.distance_to_even or 0.0
        best_distance = best_candidate.distance_to_even or 0.0
        review_score_loss = max(0.0, played_distance - best_distance)
        is_mistake, mistake_severity, _ = _normalize_mistake(review_score_loss)
        if review_score_loss < WIN_PROBABILITY_LOSS_THRESHOLDS["minor"]:
            mistake_reason = (
                f"Opening move stayed close to swap balance at {played_distance * 100:.1f} points from 50%"
            )
        else:
            mistake_reason = (
                f"Opening move landed {played_distance * 100:.1f} points from 50%; "
                f"best reviewed option was {best_distance * 100:.1f} points away"
            )

        return MoveAnalysis(
            ply=ply,
            move_number=ply + 1,
            player=_summarize_player(state.current_player_enum),
            game_phase=_determine_game_phase(ply, total_moves),
            position_trmph=state.to_trmph(),
            board_before=_board_to_display_slice(state.board, self.display_board_size),
            move_played=move_played,
            move_played_trmph=move_played_trmph,
            position_win_probability_for_player=position_win_probability_for_player,
            move_played_win_probability_for_player=played_candidate.win_probability_for_player,
            best_move=(best_candidate.row, best_candidate.col),
            best_move_trmph=best_candidate.move_trmph,
            best_move_win_probability_for_player=best_candidate.win_probability_for_player,
            review_metric="swap_evenness",
            review_metric_label="Distance from 50%",
            review_score_played=played_candidate.review_score,
            best_review_score=best_candidate.review_score,
            review_score_loss=review_score_loss,
            win_probability_loss=review_score_loss,
            move_played_policy_probability=played_candidate.policy_probability,
            move_played_policy_rank=played_candidate.policy_rank,
            legal_move_count=len(legal_moves),
            candidate_move_count=len(candidate_by_move),
            suggestions=suggestions,
            move_played_distance_to_even=played_candidate.distance_to_even,
            best_move_distance_to_even=best_candidate.distance_to_even,
            is_mistake=is_mistake,
            mistake_severity=mistake_severity,
            mistake_reason=mistake_reason,
            is_losing_move=False,
            was_winning_before=False,
            is_winning_after=False,
            blue_win_probability_before=blue_win_probability_before,
            blue_win_probability_after_played=played_candidate.blue_win_probability,
            blue_win_probability_after_best=best_candidate.blue_win_probability,
        )

    def _select_candidate_moves(
        self,
        ranked_legal_moves: Sequence[Tuple[int, int]],
        policy_probability_by_move: Mapping[str, float],
        move_played: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        if not ranked_legal_moves:
            raise ValueError("Cannot review a position with no legal moves")

        selected_moves = select_top_k_moves(
            [policy_probability_by_move[rowcol_to_trmph(*move, board_size=BOARD_SIZE)] for move in ranked_legal_moves],
            list(ranked_legal_moves),
            self.candidate_policy_top_k,
        )
        if move_played not in selected_moves:
            selected_moves.append(move_played)
        return selected_moves

    def _evaluate_candidate_moves(
        self,
        state: HexGameState,
        current_player: Player,
        selected_moves: Sequence[Tuple[int, int]],
        policy_probability_by_move: Mapping[str, float],
        policy_rank_by_move: Mapping[str, int],
        move_played: Tuple[int, int],
    ) -> Dict[str, MoveCandidate]:
        candidate_states: List[str] = []
        pending_moves: List[Tuple[int, int]] = []
        finished: Dict[str, MoveCandidate] = {}

        for row, col in selected_moves:
            move_trmph = rowcol_to_trmph(row, col, board_size=BOARD_SIZE)
            next_state = apply_move_to_state_trmph(state, move_trmph)

            if next_state.game_over and next_state.winner is not None:
                blue_win_probability = 1.0 if next_state.winner == Winner.BLUE else 0.0
                winning_player = Winner.BLUE if current_player == Player.BLUE else Winner.RED
                if next_state.winner == winning_player:
                    player_win_probability = 1.0
                else:
                    player_win_probability = 0.0
                finished[move_trmph] = MoveCandidate(
                    row=row,
                    col=col,
                    move_trmph=move_trmph,
                    win_probability_for_player=player_win_probability,
                    blue_win_probability=blue_win_probability,
                    policy_probability=float(policy_probability_by_move[move_trmph]),
                    policy_rank=int(policy_rank_by_move[move_trmph]),
                    is_played_move=(row, col) == move_played,
                    review_score=player_win_probability,
                    distance_to_even=None,
                )
                continue

            candidate_states.append(next_state.to_trmph())
            pending_moves.append((row, col))

        if candidate_states:
            _, value_signed_list = self.model.batch_infer(candidate_states)
            for (row, col), value_signed in zip(pending_moves, value_signed_list):
                move_trmph = rowcol_to_trmph(row, col, board_size=BOARD_SIZE)
                value_signed = float(value_signed)
                next_player = Player.RED if current_player == Player.BLUE else Player.BLUE
                player_win_probability = 1.0 - ValuePredictor.get_win_probability(value_signed, next_player)
                blue_win_probability = ValuePredictor.get_win_probability(value_signed, Player.BLUE)
                finished[move_trmph] = MoveCandidate(
                    row=row,
                    col=col,
                    move_trmph=move_trmph,
                    win_probability_for_player=player_win_probability,
                    blue_win_probability=blue_win_probability,
                    policy_probability=float(policy_probability_by_move[move_trmph]),
                    policy_rank=int(policy_rank_by_move[move_trmph]),
                    is_played_move=(row, col) == move_played,
                    review_score=player_win_probability,
                    distance_to_even=None,
                )

        return finished

    def _get_blue_win_probability(self, state: HexGameState) -> float:
        _, value_signed = self.model.simple_infer(state.to_trmph())
        return ValuePredictor.get_win_probability(float(value_signed), Player.BLUE)

    @staticmethod
    def _signed_value_to_player_probability(value_signed: float, player: Player) -> float:
        return ValuePredictor.get_win_probability(float(value_signed), player)

    @staticmethod
    def _create_summary(move_analyses: Sequence[MoveAnalysis]) -> Dict[str, Any]:
        total_mistakes = sum(1 for analysis in move_analyses if analysis.is_mistake)
        major_mistakes = sum(1 for analysis in move_analyses if analysis.mistake_severity == "major")
        losing_moves = sum(1 for analysis in move_analyses if analysis.is_losing_move)

        mistake_by_player = {"blue": 0, "red": 0}
        mistake_by_phase = {"opening": 0, "middle": 0, "endgame": 0}

        for analysis in move_analyses:
            if not analysis.is_mistake:
                continue
            mistake_by_player[analysis.player] += 1
            mistake_by_phase[analysis.game_phase] += 1

        return {
            "total_mistakes": total_mistakes,
            "major_mistakes": major_mistakes,
            "losing_moves": losing_moves,
            "mistake_by_player": mistake_by_player,
            "mistake_by_phase": mistake_by_phase,
        }

    @staticmethod
    def _identify_critical_moments(move_analyses: Sequence[MoveAnalysis]) -> List[Dict[str, Any]]:
        ranked = sorted(
            (
                analysis
                for analysis in move_analyses
                if analysis.is_mistake or analysis.is_losing_move
            ),
            key=lambda analysis: (
                analysis.review_score_loss,
                1 if analysis.is_losing_move else 0,
            ),
            reverse=True,
        )

        critical_moments: List[Dict[str, Any]] = []
        for analysis in ranked[:8]:
            critical_moments.append(
                {
                    "ply": analysis.ply,
                    "move_number": analysis.move_number,
                    "player": analysis.player,
                    "move_played": analysis.move_played_trmph,
                    "best_move": analysis.best_move_trmph,
                    "mistake_severity": analysis.mistake_severity,
                    "is_losing_move": analysis.is_losing_move,
                    "review_metric": analysis.review_metric,
                    "review_score_loss": analysis.review_score_loss,
                    "description": analysis.mistake_reason,
                }
            )
        return critical_moments


def review_to_json(review: GameReview) -> Dict[str, Any]:
    return {
        "game_metadata": review.game_metadata,
        "analysis_metadata": review.analysis_metadata,
        "summary": {
            "total_moves": review.total_moves,
            "total_mistakes": review.total_mistakes,
            "major_mistakes": review.major_mistakes,
            "losing_moves": review.losing_moves,
            "mistake_by_player": review.mistake_by_player,
            "mistake_by_phase": review.mistake_by_phase,
        },
        "win_probability_trajectory": review.win_probability_trajectory,
        "critical_moments": review.critical_moments,
        "move_analyses": [asdict(analysis) for analysis in review.move_analyses],
    }


def format_review_as_json(review: GameReview) -> Dict[str, Any]:
    return review_to_json(review)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _asset_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_standalone_review_html(review_payload: Mapping[str, Any]) -> str:
    root = _project_root()
    base_css = _asset_text(root / "hex_ai/web/static_public/style.css")
    review_css = _asset_text(root / "hex_ai/web/static_public/review.css")
    board_renderer_js = _asset_text(root / "hex_ai/web/static_shared/board_renderer.js")
    review_ui_js = _asset_text(root / "hex_ai/web/static_shared/game_review_ui.js")
    payload_json = json.dumps(review_payload)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Snowflake 25 Review</title>
  <style>{base_css}\n{review_css}</style>
</head>
<body>
  <div id="review-page" class="review-page-shell">
    <div id="review-root"></div>
  </div>
  <script>
    window.__HEX_GAME_REVIEW_PAYLOAD__ = {payload_json};
  </script>
  <script>{board_renderer_js}</script>
  <script>{review_ui_js}</script>
  <script>
    window.HexGameReviewUi.mountStandalone(
      document.getElementById('review-root'),
      window.__HEX_GAME_REVIEW_PAYLOAD__
    );
  </script>
</body>
</html>
"""


def format_review_as_html(review: GameReview) -> str:
    return build_standalone_review_html(review_to_json(review))


def create_index_html(reviews: Sequence[GameReview]) -> str:
    links = "\n".join(
        f'<li><a href="game_{index + 1}_review.html">Game {index + 1}</a></li>'
        for index, _ in enumerate(reviews)
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Snowflake 25 Reviews</title>
</head>
<body>
  <h1>Snowflake 25 Reviews</h1>
  <ul>
    {links}
  </ul>
</body>
</html>
"""


def save_review_files(reviews: Sequence[GameReview], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, review in enumerate(reviews, start=1):
        file_counter = index
        while True:
            json_file = output_dir / f"game_{file_counter}_review.json"
            html_file = output_dir / f"game_{file_counter}_review.html"
            if json_file.exists() or html_file.exists():
                file_counter += 1
                continue
            break

        json_file.write_text(
            json.dumps(format_review_as_json(review), indent=2),
            encoding="utf-8",
        )
        html_file.write_text(format_review_as_html(review), encoding="utf-8")
        logger.info("Saved review to %s and %s", json_file, html_file)

    if len(reviews) > 1:
        (output_dir / "index.html").write_text(create_index_html(reviews), encoding="utf-8")


def reviewer_from_model_path(
    model_path: str,
    *,
    model_label: str,
    display_board_size: int = BOARD_SIZE,
    candidate_policy_top_k: int = DEFAULT_CANDIDATE_POLICY_TOP_K,
    suggestion_count: int = DEFAULT_SUGGESTION_COUNT,
    policy_temperature: float = DEFAULT_POLICY_TEMPERATURE,
    verbose: int = 1,
) -> GameReviewer:
    model = SimpleModelInference(model_path, verbose=verbose)
    return GameReviewer(
        model,
        model_label=model_label,
        display_board_size=display_board_size,
        candidate_policy_top_k=candidate_policy_top_k,
        suggestion_count=suggestion_count,
        policy_temperature=policy_temperature,
    )


def parse_single_trmph_game(game_text: str) -> GameRecord:
    trmph_string = normalize_game_input(game_text, board_size=BOARD_SIZE)
    if not trmph_string.startswith(f"#{BOARD_SIZE},"):
        trmph_string = f"#{BOARD_SIZE},{trmph_string}"
    return parse_trmph_to_gamerecord(trmph_string)


def parse_games_from_json_payload(data: Any) -> List[GameRecord]:
    if not isinstance(data, list):
        raise ValueError("JSON file must contain a list of games")

    games: List[GameRecord] = []
    for index, game_data in enumerate(data):
        if not isinstance(game_data, dict):
            raise ValueError(f"Game {index} must be an object")

        board_size = int(game_data.get("board_size", BOARD_SIZE))
        starting_player = Player(game_data.get("starting_player", Player.BLUE.value))
        moves = game_data.get("moves", [])
        game_moves = []

        for move in moves:
            if isinstance(move, dict):
                row = int(move["row"])
                col = int(move["col"])
                player = Player(move["player"])
            elif isinstance(move, list) and len(move) >= 3:
                row = int(move[0])
                col = int(move[1])
                player = Player(move[2])
            else:
                raise ValueError(f"Invalid move format in game {index}: {move}")
            game_moves.append((row, col, player))

        games.append(
            GameRecord(
                board_size=board_size,
                moves=game_moves,
                starting_player=starting_player,
                metadata=game_data.get("metadata", {}),
            )
        )

    return games


def build_initial_state_for_player(player: Player) -> HexGameState:
    if player == Player.BLUE:
        return make_empty_hex_state()
    return HexGameState(player)
