"""
Opening strategies for self-play training.

This module provides utilities for generating boards with specific opening moves
that are realistic for pie rule play. The goal is to ensure self-play games
explore the kinds of configurations that high-level play would focus on.
"""

import random
from typing import Dict, List, Mapping, Optional, Tuple
from hex_ai.config import BOARD_SIZE
from hex_ai.inference.game_engine import HexGameState, make_empty_hex_state
from hex_ai.utils.format_conversion import rowcol_to_trmph, trmph_move_to_rowcol


# Keep this much softer than web/app.py (k=10) so self-play explores broader openings.
PIE_RULE_VALUE_BALANCED_WEIGHT_EXPONENT = 3.0
PIE_RULE_VALUE_BALANCED_MIN_MOVE_PROBABILITY = 0.0003  # 0.03%
PIE_RULE_VALUE_BALANCED_CYCLE_LENGTH = 10_000
PIE_RULE_VALUE_BALANCED_SHUFFLE_SEED = 20260226

# Rounded first-move Blue win probabilities from the value network (13x13),
# captured from checkpoints/hyperparameter_tuning/pipeline_20260225_024817/epoch164_mini15.pt.gz.
# Rounded to ~3 significant digits to keep tuning simple and avoid overfitting to noisy precision.
PIE_RULE_VALUE_BALANCED_OPENING_WIN_RATES_13X13: Dict[str, float] = {
    "a1":  0.187,  "a2":  0.383,  "a3":  0.317,  "a4":  0.688,  "a5":  0.715,
    "a6":  0.606,  "a7":  0.598,  "a8":  0.628,  "a9":  0.525,  "a10": 0.559,
    "a11": 0.297,  "a12": 0.675,  "a13": 0.676,  "b1":  0.203,  "b2":  0.392,
    "b3":  0.769,  "b4":  0.711,  "b5":  0.773,  "b6":  0.793,  "b7":  0.808,
    "b8":  0.744,  "b9":  0.787,  "b10": 0.828,  "b11": 0.685,  "b12": 0.750,
    "b13": 0.366,  "c1":  0.244,  "c2":  0.462,  "c3":  0.722,  "c4":  0.758,
    "c5":  0.786,  "c6":  0.791,  "c7":  0.782,  "c8":  0.715,  "c9":  0.746,
    "c10": 0.743,  "c11": 0.782,  "c12": 0.376,  "c13": 0.293,  "d1":  0.242,
    "d2":  0.412,  "d3":  0.638,  "d4":  0.764,  "d5":  0.815,  "d6":  0.812,
    "d7":  0.806,  "d8":  0.762,  "d9":  0.694,  "d10": 0.825,  "d11": 0.607,
    "d12": 0.300,  "d13": 0.218,  "e1":  0.225,  "e2":  0.282,  "e3":  0.547,
    "e4":  0.752,  "e5":  0.808,  "e6":  0.829,  "e7":  0.795,  "e8":  0.798,
    "e9":  0.794,  "e10": 0.743,  "e11": 0.624,  "e12": 0.320,  "e13": 0.183,
    "f1":  0.195,  "f2":  0.280,  "f3":  0.410,  "f4":  0.740,  "f5":  0.827,
    "f6":  0.809,  "f7":  0.822,  "f8":  0.809,  "f9":  0.806,  "f10": 0.740,
    "f11": 0.569,  "f12": 0.298,  "f13": 0.173,  "g1":  0.201,  "g2":  0.254,
    "g3":  0.427,  "g4":  0.766,  "g5":  0.779,  "g6":  0.807,  "g7":  0.823,
    "g8":  0.802,  "g9":  0.807,  "g10": 0.782,  "g11": 0.487,  "g12": 0.246,
    "g13": 0.174,  "h1":  0.219,  "h2":  0.289,  "h3":  0.541,  "h4":  0.697,
    "h5":  0.794,  "h6":  0.817,  "h7":  0.837,  "h8":  0.835,  "h9":  0.832,
    "h10": 0.774,  "h11": 0.449,  "h12": 0.272,  "h13": 0.181,  "i1":  0.242,
    "i2":  0.334,  "i3":  0.577,  "i4":  0.765,  "i5":  0.822,  "i6":  0.822,
    "i7":  0.838,  "i8":  0.846,  "i9":  0.826,  "i10": 0.722,  "i11": 0.447,
    "i12": 0.273,  "i13": 0.161,  "j1":  0.296,  "j2":  0.305,  "j3":  0.668,
    "j4":  0.834,  "j5":  0.823,  "j6":  0.847,  "j7":  0.834,  "j8":  0.829,
    "j9":  0.830,  "j10": 0.780,  "j11": 0.489,  "j12": 0.401,  "j13": 0.158,
    "k1":  0.304,  "k2":  0.394,  "k3":  0.786,  "k4":  0.821,  "k5":  0.807,
    "k6":  0.836,  "k7":  0.836,  "k8":  0.821,  "k9":  0.807,  "k10": 0.781,
    "k11": 0.717,  "k12": 0.437,  "k13": 0.228,  "l1":  0.445,  "l2":  0.765,
    "l3":  0.704,  "l4":  0.832,  "l5":  0.770,  "l6":  0.720,  "l7":  0.818,
    "l8":  0.775,  "l9":  0.761,  "l10": 0.564,  "l11": 0.717,  "l12": 0.388,
    "l13": 0.194,  "m1":  0.664,  "m2":  0.743,  "m3":  0.293,  "m4":  0.535,
    "m5":  0.553,  "m6":  0.610,  "m7":  0.609,  "m8":  0.608,  "m9":  0.714,
    "m10": 0.676,  "m11": 0.352,  "m12": 0.387,  "m13": 0.222,
}


def _all_first_moves_trmph(board_size: int) -> List[str]:
    return [
        rowcol_to_trmph(row, col, board_size=board_size)
        for row in range(board_size)
        for col in range(board_size)
    ]


def _pie_rule_opening_weight_from_probability(
    opening_win_prob: float,
    *,
    exponent: float = PIE_RULE_VALUE_BALANCED_WEIGHT_EXPONENT,
) -> float:
    """Weight based on closeness to 50% using min(p, 1-p)^k."""
    p = max(0.0, min(1.0, float(opening_win_prob)))
    symmetry_distance = min(p, 1.0 - p)
    if symmetry_distance <= 0.0:
        return 0.0
    return float(symmetry_distance ** exponent)


def build_value_balanced_opening_probability_map(
    opening_win_rates: Mapping[str, float],
    *,
    board_size: int = BOARD_SIZE,
    weight_exponent: float = PIE_RULE_VALUE_BALANCED_WEIGHT_EXPONENT,
    min_move_probability: float = PIE_RULE_VALUE_BALANCED_MIN_MOVE_PROBABILITY,
) -> Dict[str, float]:
    """
    Build normalized first-move probabilities from value-head win rates.

    Every move receives at least ``min_move_probability``. Remaining mass is distributed
    proportionally using min(p, 1-p)^weight_exponent, which smoothly favors near-even moves.
    """
    if board_size <= 0:
        raise ValueError(f"board_size must be positive, got {board_size}")
    if weight_exponent <= 0.0:
        raise ValueError(f"weight_exponent must be > 0, got {weight_exponent}")
    if min_move_probability < 0.0:
        raise ValueError(
            f"min_move_probability must be >= 0, got {min_move_probability}"
        )

    legal_moves = _all_first_moves_trmph(board_size)
    legal_set = set(legal_moves)
    provided_set = set(opening_win_rates.keys())
    missing_moves = sorted(legal_set - provided_set)
    unexpected_moves = sorted(provided_set - legal_set)
    if missing_moves:
        raise ValueError(
            f"Missing win-rate entries for {len(missing_moves)} legal moves: {missing_moves[:5]}"
        )
    if unexpected_moves:
        raise ValueError(
            f"Unexpected win-rate entries for {len(unexpected_moves)} illegal moves: {unexpected_moves[:5]}"
        )

    floor_mass = min_move_probability * len(legal_moves)
    if floor_mass >= 1.0:
        raise ValueError(
            "min_move_probability is too large for board size. "
            f"Need min_move_probability * num_moves < 1.0, got {floor_mass:.6f}."
        )

    weights: List[float] = []
    for move in legal_moves:
        weight = _pie_rule_opening_weight_from_probability(
            opening_win_rates[move], exponent=weight_exponent
        )
        weights.append(weight)

    total_weight = float(sum(weights))
    if total_weight <= 0.0:
        raise ValueError(
            "All opening weights are zero; check opening_win_rates and weight_exponent."
        )

    remaining_mass = 1.0 - floor_mass
    probabilities: Dict[str, float] = {}
    for move, weight in zip(legal_moves, weights):
        probabilities[move] = float(
            min_move_probability + remaining_mass * (weight / total_weight)
        )
    return probabilities


def build_value_balanced_opening_moves(
    opening_win_rates: Mapping[str, float],
    *,
    board_size: int = BOARD_SIZE,
    weight_exponent: float = PIE_RULE_VALUE_BALANCED_WEIGHT_EXPONENT,
    min_move_probability: float = PIE_RULE_VALUE_BALANCED_MIN_MOVE_PROBABILITY,
    cycle_length: int = PIE_RULE_VALUE_BALANCED_CYCLE_LENGTH,
    shuffle_seed: int = PIE_RULE_VALUE_BALANCED_SHUFFLE_SEED,
) -> List[Tuple[int, int]]:
    """
    Build a deterministic cycle of opening moves sampled from value-balanced probabilities.

    The returned cycle is intended for repeated modulo indexing by game number.
    """
    if cycle_length <= 0:
        raise ValueError(f"cycle_length must be > 0, got {cycle_length}")

    probabilities = build_value_balanced_opening_probability_map(
        opening_win_rates,
        board_size=board_size,
        weight_exponent=weight_exponent,
        min_move_probability=min_move_probability,
    )
    ordered_moves = sorted(
        probabilities.keys(), key=lambda move: trmph_move_to_rowcol(move, board_size)
    )

    expected_counts = [probabilities[move] * cycle_length for move in ordered_moves]
    counts = [int(value) for value in expected_counts]
    remainder = cycle_length - sum(counts)
    if remainder < 0:
        raise RuntimeError(
            "Internal error while allocating opening counts: negative remainder."
        )

    if remainder:
        fractional_order = sorted(
            range(len(ordered_moves)),
            key=lambda idx: (-(expected_counts[idx] - counts[idx]), ordered_moves[idx]),
        )
        for idx in fractional_order[:remainder]:
            counts[idx] += 1

    zero_count_moves = [
        ordered_moves[idx] for idx, count in enumerate(counts) if count <= 0
    ]
    if zero_count_moves:
        raise ValueError(
            "cycle_length too small for configured min_move_probability. "
            f"Moves with zero samples: {zero_count_moves[:5]}"
        )

    opening_moves: List[Tuple[int, int]] = []
    for move, count in zip(ordered_moves, counts):
        row, col = trmph_move_to_rowcol(move, board_size=board_size)
        opening_moves.extend([(row, col)] * count)

    if len(opening_moves) != cycle_length:
        raise RuntimeError(
            f"Opening cycle construction mismatch: expected {cycle_length}, got {len(opening_moves)}"
        )

    random.Random(shuffle_seed).shuffle(opening_moves)
    return opening_moves


class OpeningStrategy:
    """Base class for opening strategies."""
    
    def __init__(self, board_size: int = BOARD_SIZE):
        if isinstance(board_size, bool):
            raise TypeError("board_size must be an integer, got bool")
        self.board_size = int(board_size)
        if self.board_size <= 0:
            raise ValueError(f"board_size must be positive, got {self.board_size}")
    
    def get_opening_move(self, game_index: int) -> Optional[Tuple[int, int]]:
        """
        Get the opening move for a given game index.
        
        Args:
            game_index: Index of the game (0-based)
            
        Returns:
            Tuple of (row, col) coordinates, or None for empty board
        """
        raise NotImplementedError
    
    def get_total_games(self) -> int:
        """Get the total number of games this strategy covers."""
        raise NotImplementedError


class PieRuleOpeningStrategy(OpeningStrategy):
    """
    Opening strategy that focuses on realistic pie rule openings.

    Modes:
    - ``value_balanced`` (default): deterministic cycle sampled from value-head first-move
      win rates, with smooth preference for near-50% openings.
    - ``legacy``: original hand-crafted pie-rule buckets.
    """

    def __init__(
        self,
        board_size: int = BOARD_SIZE,
        strategy_mode: str = "value_balanced",
        weight_exponent: float = PIE_RULE_VALUE_BALANCED_WEIGHT_EXPONENT,
        min_move_probability: float = PIE_RULE_VALUE_BALANCED_MIN_MOVE_PROBABILITY,
        cycle_length: int = PIE_RULE_VALUE_BALANCED_CYCLE_LENGTH,
    ):
        super().__init__(board_size)
        if self.board_size != BOARD_SIZE:
            raise ValueError(
                f"PieRuleOpeningStrategy is currently tuned for {BOARD_SIZE}x{BOARD_SIZE}. "
                f"Got {self.board_size}."
            )
        if strategy_mode not in {"value_balanced", "legacy"}:
            raise ValueError(
                "strategy_mode must be 'value_balanced' or 'legacy', "
                f"got {strategy_mode!r}"
            )

        self.strategy_mode = strategy_mode
        self.weight_exponent = float(weight_exponent)
        self.min_move_probability = float(min_move_probability)
        self.cycle_length = int(cycle_length)

        self.opening_moves: List[Tuple[int, int]] = []
        self.balanced_moves: List[Tuple[int, int]] = []
        self.somewhat_unbalanced_moves: List[Tuple[int, int]] = []
        self.bad_moves: List[Tuple[int, int]] = []
        self._legacy_bad_move_indices: List[int] = []
        self.balanced_games = 0
        self.unbalanced_games = 0
        self.network_games = 0
        self.bad_games = 0

        if self.strategy_mode == "value_balanced":
            self.opening_moves = build_value_balanced_opening_moves(
                PIE_RULE_VALUE_BALANCED_OPENING_WIN_RATES_13X13,
                board_size=self.board_size,
                weight_exponent=self.weight_exponent,
                min_move_probability=self.min_move_probability,
                cycle_length=self.cycle_length,
                shuffle_seed=PIE_RULE_VALUE_BALANCED_SHUFFLE_SEED,
            )
            self._total_games = len(self.opening_moves)
        else:
            self._initialize_legacy_openings()

    def _initialize_legacy_openings(self) -> None:
        # TODO(2026-03-31): Remove legacy pie-rule mode if it is not useful in practice.
        first_move_bad_move_frequency = 0.1

        # Balanced moves: (a2-a13), (b5-b11), (b2-k2)
        for row in range(1, self.board_size):  # a2 to a13
            self.balanced_moves.append((row, 0))  # col 0 = 'a'

        for row in range(4, 11):  # b5 to b11
            self.balanced_moves.append((row, 1))  # col 1 = 'b'

        for col in range(1, 11):  # b2 to k2
            self.balanced_moves.append((1, col))

        self.somewhat_unbalanced_moves = [
            (0, 0),   # a1
            (11, 1),  # b12
            (2, 2),   # c3
        ]

        # Bad moves: b1, c1, d1, ..., l1 (edge moves that are too strong)
        for col in range(1, 12):  # b1 to l1
            self.bad_moves.append((0, col))

        self.balanced_games = len(self.balanced_moves)
        self.unbalanced_games = len(self.somewhat_unbalanced_moves)
        self.network_games = 2  # 2 games with network-chosen moves
        self.bad_games = int(
            len(self.bad_moves) * first_move_bad_move_frequency
        )
        self._legacy_bad_move_indices = [
            int(game_index / first_move_bad_move_frequency)
            for game_index in range(self.bad_games)
        ]
        self._total_games = (
            self.balanced_games
            + self.unbalanced_games
            + self.network_games
            + self.bad_games
        )
    
    def get_opening_move(self, game_index: int) -> Optional[Tuple[int, int]]:
        """
        Get the opening move for a given game index.
        
        Args:
            game_index: Index of the game (0-based)
            
        Returns:
            Tuple of (row, col) coordinates, or None for empty board
        """
        if self._total_games <= 0:
            raise RuntimeError("PieRuleOpeningStrategy has no configured opening moves")
        game_index = game_index % self._total_games

        if self.strategy_mode == "value_balanced":
            return self.opening_moves[game_index]

        # Legacy mode
        if game_index < self.balanced_games:
            return self.balanced_moves[game_index]
        game_index -= self.balanced_games

        if game_index < self.unbalanced_games:
            return self.somewhat_unbalanced_moves[game_index]
        game_index -= self.unbalanced_games

        if game_index < self.network_games:
            return None
        game_index -= self.network_games

        if game_index < self.bad_games:
            bad_move_index = self._legacy_bad_move_indices[game_index]
            if bad_move_index < len(self.bad_moves):
                return self.bad_moves[bad_move_index]
        return None
    
    def get_total_games(self) -> int:
        """Get the total number of games this strategy covers."""
        return self._total_games


class RandomOpeningStrategy(OpeningStrategy):
    """Opening strategy that randomly selects from a set of moves."""
    
    def __init__(self, moves: List[Tuple[int, int]], board_size: int = BOARD_SIZE, 
                 empty_board_prob: float = 0.1, rng_seed: Optional[int] = None):
        super().__init__(board_size)
        self.moves = moves
        self.empty_board_prob = empty_board_prob
        # Keep opening randomness independent from global random reseeding in self-play.
        self._rng = random.Random(rng_seed)
    
    def get_opening_move(self, game_index: int) -> Optional[Tuple[int, int]]:
        """Get a random opening move."""
        if self._rng.random() < self.empty_board_prob:
            return None
        return self._rng.choice(self.moves)
    
    def get_total_games(self) -> int:
        """This strategy can be used for any number of games."""
        return float('inf')  # Infinite games


def create_board_with_opening(opening_move: Optional[Tuple[int, int]], 
                            board_size: int = BOARD_SIZE) -> HexGameState:
    """
    Create a game state with a specific opening move.
    
    Args:
        opening_move: Tuple of (row, col) coordinates, or None for empty board
        board_size: Size of the board
        
    Returns:
        HexGameState with the opening move applied
    """
    state = make_empty_hex_state(board_size=board_size)
    
    if opening_move is not None:
        row, col = opening_move
        # Validate coordinates
        if not (0 <= row < board_size and 0 <= col < board_size):
            raise ValueError(f"Invalid coordinates ({row}, {col}) for board size {board_size}")
        
        # Apply the opening move
        state = state.make_move(row, col)
    
    return state


def get_trmph_opening_move(opening_move: Optional[Tuple[int, int]], 
                          board_size: int = BOARD_SIZE) -> Optional[str]:
    """
    Convert opening move coordinates to TRMPH format.
    
    Args:
        opening_move: Tuple of (row, col) coordinates, or None
        board_size: Size of the board
        
    Returns:
        TRMPH move string (e.g., "a2"), or None for empty board
    """
    if opening_move is None:
        return None
    
    row, col = opening_move
    return rowcol_to_trmph(row, col, board_size)


def create_pie_rule_strategy(
    board_size: int = BOARD_SIZE,
    strategy_mode: str = "value_balanced",
    weight_exponent: float = PIE_RULE_VALUE_BALANCED_WEIGHT_EXPONENT,
    min_move_probability: float = PIE_RULE_VALUE_BALANCED_MIN_MOVE_PROBABILITY,
    cycle_length: int = PIE_RULE_VALUE_BALANCED_CYCLE_LENGTH,
) -> PieRuleOpeningStrategy:
    """
    Create a pie rule opening strategy.
    
    Args:
        board_size: Size of the board
        strategy_mode: "value_balanced" (default) or "legacy"
        weight_exponent: Weight exponent for value_balanced mode
        min_move_probability: Minimum probability floor per move for value_balanced mode
        cycle_length: Deterministic cycle length for value_balanced mode
        
    Returns:
        PieRuleOpeningStrategy instance
    """
    return PieRuleOpeningStrategy(
        board_size=board_size,
        strategy_mode=strategy_mode,
        weight_exponent=weight_exponent,
        min_move_probability=min_move_probability,
        cycle_length=cycle_length,
    )
