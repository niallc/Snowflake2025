"""
Self-play engine for generating training data using the Hex AI model.
"""

import logging
import numpy as np
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hex_ai.config import (
    BOARD_SIZE,
    DEFAULT_CACHE_SIZE,
    DEFAULT_C_PUCT,
    DEFAULT_MCTS_SIMS,
    DEFAULT_SELFPLAY_BASE_FRACTION_MCTS_MOVES,
    DEFAULT_SELFPLAY_SMALL_BOARD_FRACTION,
    DEFAULT_SELFPLAY_SMALL_BOARD_MAX_DISPLAY_SIZE,
    DEFAULT_SELFPLAY_SMALL_BOARD_MIN_DISPLAY_SIZE,
    DEFAULT_TEMPERATURE_END,
    DEFAULT_TEMPERATURE_START,
    TRMPH_BLUE_WIN,
    TRMPH_PREFIX,
    TRMPH_RED_WIN,
)
from hex_ai.enums import Winner
from hex_ai.inference.game_engine import HexGameEngine, make_empty_hex_state
from hex_ai.inference.mcts import BaselineMCTS, create_mcts_config
from hex_ai.inference.model_wrapper import ModelWrapper
from hex_ai.inference.simple_model_inference import SimpleModelInference
from hex_ai.move_provenance import (
    MOVE_CODE_CONFIDENCE_TERMINATION,
    MOVE_CODE_GUMBEL_ROOT,
    MOVE_CODE_TERMINAL_TERMINATION,
    MOVE_CODE_VISIT_COUNT,
    make_move_provenance_record,
    sidecar_path_for_trmph,
)
from hex_ai.selfplay.generation_summary import SelfPlayGenerationSummary
from hex_ai.system_utils import get_git_commit_info
from hex_ai.training_utils import get_device
from hex_ai.utils.format_conversion import count_trmph_moves, rowcol_to_trmph
from hex_ai.utils.tournament_logging import write_trmph_header
from hex_ai.utils.temperature import calculate_mcts_root_temperature
from hex_ai.value_utils import select_policy_move, validate_trmph_winner
from hex_ai.virtual_board import (
    MAX_VIRTUAL_DISPLAY_BOARD_SIZE,
    MIN_VIRTUAL_DISPLAY_BOARD_SIZE,
    get_virtual_prefill_move_coords,
)

DEFAULT_SELFPLAY_CONFIDENCE_TERMINATION_THRESHOLD = 0.85
# Policy-only moves use the non-trainable provenance code path.
POLICY_ONLY_MOVE_PROVENANCE_CODE = MOVE_CODE_CONFIDENCE_TERMINATION
# Virtual-board prefill moves are externally injected and should not be policy-trainable.
VIRTUAL_PREFILL_MOVE_PROVENANCE_CODE = MOVE_CODE_CONFIDENCE_TERMINATION
SELECTED_MOVE_SOURCE_TO_PROVENANCE_CODE = {
    "visit_counts": MOVE_CODE_VISIT_COUNT,
    "gumbel_root": MOVE_CODE_GUMBEL_ROOT,
    "neural_network_confidence": MOVE_CODE_CONFIDENCE_TERMINATION,
    "terminal_move": MOVE_CODE_TERMINAL_TERMINATION,
}


class SelfPlayEngine:
    """High-performance self-play engine with optimized inference and logging."""
    
    @staticmethod
    def _normalize_board_size(board_size: int, *, source: str) -> int:
        """Normalize and validate integer board-size inputs."""
        if isinstance(board_size, bool):
            raise TypeError(f"{source} must be an integer, got bool")
        try:
            size = int(board_size)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{source} must be an integer, got {type(board_size)}") from exc
        if size <= 0:
            raise ValueError(f"{source} must be positive, got {size}")
        return size

    @staticmethod
    def _ensure_supported_board_size(board_size: int) -> None:
        """
        Fail fast for unsupported board sizes until engine/model are fully parameterized.

        This avoids silently running 13x13 logic when callers believe another board size is in use.
        """
        if board_size != BOARD_SIZE:
            raise ValueError(
                f"Unsupported self-play board size {board_size}. "
                f"Current self-play runtime supports only {BOARD_SIZE}x{BOARD_SIZE}. "
                "Engine/model parameterization is required before enabling other sizes."
            )

    def _resolve_generation_board_size(self, board_size: Optional[int]) -> int:
        """Resolve per-call board size while enforcing engine-level consistency."""
        if board_size is None:
            return self.board_size
        requested_size = self._normalize_board_size(
            board_size, source="self-play generation board_size"
        )
        if requested_size != self.board_size:
            raise ValueError(
                "generate_games* board_size must match engine board_size. "
                f"Got {requested_size} vs {self.board_size}."
            )
        self._ensure_supported_board_size(requested_size)
        return requested_size

    @staticmethod
    def _normalize_base_fraction_mcts_moves(value: float) -> float:
        """Normalize and validate MCTS-move fraction in [0, 1]."""
        if isinstance(value, bool):
            raise TypeError("base_fraction_mcts_moves must be numeric, got bool")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"base_fraction_mcts_moves must be numeric, got {type(value)}"
            ) from exc
        if not 0.0 <= numeric <= 1.0:
            raise ValueError(
                f"base_fraction_mcts_moves must be in [0, 1], got {numeric}"
            )
        return numeric

    @staticmethod
    def _normalize_small_board_fraction(value: float) -> float:
        """Normalize and validate virtual small-board sampling fraction in [0, 1]."""
        if isinstance(value, bool):
            raise TypeError("small_board_fraction must be numeric, got bool")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"small_board_fraction must be numeric, got {type(value)}"
            ) from exc
        if not 0.0 <= numeric <= 1.0:
            raise ValueError(f"small_board_fraction must be in [0, 1], got {numeric}")
        return numeric

    @staticmethod
    def _normalize_small_board_size_range(
        min_display_board_size: int,
        max_display_board_size: int,
        *,
        network_board_size: int,
    ) -> Tuple[int, int]:
        """Normalize and validate configured virtual small-board size range."""
        min_size = SelfPlayEngine._normalize_board_size(
            min_display_board_size,
            source="small_board_min_display_size",
        )
        max_size = SelfPlayEngine._normalize_board_size(
            max_display_board_size,
            source="small_board_max_display_size",
        )
        if min_size < MIN_VIRTUAL_DISPLAY_BOARD_SIZE:
            raise ValueError(
                "small_board_min_display_size must be >= "
                f"{MIN_VIRTUAL_DISPLAY_BOARD_SIZE}, got {min_size}"
            )
        if max_size > MAX_VIRTUAL_DISPLAY_BOARD_SIZE:
            raise ValueError(
                "small_board_max_display_size must be <= "
                f"{MAX_VIRTUAL_DISPLAY_BOARD_SIZE}, got {max_size}"
            )
        if max_size >= network_board_size:
            raise ValueError(
                "small_board_max_display_size must be strictly smaller than "
                f"network board size {network_board_size}, got {max_size}"
            )
        if min_size > max_size:
            raise ValueError(
                "small_board_min_display_size must be <= "
                f"small_board_max_display_size, got {min_size} > {max_size}"
            )
        return min_size, max_size

    @staticmethod
    def _build_small_board_sampling_plan(
        min_display_board_size: int,
        max_display_board_size: int,
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """
        Build display-size candidates and linear preference weights.

        Weights are `(size - min_size + 1)`, which yields 12x12 probability 25%
        for the default 6..12 range.
        """
        sizes = tuple(range(min_display_board_size, max_display_board_size + 1))
        if not sizes:
            raise ValueError(
                "No small-board display sizes configured for sampling "
                f"({min_display_board_size}..{max_display_board_size})"
            )
        weights = tuple(size - min_display_board_size + 1 for size in sizes)
        if any(weight <= 0 for weight in weights):
            raise RuntimeError(f"Invalid non-positive small-board weights: {weights}")
        return sizes, weights

    def _compute_game_seed(self, game_id: Optional[int]) -> int:
        """Compute deterministic per-game seed used across all game-level sampling."""
        if game_id is not None:
            return int(self.run_seed + int(game_id) * 1000)
        return int(time.time() * 1000000) % (2**32)

    def _sample_virtual_display_board_size(self, game_id: Optional[int]) -> Optional[int]:
        """Sample optional virtual display board size for one game."""
        if self.small_board_fraction <= 0.0:
            return None
        seed = self._compute_game_seed(game_id)
        rng = random.Random(seed + 104729)
        if rng.random() >= self.small_board_fraction:
            return None
        sampled = rng.choices(
            self.small_board_display_sizes,
            weights=self.small_board_display_size_weights,
            k=1,
        )[0]
        return int(sampled)

    @staticmethod
    def _maybe_filter_opening_for_virtual_display(
        opening_move: Optional[Tuple[int, int]],
        display_board_size: Optional[int],
    ) -> Optional[Tuple[int, int]]:
        """Drop opening moves that are outside the active virtual display board."""
        if opening_move is None or display_board_size is None:
            return opening_move
        row, col = opening_move
        if row < 0 or col < 0:
            raise ValueError(f"Opening move contains negative coordinates: {opening_move}")
        if row >= display_board_size or col >= display_board_size:
            return None
        return opening_move

    @staticmethod
    def _format_small_board_sampling_description(
        counts_by_size: Dict[int, int],
        total_games: int,
    ) -> str:
        """Build a compact human-readable summary for virtual small-board sampling."""
        sampled_games = sum(counts_by_size.values())
        sampled_fraction = (sampled_games / total_games) if total_games > 0 else 0.0
        parts = [
            f"{size}x{size}={count}" for size, count in sorted(counts_by_size.items())
        ]
        distribution = ", ".join(parts)
        return (
            "Virtual small-board games: "
            f"{sampled_games}/{total_games} ({sampled_fraction:.2%})"
            + (f" [{distribution}]" if distribution else "")
        )

    def __init__(self, model_path: str,
                 cache_size: int = DEFAULT_CACHE_SIZE, temperature: float = DEFAULT_TEMPERATURE_START, temperature_end: float = DEFAULT_TEMPERATURE_END, 
                 verbose: int = 1, streaming_save: bool = False, streaming_file: str = None,
                 output_dir: str = None,
                 mcts_sims: int = DEFAULT_MCTS_SIMS, c_puct: float = DEFAULT_C_PUCT, enable_gumbel: bool = True,
                 enable_dead_cell_pruning: bool = False,
                 dead_cell_enable_four_run: bool = True,
                 dead_cell_enable_two_two_split: bool = True,
                 dead_cell_enable_three_plus_one: bool = True,
                 dead_cell_three_plus_one_requires_adjacent_opposite: bool = False,
                 dead_cell_enable_a1b2a3_discouraged: bool = True,
                 dead_cell_enable_double_dead_pairs: bool = False,
                 base_fraction_mcts_moves: float = DEFAULT_SELFPLAY_BASE_FRACTION_MCTS_MOVES,
                 small_board_fraction: float = DEFAULT_SELFPLAY_SMALL_BOARD_FRACTION,
                 small_board_min_display_size: int = DEFAULT_SELFPLAY_SMALL_BOARD_MIN_DISPLAY_SIZE,
                 small_board_max_display_size: int = DEFAULT_SELFPLAY_SMALL_BOARD_MAX_DISPLAY_SIZE,
                 confidence_termination_threshold: float = DEFAULT_SELFPLAY_CONFIDENCE_TERMINATION_THRESHOLD,
                 write_provenance: bool = True,
                 command_line: str = None,
                 mcts_profile: bool = False,
                 mcts_profile_every: int = 10,
                 mcts_profile_max_calls: int = 50,
                 board_size: int = BOARD_SIZE):
        
        # Generate a unique run seed based on current time
        self.run_seed = int(time.time() * 1000000) % (2**32)
        if verbose >= 1:
            print(f"SelfPlayEngine run seed: {self.run_seed}")
        """
        Initialize the self-play engine.
        
        Args:
            model_path: Path to the model checkpoint
            cache_size: Size of the LRU cache
            temperature: Starting temperature for move sampling
            temperature_end: Final temperature for move sampling (for decay)
            verbose: Verbosity level (0=quiet, 1=normal, 2=detailed, 3+=debug)
            streaming_save: Save games incrementally to avoid data loss
            streaming_file: File path for streaming save (auto-generated if None)
            output_dir: Output directory for streaming files (used if streaming_file is None)
            mcts_sims: Number of MCTS simulations per move
            c_puct: PUCT exploration constant for MCTS
            enable_gumbel: Enable Gumbel-AlphaZero root selection for MCTS
            enable_dead_cell_pruning: Enable dead-cell hard masking in MCTS
            dead_cell_enable_four_run: Enable dead-cell D1 (4-run) motif
            dead_cell_enable_two_two_split: Enable dead-cell D2 motif
            dead_cell_enable_three_plus_one: Enable dead-cell D3 motif
            dead_cell_three_plus_one_requires_adjacent_opposite: Enable strict D3 variant
            dead_cell_enable_a1b2a3_discouraged: Enable A1B2A3 discouraged motif
            dead_cell_enable_double_dead_pairs: Enable two-cell dead-pair motif
            base_fraction_mcts_moves: Fraction of moves that should run full MCTS
            small_board_fraction: Fraction of games that start from a virtual small-board prefill
            small_board_min_display_size: Small-board display-size lower bound (inclusive)
            small_board_max_display_size: Small-board display-size upper bound (inclusive)
            confidence_termination_threshold: Early-termination confidence threshold
            write_provenance: Whether to write move-provenance sidecar data
            board_size: Board size for generated games (currently fail-fast restricted to 13)
        """
        self.model_path = model_path
        self.cache_size = cache_size
        self.temperature = temperature
        self.temperature_end = temperature_end
        self.verbose = verbose
        self.streaming_save = streaming_save
        self.streaming_file = streaming_file
        self.output_dir = output_dir
        self.mcts_sims = mcts_sims
        self.c_puct = c_puct
        self.enable_gumbel = enable_gumbel
        self.enable_dead_cell_pruning = bool(enable_dead_cell_pruning)
        self.dead_cell_enable_four_run = bool(dead_cell_enable_four_run)
        self.dead_cell_enable_two_two_split = bool(dead_cell_enable_two_two_split)
        self.dead_cell_enable_three_plus_one = bool(dead_cell_enable_three_plus_one)
        self.dead_cell_three_plus_one_requires_adjacent_opposite = bool(
            dead_cell_three_plus_one_requires_adjacent_opposite
        )
        self.dead_cell_enable_a1b2a3_discouraged = bool(dead_cell_enable_a1b2a3_discouraged)
        self.dead_cell_enable_double_dead_pairs = bool(dead_cell_enable_double_dead_pairs)
        self.base_fraction_mcts_moves = self._normalize_base_fraction_mcts_moves(
            base_fraction_mcts_moves
        )
        self.confidence_termination_threshold = confidence_termination_threshold
        self.write_provenance = write_provenance
        self.command_line = command_line
        self.mcts_profile = mcts_profile
        self.mcts_profile_every = mcts_profile_every
        self.mcts_profile_max_calls = mcts_profile_max_calls
        self.board_size = self._normalize_board_size(
            board_size, source="SelfPlayEngine.board_size"
        )
        self._ensure_supported_board_size(self.board_size)
        self.small_board_fraction = self._normalize_small_board_fraction(
            small_board_fraction
        )
        (
            self.small_board_min_display_size,
            self.small_board_max_display_size,
        ) = self._normalize_small_board_size_range(
            small_board_min_display_size,
            small_board_max_display_size,
            network_board_size=self.board_size,
        )
        (
            self.small_board_display_sizes,
            self.small_board_display_size_weights,
        ) = self._build_small_board_sampling_plan(
            self.small_board_min_display_size,
            self.small_board_max_display_size,
        )
        self._small_board_prefill_moves_by_size: Dict[int, Tuple[Tuple[int, int], ...]] = {}
        for display_size in self.small_board_display_sizes:
            prefill_moves = get_virtual_prefill_move_coords(
                display_size,
                network_board_size=self.board_size,
            )
            if not prefill_moves:
                raise ValueError(
                    "Virtual small-board prefill must be non-empty for display size "
                    f"{display_size}"
                )
            self._small_board_prefill_moves_by_size[display_size] = prefill_moves

        if self.small_board_fraction > 0.0:
            self._validate_small_board_prefill_sequences()

        self._mcts_profile_calls = 0
        self.streaming_provenance_file: Optional[str] = None
        self._streaming_games_written = 0
        
        # Initialize model inference once and reuse its wrapper for MCTS to avoid
        # loading the same checkpoint twice in one process.
        self.model = SimpleModelInference(
            model_path,
            device=get_device(),
            cache_size=cache_size,
            board_size=self.board_size,
        )
        model_board_size = self._normalize_board_size(
            getattr(self.model, "board_size", BOARD_SIZE),
            source="SimpleModelInference.board_size",
        )
        if model_board_size != self.board_size:
            raise ValueError(
                "Self-play board size does not match model board size: "
                f"{self.board_size} vs {model_board_size}."
            )

        model_wrapper = getattr(self.model, "model", None)
        if not isinstance(model_wrapper, ModelWrapper):
            raise TypeError(
                "SelfPlayEngine expected SimpleModelInference.model to be a ModelWrapper, "
                f"got {type(model_wrapper)!r}"
            )
        self.model_wrapper = model_wrapper

        # Initialize MCTS components
        self.game_engine = HexGameEngine(board_size=self.board_size)
        # Create MCTS configuration optimized for self-play with confidence termination
        self.mcts_config = create_mcts_config("selfplay",
            sims=self.mcts_sims,
            # Aggressive confidence termination for speed.
            confidence_termination_threshold=self.confidence_termination_threshold,
            cache_size=self.cache_size,  # Use same cache size as SimpleModelInference
            c_puct=self.c_puct,  # Use specified PUCT exploration constant
            # Used by non-Gumbel visit-count move selection (and its reporting path).
            temperature_start=self.temperature,
            temperature_end=self.temperature_end,
            enable_gumbel_root_selection=self.enable_gumbel,  # Enable/disable Gumbel root selection
            enable_dead_cell_pruning=self.enable_dead_cell_pruning,
            dead_cell_enable_four_run=self.dead_cell_enable_four_run,
            dead_cell_enable_two_two_split=self.dead_cell_enable_two_two_split,
            dead_cell_enable_three_plus_one=self.dead_cell_enable_three_plus_one,
            dead_cell_three_plus_one_requires_adjacent_opposite=(
                self.dead_cell_three_plus_one_requires_adjacent_opposite
            ),
            dead_cell_enable_a1b2a3_discouraged=self.dead_cell_enable_a1b2a3_discouraged,
            dead_cell_enable_double_dead_pairs=self.dead_cell_enable_double_dead_pairs,
        )
        # Reuse one MCTS instance so eval_cache can persist across moves/games in this process.
        self.mcts = BaselineMCTS(self.game_engine, self.model_wrapper, self.mcts_config)
        
        # Performance tracking
        self.stats = {
            # Keep this dict scoped to values we actually update in self-play.
            'total_time': 0.0,
            'games_generated': 0,
            'total_moves': 0,
            'games_per_second': 0.0
        }

        # Aggregate MCTS runtime stats across all searched moves in this engine process.
        self.mcts_run_stats = {
            'moves_searched': 0,
            'total_search_time_s': 0.0,
            'total_effective_simulations': 0,
            'total_unique_evals': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'batch_count': 0,
            'batch_size_sum': 0.0,
            'algorithm_termination_reason_counts': {},
        }
        
        # Streaming save setup
        if self.streaming_save and self.streaming_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if self.output_dir:
                self.streaming_file = f"{self.output_dir}/streaming_selfplay_{timestamp}.trmph"
            else:
                self.streaming_file = f"data/sf25/selfplay_default/streaming_selfplay_{timestamp}.trmph"
        
        if self.streaming_save:
            streaming_path = Path(self.streaming_file)
            streaming_path.parent.mkdir(parents=True, exist_ok=True)

            metadata = {
                "Model": model_path,
                "Board size": self.board_size,
                "MCTS simulations": mcts_sims,
                "C_PUCT": c_puct,
                "Gumbel root selection": enable_gumbel,
                "Dead-cell pruning": self.enable_dead_cell_pruning,
                "Dead-cell D1 four-run": self.dead_cell_enable_four_run,
                "Dead-cell D2 two-two split": self.dead_cell_enable_two_two_split,
                "Dead-cell D3 three-plus-one": self.dead_cell_enable_three_plus_one,
                "Dead-cell D3 strict adjacent opposite": (
                    self.dead_cell_three_plus_one_requires_adjacent_opposite
                ),
                "Dead-cell A1B2A3 discouraged": self.dead_cell_enable_a1b2a3_discouraged,
                "Dead-cell two-cell dead pairs": self.dead_cell_enable_double_dead_pairs,
                "Base MCTS move fraction": self.base_fraction_mcts_moves,
                "Early termination threshold": confidence_termination_threshold,
                "Temperature": temperature,
                "Temperature end": temperature_end,
            }

            if self.write_provenance:
                self.streaming_provenance_file = str(
                    sidecar_path_for_trmph(self.streaming_file)
                )
            if streaming_path.exists():
                existing_games = self._count_trmph_game_lines(self.streaming_file)
                self._streaming_games_written = existing_games
                if self.write_provenance:
                    if self.streaming_provenance_file is None:
                        raise RuntimeError(
                            "streaming_provenance_file is not initialized while write_provenance=True"
                        )
                    provenance_path = Path(self.streaming_provenance_file)
                    if not provenance_path.exists():
                        raise RuntimeError(
                            "Cannot append to existing streaming TRMPH file without "
                            f"matching provenance sidecar: {provenance_path}"
                        )
                    existing_provenance_records = self._count_nonempty_lines(
                        self.streaming_provenance_file
                    )
                    if existing_provenance_records != existing_games:
                        raise RuntimeError(
                            "Existing streaming TRMPH/provenance files are inconsistent "
                            f"and cannot be appended safely: {existing_games} game lines vs "
                            f"{existing_provenance_records} provenance records."
                        )
                if self.verbose >= 1:
                    print(
                        "Appending to existing streaming output: "
                        f"{self.streaming_file} (existing games: {existing_games})"
                    )
            else:
                write_trmph_header(
                    self.streaming_file,
                    "Self-play games",
                    metadata,
                    self.run_seed,
                    self.command_line,
                )
                if self.write_provenance:
                    if self.streaming_provenance_file is None:
                        raise RuntimeError(
                            "streaming_provenance_file is not initialized while write_provenance=True"
                        )
                    Path(self.streaming_provenance_file).parent.mkdir(
                        parents=True, exist_ok=True
                    )
                    with open(self.streaming_provenance_file, "w", encoding="utf-8"):
                        pass
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        if self.verbose >= 1:
            print("SelfPlayEngine initialized:")
            print(f"  Model: {model_path}")
            print(f"  Board size: {self.board_size}")
            print(f"  Cache size: {cache_size}")
            print(f"  Search method: hybrid policy+MCTS ({mcts_sims} simulations on MCTS moves)")
            print(f"  Base MCTS move fraction: {self.base_fraction_mcts_moves:.3f}")
            print(f"  C_PUCT: {c_puct}")
            print(f"  Gumbel root selection: {enable_gumbel}")
            print(f"  Dead-cell pruning: {self.enable_dead_cell_pruning}")
            if self.enable_dead_cell_pruning:
                print(f"    D1 four-run: {self.dead_cell_enable_four_run}")
                print(f"    D2 two-two split: {self.dead_cell_enable_two_two_split}")
                print(f"    D3 three-plus-one: {self.dead_cell_enable_three_plus_one}")
                print(
                    "    D3 strict adjacent opposite: "
                    f"{self.dead_cell_three_plus_one_requires_adjacent_opposite}"
                )
                print(f"    A1B2A3 discouraged: {self.dead_cell_enable_a1b2a3_discouraged}")
                print(f"    Two-cell dead pairs: {self.dead_cell_enable_double_dead_pairs}")
            print(f"  Early termination threshold: {confidence_termination_threshold}")
            print(f"  Temperature: {temperature} -> {temperature_end}")
            if enable_gumbel and self.mcts_sims <= self.mcts_config.gumbel_sim_threshold:
                print(
                    "  Note: Gumbel root selection uses fixed root temperature=1.0; "
                    "configured temperature applies to non-Gumbel visit-count sampling."
                )
            print(
                "  Virtual small-board sampling: "
                f"{self.small_board_fraction:.2%} on "
                f"{self.small_board_min_display_size}..{self.small_board_max_display_size} "
                "(linear larger-board preference)"
            )
            print(f"  Write provenance sidecar: {write_provenance}")
            print(f"  Verbose: {verbose}")
            

    def _validate_small_board_prefill_sequences(self) -> None:
        """Fail fast if configured virtual-prefill sequences are invalid or terminal."""
        for display_size, prefill_moves in self._small_board_prefill_moves_by_size.items():
            state = make_empty_hex_state(board_size=self.board_size)
            for move_index, move in enumerate(prefill_moves):
                row, col = move
                state = state.make_move(row, col)
                if state.game_over:
                    raise RuntimeError(
                        "Virtual prefill reached terminal state unexpectedly for "
                        f"display size {display_size} at move index {move_index}: "
                        f"{move}"
                    )

    def _should_use_mcts_for_move(self) -> bool:
        """Decide whether to run full MCTS for the next move."""
        if self.base_fraction_mcts_moves >= 1.0:
            return True
        if self.base_fraction_mcts_moves <= 0.0:
            return False
        return random.random() < self.base_fraction_mcts_moves

    def _select_policy_only_move(self, state) -> Tuple[Tuple[int, int], float]:
        """
        Select a move directly from the policy head for fast rollout moves.

        Returns:
            ((row, col), temperature_used)
        """
        move_idx = len(state.move_history)
        policy_temperature = calculate_mcts_root_temperature(
            move_count=move_idx,
            cfg=self.mcts_config,
            board_size=self.board_size,
        )
        move = select_policy_move(state, self.model, policy_temperature)
        return move, policy_temperature

    def _generate_single_game(
        self,
        board_size: int,
        opening_move: Optional[Tuple[int, int]] = None,
        game_id: Optional[int] = None,
        *,
        prefill_moves: Optional[Tuple[Tuple[int, int], ...]] = None,
        virtual_display_board_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate a single self-play game.
        
        Args:
            board_size: Size of the board
            opening_move: Optional opening move as (row, col) tuple
            game_id: Optional game ID for setting unique random seed
            prefill_moves: Optional virtual-board prefill move sequence applied before play
            virtual_display_board_size: Optional virtual display size metadata for diagnostics
            
        Returns:
            Dictionary containing game data with TRMPH string and winner
        """
        self._ensure_supported_board_size(board_size)

        # Set deterministic per-game seed for reproducible stochastic decisions.
        seed = self._compute_game_seed(game_id)
        
        # Set both Python and numpy random seeds to ensure MCTS uses the correct randomness
        random.seed(seed)
        np.random.seed(seed)
        
        if self.verbose >= 3:
            print(f"🎮 SELF-PLAY: Game {game_id} using seed {seed}")
            if virtual_display_board_size is not None:
                print(
                    "🎮 SELF-PLAY: Virtual display board "
                    f"{virtual_display_board_size}x{virtual_display_board_size}"
                )
        
        state = make_empty_hex_state(board_size=board_size)
        move_provenance_codes: List[str] = []

        if prefill_moves:
            if self.verbose >= 3:
                print(
                    "🎮 SELF-PLAY: Applying virtual-board prefill "
                    f"({len(prefill_moves)} moves)"
                )
            for prefill_index, (row, col) in enumerate(prefill_moves):
                if not (0 <= row < board_size and 0 <= col < board_size):
                    raise ValueError(
                        "Virtual prefill move is out of bounds for "
                        f"{board_size}x{board_size} board: {(row, col)}"
                    )
                state = state.make_move(row, col)
                if self.write_provenance:
                    move_provenance_codes.append(VIRTUAL_PREFILL_MOVE_PROVENANCE_CODE)
                if state.game_over:
                    raise RuntimeError(
                        "Virtual prefill reached terminal state unexpectedly at "
                        f"prefill index {prefill_index}: {(row, col)}"
                    )
        
        # Apply opening move if provided
        if opening_move is not None:
            row, col = opening_move
            if not (0 <= row < board_size and 0 <= col < board_size):
                raise ValueError(
                    f"Opening move {opening_move} is out of bounds for "
                    f"{board_size}x{board_size} board."
                )
            if self.verbose >= 3:
                trmph_move = rowcol_to_trmph(row, col, board_size=board_size)
                print(f"🎮 SELF-PLAY: Starting with opening move {trmph_move} ({row}, {col})")
            state = state.make_move(row, col)
            if self.write_provenance:
                # Opening moves are externally provided and have no MCTS source in schema v1.
                # We mark them as trainable visit-style moves to keep move-level alignment.
                move_provenance_codes.append(MOVE_CODE_VISIT_COUNT)
        
        if self.verbose >= 3:
            print(
                "🎮 SELF-PLAY: Starting new game with hybrid policy+MCTS "
                f"(mcts_sims={self.mcts_sims}, base_fraction={self.base_fraction_mcts_moves:.3f})"
            )
            print(f"🎮 SELF-PLAY: MCTS config - decay_type: {self.mcts_config.temperature_decay_type}, "
                  f"start_temp: {self.mcts_config.temperature_start}, "
                  f"end_temp: {self.mcts_config.temperature_end}")
        
        while not state.game_over:
            move_idx = len(state.move_history)
            legal_moves = state.get_legal_moves()
            if self.verbose >= 3:
                print(
                    f"🎮 SELF-PLAY: Move {move_idx}, player {state.current_player}, "
                    f"legal moves: {len(legal_moves)}"
                )

            use_mcts = self._should_use_mcts_for_move()
            if use_mcts:
                if self.verbose >= 3:
                    print(f"🎮 SELF-PLAY: Running MCTS with {self.mcts_sims} simulations")

                start_time = time.perf_counter()
                mcts_result = self.mcts.run(state)
                search_time = time.perf_counter() - start_time
                self._accumulate_mcts_run_stats(mcts_result.stats, search_time)

                if self.mcts_profile and self._mcts_profile_calls < self.mcts_profile_max_calls:
                    self._mcts_profile_calls += 1
                    if (self._mcts_profile_calls % self.mcts_profile_every) == 0:
                        try:
                            stats = mcts_result.stats or {}
                            h2d_ms = float(stats.get("h2d_ms", 0.0))
                            forward_ms = float(stats.get("forward_ms", 0.0))
                            d2h_ms = float(stats.get("d2h_ms", 0.0))
                            nn_ms = h2d_ms + forward_ms + d2h_ms
                            cpu_ms = 0.0
                            for k in (
                                "select_ms",
                                "encode_ms",
                                "stack_ms",
                                "expand_ms",
                                "backprop_ms",
                                "cache_lookup_ms",
                                "state_creation_ms",
                            ):
                                cpu_ms += float(stats.get(k, 0.0))
                            total_ms = nn_ms + cpu_ms
                            nn_pct = (nn_ms / total_ms * 100.0) if total_ms > 0 else 0.0
                            batch_sizes = stats.get("batch_sizes", []) or []
                            try:
                                bs = [float(x) for x in batch_sizes]
                            except Exception:
                                bs = []
                            avg_batch = (sum(bs) / max(1, len(bs))) if bs else 0.0
                            device = stats.get("device", None)
                            device_s = str(device) if device is not None else "unknown"
                            print(
                                f"[MCTS_PROFILE] device={device_s} move={move_idx} "
                                f"batches={int(stats.get('batch_count', 0))} avg_batch={avg_batch:.1f} "
                                f"NN_ms={nn_ms:.1f} CPU_ms={cpu_ms:.1f} NN%={nn_pct:.1f} "
                                f"search_time_s={search_time:.3f}"
                            )
                        except Exception as e:
                            print(f"[MCTS_PROFILE] failed to summarize stats: {e}")

                move = mcts_result.move
                if self.write_provenance:
                    move_provenance_codes.append(
                        self._get_move_provenance_code(mcts_result.stats)
                    )

                # Get root value (approximate from MCTS)
                tree_data = mcts_result.tree_data
                search_value = tree_data.get('v_curr_signed_root', 0.0)

                # Log MCTS statistics
                if self.verbose >= 2:
                    cache_hit_rate = self.mcts.cache_hits / max(
                        1, self.mcts.cache_hits + self.mcts.cache_misses
                    )
                    print(
                        f"[Move {move_idx}] MCTS: sims={self.mcts_sims}, "
                        f"inferences={mcts_result.stats.get('total_simulations', 0)}, "
                        f"cache_hit_rate={cache_hit_rate:.1%}, "
                        f"time={search_time:.4f}s"
                    )

                if self.verbose >= 3:
                    print(f"🎮 SELF-PLAY: Selected move {move}, value {search_value:.4f}")
            else:
                move, policy_temperature = self._select_policy_only_move(state)
                if self.write_provenance:
                    # Policy-only rollout moves are intentionally excluded from policy-target training.
                    move_provenance_codes.append(POLICY_ONLY_MOVE_PROVENANCE_CODE)
                if self.verbose >= 2:
                    print(
                        f"[Move {move_idx}] POLICY_ONLY: temp={policy_temperature:.3f}, "
                        f"legal_moves={len(legal_moves)}"
                    )
                if self.verbose >= 3:
                    print(f"🎮 SELF-PLAY: Selected policy-only move {move}")
            
            # Apply move
            state = state.make_move(*move)
            self.stats['total_moves'] += 1
        
        # Game data - TRMPH string and winner
        # Only handle enum case - fail fast on legacy values
        if not isinstance(state.winner, Winner):
            raise ValueError(f"Expected Winner enum, got: {state.winner!r} (type: {type(state.winner)})")
        
        if state.winner == Winner.RED:
            winner_char = TRMPH_RED_WIN
        elif state.winner == Winner.BLUE:
            winner_char = TRMPH_BLUE_WIN
        else:
            raise ValueError(f"Unexpected winner enum: {state.winner!r}")
        
        game_data = {
            'trmph': state.to_trmph(),
            'winner': winner_char
        }
        if virtual_display_board_size is not None:
            game_data['virtual_display_board_size'] = int(virtual_display_board_size)
        if self.write_provenance:
            game_data['move_provenance_codes'] = ''.join(move_provenance_codes)
            expected_move_count = count_trmph_moves(game_data['trmph'])
            if len(game_data['move_provenance_codes']) != expected_move_count:
                raise RuntimeError(
                    "Move provenance length mismatch for generated game. "
                    f"Expected {expected_move_count}, got {len(game_data['move_provenance_codes'])}."
                )
        
        if self.verbose >= 3:
            print(f"🎮 SELF-PLAY: Game complete, winner: {state.winner}, moves: {len(state.move_history)}")
        
        return game_data

    def _get_move_provenance_code(self, move_stats: Optional[Dict[str, Any]]) -> str:
        """Translate MCTS-selected move source into provenance code."""
        stats = move_stats or {}
        source_raw = stats.get("selected_move_source")
        source = str(source_raw).strip() if source_raw is not None else ""
        if not source:
            raise ValueError(
                "MCTS result missing selected_move_source; cannot build move provenance sidecar."
            )

        code = SELECTED_MOVE_SOURCE_TO_PROVENANCE_CODE.get(source)
        if code is None:
            raise ValueError(
                f"Unsupported selected_move_source {source!r}. "
                f"Expected one of {sorted(SELECTED_MOVE_SOURCE_TO_PROVENANCE_CODE.keys())}."
            )
        return code

    def _validate_game_data(self, game_data: Dict[str, Any], game_id: Optional[int] = None) -> None:
        """
        Validate game data structure and content.
        
        Args:
            game_data: Game data dictionary to validate
            game_id: Optional game ID for error reporting
            
        Raises:
            ValueError: If game data is invalid
        """
        if not isinstance(game_data, dict):
            raise ValueError(f"Game data must be a dictionary, got {type(game_data)}")
        
        required_keys = {'trmph', 'winner'}
        missing_keys = required_keys - set(game_data.keys())
        if missing_keys:
            raise ValueError(f"Game data missing required keys: {missing_keys}")
        
        winner = game_data.get('winner')
        trmph = game_data.get('trmph')
        
        # Use constants for winner validation and check for legacy values
        try:
            validate_trmph_winner(winner)
        except ValueError as e:
            game_info = f" (game {game_id})" if game_id is not None else ""
            raise ValueError(f"Invalid winner{game_info}: {e}")
        
        valid_winners = {TRMPH_RED_WIN, TRMPH_BLUE_WIN}
        if winner not in valid_winners:
            game_info = f" (game {game_id})" if game_id is not None else ""
            raise ValueError(f"Invalid winner{game_info}: {winner!r} (expected {TRMPH_RED_WIN!r} or {TRMPH_BLUE_WIN!r})")
        
        if not trmph or not isinstance(trmph, str):
            game_info = f" (game {game_id})" if game_id is not None else ""
            raise ValueError(f"Invalid TRMPH{game_info}: {trmph!r} (expected non-empty string)")
        
        if not trmph.startswith(TRMPH_PREFIX):
            game_info = f" (game {game_id})" if game_id is not None else ""
            raise ValueError(f"Invalid TRMPH format{game_info}: must start with {TRMPH_PREFIX!r}")

        move_count = count_trmph_moves(trmph)
        move_codes = game_data.get('move_provenance_codes')

        if self.write_provenance and move_codes is None:
            game_info = f" (game {game_id})" if game_id is not None else ""
            raise ValueError(
                f"Missing move_provenance_codes{game_info} while write_provenance=True"
            )

        if move_codes is not None:
            if not isinstance(move_codes, str):
                game_info = f" (game {game_id})" if game_id is not None else ""
                raise ValueError(
                    f"Invalid move_provenance_codes{game_info}: expected str, got {type(move_codes)}"
                )
            if len(move_codes) != move_count:
                game_info = f" (game {game_id})" if game_id is not None else ""
                raise ValueError(
                    f"Invalid move_provenance_codes length{game_info}: "
                    f"expected {move_count}, got {len(move_codes)}"
                )

    def _print_generation_progress(
        self,
        game_index: int,
        num_games: int,
        start_time: float,
        start_move_count: int,
        last_report_time: float,
        last_report_game_count: int,
        last_report_move_count: int,
        progress_interval: int,
    ) -> Tuple[float, int, int]:
        """
        Print periodic generation progress updates and return updated report state.

        Returns:
            Tuple of (last_report_time, last_report_game_count, last_report_move_count)
            where move counts are tracked relative to this generation call's start.
        """
        if (game_index + 1) % progress_interval == 0 or (game_index + 1) == num_games:
            now = time.time()
            elapsed = max(now - start_time, 1e-9)
            games_done = game_index + 1
            moves_done = max(0, int(self.stats['total_moves']) - int(start_move_count))

            cumulative_games_per_sec = games_done / elapsed
            cumulative_moves_per_sec = moves_done / elapsed

            window_elapsed = max(now - last_report_time, 1e-9)
            window_games = max(1, games_done - last_report_game_count)
            window_moves = max(0, moves_done - last_report_move_count)
            window_games_per_sec = window_games / window_elapsed
            window_moves_per_sec = window_moves / window_elapsed
            if self.verbose >= 1:
                print(
                    f"  Generated {games_done}/{num_games} games "
                    f"(cum: {cumulative_games_per_sec:.2f} games/s, "
                    f"last{window_games}: {window_games_per_sec:.2f} games/s, "
                    f"cum: {cumulative_moves_per_sec:.2f} moves/s, "
                    f"last{window_games}: {window_moves_per_sec:.2f} moves/s)"
                )
            return now, games_done, moves_done
        elif self.verbose >= 1:
            print(".", end="", flush=True)  # Progress dot for each game
        return last_report_time, last_report_game_count, last_report_move_count

    def _update_generation_stats(self, num_games: int, total_time: float) -> None:
        """Update process-level generation metrics."""
        self.stats['games_generated'] += num_games
        self.stats['total_time'] += total_time
        self.stats['games_per_second'] = num_games / total_time if total_time > 0 else 0

    def _generate_games_common(
        self,
        num_games: int,
        board_size: int,
        progress_interval: int,
        opening_strategy=None,
        *,
        collect_games: bool,
        stream_save: bool,
    ) -> Tuple[List[Dict[str, Any]], SelfPlayGenerationSummary, float]:
        """Run the shared self-play generation loop."""
        start_time = time.time()
        games: List[Dict[str, Any]] = []
        games_generated = 0
        red_wins = 0
        blue_wins = 0
        start_move_count = int(self.stats['total_moves'])
        last_report_time = start_time
        last_report_game_count = 0
        last_report_move_count = 0
        virtual_small_board_counts: Dict[int, int] = {}

        for i in range(num_games):
            virtual_display_board_size = self._sample_virtual_display_board_size(i)
            prefill_moves = None
            if virtual_display_board_size is not None:
                prefill_moves = self._small_board_prefill_moves_by_size[
                    virtual_display_board_size
                ]
                virtual_small_board_counts[virtual_display_board_size] = (
                    virtual_small_board_counts.get(virtual_display_board_size, 0) + 1
                )

            opening_move = None
            if opening_strategy is not None:
                opening_move = opening_strategy.get_opening_move(i)
                opening_move = self._maybe_filter_opening_for_virtual_display(
                    opening_move,
                    virtual_display_board_size,
                )

            game_data = self._generate_single_game(
                board_size,
                opening_move,
                game_id=i,
                prefill_moves=prefill_moves,
                virtual_display_board_size=virtual_display_board_size,
            )
            self._validate_game_data(game_data, i)
            winner = game_data['winner']
            if winner == TRMPH_RED_WIN:
                red_wins += 1
            elif winner == TRMPH_BLUE_WIN:
                blue_wins += 1
            else:
                raise ValueError(f"Unexpected winner while generating game {i}: {winner!r}")

            if collect_games:
                games.append(game_data)

            if stream_save:
                # Save immediately to avoid data loss on interruptions.
                self.save_game_to_stream(game_data)

            games_generated += 1
            (
                last_report_time,
                last_report_game_count,
                last_report_move_count,
            ) = self._print_generation_progress(
                i,
                num_games,
                start_time,
                start_move_count,
                last_report_time,
                last_report_game_count,
                last_report_move_count,
                progress_interval,
            )

        if self.verbose >= 1 and self.small_board_fraction > 0.0:
            print(
                self._format_small_board_sampling_description(
                    virtual_small_board_counts,
                    num_games,
                )
            )

        total_time = time.time() - start_time
        summary = SelfPlayGenerationSummary(
            num_games=games_generated,
            red_wins=red_wins,
            blue_wins=blue_wins,
        )
        return games, summary, total_time

    def generate_games_with_monitoring(
        self,
        num_games: int,
        board_size: Optional[int] = None,
        progress_interval: int = 10,
        opening_strategy=None,
    ) -> Tuple[List[Dict[str, Any]], SelfPlayGenerationSummary]:
        """
        Generate self-play games with monitoring and statistics.
        
        Args:
            num_games: Number of games to generate
            board_size: Optional board size override (must match engine board size)
            progress_interval: How often to print progress updates
            
        Returns:
            Tuple of generated games and typed generation summary
        """
        effective_board_size = self._resolve_generation_board_size(board_size)
        print(f"Generating {num_games} games...")

        games, summary, total_time = self._generate_games_common(
            num_games,
            effective_board_size,
            progress_interval,
            opening_strategy=opening_strategy,
            collect_games=True,
            stream_save=False,
        )
        self._update_generation_stats(summary.num_games, total_time)
        print(
            f"Generated {summary.num_games} games in {total_time:.1f}s "
            f"({self.stats['games_per_second']:.2f} games/s)"
        )
        return games, summary

    def generate_games_streaming(
        self,
        num_games: int,
        board_size: Optional[int] = None,
        progress_interval: int = 10,
        opening_strategy=None,
    ) -> SelfPlayGenerationSummary:
        """
        Generate games with streaming save to avoid data loss on interruption.
        
        Args:
            num_games: Number of games to generate
            board_size: Optional board size override (must match engine board size)
            progress_interval: How often to print progress updates
            
        Returns:
            Typed summary with game count, winner totals, and output paths
        """
        effective_board_size = self._resolve_generation_board_size(board_size)
        if not self.streaming_save:
            raise RuntimeError("Streaming save is not enabled. Set self.streaming_save=True to use generate_games_streaming.")
        
        print(f"Generating {num_games} games with streaming save...")
        _, summary, total_time = self._generate_games_common(
            num_games,
            effective_board_size,
            progress_interval,
            opening_strategy=opening_strategy,
            collect_games=False,
            stream_save=True,
        )
        self._update_generation_stats(summary.num_games, total_time)

        print(
            f"Generated {summary.num_games} games in {total_time:.1f}s "
            f"({self.stats['games_per_second']:.2f} games/s)"
        )
        print(f"Games saved to: {self.streaming_file}")
        if self.write_provenance and self.streaming_provenance_file:
            print(f"Move provenance sidecar: {self.streaming_provenance_file}")
        
        return summary.with_files(
            trmph_file=self.streaming_file,
            provenance_file=(
                self.streaming_provenance_file
                if self.write_provenance
                else None
            ),
        )

    def generate_games_with_opening_strategy(
        self,
        opening_strategy,
        num_games: int,
        board_size: Optional[int] = None,
        progress_interval: int = 10,
    ) -> Tuple[List[Dict[str, Any]], SelfPlayGenerationSummary]:
        """
        Generate self-play games using a specific opening strategy.
        
        Args:
            opening_strategy: OpeningStrategy instance that provides opening moves
            num_games: Number of games to generate
            board_size: Optional board size override (must match engine board size)
            progress_interval: How often to print progress updates
            
        Returns:
            Tuple of generated games and typed generation summary
        """
        effective_board_size = self._resolve_generation_board_size(board_size)
        print(f"Generating {num_games} games with opening strategy...")
        print(f"Strategy covers {opening_strategy.get_total_games()} games")

        games, summary, total_time = self._generate_games_common(
            num_games,
            effective_board_size,
            progress_interval,
            opening_strategy=opening_strategy,
            collect_games=True,
            stream_save=False,
        )
        self._update_generation_stats(summary.num_games, total_time)
        print(
            f"Generated {summary.num_games} games in {total_time:.1f}s "
            f"({self.stats['games_per_second']:.2f} games/s)"
        )
        return games, summary

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.stats.copy()
        total_moves = self._safe_int(stats.get('total_moves', 0))
        games_generated = self._safe_int(stats.get('games_generated', 0))
        total_time = self._safe_float(stats.get('total_time', 0.0))

        stats['moves_per_second'] = (
            total_moves / total_time if total_time > 0.0 else 0.0
        )
        stats['avg_moves_per_game'] = (
            total_moves / games_generated if games_generated > 0 else 0.0
        )

        # MCTS is the primary runtime path for self-play move generation.
        stats['mcts'] = self._build_mcts_summary_stats()

        # Keep optional SimpleModelInference diagnostics only when there is activity.
        model_stats = self.model.get_performance_stats()
        total_inferences = self._safe_int(model_stats.get("total_inferences", 0))
        total_batch_inferences = self._safe_int(model_stats.get("total_batch_inferences", 0))
        if total_inferences > 0 or total_batch_inferences > 0:
            stats['model'] = model_stats
        
        return stats

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """Best-effort numeric conversion for stats aggregation."""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return default
        if not np.isfinite(numeric):
            return default
        return numeric

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        """Best-effort integer conversion for stats aggregation."""
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _accumulate_mcts_run_stats(self, move_stats: Optional[Dict[str, Any]], search_time_s: float) -> None:
        """
        Accumulate per-move MCTS stats into process-level totals.

        This uses MCTS result stats as the primary source and falls back to wall-clock
        search time measurement when per-move timing is missing.
        """
        stats = move_stats or {}
        aggregate = self.mcts_run_stats

        aggregate['moves_searched'] += 1

        total_search_time_s = self._safe_float(
            stats.get('total_search_time', search_time_s),
            default=max(0.0, float(search_time_s)),
        )
        aggregate['total_search_time_s'] += max(0.0, total_search_time_s)

        aggregate['total_effective_simulations'] += max(
            0, self._safe_int(stats.get('effective_sims_total', 0))
        )
        aggregate['total_unique_evals'] += max(
            0, self._safe_int(stats.get('unique_evals_total', 0))
        )

        cache_hits = max(0, self._safe_int(stats.get('cache_hits', 0)))
        cache_misses = max(0, self._safe_int(stats.get('cache_misses', 0)))
        aggregate['cache_hits'] += cache_hits
        aggregate['cache_misses'] += cache_misses

        batch_count = max(0, self._safe_int(stats.get('batch_count', 0)))
        aggregate['batch_count'] += batch_count

        batch_sizes = stats.get('batch_sizes', []) or []
        if isinstance(batch_sizes, (list, tuple)):
            for batch_size in batch_sizes:
                aggregate['batch_size_sum'] += max(0.0, self._safe_float(batch_size, 0.0))

        reason_raw = stats.get('algorithm_termination_reason', 'unknown')
        reason = str(reason_raw).strip() if reason_raw is not None else ""
        if not reason:
            reason = 'unknown'
        counts = aggregate['algorithm_termination_reason_counts']
        counts[reason] = counts.get(reason, 0) + 1

    def _build_mcts_summary_stats(self) -> Dict[str, Any]:
        """Return normalized process-level MCTS summary stats."""
        aggregate = self.mcts_run_stats
        moves_searched = int(aggregate['moves_searched'])
        total_search_time_s = float(aggregate['total_search_time_s'])
        cache_hits = int(aggregate['cache_hits'])
        cache_misses = int(aggregate['cache_misses'])
        total_batches = int(aggregate['batch_count'])
        batch_size_sum = float(aggregate['batch_size_sum'])
        reason_counts = aggregate['algorithm_termination_reason_counts']
        sorted_reason_counts = dict(sorted(reason_counts.items()))

        return {
            'moves_searched': moves_searched,
            'total_search_time_s': total_search_time_s,
            'avg_search_time_s': total_search_time_s / max(1, moves_searched),
            'total_effective_simulations': int(aggregate['total_effective_simulations']),
            'total_unique_evals': int(aggregate['total_unique_evals']),
            'cache_hits': cache_hits,
            'cache_misses': cache_misses,
            'cache_hit_rate': cache_hits / max(1, cache_hits + cache_misses),
            'batch_count': total_batches,
            'avg_batch_size': batch_size_sum / max(1, total_batches),
            'algorithm_termination_reason_counts': sorted_reason_counts,
        }

    def print_mcts_run_summary(self) -> None:
        """Print aggregate MCTS runtime stats for this self-play process."""
        stats = self._build_mcts_summary_stats()
        print("\n=== Self-Play MCTS Summary ===")
        print(f"Moves searched: {stats['moves_searched']}")
        print(
            f"Search time: total={stats['total_search_time_s']:.2f}s "
            f"avg={stats['avg_search_time_s']:.4f}s"
        )
        print(f"Effective simulations: total={stats['total_effective_simulations']}")
        print(f"Unique evals: total={stats['total_unique_evals']}")
        print(
            f"Cache: hits={stats['cache_hits']} misses={stats['cache_misses']} "
            f"hit_rate={stats['cache_hit_rate']:.1%}"
        )
        print(
            f"Batches: total={stats['batch_count']} "
            f"avg_batch_size={stats['avg_batch_size']:.1f}"
        )
        reason_counts = stats['algorithm_termination_reason_counts']
        if reason_counts:
            reason_summary = ", ".join(
                f"{reason}={count}" for reason, count in reason_counts.items()
            )
        else:
            reason_summary = "none"
        print(f"Algorithm termination reasons: {reason_summary}")
        print("================================\n")

    def _simple_inference_summary_has_activity(self) -> bool:
        """
        Return whether SimpleModelInference tracked any actual inference work.

        Self-play move generation uses BaselineMCTS + ModelWrapper, so these counters are
        often zero in this code path.
        """
        model_stats = self.model.get_performance_stats()
        total_inferences = self._safe_int(model_stats.get("total_inferences", 0))
        total_batch_inferences = self._safe_int(model_stats.get("total_batch_inferences", 0))
        return total_inferences > 0 or total_batch_inferences > 0

    @staticmethod
    def _count_trmph_game_lines(file_path: str) -> int:
        """Count non-empty TRMPH game lines in a .trmph file."""
        count = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if line.startswith(TRMPH_PREFIX):
                    count += 1
        return count

    @staticmethod
    def _count_nonempty_lines(file_path: str) -> int:
        """Count non-empty lines in a UTF-8 text file."""
        count = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                if raw_line.strip():
                    count += 1
        return count

    def save_games_simple(self, games: List[Dict[str, Any]], base_filename: str) -> str:
        """
        Save-or-append games to a TRMPH text file and optional provenance sidecar.

        Args:
            games: List of game data dictionaries
            base_filename: Base filename (without extension)
            
        Returns:
            The TRMPH file path.

        Notes:
            - Creates new files (with headers) when none exist.
            - Appends games/records when files already exist.
            - Fails fast on inconsistent existing TRMPH/provenance pairs.
        """
        trmph_file = f"{base_filename}.trmph"
        trmph_path = Path(trmph_file)
        trmph_path.parent.mkdir(parents=True, exist_ok=True)

        trmph_exists = trmph_path.exists()
        existing_trmph_games = (
            self._count_trmph_game_lines(trmph_file) if trmph_exists else 0
        )
        trmph_mode = "a" if trmph_exists else "w"

        with open(trmph_file, trmph_mode, encoding="utf-8") as f:
            if not trmph_exists:
                git_info = get_git_commit_info()
                f.write(f"# Self-play games - {datetime.now().isoformat()}\n")
                f.write(f"# Model: {self.model_path}\n")
                f.write(f"# MCTS simulations: {self.mcts_sims}\n")
                f.write(f"# C_PUCT: {self.c_puct}\n")
                f.write(f"# Gumbel root selection: {self.enable_gumbel}\n")
                f.write(f"# Base MCTS move fraction: {self.base_fraction_mcts_moves}\n")
                f.write(f"# Temperature: {self.temperature}\n")
                f.write(f"# Temperature end: {self.temperature_end}\n")
                f.write(f"# Git commit: {git_info['status']}\n")
                f.write("# Format: trmph_string winner\n")

            for game in games:
                self._validate_game_data(game)
                f.write(f"{game['trmph']} {game['winner']}\n")

        if self.write_provenance:
            provenance_file = str(sidecar_path_for_trmph(trmph_file))
            provenance_path = Path(provenance_file)
            provenance_exists = provenance_path.exists()
            if trmph_exists and not provenance_exists:
                raise RuntimeError(
                    "Cannot append games: existing TRMPH file has no matching "
                    f"provenance sidecar ({provenance_file})."
                )
            if provenance_exists and not trmph_exists:
                raise RuntimeError(
                    "Found provenance sidecar without TRMPH file; refusing to append "
                    f"due to ambiguous run state: {provenance_file}"
                )

            existing_provenance_records = (
                self._count_nonempty_lines(provenance_file)
                if provenance_exists
                else 0
            )
            if trmph_exists and existing_provenance_records != existing_trmph_games:
                raise RuntimeError(
                    "Cannot append games: existing TRMPH/provenance pair is inconsistent "
                    f"({existing_trmph_games} game lines vs {existing_provenance_records} "
                    "provenance records)."
                )

            provenance_mode = "a" if provenance_exists else "w"
            with open(provenance_file, provenance_mode, encoding='utf-8') as f:
                for game_index, game in enumerate(games):
                    move_codes = game.get('move_provenance_codes')
                    if move_codes is None:
                        raise ValueError(
                            f"Missing move_provenance_codes for game index {game_index} while writing provenance"
                        )
                    record = make_move_provenance_record(
                        existing_provenance_records + game_index,
                        move_codes,
                    )
                    f.write(record.to_json_line())
                    f.write("\n")
        
        if self.verbose >= 1:
            if trmph_exists:
                print(
                    f"Appended {len(games)} games to {trmph_file} "
                    f"(existing={existing_trmph_games}, total={existing_trmph_games + len(games)})"
                )
            else:
                print(f"Saved {len(games)} games to {trmph_file}")
            if self.write_provenance:
                if trmph_exists:
                    print(
                        f"Appended move provenance sidecar records to {provenance_file} "
                        f"(existing={existing_provenance_records}, total={existing_provenance_records + len(games)})"
                    )
                else:
                    print(f"Saved move provenance sidecar to {provenance_file}")
        
        return trmph_file

    def save_game_to_stream(self, game_data: Dict[str, Any]):
        """Save a single game to the streaming file."""
        if not self.streaming_save:
            return
            
        self._validate_game_data(game_data)
        
        with open(self.streaming_file, 'a') as f:
            f.write(f"{game_data['trmph']} {game_data['winner']}\n")

        if self.write_provenance:
            if self.streaming_provenance_file is None:
                raise RuntimeError(
                    "streaming_provenance_file is not initialized while write_provenance=True"
                )
            move_codes = game_data.get('move_provenance_codes')
            if move_codes is None:
                raise ValueError(
                    "Missing move_provenance_codes while writing streaming provenance sidecar"
                )
            record = make_move_provenance_record(
                self._streaming_games_written, move_codes
            )
            with open(self.streaming_provenance_file, 'a', encoding='utf-8') as f:
                f.write(record.to_json_line())
                f.write("\n")
            self._streaming_games_written += 1

    def shutdown(self):
        """Clean shutdown of the engine."""
        print("Shutting down SelfPlayEngine...")
        if self._simple_inference_summary_has_activity():
            self.model.print_performance_summary()
        self.print_mcts_run_summary()
