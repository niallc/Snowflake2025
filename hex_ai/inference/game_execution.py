"""
Game execution utilities for tournaments.

This module provides functions for executing games between strategies,
including opening position generation and deterministic game play.
"""

import glob
import itertools
import json
import logging
import os
import random
import time
import warnings
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import gc

from hex_ai.config import (
    BOARD_SIZE,
    EMPTY_PIECE,
    POLICY_TARGET_CONSTRUCTION_VERSION,
    TRMPH_BLUE_WIN,
    TRMPH_RED_WIN,
    TRMPH_PREFIX,
)
from hex_ai.data_processing import parse_trmph_line_flexible
from hex_ai.enums import Player, Piece
from hex_ai.inference.game_engine import HexGameState, apply_move_to_state
from hex_ai.inference.move_selection import get_strategy, MoveSelectionConfig
from hex_ai.inference.strategy_config import StrategyConfig
from hex_ai.inference.tournament import TournamentResult as BaseTournamentResult
from hex_ai.inference.model_cache import create_temporary_model_cache
from hex_ai.move_provenance import (
    MOVE_CODE_CONFIDENCE_TERMINATION,
    MOVE_CODE_GUMBEL_ROOT,
    MOVE_CODE_TERMINAL_TERMINATION,
    MOVE_CODE_VISIT_COUNT,
)
from hex_ai.policy_target_construction import (
    build_policy_target_vector_from_gumbel_candidate_scores,
    build_policy_target_vector_from_mcts_result,
)
from hex_ai.utils.format_conversion import (
    rowcol_to_tensor_with_size,
    rowcol_to_trmph,
    trmph_to_moves,
)
from hex_ai.utils.tournament_logging import append_trmph_winner_line, write_tournament_trmph_header, find_available_csv_filename
from hex_ai.utils.deterministic_tournament_utils import (
    setup_tournament_output,
    save_opening_positions,
    setup_strategy_pair_files,
    create_play_config_for_pair,
    GameDuplicateTracker,
    play_strategy_pair_games,
    report_strategy_pair_results
)
from hex_ai.memory_profiler import get_profiler

logger = logging.getLogger(__name__)

# PyTorch can emit this warning from internal legacy distributed symbols even
# when project code does not call torch.distributed.reduce_op directly.
warnings.filterwarnings(
    "ignore",
    message=r"`torch\.distributed\.reduce_op` is deprecated, please use `torch\.distributed\.ReduceOp` instead",
    category=FutureWarning,
)

# Constants
DEFAULT_OPENING_LENGTH = 5
DEFAULT_TEMPERATURE = 0.0
DEFAULT_VERBOSE = 1
OUTPUT_DIR_PREFIX = "data/tournament_play/tournament_"
TRMPH_SOURCE_DIR = "data/sf25/sep28"
TRMPH_FILE_PATTERN = "*.trmph"

TOURNAMENT_NON_TRAINABLE_PROVENANCE_CODE = MOVE_CODE_CONFIDENCE_TERMINATION
TOURNAMENT_SELECTED_MOVE_SOURCE_TO_PROVENANCE_CODE = {
    "visit_counts": MOVE_CODE_VISIT_COUNT,
    "gumbel_root": MOVE_CODE_GUMBEL_ROOT,
    "neural_network_confidence": MOVE_CODE_CONFIDENCE_TERMINATION,
    "terminal_move": MOVE_CODE_TERMINAL_TERMINATION,
    # Non-selfplay strategy paths used by tournament wrappers.
    "policy_only": MOVE_CODE_CONFIDENCE_TERMINATION,
    "fixed_tree_search": MOVE_CODE_CONFIDENCE_TERMINATION,
}


class DeterministicTournamentResult(BaseTournamentResult):
    """Extended tournament result with timing tracking."""
    
    def __init__(self, participants: List[str]):
        super().__init__(participants)
        # Track timing data for each strategy
        self.strategy_timings = {name: 0.0 for name in participants}
        self.strategy_move_counts = {name: 0 for name in participants}
        # Memory note: This is a small amount of information per game, unlikely to be a significant memory issue.
        self.game_timings = []  # List of individual game timing data
        self.start_time = time.time()
        self.end_time: Optional[float] = None
    
    def record_game_with_timing(self, winner: str, loser: str, game_timing_data: Dict[str, Any]):
        """Record a game result with timing information."""
        # Record the basic game result
        self.record_game(winner, loser)
        
        # Record timing data
        strategy_timings = game_timing_data.get('strategy_timings', {})
        for strategy_name, time_taken in strategy_timings.items():
            if strategy_name in self.strategy_timings:
                self.strategy_timings[strategy_name] += time_taken
        
        # Record move counts
        total_moves = game_timing_data.get('total_moves', 0)
        for strategy_name in strategy_timings:
            if strategy_name in self.strategy_move_counts:
                self.strategy_move_counts[strategy_name] += total_moves
        
        # Store individual game timing data
        self.game_timings.append(game_timing_data)
    
    def get_timing_summary(self) -> Dict[str, Any]:
        """Get a summary of timing statistics."""
        summary = {}
        
        for strategy_name in self.participants:
            total_time = self.strategy_timings.get(strategy_name, 0.0)
            total_moves = self.strategy_move_counts.get(strategy_name, 0)
            
            summary[strategy_name] = {
                'total_time': total_time,
                'total_moves': total_moves,
                'avg_time_per_move': total_time / max(1, total_moves),
                'total_games': sum(1 for game in self.game_timings 
                                 if strategy_name in game.get('strategy_timings', {}))
            }
        
        return summary
    
    def print_timing_summary(self):
        """Print a formatted timing summary."""
        summary = self.get_timing_summary()
        
        print("\n" + "="*30)
        print("TIMING SUMMARY")
        print("="*30)
        
        # Sort strategies by total time
        sorted_strategies = sorted(summary.items(), key=lambda x: x[1]['total_time'], reverse=True)
        
        for strategy_name, stats in sorted_strategies:
            print(f"{strategy_name}:")
            print(f"  Total time: {stats['total_time']:.3f}s")
            print(f"  Total moves: {stats['total_moves']}")
            print(f"  Average time per move: {stats['avg_time_per_move']:.3f}s")
            print(f"  Games played: {stats['total_games']}")
            print()
        
        # Print overall tournament timing
        total_tournament_time = sum(stats['total_time'] for stats in summary.values())
        print(f"Total tournament time: {total_tournament_time:.3f}s")
        print("="*60)
    
    def finish(self):
        """Mark tournament as finished and record end time."""
        self.end_time = time.time()
    
    def get_duration(self) -> float:
        """Get tournament duration in seconds."""
        end_time = self.end_time or time.time()
        return end_time - self.start_time


class OpeningPosition:
    """Represents an opening position with moves and metadata."""
    
    def __init__(self, moves: List[Tuple[int, int]], source_game: str = "", 
                 opening_length: int = DEFAULT_OPENING_LENGTH):
        self.moves = moves
        self.source_game = source_game
        self.opening_length = opening_length
    
    def get_state(self, board_size: int = BOARD_SIZE) -> HexGameState:
        """Create a game state from this opening position."""
        # Initialize empty board
        board = np.full((board_size, board_size), EMPTY_PIECE, dtype='U1')
        state = HexGameState(board=board, _current_player=Player.BLUE)
        
        # Apply the opening moves
        for row, col in self.moves:
            state = apply_move_to_state(state, row, col)
        
        return state
    
    def get_trmph_string(self, board_size: int = BOARD_SIZE) -> str:
        """Get TRMPH representation of the opening moves."""
        trmph_moves = ''.join([rowcol_to_trmph(r, c, board_size) for r, c in self.moves])
        return f"{TRMPH_PREFIX}{trmph_moves}"
    
    def __str__(self) -> str:
        return f"Opening({len(self.moves)} moves from {self.source_game})"


def get_move_config_for_strategy(strategy_config: StrategyConfig, global_temperature: float = DEFAULT_TEMPERATURE) -> MoveSelectionConfig:
    """Create a MoveSelectionConfig for a strategy with specified temperature."""
    config_dict = strategy_config.config.copy()
    # Use strategy-specific temperature if available, otherwise use global temperature
    temperature = strategy_config.temperature if strategy_config.temperature is not None else global_temperature
    config_dict['temperature'] = temperature
    return MoveSelectionConfig(**config_dict)


def _normalize_opening_line(raw_line: str) -> Optional[str]:
    """
    Normalize an opening source line into a TRMPH payload, or return None to skip.

    Supported input lines:
    - TRMPH game/opening lines (e.g. "#13,...", optionally with winner)
    - "Opening N: #13,..." lines from saved opening lists
    - Header/comment lines beginning with "#" (except TRMPH "#13,") are skipped
    """
    line = raw_line.strip()
    if not line:
        return None

    # Saved openings files commonly use: "Opening 1: #13,..."
    if line.startswith("Opening "):
        if ":" not in line:
            raise ValueError(f"Invalid opening line format: {raw_line.rstrip()!r}")
        line = line.split(":", 1)[1].strip()
        if not line:
            return None

    # Skip human-readable comments/headers, but keep TRMPH lines like "#13,..."
    if line.startswith("#") and not line.startswith(TRMPH_PREFIX):
        return None

    return line


def load_openings_from_file(
    file_path: str,
    opening_length: int = DEFAULT_OPENING_LENGTH,
    *,
    max_openings: Optional[int] = None,
    require_winner: bool = False,
    strict: bool = True,
) -> List[OpeningPosition]:
    """
    Load opening positions from a file using a single parsing path.

    This accepts TRMPH files (with metadata headers), plain opening lists, and
    saved "Opening N: #13,..." files.

    Args:
        file_path: Path to input file
        opening_length: Number of moves to keep per opening
        max_openings: Optional cap on number of openings returned
        require_winner: If True, only keep lines with explicit winner indicators
        strict: If True, raise on malformed non-comment lines; if False, warn/skip

    Returns:
        List of OpeningPosition objects

    Raises:
        ValueError: If parsing fails in strict mode or no valid openings are found
    """
    openings: List[OpeningPosition] = []

    with open(file_path, 'r') as f:
        for line_num, raw_line in enumerate(f, 1):
            if max_openings is not None and len(openings) >= max_openings:
                break

            try:
                normalized = _normalize_opening_line(raw_line)
                if normalized is None:
                    continue

                trmph_string, winner_indicator = parse_trmph_line_flexible(normalized)
                if require_winner and winner_indicator is None:
                    continue

                moves = trmph_to_moves(trmph_string, BOARD_SIZE)
                if len(moves) < opening_length:
                    logger.warning(
                        "Skipping short opening in %s line %d: got %d moves, need %d",
                        file_path, line_num, len(moves), opening_length
                    )
                    continue

                opening_moves = moves[:opening_length]
                # Guard against malformed inputs with duplicate coordinates in one opening.
                if len(set(opening_moves)) != len(opening_moves):
                    logger.warning(
                        "Skipping opening with duplicate moves in %s line %d: %s",
                        file_path, line_num, opening_moves
                    )
                    continue

                source_game = f"{os.path.basename(file_path)}:line{line_num}"
                openings.append(OpeningPosition(opening_moves, source_game, opening_length))

            except Exception as e:
                if strict:
                    raise ValueError(f"Error parsing line {line_num} in {file_path}: {e}")
                logger.warning("Could not parse line %d in %s: %s", line_num, file_path, e)
                continue

    if not openings:
        raise ValueError(f"No valid openings found in {file_path}")

    return openings


def extract_openings_from_trmph_file(file_path: str, opening_length: int = DEFAULT_OPENING_LENGTH, 
                                   max_openings: int = 500) -> List[OpeningPosition]:
    """
    Extract diverse opening positions from a TRMPH file.
    
    Args:
        file_path: Path to TRMPH file
        opening_length: Number of moves to extract for each opening
        max_openings: Maximum number of openings to extract
    
    Returns:
        List of OpeningPosition objects
    """
    try:
        return load_openings_from_file(
            file_path=file_path,
            opening_length=opening_length,
            max_openings=max_openings,
            require_winner=True,
            strict=False,
        )
    except ValueError:
        # Non-strict parsing can still yield no valid openings; callers of this helper
        # expect an empty list in that case rather than an exception.
        return []


def find_trmph_files(source_dir: str) -> List[str]:
    """Find all TRMPH files in the source directory."""
    pattern = os.path.join(source_dir, TRMPH_FILE_PATTERN)
    files = glob.glob(pattern)
    logger.info(f"Found {len(files)} TRMPH files in {source_dir}")
    return sorted(files)


def generate_diverse_openings(trmph_files: List[str], opening_length: int = DEFAULT_OPENING_LENGTH,
                            target_count: int = 500, cache_file: str = None) -> List[OpeningPosition]:
    """
    Generate diverse opening positions from multiple TRMPH files.
    
    This function ensures uniqueness by checking each opening against previously
    collected ones before adding it to the list.
    
    Args:
        trmph_files: List of TRMPH file paths
        opening_length: Number of moves per opening
        target_count: Target number of openings to generate
        cache_file: Optional file to save/load openings for faster subsequent runs
    
    Returns:
        List of diverse OpeningPosition objects
    """
    # Try to load from cache first
    if cache_file and os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                if (cached_data.get('opening_length') == opening_length and 
                    len(cached_data.get('openings', [])) >= target_count):
                    logger.info(f"Loading {target_count} openings from cache: {cache_file}")
                    openings = []
                    for i, opening_data in enumerate(cached_data['openings'][:target_count]):
                        opening = OpeningPosition(
                            moves=opening_data['moves'],
                            source_game=opening_data['source'],
                            opening_length=opening_length
                        )
                        openings.append(opening)
                    return openings
        except Exception as e:
            logger.warning(f"Could not load cache file {cache_file}: {e}")
    
    logger.info(f"Generating {target_count} unique openings...")
    
    # Set to track unique opening move sequences
    unique_openings = set()
    diverse_openings = []
    
    # Process files until we have enough unique openings
    for file_path in trmph_files:
        if len(diverse_openings) >= target_count:
            break
            
        if not os.path.exists(file_path):
            continue
            
        logger.info(f"Processing {os.path.basename(file_path)}...")
        
        # Extract all openings from this file
        file_openings = extract_openings_from_trmph_file(
            file_path, opening_length, max_openings=1000  # Extract many to find unique ones
        )
        
        # Check each opening for uniqueness
        for opening in file_openings:
            if len(diverse_openings) >= target_count:
                break
                
            # Create a tuple of moves for comparison (tuples are hashable)
            moves_tuple = tuple(opening.moves)
            
            if moves_tuple not in unique_openings:
                unique_openings.add(moves_tuple)
                diverse_openings.append(opening)
                
                if len(diverse_openings) % 50 == 0:
                    logger.info(f"  Found {len(diverse_openings)} unique openings so far...")
    
    logger.info(f"Generated {len(diverse_openings)} unique openings from {len(trmph_files)} files")
    
    # Save to cache if requested
    if cache_file and diverse_openings:
        try:
            cache_data = {
                'opening_length': opening_length,
                'openings': [
                    {
                        'moves': opening.moves,
                        'source': opening.source_game
                    }
                    for opening in diverse_openings
                ]
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            logger.info(f"Saved {len(diverse_openings)} openings to cache: {cache_file}")
        except Exception as e:
            logger.warning(f"Could not save cache file {cache_file}: {e}")
    
    return diverse_openings


def select_random_openings(
    openings: List[OpeningPosition],
    num_openings: int,
    seed: Optional[int] = None
) -> List[OpeningPosition]:
    """
    Randomly select a subset of openings without replacement.

    Args:
        openings: Full opening pool
        num_openings: Number of openings requested
        seed: Optional seed for reproducible selection

    Returns:
        Selected opening subset
    """
    if num_openings >= len(openings):
        logger.info(f"Requested {num_openings} openings, returning all {len(openings)} available")
        return openings.copy()

    rng = random.Random(seed) if seed is not None else random
    selected_indices = rng.sample(range(len(openings)), num_openings)
    selected_openings = [openings[i] for i in selected_indices]

    logger.info(f"Randomly selected {len(selected_openings)} unique openings from pool of {len(openings)}")
    return selected_openings


def _zero_policy_target_vector(board_size: int) -> np.ndarray:
    """Return all-zero policy-target row for non-trainable moves."""
    return np.zeros(board_size * board_size, dtype=np.float32)


def _one_hot_policy_target_vector(move: Tuple[int, int], board_size: int) -> np.ndarray:
    """Return one-hot policy target row for a selected move."""
    row, col = move
    idx = rowcol_to_tensor_with_size(int(row), int(col), board_size)
    vec = np.zeros(board_size * board_size, dtype=np.float32)
    vec[idx] = 1.0
    return vec


def _build_policy_target_vector_from_mcts_result(
    mcts_result: Any, *, board_size: int
) -> np.ndarray:
    """Build dense policy-target row from MCTS root visit distribution."""
    return build_policy_target_vector_from_mcts_result(
        mcts_result,
        board_size=board_size,
    )


def _build_policy_target_vector_from_gumbel_candidate_scores(
    mcts_result: Any,
    *,
    board_size: int,
    gumbel_c_visit: float,
    gumbel_c_scale: float,
) -> np.ndarray:
    """Build v2 dense policy target from the full Gumbel top-m candidate set."""
    return build_policy_target_vector_from_gumbel_candidate_scores(
        mcts_result,
        board_size=board_size,
        gumbel_c_visit=gumbel_c_visit,
        gumbel_c_scale=gumbel_c_scale,
    )


def _build_policy_target_vector_from_gumbel_final_scores(
    mcts_result: Any, *, board_size: int
) -> np.ndarray:
    """
    Legacy v1 Gumbel policy target helper retained for debugging/comparison.

    Build dense policy target from final noise-free Gumbel ranking scores.

    Contract:
    - Uses `stats["gumbel_final_rank_rows"]` entries (tensor_action + score_without_gumbel).
    - Applies softmax over the scored action set only.
    - Leaves all non-scored legal actions at probability 0.
    - Enforces that the top score action is exactly the selected move.
    """
    stats = getattr(mcts_result, "stats", {}) or {}
    rows = stats.get("gumbel_final_rank_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(
            "Gumbel move is missing gumbel_final_rank_rows in MCTS stats; "
            "cannot build Gumbel-specific policy target."
        )

    action_count = board_size * board_size
    action_indices: List[int] = []
    raw_scores: List[float] = []
    seen_indices: set[int] = set()

    for row in rows:
        if not isinstance(row, dict):
            raise TypeError(
                f"gumbel_final_rank_rows entries must be dict, got {type(row)}"
            )
        action_raw = row.get("tensor_action")
        score_raw = row.get("score_without_gumbel")
        if action_raw is None or score_raw is None:
            raise ValueError(
                "gumbel_final_rank_rows entries must include tensor_action and score_without_gumbel"
            )
        action_idx = int(action_raw)
        if action_idx < 0 or action_idx >= action_count:
            raise ValueError(
                f"Invalid tensor_action {action_idx} for board size {board_size}"
            )
        if action_idx in seen_indices:
            raise ValueError(
                f"Duplicate tensor_action {action_idx} in gumbel_final_rank_rows"
            )
        score = float(score_raw)
        if not np.isfinite(score):
            raise ValueError(
                f"Non-finite score_without_gumbel for tensor_action {action_idx}: {score_raw!r}"
            )
        seen_indices.add(action_idx)
        action_indices.append(action_idx)
        raw_scores.append(score)

    selected_move = mcts_result.move
    selected_idx = rowcol_to_tensor_with_size(
        int(selected_move[0]), int(selected_move[1]), board_size
    )
    top_idx = action_indices[int(np.argmax(np.asarray(raw_scores, dtype=np.float64)))]
    if selected_idx != top_idx:
        raise RuntimeError(
            "Gumbel target construction mismatch: selected move is not the top noise-free "
            f"Gumbel score action (selected={selected_idx}, top={top_idx})."
        )

    scores_arr = np.asarray(raw_scores, dtype=np.float64)
    max_score = float(np.max(scores_arr))
    exp_scores = np.exp(scores_arr - max_score)
    exp_sum = float(np.sum(exp_scores))
    if exp_sum <= 0.0 or not np.isfinite(exp_sum):
        raise RuntimeError(
            "Invalid Gumbel final-score normalization (non-positive/invalid softmax denominator)."
        )
    probs = exp_scores / exp_sum

    vec = np.zeros(action_count, dtype=np.float32)
    for action_idx, prob in zip(action_indices, probs):
        vec[action_idx] = float(prob)

    total = float(vec.sum())
    if total <= 0.0:
        raise RuntimeError("Gumbel policy target vector has zero mass.")
    if not np.isclose(total, 1.0, atol=1e-6):
        vec /= total
    return vec


def _resolve_provenance_code_from_selected_move_source(source_raw: Any) -> str:
    source = str(source_raw).strip() if source_raw is not None else ""
    if not source:
        raise ValueError("Missing selected_move_source while building tournament policy targets.")
    code = TOURNAMENT_SELECTED_MOVE_SOURCE_TO_PROVENANCE_CODE.get(source)
    if code is None:
        raise ValueError(
            f"Unsupported selected_move_source {source!r}. "
            f"Expected one of {sorted(TOURNAMENT_SELECTED_MOVE_SOURCE_TO_PROVENANCE_CODE.keys())}."
        )
    return code


def play_deterministic_game(
    model_cache,
    strategy_a: StrategyConfig,
    strategy_b: StrategyConfig,
    opening: OpeningPosition,
    temperature: float = DEFAULT_TEMPERATURE,
    board_size: int = BOARD_SIZE,
    verbose: int = 0,
    strategy_a_is_blue: bool = True
) -> Dict[str, Any]:
    """
    Play a deterministic game from an opening position.
    
    Args:
        model_cache: Model cache to get models for each strategy
        strategy_a: Strategy configuration for player A
        strategy_b: Strategy configuration for player B
        opening: Opening position to start from
        temperature: Temperature for move selection (0.0 = deterministic)
        board_size: Board size for the game
        verbose: Verbosity level
        strategy_a_is_blue: Whether strategy_a plays as Blue (True) or Red (False)
    
    Returns:
        Dictionary with game results including timing information
    """
    # Initialize timing tracking
    strategy_timings = {
        strategy_a.name: 0.0,
        strategy_b.name: 0.0
    }
    move_count = 0
    
    # Start from the opening position
    state = opening.get_state(board_size)
    
    # Create strategy configurations with specified temperature
    config_a = get_move_config_for_strategy(strategy_a, temperature)
    config_b = get_move_config_for_strategy(strategy_b, temperature)
    
    # Get strategy objects
    strategy_a_obj = get_strategy(strategy_a.strategy_type)
    strategy_b_obj = get_strategy(strategy_b.strategy_type)
    
    # Play the game from the opening position
    move_sequence = list(opening.moves)  # Start with opening moves
    move_provenance_codes: List[str] = []
    policy_target_rows: List[np.ndarray] = []
    for opening_move in opening.moves:
        move_provenance_codes.append(MOVE_CODE_VISIT_COUNT)
        policy_target_rows.append(
            _one_hot_policy_target_vector(opening_move, board_size)
        )
    
    logger.debug(f"Starting game: {strategy_a.name} vs {strategy_b.name}")
    logger.debug(f"Opening moves: {opening.moves}")
    logger.debug(f"Initial state current player: {state.current_player_enum}")
    logger.debug(f"Strategy A is Blue: {strategy_a_is_blue}")
    
    while not state.game_over:
        # Determine which strategy to use based on current player and color assignment
        current_player = state.current_player_enum
        if current_player == Player.BLUE:
            if strategy_a_is_blue:
                strategy_obj = strategy_a_obj
                strategy_config = config_a
                strategy_name = strategy_a.name
                model = model_cache.get_simple_model(strategy_a.model_path)
            else:
                strategy_obj = strategy_b_obj
                strategy_config = config_b
                strategy_name = strategy_b.name
                model = model_cache.get_simple_model(strategy_b.model_path)
        else:  # Player.RED
            if strategy_a_is_blue:
                strategy_obj = strategy_b_obj
                strategy_config = config_b
                strategy_name = strategy_b.name
                model = model_cache.get_simple_model(strategy_b.model_path)
            else:
                strategy_obj = strategy_a_obj
                strategy_config = config_a
                strategy_name = strategy_a.name
                model = model_cache.get_simple_model(strategy_a.model_path)
    
        # Time the move selection
        start_time = time.perf_counter()
        move = strategy_obj.select_move(state, model, strategy_config, verbose=verbose)
        end_time = time.perf_counter()
        
        if move is None:
            raise ValueError(
                f"Move selection returned None for {strategy_obj.get_name()}. "
                f"This indicates a bug in the strategy implementation. "
                f"Please check the strategy code and ensure it always returns a valid move."
            )
        
        # Record timing for this strategy
        move_time = end_time - start_time
        strategy_timings[strategy_name] += move_time
        move_count += 1

        move_metadata = strategy_obj.pop_last_move_metadata()
        if move_metadata is None:
            # Conservative fallback for strategy implementations that do not expose
            # per-move diagnostics: keep row non-trainable.
            provenance_code = TOURNAMENT_NON_TRAINABLE_PROVENANCE_CODE
            policy_target = _zero_policy_target_vector(board_size)
        else:
            provenance_code = _resolve_provenance_code_from_selected_move_source(
                move_metadata.get("selected_move_source")
            )
            if provenance_code == MOVE_CODE_GUMBEL_ROOT:
                mcts_result = move_metadata.get("mcts_result")
                if mcts_result is None:
                    raise RuntimeError(
                        "Gumbel-root tournament move missing MCTS result payload."
                    )
                policy_target = _build_policy_target_vector_from_gumbel_candidate_scores(
                    mcts_result,
                    board_size=board_size,
                    gumbel_c_visit=float(strategy_config.gumbel_c_visit),
                    gumbel_c_scale=float(strategy_config.gumbel_c_scale),
                )
            elif provenance_code in {
                MOVE_CODE_VISIT_COUNT,
                MOVE_CODE_TERMINAL_TERMINATION,
            }:
                mcts_result = move_metadata.get("mcts_result")
                if mcts_result is None:
                    raise RuntimeError(
                        "Trainable tournament move missing MCTS result payload."
                    )
                policy_target = _build_policy_target_vector_from_mcts_result(
                    mcts_result, board_size=board_size
                )
            elif provenance_code == MOVE_CODE_CONFIDENCE_TERMINATION:
                policy_target = _zero_policy_target_vector(board_size)
            else:
                raise RuntimeError(
                    f"Unsupported tournament provenance code: {provenance_code!r}"
                )

        move_provenance_codes.append(provenance_code)
        policy_target_rows.append(policy_target)
        
        logger.debug(f"Player {current_player.name} ({strategy_name}) plays move {move} in {move_time:.3f}s")
        
        # Apply move
        move_sequence.append(move)
        state = apply_move_to_state(state, *move)
        
        if verbose >= 2:
            print("-", end="", flush=True)
    
    # Convert to TRMPH format
    trmph_moves = ''.join([rowcol_to_trmph(r, c, board_size) for r, c in move_sequence])
    trmph_str = f"{TRMPH_PREFIX}{trmph_moves}"
    
    # Determine winner
    winner_enum = state.winner_enum
    if winner_enum is None:
        raise ValueError(
            "Game is not over or winner missing. "
            "This indicates a bug in the game engine. "
            "Please check the game state and ensure the game has properly ended."
        )
    
    if winner_enum.name == 'BLUE':
        winner_strategy = strategy_a.name if strategy_a_is_blue else strategy_b.name
        winner_char = TRMPH_BLUE_WIN
    elif winner_enum.name == 'RED':
        winner_strategy = strategy_b.name if strategy_a_is_blue else strategy_a.name
        winner_char = TRMPH_RED_WIN
    else:
        raise ValueError(f"Unknown winner enum: {winner_enum}")
    
    logger.debug(f"Game complete: {winner_strategy} wins with {len(move_sequence)} moves")
    logger.debug(f"Final TRMPH: {trmph_str}")
    logger.debug(f"Timing summary: {strategy_a.name}={strategy_timings[strategy_a.name]:.3f}s, {strategy_b.name}={strategy_timings[strategy_b.name]:.3f}s")

    expected_move_count = len(move_sequence)
    if len(move_provenance_codes) != expected_move_count:
        raise RuntimeError(
            "Tournament move provenance length mismatch: "
            f"expected {expected_move_count}, got {len(move_provenance_codes)}"
        )
    if len(policy_target_rows) != expected_move_count:
        raise RuntimeError(
            "Tournament policy-target row count mismatch: "
            f"expected {expected_move_count}, got {len(policy_target_rows)}"
        )
    if policy_target_rows:
        policy_targets_matrix = np.stack(policy_target_rows, axis=0).astype(
            np.float32, copy=False
        )
    else:
        policy_targets_matrix = np.zeros(
            (0, board_size * board_size), dtype=np.float32
        )
    
    return {
        'winner_strategy': winner_strategy,
        'winner_char': winner_char,
        'trmph_str': trmph_str,
        'move_sequence': move_sequence,
        'num_moves': len(move_sequence),
        'opening': opening,
        'strategy_timings': strategy_timings,
        'total_moves': move_count,
        'move_provenance_codes': ''.join(move_provenance_codes),
        'policy_targets_matrix': policy_targets_matrix,
        'policy_target_source_codes': ''.join(move_provenance_codes),
        'policy_target_version': POLICY_TARGET_CONSTRUCTION_VERSION,
    }


def run_round_robin_tournament(
    strategy_configs: List[StrategyConfig],
    openings: List[OpeningPosition],
    temperature: float = DEFAULT_TEMPERATURE,
    verbose: int = DEFAULT_VERBOSE,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
    command_line: str = None,
    run_desc: Optional[str] = None,
    mps_empty_cache_per_pair: bool = False,
) -> DeterministicTournamentResult:
    """
    Run a round-robin tournament using pre-generated opening positions.
    
    This function executes a round-robin tournament where each strategy plays
    against every other strategy using the same set of opening positions.
    
    Args:
        strategy_configs: List of strategy configurations (each with its own model)
        openings: List of opening positions to use
        temperature: Temperature for move selection (0.0 = deterministic)
        verbose: Verbosity level
        seed: Random seed for reproducibility (default: None, uses time-based seed)
        output_dir: Output directory for tournament files (default: None, auto-generated)
        command_line: Command line that was used to run the tournament
        run_desc: Optional description of this tournament run (e.g., "Testing c_scale = 1.5")
    
    Returns:
        DeterministicTournamentResult with results
    """
    # TODO: Add progress tracking and resume functionality
    # TODO: Add parallel processing for multiple strategy pairs
    # TODO: Add memory usage monitoring for large tournaments
    # TODO: Consider adding early termination if one strategy dominates
    
    # Create tournament result tracking strategy names
    # Use unique strategy names for tournament tracking (after parameter modifications)
    unique_strategy_names = [config.name for config in strategy_configs]
    result = DeterministicTournamentResult(unique_strategy_names)
        
    # Set up tournament output using utilities
    if output_dir is None:
        output_dir, openings_file = setup_tournament_output(OUTPUT_DIR_PREFIX)
    else:
        # Use provided output directory
        os.makedirs(output_dir, exist_ok=True)
        openings_file = os.path.join(output_dir, "openings.txt")
    save_opening_positions(openings, openings_file)
    
    # Initialize game duplicate tracker
    duplicate_tracker = GameDuplicateTracker()
    
    # Run round-robin between all strategy pairs
    for strategy_a, strategy_b in itertools.combinations(strategy_configs, 2):
        profiler = get_profiler()
        if profiler is not None:
            profiler.log_measurement(label=f"pair_start:{strategy_a.name}_vs_{strategy_b.name}")

        openings_per_pair = len(openings)
        games_per_pair = openings_per_pair * 2  # Each opening is played twice with swapped colors.
        logger.info(
            f"\nPlaying {games_per_pair} games ({openings_per_pair} openings x 2 colors): "
            f"{strategy_a.name} vs {strategy_b.name}"
        )
        
        # Load models temporarily for this match only (keeps peak memory lower).
        match_model_paths = [strategy_a.model_path, strategy_b.model_path]
        model_cache = create_temporary_model_cache(match_model_paths, verbose=0)

        # Best-effort census: counts of live model objects (helps distinguish true retention vs allocator high-water).
        if profiler is not None:
            try:
                from hex_ai.inference.simple_model_inference import SimpleModelInference
                from hex_ai.inference.model_wrapper import ModelWrapper
                n_simple = 0
                n_wrapper = 0
                for o in gc.get_objects():
                    if isinstance(o, SimpleModelInference):
                        n_simple += 1
                    elif isinstance(o, ModelWrapper):
                        n_wrapper += 1
                profiler.log_object_census(
                    {"live_simple_model_inference": n_simple, "live_model_wrapper": n_wrapper},
                    label=f"after_model_load:{strategy_a.name}_vs_{strategy_b.name}",
                )
            except Exception:
                # Diagnostics should never break tournament execution.
                pass
        
        # Set up output files for this strategy pair
        trmph_file, csv_file = setup_strategy_pair_files(output_dir, strategy_a, strategy_b)
        
        # Create play configuration
        play_config = create_play_config_for_pair(strategy_a, strategy_b, temperature, seed, command_line, run_desc)
        
        # Write TRMPH header
        pair_model_paths = [strategy_a.model_path, strategy_b.model_path]
        pair_strategy_configs = [strategy_a, strategy_b]
        actual_trmph_file = write_tournament_trmph_header(
            trmph_file, pair_model_paths, games_per_pair, play_config, BOARD_SIZE,
            strategy_configs=pair_strategy_configs
        )
        
        # Find available CSV filename
        actual_csv_file = find_available_csv_filename(csv_file)
        
        # Play all games for this strategy pair using utility function
        game_results = play_strategy_pair_games(
            model_cache, strategy_a, strategy_b, openings, temperature, verbose,
            duplicate_tracker, actual_trmph_file, actual_csv_file, play_deterministic_game, result
        )
        
        # Report results for this pair
        print()  # Add line break before match summary
        report_strategy_pair_results(verbose, strategy_a, strategy_b, result, duplicate_tracker)
        
        # Clean up temporary models to free memory
        # The temporary models will be garbage collected when this iteration ends
        logger.debug(f"Cleaning up temporary models for match: {strategy_a.name} vs {strategy_b.name}")

        # Diagnostic only: on MPS, empty the backend cache between pairs to test allocator behavior.
        if mps_empty_cache_per_pair:
            try:
                import torch
                if torch.backends.mps.is_available() and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
            except Exception:
                pass

        if profiler is not None:
            profiler.log_measurement(label=f"pair_end:{strategy_a.name}_vs_{strategy_b.name}")
            try:
                from hex_ai.inference.simple_model_inference import SimpleModelInference
                from hex_ai.inference.model_wrapper import ModelWrapper
                n_simple = 0
                n_wrapper = 0
                for o in gc.get_objects():
                    if isinstance(o, SimpleModelInference):
                        n_simple += 1
                    elif isinstance(o, ModelWrapper):
                        n_wrapper += 1
                profiler.log_object_census(
                    {"live_simple_model_inference": n_simple, "live_model_wrapper": n_wrapper},
                    label=f"after_pair_end:{strategy_a.name}_vs_{strategy_b.name}",
                )
            except Exception:
                pass
    
    logger.info(f"Tournament complete. Total unique games played: {len(duplicate_tracker.seen_games)}")
    return result
