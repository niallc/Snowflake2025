"""
Self-play engine for generating training data using the Hex AI model.
"""

import logging
import numpy as np
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hex_ai.config import TRMPH_BLUE_WIN, TRMPH_PREFIX, TRMPH_RED_WIN, DEFAULT_C_PUCT, DEFAULT_MCTS_SIMS, DEFAULT_CACHE_SIZE, BOARD_SIZE, DEFAULT_TEMPERATURE_START, DEFAULT_TEMPERATURE_END
from hex_ai.enums import Winner
from hex_ai.inference.game_engine import HexGameEngine, HexGameState, make_empty_hex_state
from hex_ai.inference.mcts import BaselineMCTS, BaselineMCTSConfig, create_mcts_config
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
from hex_ai.system_utils import get_git_commit_info
from hex_ai.training_utils import get_device
from hex_ai.utils.format_conversion import count_trmph_moves, rowcol_to_trmph
from hex_ai.utils.tournament_logging import write_trmph_header
from hex_ai.value_utils import validate_trmph_winner

DEFAULT_SELFPLAY_CONFIDENCE_TERMINATION_THRESHOLD = 0.85
SELECTED_MOVE_SOURCE_TO_PROVENANCE_CODE = {
    "visit_counts": MOVE_CODE_VISIT_COUNT,
    "gumbel_root": MOVE_CODE_GUMBEL_ROOT,
    "neural_network_confidence": MOVE_CODE_CONFIDENCE_TERMINATION,
    "terminal_move": MOVE_CODE_TERMINAL_TERMINATION,
}


class SelfPlayEngine:
    """High-performance self-play engine with optimized inference and logging."""
    
    def __init__(self, model_path: str, batch_size: int = 32, 
                 cache_size: int = DEFAULT_CACHE_SIZE, temperature: float = DEFAULT_TEMPERATURE_START, temperature_end: float = DEFAULT_TEMPERATURE_END, 
                 verbose: int = 1, streaming_save: bool = False, streaming_file: str = None,
                 use_batched_inference: bool = True, output_dir: str = None,
                 mcts_sims: int = DEFAULT_MCTS_SIMS, c_puct: float = DEFAULT_C_PUCT, enable_gumbel: bool = True,
                 confidence_termination_threshold: float = DEFAULT_SELFPLAY_CONFIDENCE_TERMINATION_THRESHOLD,
                 write_provenance: bool = True,
                 command_line: str = None,
                 mcts_profile: bool = False,
                 mcts_profile_every: int = 10,
                 mcts_profile_max_calls: int = 50):
        
        # Generate a unique run seed based on current time
        self.run_seed = int(time.time() * 1000000) % (2**32)
        if verbose >= 1:
            print(f"SelfPlayEngine run seed: {self.run_seed}")
        """
        Initialize the self-play engine.
        
        Args:
            model_path: Path to the model checkpoint
            batch_size: Batch size for inference
            cache_size: Size of the LRU cache
            temperature: Starting temperature for move sampling
            temperature_end: Final temperature for move sampling (for decay)
            verbose: Verbosity level (0=quiet, 1=normal, 2=detailed)
            streaming_save: Save games incrementally to avoid data loss
            streaming_file: File path for streaming save (auto-generated if None)
            use_batched_inference: Whether to use batched inference for better performance
            output_dir: Output directory for streaming files (used if streaming_file is None)
            mcts_sims: Number of MCTS simulations per move
            c_puct: PUCT exploration constant for MCTS
            enable_gumbel: Enable Gumbel-AlphaZero root selection for MCTS
            confidence_termination_threshold: Early-termination confidence threshold
            write_provenance: Whether to write move-provenance sidecar data
        """
        self.model_path = model_path
        self.batch_size = batch_size
        self.cache_size = cache_size
        self.temperature = temperature
        self.temperature_end = temperature_end
        self.verbose = verbose
        self.streaming_save = streaming_save
        self.streaming_file = streaming_file
        self.use_batched_inference = use_batched_inference
        self.output_dir = output_dir
        self.mcts_sims = mcts_sims
        self.c_puct = c_puct
        self.enable_gumbel = enable_gumbel
        self.confidence_termination_threshold = confidence_termination_threshold
        self.write_provenance = write_provenance
        self.command_line = command_line
        self.mcts_profile = mcts_profile
        self.mcts_profile_every = mcts_profile_every
        self.mcts_profile_max_calls = mcts_profile_max_calls
        self._mcts_profile_calls = 0
        self.streaming_provenance_file: Optional[str] = None
        self._streaming_games_written = 0
        
        # Initialize model
        self.model = SimpleModelInference(model_path, device=get_device(), cache_size=cache_size)
        
        # Initialize MCTS components
        self.game_engine = HexGameEngine()
        # Create ModelWrapper for MCTS
        self.model_wrapper = ModelWrapper(model_path, device=get_device())
        # Create MCTS configuration optimized for self-play with confidence termination
        self.mcts_config = create_mcts_config("selfplay",
            sims=self.mcts_sims,
            # Aggressive confidence termination for speed.
            confidence_termination_threshold=self.confidence_termination_threshold,
            cache_size=self.cache_size,  # Use same cache size as SimpleModelInference
            c_puct=self.c_puct,  # Use specified PUCT exploration constant
            enable_gumbel_root_selection=self.enable_gumbel  # Enable/disable Gumbel root selection
        )
        
        # Performance tracking
        self.stats = {
            'total_inferences': 0,
            'total_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
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
            os.makedirs(os.path.dirname(self.streaming_file), exist_ok=True)
            # Write header using generic function
            metadata = {
                "Model": model_path,
                "MCTS simulations": mcts_sims,
                "C_PUCT": c_puct,
                "Gumbel root selection": enable_gumbel,
                "Early termination threshold": confidence_termination_threshold,
                "Temperature": temperature,
            }
            write_trmph_header(self.streaming_file, "Self-play games", metadata, self.run_seed, self.command_line)
            if self.write_provenance:
                self.streaming_provenance_file = str(
                    sidecar_path_for_trmph(self.streaming_file)
                )
                Path(self.streaming_provenance_file).parent.mkdir(
                    parents=True, exist_ok=True
                )
                with open(self.streaming_provenance_file, "w", encoding="utf-8"):
                    pass
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        if self.verbose >= 1:
            print(f"SelfPlayEngine initialized:")
            print(f"  Model: {model_path}")
            print(f"  Batch size: {batch_size}")
            print(f"  Cache size: {cache_size}")
            print(f"  Search method: MCTS ({mcts_sims} simulations)")
            print(f"  C_PUCT: {c_puct}")
            print(f"  Gumbel root selection: {enable_gumbel}")
            print(f"  Early termination threshold: {confidence_termination_threshold}")
            print(f"  Temperature: {temperature} -> {temperature_end}")
            print(f"  Write provenance sidecar: {write_provenance}")
            print(f"  Verbose: {verbose}")
            print(f"  Batched inference: {use_batched_inference}")
            


    def _generate_single_game(self, board_size: int, opening_move: Optional[Tuple[int, int]] = None, game_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a single self-play game.
        
        Args:
            board_size: Size of the board (ignored, always uses 13)
            opening_move: Optional opening move as (row, col) tuple
            game_id: Optional game ID for setting unique random seed
            
        Returns:
            Dictionary containing game data with TRMPH string and winner
        """
        # Set unique random seed for this game to ensure diversity
        if game_id is not None:
            # Combine run seed with game_id to ensure uniqueness across runs
            seed = self.run_seed + game_id * 1000
        else:
            # Use time-based seed for uniqueness
            seed = int(time.time() * 1000000) % (2**32)
        
        # Set both Python and numpy random seeds to ensure MCTS uses the correct randomness
        random.seed(seed)
        np.random.seed(seed)
        
        if self.verbose >= 3:
            print(f"🎮 SELF-PLAY: Game {game_id} using seed {seed}")
        
        state = make_empty_hex_state()  # Always uses 13x13 board
        move_provenance_codes: List[str] = []
        
        # Apply opening move if provided
        if opening_move is not None:
            row, col = opening_move
            if self.verbose >= 3:
                trmph_move = rowcol_to_trmph(row, col)
                print(f"🎮 SELF-PLAY: Starting with opening move {trmph_move} ({row}, {col})")
            state = state.make_move(row, col)
            if self.write_provenance:
                # Opening moves are externally provided and have no MCTS source in schema v1.
                # We mark them as trainable visit-style moves to keep move-level alignment.
                move_provenance_codes.append(MOVE_CODE_VISIT_COUNT)
        
        if self.verbose >= 3:
            print(f"🎮 SELF-PLAY: Starting new game with MCTS ({self.mcts_sims} simulations)")
            print(f"🎮 SELF-PLAY: MCTS config - decay_type: {self.mcts_config.temperature_decay_type}, "
                  f"start_temp: {self.mcts_config.temperature_start}, "
                  f"end_temp: {self.mcts_config.temperature_end}")
        
        while not state.game_over:
            # Use MCTS for move generation
            if self.verbose >= 3:
                print(f"🎮 SELF-PLAY: Move {len(state.move_history)}, player {state.current_player}, legal moves: {len(state.get_legal_moves())}")
            
            # Use natural MCTS interface
            mcts = BaselineMCTS(self.game_engine, self.model_wrapper, self.mcts_config)
            
            # Run MCTS
            if self.verbose >= 3:
                print(f"🎮 SELF-PLAY: Running MCTS with {self.mcts_sims} simulations")
            
            start_time = time.perf_counter()
            mcts_result = mcts.run(state)
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
                        for k in ("select_ms", "encode_ms", "stack_ms", "expand_ms", "backprop_ms", "cache_lookup_ms", "state_creation_ms"):
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
                            f"[MCTS_PROFILE] device={device_s} move={len(state.move_history)} "
                            f"batches={int(stats.get('batch_count', 0))} avg_batch={avg_batch:.1f} "
                            f"NN_ms={nn_ms:.1f} CPU_ms={cpu_ms:.1f} NN%={nn_pct:.1f} "
                            f"search_time_s={search_time:.3f}"
                        )
                    except Exception as e:
                        print(f"[MCTS_PROFILE] failed to summarize stats: {e}")
            
            # Get the best move from the result
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
                cache_hit_rate = mcts.cache_hits / max(1, mcts.cache_hits + mcts.cache_misses)
                print(
                    f"[Move {len(state.move_history)}] MCTS: sims={self.mcts_sims}, "
                    f"inferences={mcts_result.stats.get('total_simulations', 0)}, "
                    f"cache_hit_rate={cache_hit_rate:.1%}, "
                    f"time={search_time:.4f}s"
                )
            
            if self.verbose >= 3:
                print(f"🎮 SELF-PLAY: Selected move {move}, value {search_value:.4f}")
            
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

    def generate_games_with_monitoring(self, num_games: int, board_size: int = BOARD_SIZE, 
                                     progress_interval: int = 10, opening_strategy=None) -> List[Dict[str, Any]]:
        """
        Generate self-play games with monitoring and statistics.
        
        Args:
            num_games: Number of games to generate
            board_size: Size of the board (default: 13)
            progress_interval: How often to print progress updates
            
        Returns:
            List of game data dictionaries
        """
        start_time = time.time()
        print(f"Generating {num_games} games...")
        print(f"Using {'batched' if self.use_batched_inference else 'individual'} inference")
        
        games = []
        
        # Generate games sequentially (single-threaded)
        for i in range(num_games):
            # Get opening move if strategy provided
            opening_move = None
            if opening_strategy is not None:
                opening_move = opening_strategy.get_opening_move(i)
            
            game_data = self._generate_single_game(board_size, opening_move, game_id=i)
            self._validate_game_data(game_data, i)
            games.append(game_data)
            
            # Progress update
            if (i + 1) % progress_interval == 0 or (i + 1) == num_games:
                elapsed = time.time() - start_time
                games_per_sec = (i + 1) / elapsed
                if self.verbose >= 1:
                    print(f"  Generated {i + 1}/{num_games} games ({games_per_sec:.1f} games/s)")
            elif self.verbose >= 1:
                print(".", end="", flush=True)  # Progress dot for each game
        
        # Update statistics
        total_time = time.time() - start_time
        self.stats['games_generated'] += len(games)
        self.stats['total_time'] += total_time
        self.stats['games_per_second'] = len(games) / total_time if total_time > 0 else 0
        
        print(f"Generated {len(games)} games in {total_time:.1f}s ({self.stats['games_per_second']:.1f} games/s)")
        
        return games

    def generate_games_streaming(self, num_games: int, board_size: int = BOARD_SIZE, 
                               progress_interval: int = 10, opening_strategy=None) -> List[Dict[str, Any]]:
        """
        Generate games with streaming save to avoid data loss on interruption.
        
        Args:
            num_games: Number of games to generate
            board_size: Size of the board (default: 13)
            progress_interval: How often to print progress updates
            
        Returns:
            List of game data dictionaries
        """
        if not self.streaming_save:
            raise RuntimeError("Streaming save is not enabled. Set self.streaming_save=True to use generate_games_streaming.")
        
        start_time = time.time()
        print(f"Generating {num_games} games with streaming save...")
        print(f"Using {'batched' if self.use_batched_inference else 'individual'} inference")
        
        games = []
        
        # Generate games sequentially (single-threaded)
        for i in range(num_games):
            # Get opening move if strategy provided
            opening_move = None
            if opening_strategy is not None:
                opening_move = opening_strategy.get_opening_move(i)
            
            game_data = self._generate_single_game(board_size, opening_move, game_id=i)
            self._validate_game_data(game_data, i)
            games.append(game_data)
            
            # Save immediately to avoid data loss
            self.save_game_to_stream(game_data)
            
            # Progress update
            if (i + 1) % progress_interval == 0 or (i + 1) == num_games:
                elapsed = time.time() - start_time
                games_per_sec = (i + 1) / elapsed
                if self.verbose >= 1:
                    print(f"  Generated {i + 1}/{num_games} games ({games_per_sec:.1f} games/s)")
            elif self.verbose >= 1:
                print(".", end="", flush=True)  # Progress dot for each game
        
        # Update statistics
        total_time = time.time() - start_time
        self.stats['games_generated'] += len(games)
        self.stats['total_time'] += total_time
        self.stats['games_per_second'] = len(games) / total_time if total_time > 0 else 0
        
        print(f"Generated {len(games)} games in {total_time:.1f}s ({self.stats['games_per_second']:.1f} games/s)")
        print(f"Games saved to: {self.streaming_file}")
        if self.write_provenance and self.streaming_provenance_file:
            print(f"Move provenance sidecar: {self.streaming_provenance_file}")
        
        return games

    def generate_games_with_opening_strategy(self, opening_strategy, num_games: int, 
                                           board_size: int = BOARD_SIZE, progress_interval: int = 10) -> List[Dict[str, Any]]:
        """
        Generate self-play games using a specific opening strategy.
        
        Args:
            opening_strategy: OpeningStrategy instance that provides opening moves
            num_games: Number of games to generate
            board_size: Size of the board (default: 13)
            progress_interval: How often to print progress updates
            
        Returns:
            List of game data dictionaries
        """
        start_time = time.time()
        print(f"Generating {num_games} games with opening strategy...")
        print(f"Strategy covers {opening_strategy.get_total_games()} games")
        print(f"Using {'batched' if self.use_batched_inference else 'individual'} inference")
        
        games = []
        
        # Generate games sequentially (single-threaded)
        for i in range(num_games):
            # Get opening move from strategy
            opening_move = opening_strategy.get_opening_move(i)
            
            game_data = self._generate_single_game(board_size, opening_move, game_id=i)
            self._validate_game_data(game_data, i)
            games.append(game_data)
            
            # Progress update
            if (i + 1) % progress_interval == 0 or (i + 1) == num_games:
                elapsed = time.time() - start_time
                games_per_sec = (i + 1) / elapsed
                if self.verbose >= 1:
                    print(f"  Generated {i + 1}/{num_games} games ({games_per_sec:.1f} games/s)")
            elif self.verbose >= 1:
                print(".", end="", flush=True)  # Progress dot for each game
        
        # Update statistics
        total_time = time.time() - start_time
        self.stats['games_generated'] += len(games)
        self.stats['total_time'] += total_time
        self.stats['games_per_second'] = len(games) / total_time if total_time > 0 else 0
        
        print(f"Generated {len(games)} games in {total_time:.1f}s ({self.stats['games_per_second']:.1f} games/s)")
        
        return games

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.stats.copy()
        
        # Add model performance stats
        model_stats = self.model.get_performance_stats()
        stats['model'] = model_stats
        stats['mcts'] = self._build_mcts_summary_stats()
        
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

    def save_games_simple(self, games: List[Dict[str, Any]], base_filename: str) -> str:
        """
        Save games to a TRMPH text file.
        
        Args:
            games: List of game data dictionaries
            base_filename: Base filename (without extension)
            
        Returns:
            The TRMPH file path
        """
        # Save as TRMPH text file
        trmph_file = f"{base_filename}.trmph"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(trmph_file), exist_ok=True)
        
        # Save as TRMPH text file using the same format as streaming
        with open(trmph_file, 'w') as f:
            # Write header with metadata
            git_info = get_git_commit_info()
            f.write(f"# Self-play games - {datetime.now().isoformat()}\n")
            f.write(f"# Model: {self.model_path}\n")
            f.write(f"# MCTS simulations: {self.mcts_sims}\n")
            f.write(f"# C_PUCT: {self.c_puct}\n")
            f.write(f"# Gumbel root selection: {self.enable_gumbel}\n")
            f.write(f"# Temperature: {self.temperature}\n")
            f.write(f"# Git commit: {git_info['status']}\n")
            f.write("# Format: trmph_string winner\n")
            
            # Write games
            for game in games:
                self._validate_game_data(game)
                f.write(f"{game['trmph']} {game['winner']}\n")

        if self.write_provenance:
            provenance_file = str(sidecar_path_for_trmph(trmph_file))
            with open(provenance_file, 'w', encoding='utf-8') as f:
                for game_index, game in enumerate(games):
                    move_codes = game.get('move_provenance_codes')
                    if move_codes is None:
                        raise ValueError(
                            f"Missing move_provenance_codes for game index {game_index} while writing provenance"
                        )
                    record = make_move_provenance_record(game_index, move_codes)
                    f.write(record.to_json_line())
                    f.write("\n")
        
        if self.verbose >= 1:
            print(f"Saved {len(games)} games to {trmph_file}")
            if self.write_provenance:
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

    def clear_cache(self):
        """Clear the model's inference cache."""
        self.model.clear_cache()

    def shutdown(self):
        """Clean shutdown of the engine."""
        print("Shutting down SelfPlayEngine...")
        if self._simple_inference_summary_has_activity():
            self.model.print_performance_summary()
        self.print_mcts_run_summary()
