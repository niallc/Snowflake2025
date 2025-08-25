"""
Self-play engine for generating training data using the Hex AI model.
"""

import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from hex_ai.config import TRMPH_BLUE_WIN, TRMPH_PREFIX, TRMPH_RED_WIN
from hex_ai.enums import Winner
from hex_ai.inference.mcts import BaselineMCTS, BaselineMCTSConfig
from hex_ai.inference.game_engine import HexGameEngine
from hex_ai.inference.model_wrapper import ModelWrapper
from hex_ai.inference.game_engine import HexGameState
from hex_ai.inference.simple_model_inference import SimpleModelInference
from hex_ai.training_utils import get_device
from hex_ai.value_utils import validate_trmph_winner


class SelfPlayEngine:
    """High-performance self-play engine with optimized inference and logging."""
    
    def __init__(self, model_path: str, batch_size: int = 32, 
                 cache_size: int = 10000, temperature: float = 1.0, verbose: int = 1, 
                 streaming_save: bool = False, streaming_file: str = None,
                 use_batched_inference: bool = True, output_dir: str = None,
                 mcts_sims: int = 500):
        """
        Initialize the self-play engine.
        
        Args:
            model_path: Path to the model checkpoint
            batch_size: Batch size for inference
            cache_size: Size of the LRU cache
            temperature: Temperature for move sampling
            verbose: Verbosity level (0=quiet, 1=normal, 2=detailed)
            streaming_save: Save games incrementally to avoid data loss
            streaming_file: File path for streaming save (auto-generated if None)
            use_batched_inference: Whether to use batched inference for better performance
            output_dir: Output directory for streaming files (used if streaming_file is None)
            mcts_sims: Number of MCTS simulations per move
        """
        self.model_path = model_path
        self.batch_size = batch_size
        self.cache_size = cache_size
        self.temperature = temperature
        self.verbose = verbose
        self.streaming_save = streaming_save
        self.streaming_file = streaming_file
        self.use_batched_inference = use_batched_inference
        self.output_dir = output_dir
        self.mcts_sims = mcts_sims
        
        # Initialize model
        self.model = SimpleModelInference(model_path, device=get_device(), cache_size=cache_size)
        
        # Initialize MCTS components
        self.game_engine = HexGameEngine()
        # Create ModelWrapper for MCTS
        self.model_wrapper = ModelWrapper(model_path, device=get_device())
        # Create MCTS configuration
        self.mcts_config = BaselineMCTSConfig(
            sims=self.mcts_sims,
            batch_cap=64,  # Optimize for throughput
            c_puct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_eps=0.25,
            add_root_noise=False,  # Disable for self-play consistency
            temperature=self.temperature,
            seed=1234  # Fixed seed for reproducibility
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
        
        # Streaming save setup
        if self.streaming_save and self.streaming_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if self.output_dir:
                self.streaming_file = f"{self.output_dir}/streaming_selfplay_{timestamp}.trmph"
            else:
                self.streaming_file = f"data/sf25/selfplay_default/streaming_selfplay_{timestamp}.trmph"
        
        if self.streaming_save:
            os.makedirs(os.path.dirname(self.streaming_file), exist_ok=True)
            # Write header
            with open(self.streaming_file, 'w') as f:
                f.write(f"# Self-play games - {datetime.now().isoformat()}\n")
                f.write(f"# Model: {model_path}\n")
                f.write(f"# MCTS simulations: {mcts_sims}\n")
                f.write(f"# Temperature: {temperature}\n")
                f.write("# Format: trmph_string winner\n")
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        if self.verbose >= 1:
            print(f"SelfPlayEngine initialized:")
            print(f"  Model: {model_path}")
            print(f"  Batch size: {batch_size}")
            print(f"  Cache size: {cache_size}")
            print(f"  Search method: MCTS ({mcts_sims} simulations)")
            print(f"  Temperature: {temperature}")
            print(f"  Verbose: {verbose}")
            print(f"  Batched inference: {use_batched_inference}")
            


    def _generate_single_game(self, board_size: int) -> Dict[str, Any]:
        """
        Generate a single self-play game.
        
        Args:
            board_size: Size of the board (ignored, always uses 13)
            
        Returns:
            Dictionary containing game data with TRMPH string and winner
        """
        state = HexGameState()  # Always uses 13x13 board
        
        if self.verbose >= 3:
            print(f"🎮 SELF-PLAY: Starting new game with MCTS ({self.mcts_sims} simulations)")
        
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
            mcts_stats = mcts.run(state)
            search_time = time.perf_counter() - start_time
            
            # Get the best move
            move = mcts.pick_move(state, temperature=self.temperature)
            
            # Get root value (approximate from MCTS)
            tree_data = mcts.get_tree_data(state)
            search_value = tree_data.get('root_value', 0.0)
            
            # Log MCTS statistics
            if self.verbose >= 2:
                cache_hit_rate = mcts.cache_hits / max(1, mcts.cache_hits + mcts.cache_misses)
                print(
                    f"[Move {len(state.move_history)}] MCTS: sims={self.mcts_sims}, "
                    f"inferences={mcts_stats.get('inferences', 0)}, "
                    f"cache_hit_rate={cache_hit_rate:.1%}, "
                    f"time={search_time:.4f}s"
                )
            
            if self.verbose >= 3:
                print(f"🎮 SELF-PLAY: Selected move {move}, value {search_value:.4f}")
            
            # Apply move
            state = state.make_move(*move)
        
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
        
        if self.verbose >= 3:
            print(f"🎮 SELF-PLAY: Game complete, winner: {state.winner}, moves: {len(state.move_history)}")
        
        return game_data

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

    def generate_games_with_monitoring(self, num_games: int, board_size: int = 13, 
                                     progress_interval: int = 10) -> List[Dict[str, Any]]:
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
            try:
                game_data = self._generate_single_game(board_size)
                self._validate_game_data(game_data, i)
                games.append(game_data)
                
                # Progress update
                if (i + 1) % progress_interval == 0 or (i + 1) == num_games:
                    elapsed = time.time() - start_time
                    games_per_sec = (i + 1) / elapsed
                    if self.verbose >= 1:
                        print(f"\nGenerated {i + 1}/{num_games} games ({games_per_sec:.1f} games/s)")
                elif self.verbose >= 1:
                    print(".", end="", flush=True)  # Progress dot for each game
                    
            except Exception as e:
                self.logger.error(f"Error generating game {i}: {e}")
        
        # Update statistics
        total_time = time.time() - start_time
        self.stats['games_generated'] += len(games)
        self.stats['total_time'] += total_time
        self.stats['games_per_second'] = len(games) / total_time if total_time > 0 else 0
        
        print(f"Generated {len(games)} games in {total_time:.1f}s ({self.stats['games_per_second']:.1f} games/s)")
        
        return games

    def generate_games_streaming(self, num_games: int, board_size: int = 13, 
                               progress_interval: int = 10) -> List[Dict[str, Any]]:
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
            try:
                game_data = self._generate_single_game(board_size)
                self._validate_game_data(game_data, i)
                games.append(game_data)
                
                # Save immediately to avoid data loss
                self.save_game_to_stream(game_data)
                
                # Progress update
                if (i + 1) % progress_interval == 0 or (i + 1) == num_games:
                    elapsed = time.time() - start_time
                    games_per_sec = (i + 1) / elapsed
                    if self.verbose >= 1:
                        print(f"\nGenerated {i + 1}/{num_games} games ({games_per_sec:.1f} games/s)")
                elif self.verbose >= 1:
                    print(".", end="", flush=True)  # Progress dot for each game
                    
            except Exception as e:
                self.logger.error(f"Error generating game {i}: {e}")
        
        # Update statistics
        total_time = time.time() - start_time
        self.stats['games_generated'] += len(games)
        self.stats['total_time'] += total_time
        self.stats['games_per_second'] = len(games) / total_time if total_time > 0 else 0
        
        print(f"Generated {len(games)} games in {total_time:.1f}s ({self.stats['games_per_second']:.1f} games/s)")
        print(f"Games saved to: {self.streaming_file}")
        
        return games

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.stats.copy()
        
        # Add model performance stats
        model_stats = self.model.get_performance_stats()
        stats['model'] = model_stats
        
        return stats

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
            for game in games:
                self._validate_game_data(game)
                f.write(f"{game['trmph']} {game['winner']}\n")
        
        if self.verbose >= 1:
            print(f"Saved {len(games)} games to {trmph_file}")
        
        return trmph_file

    def save_game_to_stream(self, game_data: Dict[str, Any]):
        """Save a single game to the streaming file."""
        if not self.streaming_save:
            return
            
        self._validate_game_data(game_data)
        
        with open(self.streaming_file, 'a') as f:
            f.write(f"{game_data['trmph']} {game_data['winner']}\n")

    def clear_cache(self):
        """Clear the model's inference cache."""
        self.model.clear_cache()

    def shutdown(self):
        """Clean shutdown of the engine."""
        print("Shutting down SelfPlayEngine...")
        self.model.print_performance_summary()