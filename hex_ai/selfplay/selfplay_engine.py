"""
Self-play engine for generating training data using the Hex AI model.
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
import pickle
import gzip
from datetime import datetime

from hex_ai.inference.simple_model_inference import SimpleModelInference
from hex_ai.inference.game_engine import HexGameState
from hex_ai.inference.fixed_tree_search import minimax_policy_value_search_with_batching
from hex_ai.config import TRMPH_PREFIX, TRMPH_RED_WIN, TRMPH_BLUE_WIN
from hex_ai.value_utils import validate_trmph_winner
from hex_ai.training_utils import get_device


class SelfPlayEngine:
    """High-performance self-play engine with optimized inference and logging."""
    
    def __init__(self, model_path: str, num_workers: int = 4, batch_size: int = 32, 
                 cache_size: int = 10000, search_widths: List[int] = None, 
                 temperature: float = 1.0, verbose: int = 1, 
                 streaming_save: bool = False, streaming_file: str = None,
                 use_batched_inference: bool = True, output_dir: str = None):
        """
        Initialize the self-play engine.
        
        Args:
            model_path: Path to the model checkpoint
            num_workers: Number of worker threads
            batch_size: Batch size for inference
            cache_size: Size of the LRU cache
            search_widths: List of search widths for minimax
            temperature: Temperature for move sampling
            verbose: Verbosity level (0=quiet, 1=normal, 2=detailed)
            streaming_save: Save games incrementally to avoid data loss
            streaming_file: File path for streaming save (auto-generated if None)
            use_batched_inference: Whether to use batched inference for better performance
            output_dir: Output directory for streaming files (used if streaming_file is None)
        """
        self.model_path = model_path
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.cache_size = cache_size
        self.search_widths = search_widths or [3, 2]
        self.temperature = temperature
        self.verbose = verbose
        self.streaming_save = streaming_save
        self.streaming_file = streaming_file
        self.use_batched_inference = use_batched_inference
        self.output_dir = output_dir
        
        # Initialize model
        self.model = SimpleModelInference(model_path, device=get_device(), cache_size=cache_size)
        
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
                f.write(f"# Search widths: {search_widths}\n")
                f.write(f"# Temperature: {temperature}\n")
                f.write("# Format: trmph_string winner\n")
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        if self.verbose >= 1:
            print(f"SelfPlayEngine initialized:")
            print(f"  Model: {model_path}")
            print(f"  Workers: {num_workers}")
            print(f"  Batch size: {batch_size}")
            print(f"  Cache size: {cache_size}")
            print(f"  Search widths: {search_widths}")
            print(f"  Temperature: {temperature}")
            print(f"  Verbose: {verbose}")
            print(f"  Batched inference: {use_batched_inference}")
            
        # Warn about multi-threading incompatibility with batched inference
        if num_workers > 1 and use_batched_inference:
            print(f"\nWARNING: Multi-threading (num_workers={num_workers}) is incompatible with batched inference.")
            print("The PositionCollector requires single-threaded usage. Consider setting --num_workers=1")
            print("or --no_batched_inference to avoid this issue.\n")

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
            print(f"🎮 SELF-PLAY: Starting new game with search widths {self.search_widths}")
        
        while not state.game_over:
            # Use batched minimax search
            if self.verbose >= 3:
                print(f"🎮 SELF-PLAY: Move {len(state.move_history)}, player {state.current_player}, legal moves: {len(state.get_legal_moves())}")
            
            move, minimax_value, _, search_stats = minimax_policy_value_search_with_batching(
                state=state,
                model=self.model,
                widths=self.search_widths,
                temperature=self.temperature,
                verbose=self.verbose
            )
            
            if self.verbose >= 3:
                print(f"🎮 SELF-PLAY: Selected move {move}, value {minimax_value:.4f}")
            
            # Log search statistics
            if self.verbose >= 2:
                print(
                    f"[Move {len(state.move_history)}] evals: policy={search_stats['policy_items_processed']} "
                    f"(waves={search_stats['policy_batches_processed']}, avg_batch={search_stats['avg_policy_batch']:.2f}, "
                    f"nn_time={search_stats['policy_nn_time']:.4f}s, total_phase={search_stats['policy_total_time']:.4f}s)"
                )
            
            # Apply move
            state = state.make_move(*move)
        
        # Game data - TRMPH string and winner
        if state.winner == "red":
            winner_char = TRMPH_RED_WIN
        elif state.winner == "blue":
            winner_char = TRMPH_BLUE_WIN
        else:
            raise ValueError(f"Unexpected winner value: {state.winner!r} (expected 'red' or 'blue')")
        
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
        print(f"Generating {num_games} games with {self.num_workers} workers...")
        print(f"Using {'batched' if self.use_batched_inference else 'individual'} inference")
        
        games = []
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit game generation tasks
            future_to_game_id = {
                executor.submit(self._generate_single_game, board_size): i 
                for i in range(num_games)
            }
            
            # Collect results
            completed = 0
            for future in as_completed(future_to_game_id):
                game_id = future_to_game_id[future]
                try:
                    game_data = future.result()
                    self._validate_game_data(game_data, game_id)
                    games.append(game_data)
                    completed += 1
                    
                    # Progress update
                    if completed % progress_interval == 0 or completed == num_games:
                        elapsed = time.time() - start_time
                        games_per_sec = completed / elapsed
                        if self.verbose >= 1:
                            print(f"\nGenerated {completed}/{num_games} games ({games_per_sec:.1f} games/s)")
                    elif self.verbose >= 1:
                        print(".", end="", flush=True)  # Progress dot for each game
                        
                except Exception as e:
                    self.logger.error(f"Error generating game {game_id}: {e}")
        
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
            # return self.generate_games_with_monitoring(num_games, board_size, progress_interval)
        
        start_time = time.time()
        print(f"Generating {num_games} games with streaming save...")
        print(f"Using {'batched' if self.use_batched_inference else 'individual'} inference")
        
        games = []
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit game generation tasks
            future_to_game_id = {
                executor.submit(self._generate_single_game, board_size): i 
                for i in range(num_games)
            }
            
            # Collect results and save immediately
            completed = 0
            for future in as_completed(future_to_game_id):
                game_id = future_to_game_id[future]
                try:
                    game_data = future.result()
                    self._validate_game_data(game_data, game_id)
                    games.append(game_data)
                    
                    # Save immediately to avoid data loss
                    self.save_game_to_stream(game_data)
                    
                    completed += 1
                    
                    # Progress update
                    if completed % progress_interval == 0 or completed == num_games:
                        elapsed = time.time() - start_time
                        games_per_sec = completed / elapsed
                        if self.verbose >= 1:
                            print(f"\nGenerated {completed}/{num_games} games ({games_per_sec:.1f} games/s)")
                    elif self.verbose >= 1:
                        print(".", end="", flush=True)  # Progress dot for each game
                        
                except Exception as e:
                    self.logger.error(f"Error generating game {game_id}: {e}")
        
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

    def save_games_to_file(self, games: List[Dict[str, Any]], filename: str):
        """
        Save games to a compressed pickle file.
        
        Args:
            games: List of game data dictionaries
            filename: Output filename
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save with compression
        with gzip.open(filename, 'wb') as f:
            pickle.dump(games, f)
        
        if self.verbose >= 1:
            print(f"Saved {len(games)} games to {filename}")

    def save_games_with_details(self, games: List[Dict[str, Any]], base_filename: str):
        """
        Save games with both compressed data and detailed CSV files.
        
        Args:
            games: List of game data dictionaries
            base_filename: Base filename (without extension)
            
        Returns:
            Tuple of (compressed_file, csv_dir)
        """
        # Save compressed data
        compressed_file = f"{base_filename}.pkl.gz"
        self.save_games_to_file(games, compressed_file)
        
        # Save detailed CSV files (simplified version)
        csv_dir = f"{base_filename}_detailed_moves"
        self._save_detailed_moves_to_csv(games, csv_dir)
        
        return compressed_file, csv_dir

    def _save_detailed_moves_to_csv(self, games: List[Dict[str, Any]], output_dir: str):
        """
        Save detailed move-by-move data to CSV files.
        
        Args:
            games: List of game data dictionaries
            output_dir: Output directory for CSV files
        """
        import csv
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save summary file
        summary_file = os.path.join(output_dir, "games_summary.csv")
        with open(summary_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['game_id', 'winner', 'final_trmph'])
            
            for i, game in enumerate(games):
                self._validate_game_data(game, i)
                writer.writerow([i, game['winner'], game['trmph']])
        
        if self.verbose >= 1:
            print(f"Saved detailed move data:")
            print(f"  Summary: {summary_file}")
            print(f"  Individual games: {len(games)} CSV files in {output_dir}")

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