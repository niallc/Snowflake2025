"""
Self-play engine for generating training data using the Hex AI model.
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import logging

from ..inference.simple_model_inference import SimpleModelInference
from ..inference.game_engine import HexGameState
from ..inference.fixed_tree_search import minimax_policy_value_search
from ..config import BLUE_PLAYER, RED_PLAYER


class SelfPlayEngine:
    """High-performance self-play engine with optimized inference."""
    
    def __init__(self, 
                 model_path: str,
                 num_workers: int = 4,
                 batch_size: int = 100,
                 cache_size: int = 10000,
                 search_widths: Optional[List[int]] = None,
                 temperature: float = 1.0,
                 enable_caching: bool = True):
        """
        Initialize the self-play engine.
        
        Args:
            model_path: Path to the model checkpoint
            num_workers: Number of parallel workers for game generation
            batch_size: Maximum batch size for inference
            cache_size: Size of the inference cache
            search_widths: List of search widths for minimax search
            temperature: Temperature for move selection
            enable_caching: Whether to enable inference caching
        """
        self.model_path = model_path
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.search_widths = search_widths or [3, 2]
        self.temperature = temperature
        
        # Initialize model with caching and performance monitoring
        self.model = SimpleModelInference(
            checkpoint_path=model_path,
            cache_size=cache_size,
            max_batch_size=batch_size,
            enable_caching=enable_caching
        )
        
        # Setup logging
        self.logger = logging.getLogger("selfplay")
        self.logger.setLevel(logging.INFO)
        
        # Performance tracking
        self.stats = {
            'games_generated': 0,
            'total_moves': 0,
            'total_time': 0.0,
            'games_per_second': 0.0
        }
        
        print(f"SelfPlayEngine initialized with {num_workers} workers, batch_size={batch_size}")
        print(f"Search widths: {search_widths}, temperature: {temperature}")
    
    def generate_games(self, num_games: int, board_size: int = 13) -> List[Dict[str, Any]]:
        """
        Generate self-play games efficiently.
        
        Args:
            num_games: Number of games to generate
            board_size: Size of the board (default: 13)
            
        Returns:
            List of game data dictionaries
        """
        start_time = time.time()
        print(f"Generating {num_games} games with {self.num_workers} workers...")
        
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
                    games.append(game_data)
                    completed += 1
                    
                    if completed % 10 == 0 or completed == num_games:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        print(f"Generated {completed}/{num_games} games ({rate:.1f} games/s)")
                        
                except Exception as e:
                    self.logger.error(f"Error generating game {game_id}: {e}")
        
        # Update statistics
        total_time = time.time() - start_time
        total_moves = sum(len(game['moves']) for game in games)
        
        self.stats['games_generated'] += len(games)
        self.stats['total_moves'] += total_moves
        self.stats['total_time'] += total_time
        self.stats['games_per_second'] = len(games) / total_time if total_time > 0 else 0
        
        print(f"Generated {len(games)} games in {total_time:.1f}s ({self.stats['games_per_second']:.1f} games/s)")
        print(f"Average moves per game: {total_moves / len(games):.1f}")
        
        return games
    
    def _generate_single_game(self, board_size: int) -> Dict[str, Any]:
        """
        Generate a single self-play game.
        
        Args:
            board_size: Size of the board (ignored, always uses 13)
            
        Returns:
            Dictionary containing game data
        """
        state = HexGameState()  # Always uses 13x13 board
        moves = []
        policies = []
        values = []
        move_times = []
        
        while not state.game_over:
            move_start = time.time()
            
            # Get model prediction
            policy, value = self.model.simple_infer(state.board)
            
            # Store game state
            moves.append(state.to_trmph())
            policies.append(policy)
            values.append(value)
            
            # Select move using search
            move, value = minimax_policy_value_search(
                state=state,
                model=self.model,
                widths=self.search_widths,
                temperature=self.temperature
            )
            
            # Apply move
            state = state.make_move(*move)
            move_times.append(time.time() - move_start)
        
        return {
            'moves': moves,
            'policies': policies,
            'values': values,
            'winner': state.winner,
            'final_trmph': state.to_trmph(),
            'num_moves': len(moves),
            'avg_move_time': np.mean(move_times) if move_times else 0
        }
    
    def generate_games_with_monitoring(self, num_games: int, board_size: int = 13, 
                                     progress_interval: int = 10) -> List[Dict[str, Any]]:
        """
        Generate games with detailed monitoring and performance tracking.
        
        Args:
            num_games: Number of games to generate
            board_size: Size of the board
            progress_interval: How often to print progress updates
            
        Returns:
            List of game data dictionaries
        """
        start_time = time.time()
        print(f"Generating {num_games} games with monitoring...")
        
        games = []
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_game_id = {
                executor.submit(self._generate_single_game, board_size): i 
                for i in range(num_games)
            }
            
            completed = 0
            for future in as_completed(future_to_game_id):
                game_id = future_to_game_id[future]
                try:
                    game_data = future.result()
                    games.append(game_data)
                    completed += 1
                    
                    if completed % progress_interval == 0 or completed == num_games:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        
                        # Get model performance stats
                        model_stats = self.model.get_performance_stats()
                        cache_stats = model_stats.get('cache', {})
                        
                        print(f"\n--- Progress Update ({completed}/{num_games}) ---")
                        print(f"Games per second: {rate:.1f}")
                        print(f"Model throughput: {model_stats.get('throughput', 0):.1f} boards/s")
                        print(f"Cache hit rate: {cache_stats.get('hit_rate', 0):.1%}")
                        print(f"Average batch size: {model_stats.get('avg_batch_size', 0):.1f}")
                        
                        # Memory usage
                        memory_stats = self.model.get_memory_usage()
                        if 'system_used_mb' in memory_stats:
                            print(f"System memory: {memory_stats['system_used_mb']:.0f}MB used")
                        if 'gpu_allocated_mb' in memory_stats:
                            print(f"GPU memory: {memory_stats['gpu_allocated_mb']:.0f}MB allocated")
                        
                except Exception as e:
                    self.logger.error(f"Error generating game {game_id}: {e}")
        
        # Final statistics
        total_time = time.time() - start_time
        total_moves = sum(len(game['moves']) for game in games)
        
        print(f"\n=== Final Statistics ===")
        print(f"Games generated: {len(games)}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Games per second: {len(games) / total_time:.1f}")
        print(f"Average moves per game: {total_moves / len(games):.1f}")
        
        # Model performance summary
        self.model.print_performance_summary()
        
        return games
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        model_stats = self.model.get_performance_stats()
        memory_stats = self.model.get_memory_usage()
        
        return {
            'selfplay': self.stats,
            'model': model_stats,
            'memory': memory_stats,
            'workers': self.num_workers,
            'search_widths': self.search_widths,
            'temperature': self.temperature
        }
    
    def save_games_to_file(self, games: List[Dict[str, Any]], filename: str):
        """Save generated games to a file."""
        import pickle
        import gzip
        
        data = {
            'games': games,
            'metadata': {
                'model_path': self.model_path,
                'search_widths': self.search_widths,
                'temperature': self.temperature,
                'generation_time': time.time(),
                'stats': self.get_performance_stats()
            }
        }
        
        with gzip.open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved {len(games)} games to {filename}")
    
    def clear_cache(self):
        """Clear the model's inference cache."""
        self.model.clear_cache()
    
    def shutdown(self):
        """Clean shutdown of the engine."""
        print("Shutting down SelfPlayEngine...")
        self.model.print_performance_summary()