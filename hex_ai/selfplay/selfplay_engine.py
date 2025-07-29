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
import os
import csv
from datetime import datetime

from ..inference.simple_model_inference import SimpleModelInference
from ..inference.game_engine import HexGameState
from ..inference.fixed_tree_search import minimax_policy_value_search
from ..config import BLUE_PLAYER, RED_PLAYER
from ..utils import format_conversion as fc
from ..value_utils import get_top_k_legal_moves, policy_logits_to_probs, get_top_k_moves_with_probs


class SelfPlayEngine:
    """High-performance self-play engine with optimized inference and detailed logging."""
    
    def __init__(self, model_path: str, num_workers: int = 4, batch_size: int = 32, 
                 cache_size: int = 10000, search_widths: List[int] = None, 
                 temperature: float = 1.0, enable_caching: bool = True, 
                 verbose: int = 1, save_essential_only: bool = False,
                 streaming_save: bool = False, streaming_file: str = None):
        """
        Initialize the self-play engine.
        
        Args:
            model_path: Path to the model checkpoint
            num_workers: Number of worker threads
            batch_size: Batch size for inference
            cache_size: Size of the LRU cache
            search_widths: List of search widths for minimax
            temperature: Temperature for move sampling
            enable_caching: Whether to enable caching
            verbose: Verbosity level (0=quiet, 1=normal, 2=detailed)
            save_essential_only: Only save essential game data (no detailed moves)
            streaming_save: Save games incrementally to avoid data loss
            streaming_file: File path for streaming save (auto-generated if None)
        """
        self.model_path = model_path
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.cache_size = cache_size
        self.search_widths = search_widths or [3, 2]
        self.temperature = temperature
        self.enable_caching = enable_caching
        self.verbose = verbose
        self.save_essential_only = save_essential_only
        self.streaming_save = streaming_save
        self.streaming_file = streaming_file
        
        # Initialize model
        self.model = SimpleModelInference(model_path, device='cpu', cache_size=cache_size)
        
        # Performance tracking
        self.stats = {
            'total_inferences': 0,
            'total_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Streaming save setup
        if self.streaming_save and self.streaming_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.streaming_file = f"data/sf25/jul29/streaming_selfplay_{timestamp}.txt"
        
        if self.streaming_save:
            os.makedirs(os.path.dirname(self.streaming_file), exist_ok=True)
            # Write header
            with open(self.streaming_file, 'w') as f:
                f.write(f"# Self-play games - {datetime.now().isoformat()}\n")
                f.write(f"# Model: {model_path}\n")
                f.write(f"# Search widths: {search_widths}\n")
                f.write(f"# Temperature: {temperature}\n")
                f.write("# Format: trmph_string winner\n")
                f.write("# Example: #13,a4g7e9e8f8f7h7h6j5 r\n")
        
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
    
    def generate_games(self, num_games: int, board_size: int = 13, progress_interval = 10) -> List[Dict[str, Any]]:
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
        Generate a single self-play game with detailed move tracking.
        
        Args:
            board_size: Size of the board (ignored, always uses 13)
            
        Returns:
            Dictionary containing game data and detailed move information
        """
        state = HexGameState()  # Always uses 13x13 board
        moves = []
        policies = []
        values = []
        move_times = []
        detailed_moves = []  # New: detailed move-by-move data
        
        move_number = 0
        
        while not state.game_over:
            move_start = time.time()
            move_number += 1
            
            # Get model prediction
            policy, value = self.model.simple_infer(state.board)
            
            # Get policy head's top 5 moves
            legal_moves = state.get_legal_moves()
            policy_probs = policy_logits_to_probs(policy, self.temperature)
            
            # Find policy head's top 5 moves
            policy_top_moves = get_top_k_moves_with_probs(
                policy, legal_moves, state.board.shape[0], k=5, temperature=self.temperature
            )
            
            # Get value predictions for top 5 policy moves
            policy_move_values = []
            for move, prob in policy_top_moves:
                # Create temporary state to evaluate this move
                temp_state = state.make_move(*move)
                _, move_value = self.model.simple_infer(temp_state.board)
                policy_move_values.append(move_value)
            
            # Get policy head's preferred move (top 1)
            policy_preferred_move = policy_top_moves[0][0] if policy_top_moves else None
            
            # Store game state
            moves.append(state.to_trmph())
            policies.append(policy)
            values.append(value)
            
            # Select move using search (minimax)
            minimax_move, minimax_value = minimax_policy_value_search(
                state=state,
                model=self.model,
                widths=self.search_widths,
                temperature=self.temperature
            )
            
            # Track agreement between policy and minimax
            if policy_preferred_move == minimax_move:
                self.stats['policy_vs_minimax_agreement'] += 1
            else:
                self.stats['policy_vs_minimax_disagreement'] += 1
            
            # Create structured policy summary
            policy_summary = []
            for i, ((move, prob), value) in enumerate(zip(policy_top_moves, policy_move_values)):
                policy_summary.append({
                    'rank': i + 1,
                    'move': move,
                    'move_trmph': fc.rowcol_to_trmph(*move),
                    'probability': prob,
                    'value': value
                })
            
            # Store detailed move information
            move_detail = {
                'move_number': move_number,
                'player': state.current_player,
                'board_state': state.to_trmph(),
                'policy_summary': policy_summary,  # Top 5 moves with values
                'policy_preferred_move': policy_preferred_move,
                'policy_preferred_move_trmph': fc.rowcol_to_trmph(*policy_preferred_move) if policy_preferred_move else None,
                'minimax_chosen_move': minimax_move,
                'minimax_chosen_move_trmph': fc.rowcol_to_trmph(*minimax_move) if minimax_move else None,
                'minimax_value': minimax_value,
                'agreement': policy_preferred_move == minimax_move,
                'legal_moves_count': len(legal_moves),
                'move_time': 0.0  # Will be set after move
            }
            
            # Apply move
            state = state.make_move(*minimax_move)
            move_time = time.time() - move_start
            move_times.append(move_time)
            move_detail['move_time'] = move_time
            
            detailed_moves.append(move_detail)
            
            if self.verbose >= 2:
                print(f"\nMove {move_number} (Player {state.current_player}):")
                print(f"  Policy top 5 moves:")
                for i, item in enumerate(policy_summary):
                    marker = " →" if item['move'] == minimax_move else "  "
                    print(f"    {i+1}. {item['move_trmph']} ({item['move']}) - prob: {item['probability']:.3f}, value: {item['value']:.3f}{marker}")
                print(f"  Minimax chose: {move_detail['minimax_chosen_move_trmph']} ({minimax_move}) - value: {minimax_value:.3f}")
                print(f"  Agreement: {move_detail['agreement']}, Time: {move_time:.3f}s")
        
        return {
            'moves': moves,
            'policies': policies,
            'values': values,
            'detailed_moves': detailed_moves,  # New: detailed move data
            'winner': state.winner,
            'final_trmph': state.to_trmph(),
            'num_moves': len(moves),
            'avg_move_time': np.mean(move_times) if move_times else 0,
            'policy_minimax_agreement_rate': (
                self.stats['policy_vs_minimax_agreement'] / 
                (self.stats['policy_vs_minimax_agreement'] + self.stats['policy_vs_minimax_disagreement'])
                if (self.stats['policy_vs_minimax_agreement'] + self.stats['policy_vs_minimax_disagreement']) > 0 else 0
            )
        }
    
    def _generate_single_game_essential(self, board_size: int) -> Dict[str, Any]:
        """
        Generate a single self-play game with only essential data.
        
        Args:
            board_size: Size of the board (ignored, always uses 13)
            
        Returns:
            Dictionary containing essential game data only
        """
        state = HexGameState()  # Always uses 13x13 board
        move_times = []
        
        while not state.game_over:
            move_start = time.time()
            
            # Get model prediction
            policy, value = self.model.simple_infer(state.board)
            
            # Get minimax move
            move, minimax_value = minimax_policy_value_search(
                state=state,
                model=self.model,
                widths=self.search_widths,
                temperature=self.temperature,
                verbose=self.verbose
            )
            
            # Apply move
            state = state.make_move(*move)
            move_times.append(time.time() - move_start)
        
        # Essential game data only - TRMPH string and winner
        game_data = {
            'trmph': state.to_trmph(),
            'winner': 'r' if state.winner == RED_PLAYER else 'b'
        }
        
        return game_data
    
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
        
        if self.verbose >= 1:
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
                    
                    # Update stats
                    self.stats['total_games'] += 1
                    self.stats['total_moves'] += len(game_data['moves'])
                    
                    if self.verbose >= 1 and (completed % progress_interval == 0 or completed == num_games):
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
                        
                        # Policy vs Minimax agreement
                        if completed > 0:
                            total_agreements = sum(1 for game in games if game.get('policy_minimax_agreement_rate', 0) > 0.5)
                            print(f"Policy-Minimax agreement rate: {total_agreements/completed:.1%}")
                        
                        # Memory usage
                        memory_stats = self.model.get_memory_usage()
                        if 'system_used_mb' in memory_stats:
                            print(f"System memory: {memory_stats['system_used_mb']:.0f}MB used")
                        if 'gpu_allocated_mb' in memory_stats:
                            print(f"GPU memory: {memory_stats['gpu_allocated_mb']:.0f}MB allocated")
                        
                except Exception as e:
                    self.logger.error(f"Error generating game {game_id}: {e}")
                    if self.verbose >= 1:
                        print(f"Error generating game {game_id}: {e}")
        
        # Final statistics
        total_time = time.time() - start_time
        total_moves = sum(len(game['moves']) for game in games)
        
        if self.verbose >= 1:
            print(f"\n=== Final Statistics ===")
            print(f"Games generated: {len(games)}")
            print(f"Total time: {total_time:.1f}s")
            if len(games) > 0:
                print(f"Games per second: {len(games) / total_time:.1f}")
                print(f"Average moves per game: {total_moves / len(games):.1f}")
                
                # Policy vs Minimax agreement
                agreement_rates = [game.get('policy_minimax_agreement_rate', 0) for game in games]
                avg_agreement = sum(agreement_rates) / len(agreement_rates)
                print(f"Average policy-minimax agreement: {avg_agreement:.1%}")
            else:
                print("No games generated successfully.")
            
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
        
        if self.verbose >= 1:
            print(f"Saved {len(games)} games to {filename}")
    
    def save_detailed_moves_to_csv(self, games: List[Dict[str, Any]], output_dir: str):
        """
        Save detailed move-by-move data to CSV files.
        One file per game, one row per move.
        
        Args:
            games: List of game data dictionaries
            output_dir: Directory to save CSV files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a summary file
        summary_file = os.path.join(output_dir, "games_summary.csv")
        with open(summary_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'game_id', 'num_moves', 'winner', 'final_trmph', 
                'avg_move_time', 'policy_minimax_agreement_rate'
            ])
            
            for i, game in enumerate(games):
                writer.writerow([
                    i,
                    game['num_moves'],
                    game['winner'],
                    game['final_trmph'],
                    game['avg_move_time'],
                    game.get('policy_minimax_agreement_rate', 0)
                ])
        
        # Create detailed move files
        for i, game in enumerate(games):
            game_file = os.path.join(output_dir, f"game_{i:04d}_moves.csv")
            
            with open(game_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'move_number', 'player', 'board_state',
                    'policy_rank1_move', 'policy_rank1_trmph', 'policy_rank1_prob', 'policy_rank1_value',
                    'policy_rank2_move', 'policy_rank2_trmph', 'policy_rank2_prob', 'policy_rank2_value',
                    'policy_rank3_move', 'policy_rank3_trmph', 'policy_rank3_prob', 'policy_rank3_value',
                    'policy_rank4_move', 'policy_rank4_trmph', 'policy_rank4_prob', 'policy_rank4_value',
                    'policy_rank5_move', 'policy_rank5_trmph', 'policy_rank5_prob', 'policy_rank5_value',
                    'minimax_chosen_move', 'minimax_chosen_move_trmph', 'minimax_value',
                    'agreement', 'legal_moves_count', 'move_time'
                ])
                
                for move_detail in game.get('detailed_moves', []):
                    # Extract policy summary data
                    policy_summary = move_detail.get('policy_summary', [])
                    
                    # Pad with empty values if less than 5 moves
                    while len(policy_summary) < 5:
                        policy_summary.append({
                            'move': None, 'move_trmph': None, 'probability': 0.0, 'value': 0.0
                        })
                    
                    writer.writerow([
                        move_detail['move_number'],
                        move_detail['player'],
                        move_detail['board_state'],
                        # Policy rank 1
                        str(policy_summary[0]['move']),
                        policy_summary[0]['move_trmph'],
                        policy_summary[0]['probability'],
                        policy_summary[0]['value'],
                        # Policy rank 2
                        str(policy_summary[1]['move']),
                        policy_summary[1]['move_trmph'],
                        policy_summary[1]['probability'],
                        policy_summary[1]['value'],
                        # Policy rank 3
                        str(policy_summary[2]['move']),
                        policy_summary[2]['move_trmph'],
                        policy_summary[2]['probability'],
                        policy_summary[2]['value'],
                        # Policy rank 4
                        str(policy_summary[3]['move']),
                        policy_summary[3]['move_trmph'],
                        policy_summary[3]['probability'],
                        policy_summary[3]['value'],
                        # Policy rank 5
                        str(policy_summary[4]['move']),
                        policy_summary[4]['move_trmph'],
                        policy_summary[4]['probability'],
                        policy_summary[4]['value'],
                        # Minimax data
                        str(move_detail['minimax_chosen_move']),
                        move_detail['minimax_chosen_move_trmph'],
                        move_detail['minimax_value'],
                        move_detail['agreement'],
                        move_detail['legal_moves_count'],
                        move_detail['move_time']
                    ])
        
        if self.verbose >= 1:
            print(f"Saved detailed move data:")
            print(f"  Summary: {summary_file}")
            print(f"  Individual games: {len(games)} CSV files in {output_dir}")
            print(f"  Format: Top 5 policy moves with probabilities and values per move")
    
    def save_games_with_details(self, games: List[Dict[str, Any]], base_filename: str):
        """
        Save games with both compressed data and detailed CSV files.
        
        Args:
            games: List of game data dictionaries
            base_filename: Base filename (without extension)
        """
        # Save compressed data
        compressed_file = f"{base_filename}.pkl.gz"
        self.save_games_to_file(games, compressed_file)
        
        # Save detailed CSV files
        csv_dir = f"{base_filename}_detailed_moves"
        self.save_detailed_moves_to_csv(games, csv_dir)
        
        return compressed_file, csv_dir
    
    def clear_cache(self):
        """Clear the model's inference cache."""
        self.model.clear_cache()
    
    def shutdown(self):
        """Clean shutdown of the engine."""
        print("Shutting down SelfPlayEngine...")
        self.model.print_performance_summary()
    
    def save_game_to_stream(self, game_data: Dict[str, Any]):
        """Save a single game to the streaming file in simple text format."""
        if self.streaming_save and self.streaming_file:
            # Simple format: trmph_string <winner>
            # Example: #13,a4g7e9e8f8f7h7h6j5 r
            line = f"{game_data['trmph']} {game_data['winner']}\n"
            with open(self.streaming_file, 'a') as f:
                f.write(line)
    
    def generate_games_streaming(self, num_games: int, board_size: int = 13, 
                               progress_interval: int = 10) -> List[Dict[str, Any]]:
        """
        Generate games with streaming save to avoid data loss.
        
        Args:
            num_games: Number of games to generate
            board_size: Size of the board
            progress_interval: How often to print progress updates
            
        Returns:
            List of game data dictionaries (also saved incrementally)
        """
        start_time = time.time()
        games = []
        
        if self.verbose >= 1:
            print(f"Generating {num_games} games with streaming save...")
        
        for i in range(num_games):
            game_data = self._generate_single_game_essential(board_size)
            games.append(game_data)
            
            # Save to stream immediately
            self.save_game_to_stream(game_data)
            
            # Progress reporting
            if self.verbose >= 1:
                if (i + 1) % progress_interval == 0:
                    print(f"\nGenerated {i + 1}/{num_games} games", end="")
                else:
                    print(".", end="", flush=True)  # Progress dot for each game
        
        total_time = time.time() - start_time
        
        if self.verbose >= 1:
            print(f"\n=== Generation Complete ===")
            print(f"Games generated: {len(games)}")
            print(f"Total time: {total_time:.1f}s")
            print(f"Games per second: {len(games) / total_time:.1f}")
            
            # Winner distribution
            winners = [game['winner'] for game in games]
            red_wins = winners.count('r')
            blue_wins = winners.count('b')
            print(f"Winner distribution: Red {red_wins}, Blue {blue_wins}")
            
            if self.streaming_save:
                print(f"Games saved to: {self.streaming_file}")
                print(f"Format: TRMPH string + winner (e.g., '#13,a4g7e9e8f8f7h7h6j5 r')")
        
        return games