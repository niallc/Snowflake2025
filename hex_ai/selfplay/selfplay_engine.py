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
from ..inference.fixed_tree_search import minimax_policy_value_search, minimax_policy_value_search_with_batching
from ..config import BLUE_PLAYER, RED_PLAYER
from ..utils import format_conversion as fc
from ..value_utils import get_top_k_legal_moves, policy_logits_to_probs, get_top_k_moves_with_probs
from ..training_utils import get_device  # Use centralized device detection

# Type annotation for PositionCollector (imported locally to avoid circular imports)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..inference.fixed_tree_search import PositionCollector


class SelfPlayEngine:
    """High-performance self-play engine with optimized inference and detailed logging."""
    
    def __init__(self, model_path: str, num_workers: int = 4, batch_size: int = 32, 
                 cache_size: int = 10000, search_widths: List[int] = None, 
                 temperature: float = 1.0, enable_caching: bool = True, 
                 verbose: int = 1, save_essential_only: bool = False,
                 streaming_save: bool = False, streaming_file: str = None,
                 use_batched_inference: bool = True):
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
            use_batched_inference: Whether to use batched inference for better performance
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
        self.use_batched_inference = use_batched_inference
        
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
            'games_per_second': 0.0,
            'policy_vs_minimax_agreement': 0,
            'policy_vs_minimax_disagreement': 0
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
            print(f"  Batched inference: {use_batched_inference}")

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
            
            if self.use_batched_inference:
                # Use batched inference for better performance
                move_data = self._generate_move_with_batching(state)
                policy, value, policy_top_moves, policy_move_values, minimax_move, minimax_value = move_data
            else:
                # Use individual inference with debug information
                move_data = self._generate_move_with_debug(state)
                policy, value, policy_top_moves, policy_move_values, minimax_move, minimax_value = move_data
            
            # Get policy head's preferred move (top 1)
            policy_preferred_move = policy_top_moves[0][0] if policy_top_moves else None
            
            # Store game state
            moves.append(state.to_trmph())
            policies.append(policy)
            values.append(value)
            
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
                'legal_moves_count': len(state.get_legal_moves()),
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
            'move_times': move_times,
            'policy_vs_minimax_agreement_rate': self.stats.get('policy_vs_minimax_agreement', 0) / len(moves) if moves else 0
        }

    def _generate_move_core(self, state: HexGameState) -> Tuple[Tuple[int, int], float]:
        """
        Generate move using minimax search (core functionality only).
        
        This is the minimal function that just gets the best move and value.
        No debug information is collected.
        
        Args:
            state: Current game state
            
        Returns:
            Tuple of (best_move, best_value)
        """
        # Get model prediction for current position
        policy, value = self.model.simple_infer(state.board)
        
        # Run minimax search to find best move
        minimax_move, minimax_value = minimax_policy_value_search(
            state=state,
            model=self.model,
            widths=self.search_widths,
            temperature=self.temperature
        )
        
        return minimax_move, minimax_value

    def _generate_move_with_debug(self, state: HexGameState) -> Tuple:
        """
        Generate move with debug information for detailed analysis.
        
        This function collects additional debug information:
        - Top policy moves and their probabilities
        - Value predictions for top policy moves
        - Policy vs minimax agreement tracking
        
        Args:
            state: Current game state
            
        Returns:
            Tuple of (policy, value, policy_top_moves, policy_move_values, minimax_move, minimax_value)
        """
        # Configuration: number of top policy moves to evaluate for debugging/logging
        POLICY_DEBUG_WIDTH = 5  # Evaluate top 5 policy moves for debugging
        
        # Get model prediction
        policy, value = self.model.simple_infer(state.board)
        
        # Get policy head's top moves for debugging/logging
        legal_moves = state.get_legal_moves()
        policy_probs = policy_logits_to_probs(policy, self.temperature)
        
        # Find policy head's top moves for debugging
        policy_top_moves = get_top_k_moves_with_probs(
            policy, legal_moves, state.board.shape[0], k=POLICY_DEBUG_WIDTH, temperature=self.temperature
        )
        
        # Get value predictions for top policy moves
        policy_move_values = []
        for move, prob in policy_top_moves:
            # Create temporary state to evaluate this move
            temp_state = state.make_move(*move)
            _, move_value = self.model.simple_infer(temp_state.board)
            policy_move_values.append(move_value)
        
        # Run minimax search to find best move
        minimax_move, minimax_value = minimax_policy_value_search(
            state=state,
            model=self.model,
            widths=self.search_widths,
            temperature=self.temperature
        )
        
        return policy, value, policy_top_moves, policy_move_values, minimax_move, minimax_value

    def _generate_move_individual(self, state: HexGameState) -> Tuple:
        """
        Generate move using individual inference (original method).
        
        DEPRECATED: Use _generate_move_core() for performance or _generate_move_with_debug() for analysis.
        This method is kept for backward compatibility but should be avoided.
        """
        return self._generate_move_with_debug(state)

    def _generate_move_with_batching(self, state: HexGameState) -> Tuple:
        """
        Generate move using batched inference for better performance.
        
        This method is a batched version of _generate_move_individual that produces
        identical results but with better GPU utilization. It follows this flow:
        1. Get policy and value for current position
        2. Get top-k policy moves for debugging/logging (separate from search width)
        3. Get values for those top-k policy moves using batched inference
        4. Run minimax search (which uses search_widths for actual tree exploration)
        5. Return all results
        
        The top-k policy moves are used for debugging/logging to understand:
        - Policy vs minimax agreement
        - Model behavior analysis
        - Detailed move analysis
        
        NOTE: This is separate from search_widths. The search uses search_widths[0] 
        for the actual tree exploration, while this evaluates top-k policy moves 
        for debugging purposes.
        
        Args:
            state: Current game state
            
        Returns:
            Tuple of (policy, value, policy_top_moves, policy_move_values, minimax_move, minimax_value)
        """
        from ..inference.fixed_tree_search import PositionCollector
        
        # Configuration: number of top policy moves to evaluate for debugging/logging
        # This is separate from search_widths - it's just for understanding model behavior
        # The actual search uses search_widths[0] for tree exploration
        POLICY_DEBUG_WIDTH = 5  # Evaluate top 5 policy moves for debugging
        
        # Create position collector for batched inference
        collector = PositionCollector(self.model)
        
        # Step 1: Get policy and value for current position
        # Note: We get both policy and value in one call since the model outputs both
        policy, value = self.model.simple_infer(state.board)
        
        # Step 2: Get top-k policy moves for debugging/logging
        policy_top_moves = self._get_top_policy_moves(policy, state, POLICY_DEBUG_WIDTH)
        
        # Step 3: Get values for top-k policy moves using batched inference
        policy_move_values = self._get_policy_move_values(collector, state, policy_top_moves)
        
        # Step 4: Run minimax search (this uses search_widths for actual tree exploration)
        minimax_move, minimax_value = minimax_policy_value_search_with_batching(
            state=state,
            model=self.model,
            widths=self.search_widths,  # This controls the actual search tree
            temperature=self.temperature
        )
        
        return policy, value, policy_top_moves, policy_move_values, minimax_move, minimax_value

    def _get_top_policy_moves(self, policy: np.ndarray, state: HexGameState, k: int = 5) -> List[Tuple[Tuple[int, int], float]]:
        """
        Get top-k policy moves from policy logits.
        
        Args:
            policy: Policy logits from model
            state: Current game state
            k: Number of top moves to return (default: 5 for debugging/logging)
            
        Returns:
            List of (move, probability) tuples for top-k moves
        """
        legal_moves = state.get_legal_moves()
        return get_top_k_moves_with_probs(
            policy, legal_moves, state.board.shape[0], k=k, temperature=self.temperature
        )

    def _get_policy_move_values(self, collector: 'PositionCollector', state: HexGameState, 
                               policy_top_moves: List[Tuple[Tuple[int, int], float]]) -> List[float]:
        """
        Get value predictions for top policy moves using batched inference.
        
        Args:
            collector: PositionCollector instance for batched processing
            state: Current game state
            policy_top_moves: List of (move, probability) tuples
            
        Returns:
            List of value predictions for each move
        """
        policy_move_values = [None] * len(policy_top_moves)
        
        def value_callback_factory(index):
            def callback(value):
                policy_move_values[index] = value
            return callback
        
        # Collect value requests for each move
        for i, (move, prob) in enumerate(policy_top_moves):
            # Create temporary state by applying the move
            # Note: make_move expects (row, col) as separate arguments, not a tuple
            temp_state = state.make_move(move[0], move[1])
            collector.request_value(temp_state.board, value_callback_factory(i))
        
        # Process all value requests in batch
        collector.process_batches()
        
        return policy_move_values

    def _generate_single_game_essential(self, board_size: int) -> Dict[str, Any]:
        """
        Generate a single self-play game with only essential data.
        
        This method uses the core move generation function for maximum performance.
        No debug information is collected.
        
        Args:
            board_size: Size of the board (ignored, always uses 13)
            
        Returns:
            Dictionary containing essential game data only
        """
        state = HexGameState()  # Always uses 13x13 board
        move_times = []
        
        while not state.game_over:
            move_start = time.time()
            
            # Use core move generation for maximum performance
            if self.use_batched_inference:
                # Use batched minimax search (no debug info)
                move, minimax_value = minimax_policy_value_search_with_batching(
                    state=state,
                    model=self.model,
                    widths=self.search_widths,
                    temperature=self.temperature,
                    verbose=self.verbose
                )
            else:
                # Use core individual inference (no debug info)
                move, minimax_value = self._generate_move_core(state)
            
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
        Generate self-play games with detailed monitoring and statistics.
        
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
                executor.submit(self._generate_single_game_essential, board_size): i 
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
        total_moves = sum(len(game.get('moves', [])) for game in games)
        
        self.stats['games_generated'] += len(games)
        self.stats['total_moves'] += total_moves
        self.stats['total_time'] += total_time
        self.stats['games_per_second'] = len(games) / total_time if total_time > 0 else 0
        
        print(f"Generated {len(games)} games in {total_time:.1f}s ({self.stats['games_per_second']:.1f} games/s)")
        print(f"Average moves per game: {total_moves / len(games):.1f}")
        
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
        import pickle
        import gzip
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save with compression
        with gzip.open(filename, 'wb') as f:
            pickle.dump(games, f)
        
        if self.verbose >= 1:
            print(f"Saved {len(games)} games to {filename}")

    def save_detailed_moves_to_csv(self, games: List[Dict[str, Any]], output_dir: str):
        """
        Save detailed move-by-move data to CSV files.
        
        Args:
            games: List of game data dictionaries
            output_dir: Output directory for CSV files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save summary file
        summary_file = os.path.join(output_dir, "games_summary.csv")
        with open(summary_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'game_id', 'num_moves', 'winner', 'final_trmph', 
                'policy_vs_minimax_agreement_rate', 'avg_move_time'
            ])
            
            for i, game in enumerate(games):
                writer.writerow([
                    i,
                    game.get('num_moves', 0),
                    game.get('winner', 'unknown'),
                    game.get('final_trmph', ''),
                    game.get('policy_vs_minimax_agreement_rate', 0),
                    np.mean(game.get('move_times', [])) if game.get('move_times') else 0
                ])
        
        # Save individual game files
        for i, game in enumerate(games):
            if 'detailed_moves' not in game:
                continue
                
            game_file = os.path.join(output_dir, f"game_{i:04d}_moves.csv")
            with open(game_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'move_number', 'player', 'board_state', 'legal_moves_count',
                    # Policy data
                    'policy_move_1', 'policy_move_1_trmph', 'policy_prob_1', 'policy_value_1',
                    'policy_move_2', 'policy_move_2_trmph', 'policy_prob_2', 'policy_value_2',
                    'policy_move_3', 'policy_move_3_trmph', 'policy_prob_3', 'policy_value_3',
                    'policy_move_4', 'policy_move_4_trmph', 'policy_prob_4', 'policy_value_4',
                    'policy_move_5', 'policy_move_5_trmph', 'policy_prob_5', 'policy_value_5',
                    # Minimax data
                    'minimax_chosen_move', 'minimax_chosen_move_trmph', 'minimax_value',
                    'agreement', 'move_time'
                ])
                
                for move_detail in game['detailed_moves']:
                    policy_summary = move_detail.get('policy_summary', [])
                    
                    # Pad policy summary to 5 moves
                    while len(policy_summary) < 5:
                        policy_summary.append({
                            'move': None, 'move_trmph': None, 'probability': 0.0, 'value': 0.0
                        })
                    
                    writer.writerow([
                        move_detail['move_number'],
                        move_detail['player'],
                        move_detail['board_state'],
                        move_detail['legal_moves_count'],
                        # Policy data
                        str(policy_summary[0]['move']),
                        policy_summary[0]['move_trmph'],
                        policy_summary[0]['probability'],
                        policy_summary[0]['value'],
                        str(policy_summary[1]['move']),
                        policy_summary[1]['move_trmph'],
                        policy_summary[1]['probability'],
                        policy_summary[1]['value'],
                        str(policy_summary[2]['move']),
                        policy_summary[2]['move_trmph'],
                        policy_summary[2]['probability'],
                        policy_summary[2]['value'],
                        str(policy_summary[3]['move']),
                        policy_summary[3]['move_trmph'],
                        policy_summary[3]['probability'],
                        policy_summary[3]['value'],
                        str(policy_summary[4]['move']),
                        policy_summary[4]['move_trmph'],
                        policy_summary[4]['probability'],
                        policy_summary[4]['value'],
                        # Minimax data
                        str(move_detail['minimax_chosen_move']),
                        move_detail['minimax_chosen_move_trmph'],
                        move_detail['minimax_value'],
                        move_detail['agreement'],
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
        """Save a single game to the streaming file."""
        if not self.streaming_save:
            return
            
        trmph = game_data.get('trmph', '')
        winner = game_data.get('winner', '')
        
        with open(self.streaming_file, 'a') as f:
            f.write(f"{trmph} {winner}\n")

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
            return self.generate_games_with_monitoring(num_games, board_size, progress_interval)
        
        start_time = time.time()
        print(f"Generating {num_games} games with streaming save...")
        print(f"Using {'batched' if self.use_batched_inference else 'individual'} inference")
        
        games = []
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit game generation tasks
            future_to_game_id = {
                executor.submit(self._generate_single_game_essential, board_size): i 
                for i in range(num_games)
            }
            
            # Collect results and save immediately
            completed = 0
            for future in as_completed(future_to_game_id):
                game_id = future_to_game_id[future]
                try:
                    game_data = future.result()
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