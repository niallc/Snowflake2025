"""
Batched MCTS-based self-play engine for generating training data.

This module provides a self-play engine that uses the new batched MCTS implementation
for significantly improved performance through batched neural network inference.
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path

from hex_ai.inference.batched_mcts import BatchedNeuralMCTS
from hex_ai.inference.game_engine import HexGameState
from hex_ai.inference.simple_model_inference import SimpleModelInference
from hex_ai.config import BLUE_PLAYER, RED_PLAYER
from hex_ai.training_utils import get_device


class BatchedMCTSSelfPlayEngine:
    """Batched self-play engine using MCTS for move selection."""
    
    def __init__(self, model_path: str, num_simulations: int = 800, 
                 exploration_constant: float = 1.4, temperature: float = 1.0,
                 optimal_batch_size: int = 64, verbose: int = 1):
        """
        Initialize the batched MCTS self-play engine.
        
        Args:
            model_path: Path to the model checkpoint
            num_simulations: Number of MCTS simulations per move
            exploration_constant: PUCT exploration constant
            temperature: Temperature for move selection
            optimal_batch_size: Optimal batch size for neural network inference
            verbose: Verbosity level (0=quiet, 1=normal, 2=detailed)
        """
        self.model_path = model_path
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.temperature = temperature
        self.optimal_batch_size = optimal_batch_size
        self.verbose = verbose
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        if verbose == 0:
            self.logger.setLevel(logging.WARNING)
        elif verbose >= 2:
            self.logger.setLevel(logging.DEBUG)
        
        # Load model
        self.logger.info(f"Loading model from {model_path}")
        self.model = SimpleModelInference(model_path)
        
        # Create batched MCTS engine
        self.mcts = BatchedNeuralMCTS(
            model=self.model,
            exploration_constant=exploration_constant,
            optimal_batch_size=optimal_batch_size,
            verbose=verbose
        )
        
        # Statistics
        self.stats = {
            'games_played': 0,
            'total_moves': 0,
            'total_search_time': 0.0,
            'total_inferences': 0,
            'total_batches_processed': 0,
            'blue_wins': 0,
            'red_wins': 0,
            'cache_hit_rate': 0.0,
            'average_batch_size': 0.0
        }
        
        self.logger.info(f"Batched MCTS Self-Play Engine initialized with {num_simulations} simulations, "
                        f"optimal_batch_size={optimal_batch_size}")
    
    def play_game(self) -> Dict[str, Any]:
        """
        Play a single game using batched MCTS.
        
        Returns:
            Dictionary containing game data and statistics
        """
        start_time = time.time()
        
        # Initialize game
        state = HexGameState()
        moves = []
        search_times = []
        inference_counts = []
        batch_counts = []
        
        self.logger.info("Starting new game")
        
        # Initialize root node for tree reuse
        root = None
        
        # Play until game is over
        while not state.game_over:
            if state.current_player == BLUE_PLAYER:
                player_name = "Blue"
            elif state.current_player == RED_PLAYER:
                player_name = "Red"
            else:
                raise ValueError(f"Unknown player value: {state.current_player}")
            
            if self.verbose >= 2:
                self.logger.debug(f"{player_name}'s turn")
            
            # Run MCTS search
            search_start = time.time()
            if root is None:
                # First move: create new root
                if self.verbose >= 2:
                    self.logger.debug("Creating new MCTS root node")
                # TODO (P3): Understand why we need to search to find the root node.
                #            Isn't the root node the current state?
                root = self.mcts.search(state, self.num_simulations)
            else:
                # Subsequent moves: reuse existing tree
                # The root should already represent the current state
                # Just run additional simulations on the existing tree
                if self.verbose >= 2:
                    self.logger.debug(f"Reusing existing MCTS tree (root visits: {root.visits})")
                root = self.mcts.search(root, self.num_simulations)
            search_time = time.time() - search_start
            
            # Get search statistics
            search_stats = self.mcts.get_search_statistics()
            
            # Reset statistics for next move
            self.mcts.reset_search_statistics()
            
            # Select move
            selected_move = self.mcts.select_move(root, temperature=self.temperature)
            
            # Record statistics
            search_times.append(search_time)
            inference_counts.append(search_stats.get('total_inferences', 0))
            batch_counts.append(search_stats.get('total_batches_processed', 0))
            
            if self.verbose >= 2:
                self.logger.debug(f"{player_name} selected move {selected_move} "
                                f"(search time: {search_time:.2f}s, "
                                f"inferences: {search_stats.get('total_inferences', 0)}, "
                                f"batches: {search_stats.get('total_batches_processed', 0)})")
            
            # Make move and update root for tree reuse
            state = state.make_move(*selected_move)
            moves.append(selected_move)
            
            # Update root to the selected child and detach from parent
            if selected_move in root.children:
                root = root.children[selected_move]
                root.detach_parent()
            else:
                # TODO (P2): If the below indicates a bug, raise an exception and crash. No fallback logic! 
                #            Find errors fast.
                # Fallback: create new root if child doesn't exist
                self.logger.warning(f"Selected move {selected_move} not found in root children, creating new root")
                root = None
            
            # Update statistics
            self.stats['total_moves'] += 1
            self.stats['total_search_time'] += search_time
            self.stats['total_inferences'] += search_stats.get('total_inferences', 0)
            self.stats['total_batches_processed'] += search_stats.get('total_batches_processed', 0)
        
        # Game finished
        game_time = time.time() - start_time
        
        # Determine winner
        # TODO (P2): Consider replacing "blue" and "red" with the relevant Enums / constants from central utilities.
        if state.winner == "blue":
            self.stats['blue_wins'] += 1
            winner = "blue"
        elif state.winner == "red":
            self.stats['red_wins'] += 1
            winner = "red"
        else:
            winner = "draw"
        
        self.stats['games_played'] += 1
        
        # Update derived statistics
        if self.stats['total_moves'] > 0:
            self.stats['cache_hit_rate'] = self.mcts.batch_processor.get_statistics()['cache_hit_rate']
            self.stats['average_batch_size'] = self.mcts.batch_processor.get_statistics()['average_batch_size']
        
        # Log game results
        self.logger.info(f"Game finished: {winner} wins in {len(moves)} moves "
                        f"(time: {game_time:.2f}s)")
        
        # Return game data
        game_data = {
            'moves': moves,
            'winner': winner,
            'num_moves': len(moves),
            'game_time': game_time,
            'search_times': search_times,
            'inference_counts': inference_counts,
            'batch_counts': batch_counts,
            'average_search_time': np.mean(search_times) if search_times else 0,
            'total_inferences': sum(inference_counts),
            'total_batches': sum(batch_counts),
            'cache_hit_rate': self.stats['cache_hit_rate'],
            'average_batch_size': self.stats['average_batch_size']
        }
        
        return game_data
    
    def play_multiple_games(self, num_games: int) -> List[Dict[str, Any]]:
        """
        Play multiple games.
        
        Args:
            num_games: Number of games to play
            
        Returns:
            List of game data dictionaries
        """
        self.logger.info(f"Playing {num_games} games")
        
        games = []
        for game_idx in range(num_games):
            if self.verbose >= 1:
                self.logger.info(f"Playing game {game_idx + 1}/{num_games}")
            
            game_data = self.play_game()
            games.append(game_data)
            
            # Log progress
            if (game_idx + 1) % 10 == 0:
                self._log_progress(game_idx + 1, num_games)
        
        self.logger.info(f"Completed {num_games} games")
        self._log_final_statistics()
        
        return games
    
    def _log_progress(self, games_completed: int, total_games: int):
        """Log progress statistics."""
        if games_completed == 0:
            return
        
        avg_moves = self.stats['total_moves'] / games_completed
        avg_search_time = self.stats['total_search_time'] / self.stats['total_moves'] if self.stats['total_moves'] > 0 else 0
        avg_inferences = self.stats['total_inferences'] / self.stats['total_moves'] if self.stats['total_moves'] > 0 else 0
        blue_win_rate = self.stats['blue_wins'] / games_completed
        red_win_rate = self.stats['red_wins'] / games_completed
        
        self.logger.info(f"Progress: {games_completed}/{total_games} games "
                        f"(avg moves: {avg_moves:.1f}, "
                        f"avg search: {avg_search_time:.2f}s, "
                        f"avg inferences: {avg_inferences:.1f}, "
                        f"Blue: {blue_win_rate:.1%}, Red: {red_win_rate:.1%})")
    
    def _log_final_statistics(self):
        """Log final statistics."""
        if self.stats['games_played'] == 0:
            return
        
        total_games = self.stats['games_played']
        avg_moves = self.stats['total_moves'] / total_games
        avg_search_time = self.stats['total_search_time'] / self.stats['total_moves'] if self.stats['total_moves'] > 0 else 0
        avg_inferences = self.stats['total_inferences'] / self.stats['total_moves'] if self.stats['total_moves'] > 0 else 0
        avg_batches = self.stats['total_batches_processed'] / self.stats['total_moves'] if self.stats['total_moves'] > 0 else 0
        
        blue_win_rate = self.stats['blue_wins'] / total_games
        red_win_rate = self.stats['red_wins'] / total_games
        
        self.logger.info("Final Statistics:")
        self.logger.info(f"  Games played: {total_games}")
        self.logger.info(f"  Total moves: {self.stats['total_moves']}")
        self.logger.info(f"  Average moves per game: {avg_moves:.1f}")
        self.logger.info(f"  Average search time per move: {avg_search_time:.3f}s")
        self.logger.info(f"  Average inferences per move: {avg_inferences:.1f}")
        self.logger.info(f"  Average batches per move: {avg_batches:.1f}")
        self.logger.info(f"  Blue win rate: {blue_win_rate:.1%}")
        self.logger.info(f"  Red win rate: {red_win_rate:.1%}")
        self.logger.info(f"  Total search time: {self.stats['total_search_time']:.1f}s")
        self.logger.info(f"  Total inferences: {self.stats['total_inferences']}")
        self.logger.info(f"  Total batches processed: {self.stats['total_batches_processed']}")
        self.logger.info(f"  Cache hit rate: {self.stats['cache_hit_rate']:.1%}")
        self.logger.info(f"  Average batch size: {self.stats['average_batch_size']:.1f}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics."""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset statistics."""
        self.stats = {
            'games_played': 0,
            'total_moves': 0,
            'total_search_time': 0.0,
            'total_inferences': 0,
            'total_batches_processed': 0,
            'blue_wins': 0,
            'red_wins': 0,
            'cache_hit_rate': 0.0,
            'average_batch_size': 0.0
        }
    
    def clear_cache(self):
        """Clear the batch processor cache."""
        self.mcts.batch_processor.clear_cache()
        self.logger.info("Cache cleared")


def main():
    """Main function for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Batched MCTS Self-Play Engine")
    parser.add_argument("model_path", help="Path to model checkpoint")
    parser.add_argument("--games", type=int, default=1, help="Number of games to play")
    parser.add_argument("--simulations", type=int, default=800, help="MCTS simulations per move")
    parser.add_argument("--batch-size", type=int, default=64, help="Optimal batch size for inference")
    parser.add_argument("--temperature", type=float, default=1.0, help="Move selection temperature")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO if args.verbose >= 1 else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create engine and play games
    engine = BatchedMCTSSelfPlayEngine(
        model_path=args.model_path,
        num_simulations=args.simulations,
        optimal_batch_size=args.batch_size,
        temperature=args.temperature,
        verbose=args.verbose
    )
    
    games = engine.play_multiple_games(args.games)
    
    print(f"\nCompleted {len(games)} games")
    for i, game in enumerate(games):
        print(f"Game {i+1}: {game['winner']} wins in {game['num_moves']} moves "
              f"(inferences: {game['total_inferences']}, batches: {game['total_batches']})")


if __name__ == "__main__":
    main()
