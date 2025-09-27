import os
import itertools
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
import numpy as np
from hex_ai.inference.simple_model_inference import SimpleModelInference
from hex_ai.utils.format_conversion import rowcol_to_trmph
from hex_ai.value_utils import (
    Winner, 
    ValuePredictor,
    # Add new utilities
    policy_logits_to_probs,
    get_legal_policy_probs,
    select_top_k_moves,
    sample_move_by_value,
    select_policy_move,  # Add the new public function
)
from hex_ai.inference.game_engine import (
    HexGameState,
    apply_move_to_state,  # Add move application utilities
)
from hex_ai.config import (
    BOARD_SIZE,
    TRMPH_BLUE_WIN, TRMPH_RED_WIN, EMPTY_PIECE
)
from hex_ai.utils.tournament_logging import append_trmph_winner_line, log_game_csv, write_tournament_trmph_header, find_available_csv_filename
from hex_ai.utils.tournament_utils import get_player_label_for_checkpoint, extract_model_name_from_label, determine_winner_labels, determine_winner_labels_simple
import random
from datetime import datetime
import csv
from pathlib import Path
from hex_ai.enums import Player, Piece
from hex_ai.value_utils import int_to_player

# NOTE: Value head terminology - We use 'value_signed' as a shorthand for [-1, 1] scores
# returned by the value head (tanh activated) and used by MCTS, as opposed to 'value_logits'
# which were the old sigmoid-based outputs.

class TournamentResult:
    def __init__(self, participants: List[str]):
        self.participants = participants
        self.results = {name: {opponent: {'wins': 0, 'losses': 0, 'games': 0} for opponent in participants if opponent != name} for name in participants}
        self.total_games = 0

    def record_game(self, winner: str, loser: str):
        self.results[winner][loser]['wins'] += 1
        self.results[winner][loser]['games'] += 1
        self.results[loser][winner]['losses'] += 1
        self.results[loser][winner]['games'] += 1
        self.total_games += 1

    def win_rates(self) -> Dict[str, float]:
        win_rates = {}
        for name in self.participants:
            wins = sum(self.results[name][op]['wins'] for op in self.results[name])
            games = sum(self.results[name][op]['games'] for op in self.results[name])
            win_rates[name] = wins / games if games > 0 else 0.0
        return win_rates

    # Optional: Elo calculation (simple version)
    def elo_ratings(self, base: float = 1500.0, k_factor: float = 32.0) -> Dict[str, float]:
        """
        Calculate Elo ratings for all participants.
        Uses a simple implementation that updates ratings after each game.
        """
        ratings = {name: base for name in self.participants}
        
        # Process all games in order
        for name in self.participants:
            for opponent in self.results[name]:
                wins = self.results[name][opponent]['wins']
                losses = self.results[name][opponent]['losses']
                games = wins + losses
                
                if games > 0:
                    # Calculate expected score
                    expected = 1 / (1 + 10 ** ((ratings[opponent] - ratings[name]) / 400))
                    actual = wins / games
                    
                    # Update rating
                    ratings[name] += k_factor * (actual - expected)
        
        return ratings
    
    def _fallback_elo_ratings(self, base=1500) -> Dict[str, float]:
        """
        Fallback Elo calculation using win rates only.
        This is used when the main Elo calculation fails.
        """
        win_rates = self.win_rates()
        
        # Convert win rates to Elo ratings
        # Win rate of 0.5 = base rating
        # Win rate of 1.0 = base + 400
        # Win rate of 0.0 = base - 400
        ratings = {}
        for name, win_rate in win_rates.items():
            if win_rate == 0.5:
                ratings[name] = base
            elif win_rate > 0.5:
                # Positive rating adjustment
                ratings[name] = base + 400 * (win_rate - 0.5) * 2
            else:
                # Negative rating adjustment
                ratings[name] = base - 400 * (0.5 - win_rate) * 2
        
        # Normalize so average is base
        avg_rating = sum(ratings.values()) / len(ratings)
        avg_rating_diff = avg_rating - base
        for name in ratings:
            ratings[name] = base + (ratings[name] - base) - avg_rating_diff
        
        return ratings

    def print_summary(self):
        """Print a summary of tournament results."""
        print("\n" + "="*60)
        print("TOURNAMENT RESULTS")
        print("="*60)
        
        win_rates = self.win_rates()
        elo_ratings = self.elo_ratings()
        
        # Sort by win rate
        sorted_participants = sorted(win_rates.items(), key=lambda x: x[1], reverse=True)
        
        print(f"{'Player':<30} {'Win Rate':<10} {'Elo':<8} {'Games':<6}")
        print("-" * 60)
        
        for name, win_rate in sorted_participants:
            elo = elo_ratings[name]
            games = sum(self.results[name][op]['games'] for op in self.results[name])
            print(f"{name:<30} {win_rate:<10.3f} {elo:<8.0f} {games:<6}")
        
        print(f"\nTotal games played: {self.total_games}")
        print("="*60)

    def print_elo(self):
        """Print Elo ratings in a formatted table."""
        print("\n" + "="*40)
        print("ELO RATINGS")
        print("="*40)
        
        elo_ratings = self.elo_ratings()
        sorted_ratings = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
        
        for i, (name, rating) in enumerate(sorted_ratings, 1):
            print(f"{i:2d}. {name:<25} {rating:6.0f}")
        
        print("="*40)

    def print_head_to_head_stats(self, player1: str, player2: str):
        """Print head-to-head statistics between two players."""
        if player1 not in self.results or player2 not in self.results[player1]:
            print(f"No games played between {player1} and {player2}")
            return
        
        stats = self.results[player1][player2]
        total_games = stats['games']
        wins1 = stats['wins']
        wins2 = stats['losses']  # losses for player1 = wins for player2
        
        print(f"\nHead-to-Head: {player1} vs {player2}")
        print(f"Games: {total_games}")
        print(f"{player1} wins: {wins1} ({wins1/total_games*100:.1f}%)")
        print(f"{player2} wins: {wins2} ({wins2/total_games*100:.1f}%)")


class TournamentPlayConfig:
    """
    Configuration for tournament play, including randomness, temperature, pie rule, and reproducibility.
    If random_seed is None, use a time-based seed for uniqueness.
    
    Temperature can be either:
    - A single float: applies to all participants
    - A list of floats: applies to participants in order (must match number of participants)
    """
    def __init__(
        self,
        temperature: Union[float, List[float]] = 0.5,
        random_seed: Optional[int] = None,
        pie_rule: bool = False,
        swap_threshold: float = 0.5,
        strategy: str = "policy",
        strategy_config: Optional[Dict[str, Any]] = None,
        participant_temperatures: Optional[Dict[str, float]] = None
    ):
        self.temperature = temperature
        self.participant_temperatures = participant_temperatures or {}
        if random_seed is None:
            # Use a time-based seed, but ensure it's in [0, 2**32 - 1] for np.random.seed
            random_seed = int(datetime.now().timestamp() * 1e6) % (2**32)
        self.random_seed = random_seed
        self.pie_rule = pie_rule
        self.swap_threshold = swap_threshold  # Red swaps if Blue's win prob >= this threshold
        self.strategy = strategy
        self.strategy_config = strategy_config or {}
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    def get_temperature_for_participant(self, participant_path: str) -> float:
        """
        Get the temperature for a specific participant.
        First checks participant_temperatures, then falls back to global temperature.
        """
        if participant_path in self.participant_temperatures:
            return self.participant_temperatures[participant_path]
        
        if isinstance(self.temperature, list):
            # If temperature is a list, we need to map participant to index
            # For now, just use the first temperature
            return self.temperature[0] if self.temperature else 0.5
        
        return self.temperature


def log_game_result(result, model_1: SimpleModelInference, 
                   model_2: SimpleModelInference, play_config: TournamentPlayConfig,
                   log_file: Optional[str], csv_file: Optional[str],
                   model_1_label: Optional[str] = None, model_2_label: Optional[str] = None) -> None:
    """
    Log game result to both TRMPH and CSV files.
    """
    if log_file:
        append_trmph_winner_line(result.trmph_str, result.winner_char, log_file)
    
    if csv_file:
        # Create CSV row
        row = {
            'model_a': model_1_label or get_player_label_for_checkpoint(model_1.checkpoint_path),
            'model_b': model_2_label or get_player_label_for_checkpoint(model_2.checkpoint_path),
            'color_a': 'blue',  # model_1 is always blue in this context
            'trmph': result.trmph_str,
            'winner': result.winner_char,
            'swap_decision': result.swap_decision or 'none',
            'temperature': play_config.temperature,
            'pie_rule': play_config.pie_rule,
            'strategy': play_config.strategy
        }
        
        log_game_csv(row, csv_file)


# Example usage (to be moved to CLI or script):
if __name__ == "__main__":
    # Example tournament result
    checkpoints = [
        "checkpoints/model1.pt.gz",
        "checkpoints/model2.pt.gz"
    ]
    
    result = TournamentResult(checkpoints)
    
    print("Example tournament result:")
    result.print_summary()