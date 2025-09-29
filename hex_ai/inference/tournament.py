import numpy as np
import random
from typing import List, Dict, Optional, Any, Union
from datetime import datetime

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

    def elo_ratings(self, base: float = 1500.0) -> Dict[str, float]:
        """
        Calculate Elo ratings using order-independent win rate analysis.
        
        This method uses the inverse of the ELO expected score formula to convert
        win rates to rating differences: rating_diff = 400 * log10(win_rate / (1 - win_rate))
        
        The approach is order-independent, mathematically sound, and robust.
        """
        import math
        
        win_rates = self.win_rates()
        
        # Convert win rates to rating differences using ELO formula
        # If win_rate = 1/(1 + 10^(-rating_diff/400)), then rating_diff = 400 * log10(win_rate/(1-win_rate))
        rating_diffs = {}
        for name, win_rate in win_rates.items():
            if win_rate == 0.5:
                rating_diffs[name] = 0.0
            elif win_rate > 0.0 and win_rate < 1.0:
                # Use ELO formula to convert win rate to rating difference
                rating_diffs[name] = 400 * math.log10(win_rate / (1 - win_rate))
            elif win_rate == 1.0:
                # Perfect win rate: assign maximum reasonable rating difference
                rating_diffs[name] = 400 * math.log10(0.99 / 0.01)  # ~800 points
            else:  # win_rate == 0.0
                # Perfect loss rate: assign minimum reasonable rating difference  
                rating_diffs[name] = 400 * math.log10(0.01 / 0.99)  # ~-800 points
        
        # Convert rating differences to absolute ratings, centered around base
        ratings = {name: base + rating_diffs[name] for name in self.participants}
        
        # Normalize so average is base (handles edge cases and ensures consistency)
        avg_rating = sum(ratings.values()) / len(ratings)
        avg_offset = avg_rating - base
        ratings = {name: rating - avg_offset for name, rating in ratings.items()}
        
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


# Legacy functions removed - game logging is now handled by the new tournament system