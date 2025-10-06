"""
Utilities for deterministic tournament management.

This module provides utilities to break down the large run_tournament()
function into smaller, focused components.
"""

import os
import itertools
from datetime import datetime
from typing import List, Dict, Any, Set, Tuple, Optional
from pathlib import Path

from hex_ai.inference.strategy_config import StrategyConfig
from hex_ai.inference.tournament import TournamentPlayConfig
from hex_ai.utils.tournament_logging import (
    write_tournament_trmph_header, 
    find_available_csv_filename,
    append_trmph_winner_line
)


def setup_tournament_output(output_dir_prefix: str) -> Tuple[str, str]:
    """
    Set up tournament output directory and return paths.
    
    Args:
        output_dir_prefix: Prefix for output directory (e.g., "data/tournament_play/deterministic_tournament_")
        
    Returns:
        Tuple of (output_dir, openings_file_path)
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_dir = f"{output_dir_prefix}{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    openings_file = os.path.join(output_dir, "openings.txt")
    
    return output_dir, openings_file


def save_opening_positions(openings: List[Any], openings_file: str) -> None:
    """
    Save opening positions to a text file.
    
    Args:
        openings: List of OpeningPosition objects
        openings_file: Path to save openings to
    """
    with open(openings_file, 'w') as f:
        for i, opening in enumerate(openings):
            f.write(f"Opening {i+1}: {opening.get_trmph_string()}\n")


def setup_strategy_pair_files(output_dir: str, strategy_a: StrategyConfig, strategy_b: StrategyConfig) -> Tuple[str, str]:
    """
    Set up output files for a strategy pair.
    
    Args:
        output_dir: Tournament output directory
        strategy_a: First strategy configuration
        strategy_b: Second strategy configuration
        
    Returns:
        Tuple of (trmph_file_path, csv_file_path)
    """
    pair_name = f"{strategy_a.name}_vs_{strategy_b.name}"
    trmph_file = os.path.join(output_dir, f"{pair_name}.trmph")
    csv_file = os.path.join(output_dir, f"{pair_name}.csv")
    
    return trmph_file, csv_file


def create_play_config_for_pair(
    strategy_a: StrategyConfig, 
    strategy_b: StrategyConfig, 
    temperature: float, 
    seed: Optional[int],
    command_line: str
) -> TournamentPlayConfig:
    """
    Create a TournamentPlayConfig for a strategy pair.
    
    Args:
        strategy_a: First strategy configuration
        strategy_b: Second strategy configuration
        temperature: Global temperature fallback
        seed: Random seed
        command_line: Command line that was used to run the tournament
        
    Returns:
        TournamentPlayConfig for the pair
    """
    # Create per-participant temperature mapping
    participant_temperatures = {}
    if strategy_a.temperature is not None:
        participant_temperatures[strategy_a.name] = strategy_a.temperature
    if strategy_b.temperature is not None:
        participant_temperatures[strategy_b.name] = strategy_b.temperature
    
    # Use the first strategy's temperature as the global temperature for the play config
    global_temp = strategy_a.temperature if strategy_a.temperature is not None else temperature
    
    
    return TournamentPlayConfig(
        temperature=global_temp,
        random_seed=seed,
        pie_rule=False,  # Deterministic tournaments don't use pie rule
        strategy="deterministic",
        participant_temperatures=participant_temperatures,
        command_line=command_line
    )


class GameDuplicateTracker:
    """
    Tracks games for duplicate detection in deterministic tournaments.
    """
    
    def __init__(self):
        self.seen_games: Set[str] = set()  # All games across all strategy pairs and openings
        self.current_pair_games: Dict[int, List[str]] = {}  # Games from current strategy pair
    
    def reset_for_new_pair(self) -> None:
        """Reset tracking for a new strategy pair."""
        self.current_pair_games = {}
    
    def check_and_record_games(
        self, 
        opening_idx: int, 
        result_1: Dict[str, Any], 
        result_2: Dict[str, Any],
        strategy_a: StrategyConfig,
        strategy_b: StrategyConfig,
        opening: Any,
        openings: List[Any]
    ) -> None:
        """
        Check for duplicate games and record them.
        
        Args:
            opening_idx: Index of current opening
            result_1: Result of first game (A vs B)
            result_2: Result of second game (B vs A)
            strategy_a: First strategy configuration
            strategy_b: Second strategy configuration
            opening: Current opening position
            openings: All opening positions (for error reporting)
            
        Raises:
            SystemExit: If duplicate games are detected (should be impossible)
        """
        # Create game keys for duplicate detection
        game_1_key = f"{result_1['trmph_str']}_{result_1['winner_char']}"
        game_2_key = f"{result_2['trmph_str']}_{result_2['winner_char']}"
        
        # Check for duplicates in seen games (can happen between different strategy pairs)
        if game_1_key in self.seen_games or game_2_key in self.seen_games:
            print(f"Warning: Duplicate game detected between strategy pairs")
        
        self.seen_games.add(game_1_key)
        self.seen_games.add(game_2_key)
        
        # Store games for this opening
        self.current_pair_games[opening_idx] = [game_1_key, game_2_key]
        
        # Case 1: Check if both strategies produced the same game from this opening
        if game_1_key == game_2_key:
            opening_trmph = opening.get_trmph_string()
            print(f"Warning: {strategy_a.name} and {strategy_b.name} both produced same game {opening_trmph}")
        
        # Case 2: Check if either game duplicates a game from a different opening (SHOULD BE IMPOSSIBLE)
        for other_opening_idx, other_games in self.current_pair_games.items():
            if other_opening_idx != opening_idx:  # Different opening
                duplicate_game = None
                if game_1_key in other_games:
                    duplicate_game = game_1_key
                elif game_2_key in other_games:
                    duplicate_game = game_2_key
                
                if duplicate_game:
                    print(f"ERROR: Game from opening {opening_idx + 1} duplicates game from opening {other_opening_idx + 1}")
                    print(f"  This should be impossible! Opening positions should guarantee unique games.")
                    print(f"  Opening {opening_idx + 1}: {opening.moves}")
                    print(f"  Opening {other_opening_idx + 1}: {openings[other_opening_idx].moves}")
                    raise SystemExit(1)


def create_csv_rows_for_games(
    result_1: Dict[str, Any],
    result_2: Dict[str, Any],
    strategy_a: StrategyConfig,
    strategy_b: StrategyConfig,
    opening_idx: int,
    opening: Any,
    temperature: float
) -> List[Dict[str, Any]]:
    """
    Create CSV rows for a pair of games.
    
    Args:
        result_1: Result of first game (A vs B)
        result_2: Result of second game (B vs A)
        strategy_a: First strategy configuration
        strategy_b: Second strategy configuration
        opening_idx: Index of current opening
        opening: Current opening position
        temperature: Temperature used
        
    Returns:
        List of CSV row dictionaries
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    return [
        {
            "timestamp": timestamp,
            "strategy_a": strategy_a.name,
            "strategy_b": strategy_b.name,
            "model_a": os.path.basename(strategy_a.model_path),
            "model_b": os.path.basename(strategy_b.model_path),
            "opening_idx": opening_idx,
            "opening_source": opening.source_game,
            "game": "A_first",
            "trmph": result_1['trmph_str'],
            "winner": result_1['winner_char'],
            "winner_strategy": result_1['winner_strategy'],
            "num_moves": result_1['num_moves'],
            "opening_length": opening.opening_length,
            "temperature": temperature,
            "strategy_a_time": result_1['strategy_timings'].get(strategy_a.name, 0.0),
            "strategy_b_time": result_1['strategy_timings'].get(strategy_b.name, 0.0),
            "total_game_time": sum(result_1['strategy_timings'].values())
        },
        {
            "timestamp": timestamp,
            "strategy_a": strategy_b.name,
            "strategy_b": strategy_a.name,
            "model_a": os.path.basename(strategy_b.model_path),
            "model_b": os.path.basename(strategy_a.model_path),
            "opening_idx": opening_idx,
            "opening_source": opening.source_game,
            "game": "B_first",
            "trmph": result_2['trmph_str'],
            "winner": result_2['winner_char'],
            "winner_strategy": result_2['winner_strategy'],
            "num_moves": result_2['num_moves'],
            "opening_length": opening.opening_length,
            "temperature": temperature,
            "strategy_a_time": result_2['strategy_timings'].get(strategy_b.name, 0.0),
            "strategy_b_time": result_2['strategy_timings'].get(strategy_a.name, 0.0),
            "total_game_time": sum(result_2['strategy_timings'].values())
        }
    ]


def write_csv_results(rows: List[Dict[str, Any]], csv_file: str) -> None:
    """
    Write CSV results to file.
    
    Args:
        rows: List of CSV row dictionaries
        csv_file: Path to CSV file
    """
    import csv
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    
    # Write header if file doesn't exist
    write_header = not os.path.exists(csv_file)
    
    with open(csv_file, 'a', newline='') as f:
        if rows:  # Only write if we have data
            fieldnames = rows[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if write_header:
                writer.writeheader()
            for row in rows:
                writer.writerow(row)


def report_progress(verbose: int, opening_idx: int, total_openings: int) -> None:
    """
    Report progress for current opening.
    
    Args:
        verbose: Verbosity level
        opening_idx: Current opening index (0-based)
        total_openings: Total number of openings
    """
    if verbose >= 1:
        if opening_idx == 0:
            print(f"  Opening {opening_idx + 1}/{total_openings}", end="", flush=True)
        else:
            print(",", opening_idx + 1, end="", flush=True)


def report_strategy_pair_results(
    verbose: int,
    strategy_a: StrategyConfig,
    strategy_b: StrategyConfig,
    result: Any,
    duplicate_tracker: GameDuplicateTracker
) -> None:
    """
    Report results for a strategy pair.
    
    Args:
        verbose: Verbosity level
        strategy_a: First strategy configuration
        strategy_b: Second strategy configuration
        result: Tournament result object
        duplicate_tracker: Game duplicate tracker for unique games count
    """
    if verbose >= 1:
        # Calculate wins and total games for this specific pair only
        wins_a = result.results[strategy_a.name][strategy_b.name]['wins']
        games_a = result.results[strategy_a.name][strategy_b.name]['games']
        wins_b = result.results[strategy_b.name][strategy_a.name]['wins']
        games_b = result.results[strategy_b.name][strategy_a.name]['games']
        
        # Calculate win rates for this specific pair
        win_rate_a = wins_a / games_a if games_a > 0 else 0.0
        win_rate_b = wins_b / games_b if games_b > 0 else 0.0
        
        print(f" {strategy_a.name}: {win_rate_a:.1%} ({wins_a}/{games_a})")
        print(f" {strategy_b.name}: {win_rate_b:.1%} ({wins_b}/{games_b})")
        print(f"  Timing: {strategy_a.name}={result.strategy_timings[strategy_a.name]:.3f}s, {strategy_b.name}={result.strategy_timings[strategy_b.name]:.3f}s")
        print(f"  Total unique games played: {len(duplicate_tracker.seen_games)}")


def play_strategy_pair_games(
    model_cache: Any,
    strategy_a: StrategyConfig,
    strategy_b: StrategyConfig,
    openings: List[Any],
    temperature: float,
    verbose: int,
    duplicate_tracker: GameDuplicateTracker,
    actual_trmph_file: str,
    actual_csv_file: str,
    play_deterministic_game_func: callable,
    result_tracker: Any = None
) -> List[Dict[str, Any]]:
    """
    Play all games for a strategy pair.
    
    Args:
        model_cache: Model cache for getting models
        strategy_a: First strategy configuration
        strategy_b: Second strategy configuration
        openings: List of opening positions
        temperature: Temperature for move selection
        verbose: Verbosity level
        duplicate_tracker: Game duplicate tracker
        actual_trmph_file: Path to TRMPH output file
        actual_csv_file: Path to CSV output file
        play_deterministic_game_func: Function to play a single game
        result_tracker: Tournament result tracker (optional)
        
    Returns:
        List of game results for this strategy pair
    """
    duplicate_tracker.reset_for_new_pair()
    game_results = []
    
    for opening_idx, opening in enumerate(openings):
        report_progress(verbose, opening_idx, len(openings))
        
        # Game 1: Strategy A (Blue) vs Strategy B (Red)
        result_1 = play_deterministic_game_func(
            model_cache, strategy_a, strategy_b, opening, temperature, verbose=verbose, strategy_a_is_blue=True
        )
        game_results.append(result_1)
        
        # Game 2: Strategy B (Blue) vs Strategy A (Red)
        result_2 = play_deterministic_game_func(
            model_cache, strategy_a, strategy_b, opening, temperature, verbose=verbose, strategy_a_is_blue=False
        )
        game_results.append(result_2)
        
        # Check for duplicates and record games
        duplicate_tracker.check_and_record_games(
            opening_idx, result_1, result_2, strategy_a, strategy_b, opening, openings
        )
        
        # Log TRMPH results
        append_trmph_winner_line(result_1['trmph_str'], result_1['winner_char'], actual_trmph_file)
        append_trmph_winner_line(result_2['trmph_str'], result_2['winner_char'], actual_trmph_file)
        
        # Log CSV results
        csv_rows = create_csv_rows_for_games(
            result_1, result_2, strategy_a, strategy_b, opening_idx, opening, temperature
        )
        write_csv_results(csv_rows, actual_csv_file)
        
        # Record results for tournament tracking
        if result_tracker:
            # Game 1: Strategy A vs Strategy B
            winner_1 = result_1['winner_strategy']
            loser_1 = strategy_b.name if winner_1 == strategy_a.name else strategy_a.name
            result_tracker.record_game_with_timing(winner_1, loser_1, result_1)
            
            # Game 2: Strategy B vs Strategy A
            winner_2 = result_2['winner_strategy']
            loser_2 = strategy_a.name if winner_2 == strategy_b.name else strategy_b.name
            result_tracker.record_game_with_timing(winner_2, loser_2, result_2)
        
        if verbose >= 1:
            print(f":{result_1['winner_char']}/{result_2['winner_char']}", end="", flush=True)
    
    return game_results
