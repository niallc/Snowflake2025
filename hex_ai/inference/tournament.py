import os
import itertools
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
import numpy as np
from hex_ai.inference.simple_model_inference import SimpleModelInference
from hex_ai.inference.fixed_tree_search import minimax_policy_value_search
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
import random
from datetime import datetime
import csv
from pathlib import Path
from hex_ai.enums import Player, Piece
from hex_ai.value_utils import int_to_player

# TODO: In the future, support different play configs (search_widths, MCTS, etc.)
# TODO: Batch model inference for efficiency (currently sequential)
# TODO: Use incremental winner detection (UnionFind) for efficiency
# TODO: Add Swiss, knockout, and other tournament types
# TODO: Add logging and checkpointing for long tournaments

# NOTE: Value head terminology - We use 'value_signed' as a shorthand for [-1, 1] scores
# returned by the value head (tanh activated) and used by MCTS, as opposed to 'value_logits'
# which were the old sigmoid-based outputs.

@dataclass
class GameResult:
    """Result of a single game."""
    winner: Piece  # Piece.BLUE or Piece.RED (color-based)
    trmph_str: str
    winner_char: str  # 'b', 'r', 'd'
    swap_decision: Optional[str]
    move_sequence: List[Tuple[int, int]]

@dataclass
class PieRuleResult:
    """Result of pie rule evaluation."""
    swap: bool
    swap_decision: str
    model_1: SimpleModelInference
    model_2: SimpleModelInference

class TournamentConfig:
    def __init__(self,
                 checkpoint_paths: List[str],
                 board_size: int = 13,
                 search_widths: Optional[List[int]] = None,
                 num_games: int = 10,
                 player_labels: Optional[List[str]] = None,
                 label_to_checkpoint: Optional[Dict[str, str]] = None):
        self.checkpoint_paths = checkpoint_paths
        self.board_size = board_size
        self.search_widths = search_widths
        self.num_games = num_games
        # Player labels for unique identification (defaults to checkpoint paths)
        self.player_labels = player_labels or checkpoint_paths
        # Mapping from player labels to actual checkpoint paths
        self.label_to_checkpoint = label_to_checkpoint or {path: path for path in checkpoint_paths}

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
    def elo_ratings(self, k=32, base=1500) -> Dict[str, float]:
        """
        Calculate ELO ratings using maximum likelihood estimation.
        This approach finds ratings that best explain all observed game results,
        making it order-independent and suitable for tournament settings.
        """
        # For now, use the fallback method which is more robust
        # TODO: Implement proper MLE optimization when we have more time to debug
        return self._fallback_elo_ratings(base)
    
    def _fallback_elo_ratings(self, base=1500) -> Dict[str, float]:
        """
        Fallback ELO calculation using simple averaging approach.
        This is more robust but less sophisticated than the MLE approach.
        """
        # Calculate win rates
        win_rates = self.win_rates()
        
        # Use a more reasonable ELO scale: 400 ELO points = 91% win probability
        # This means 200 ELO points = 75% win probability, which is standard
        # For small win rate differences, we use a linear approximation
        elo_scale = 400 / 0.41  # 400 ELO points per 41% win rate difference (0.5 to 0.91)
        
        ratings = {}
        total_rating_diff = 0
        
        for player, win_rate in win_rates.items():
            # Center around base rating with more reasonable scaling
            rating_diff = (win_rate - 0.5) * elo_scale
            ratings[player] = base + rating_diff
            total_rating_diff += rating_diff
        
        # Ensure zero-sum property by centering around the base
        avg_rating_diff = total_rating_diff / len(ratings)
        for player in ratings:
            ratings[player] = base + (ratings[player] - base) - avg_rating_diff
        
        return ratings



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
        search_widths: Optional[list] = None,
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
        self.search_widths = search_widths  # Legacy support
        self.strategy = strategy
        self.strategy_config = strategy_config or {}
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    def get_temperature_for_participant(self, participant_path: str) -> float:
        """
        Get the temperature for a specific participant.
        If participant_temperatures is set, use that. Otherwise, use the global temperature.
        """
        if participant_path in self.participant_temperatures:
            return self.participant_temperatures[participant_path]
        elif isinstance(self.temperature, list):
            # This shouldn't happen if setup is correct, but provide fallback
            return self.temperature[0] if self.temperature else 0.5
        else:
            return self.temperature

def select_move(state: HexGameState, model: SimpleModelInference, 
                play_config: TournamentPlayConfig, model_path: Optional[str] = None) -> Optional[Tuple[int, int]]:
    """
    Select a move using the configured strategy.
    """
    from hex_ai.inference.move_selection import get_strategy, MoveSelectionConfig
    
    # Get temperature for this specific model
    if model_path is not None:
        temperature = play_config.get_temperature_for_participant(model_path)
    else:
        # Fallback to global temperature
        temperature = play_config.temperature if isinstance(play_config.temperature, float) else 0.5
    
    # Create strategy configuration
    strategy_config = MoveSelectionConfig(
        temperature=temperature,
        **play_config.strategy_config
    )
    
    # Get the strategy and select move
    strategy = get_strategy(play_config.strategy)
    return strategy.select_move(state, model, strategy_config)

def handle_pie_rule(state: HexGameState, model_1: SimpleModelInference, 
                   model_2: SimpleModelInference, play_config: TournamentPlayConfig,
                   verbose: int, model_1_path: Optional[str] = None, 
                   model_2_path: Optional[str] = None, model_1_label: Optional[str] = None,
                   model_2_label: Optional[str] = None) -> PieRuleResult:
    """
    Handle pie rule logic: first move, evaluation, and potential swap.
    """
    # Always play the first move by model_1 (Blue) for consistency
    move = select_move(state, model_1, play_config, model_1_label)
    state = apply_move_to_state(state, *move)
    
    if not play_config.pie_rule:
        return PieRuleResult(swap=False, swap_decision=None, model_1=model_1, model_2=model_2)
    
    # Evaluate win prob for blue after first move
    _, value_signed = model_2.simple_infer(state.board)
    win_prob_blue = ValuePredictor.get_win_probability_for_winner(value_signed, Winner.BLUE)
    
    # Decide whether to swap: Red swaps if Blue's position is too good (>= threshold)
    if win_prob_blue >= play_config.swap_threshold:
        swap = True
        swap_decision = 'swap'
        # Swap model assignments: model_2 becomes blue, model_1 becomes red
        model_1, model_2 = model_2, model_1
    else:
        swap = False
        swap_decision = 'no_swap'
    
    if verbose >= 2:
        print(f"Pie rule: win_prob_blue={win_prob_blue:.3f}, swap={swap}")
    
    return PieRuleResult(swap=swap, swap_decision=swap_decision, model_1=model_1, model_2=model_2)

def play_game_loop(state: HexGameState, model_1: SimpleModelInference, 
                  model_2: SimpleModelInference, play_config: TournamentPlayConfig, 
                  verbose: int, model_1_path: Optional[str] = None, 
                  model_2_path: Optional[str] = None, model_1_label: Optional[str] = None,
                  model_2_label: Optional[str] = None) -> List[Tuple[int, int]]:
    """
    Play the main game loop, returning the sequence of moves.
    """
    move_sequence = []
    
    while not state.game_over:
        # Determine which model to use using Player enum for safety
        current_player_enum = state.current_player_enum
        model = model_1 if current_player_enum == Player.BLUE else model_2
        model_path = model_1_path if current_player_enum == Player.BLUE else model_2_path
        model_label = model_1_label if current_player_enum == Player.BLUE else model_2_label
        
        # Fail fast if there are no legal moves while not game over (shouldn't happen in Hex)
        if not state.get_legal_moves():
            raise ValueError("No legal moves available while game is not over. This indicates a bug.")

        # Select and apply move
        move = select_move(state, model, play_config, model_label)
        if move is None:
            raise ValueError("Move selection returned None. This indicates a model or selection failure.")
        
        move_sequence.append(move)
        state = apply_move_to_state(state, *move)
        
        if verbose >= 2:
            print("-", end="", flush=True)
    
    if verbose >= 3:
        print(f"Game loop finished after {len(move_sequence)} moves")
        print(f"Final state: moves={len(state.move_history)}, game_over={state.game_over}, winner={state.winner}")
        print(f"Move sequence length: {len(move_sequence)}")
    
    return move_sequence, state

def determine_winner(state: HexGameState, model_1: SimpleModelInference, 
                    model_2: SimpleModelInference, swap: bool) -> Tuple[Piece, str]:
    """
    Determine the winner of the game.
    
    Returns:
        Tuple of (winner_color, winner_char) where:
        - winner_color: Piece.BLUE or Piece.RED (color-based)
        - winner_char: "b" or "r" (color-based)
    """
    # Fail fast and use Enum internally; convert at IO boundary
    winner_enum = state.winner_enum
    if winner_enum is None:
        raise ValueError("Game is not over or winner missing")
    if winner_enum.name == 'BLUE':
        winner_color = Piece.BLUE
        winner_char = Piece.BLUE.value
    elif winner_enum.name == 'RED':
        winner_color = Piece.RED
        winner_char = Piece.RED.value
    else:
        raise ValueError(f"Unknown winner enum: {winner_enum}")
    
    return winner_color, winner_char

def log_game_result(result: GameResult, model_1: SimpleModelInference, 
                   model_2: SimpleModelInference, play_config: TournamentPlayConfig,
                   log_file: Optional[str], csv_file: Optional[str],
                   model_1_label: Optional[str] = None, model_2_label: Optional[str] = None) -> None:
    """
    Log the game result to files.
    """
    if log_file:
        append_trmph_winner_line(result.trmph_str, result.winner_char, log_file)
    
    if csv_file:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        
        # Extract model names from labels or use checkpoint paths
        if model_1_label:
            model_1_name = os.path.basename(model_1_label) if model_1_label != getattr(model_1, 'checkpoint_path', str(model_1)) else os.path.basename(getattr(model_1, 'checkpoint_path', str(model_1)))
        else:
            model_1_name = os.path.basename(getattr(model_1, 'checkpoint_path', str(model_1)))
            
        if model_2_label:
            model_2_name = os.path.basename(model_2_label) if model_2_label != getattr(model_2, 'checkpoint_path', str(model_2)) else os.path.basename(getattr(model_2, 'checkpoint_path', str(model_2)))
        else:
            model_2_name = os.path.basename(getattr(model_2, 'checkpoint_path', str(model_2)))
        
        row = {
            "timestamp": timestamp,
            "model_1": model_1_name,
            "model_2": model_2_name,
            "color_1": "blue",
            "trmph": result.trmph_str,
            "winner": result.winner_char,
            "pie_rule": play_config.pie_rule,
            "swap": result.swap_decision,
            "temperature": play_config.temperature,
            "strategy": play_config.strategy,
            "strategy_config": str(play_config.strategy_config),
            "search_widths": str(play_config.search_widths),  # Legacy
            "seed": play_config.random_seed
        }
        
        # Check if CSV exists and has the right columns
        csv_path = Path(csv_file)
        write_header = not csv_path.exists()
        headers = list(row.keys())
        if not write_header:
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                existing_headers = next(reader, None)
                if existing_headers is None or set(headers) != set(existing_headers):
                    write_header = True
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

def play_single_game(model_1: SimpleModelInference, 
                     model_2: SimpleModelInference, 
                     board_size: int, 
                     verbose: int = 1, 
                     log_file: str = None, 
                     csv_file: str = None, 
                     play_config: Optional[TournamentPlayConfig] = None,
                     model_1_path: Optional[str] = None,
                     model_2_path: Optional[str] = None,
                     model_1_label: Optional[str] = None,
                     model_2_label: Optional[str] = None) -> GameResult:
    """
    Play a single game between model_1 (Blue, first) and model_2 (Red, second).
    Supports the pie rule, configurable temperature, and logging.
    Returns a GameResult with all game information.
    """
    if play_config is None:
        play_config = TournamentPlayConfig()
    
    # Initialize game state
    state = HexGameState(
        board=np.full((board_size, board_size), EMPTY_PIECE, dtype='U1'), 
        _current_player=Player.BLUE
    )
    
    # Handle pie rule
    pie_result = handle_pie_rule(state, model_1, model_2, play_config, verbose, model_1_path, model_2_path, model_1_label, model_2_label)
    model_1, model_2 = pie_result.model_1, pie_result.model_2
    
    # Play main game loop
    move_sequence, state = play_game_loop(
        state, model_1, model_2, play_config, verbose, model_1_path, model_2_path, model_1_label, model_2_label
    )
    
    # Convert move sequence to TRMPH string
    trmph_moves = ''.join([rowcol_to_trmph(r, c, board_size) for r, c in move_sequence])
    trmph_str = f"#{board_size},{trmph_moves}"
    
    # Determine winner
    winner_color, winner_char = determine_winner(state, model_1, model_2, pie_result.swap)
    
    # Create result
    result = GameResult(
        winner=winner_color,
        trmph_str=trmph_str,
        winner_char=winner_char,
        swap_decision=pie_result.swap_decision,
        move_sequence=move_sequence
    )
    
    if verbose >= 3:
        model_name_1 = os.path.basename(getattr(model_1, 'checkpoint_path', str(model_1)))
        model_name_2 = os.path.basename(getattr(model_2, 'checkpoint_path', str(model_2)))
        print("".join([
            "*Winner*:", str(winner_char), "(Model ", str(winner_color),
            "). Swapped=", str(pie_result.swap), ".\n",
            "Model_1=", str(model_name_1), ",Model_2=", str(model_name_2)
        ]))
    # elif verbose >= 1:
    #     print(".", end="", flush=True)

    # Log results    
    log_game_result(result, model_1, model_2, play_config, log_file, csv_file, model_1_label, model_2_label)
    
    return result

def play_games_with_each_first(
    model_a_path: str, 
    model_b_path: str, 
    models: Dict[str, SimpleModelInference], 
    config: TournamentConfig, 
    play_config: TournamentPlayConfig, 
    verbose: int, 
    log_file: Optional[str], 
    csv_file: Optional[str],
    model_a_label: Optional[str] = None,
    model_b_label: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Play two games between model_a and model_b, with each going first once.
    Returns a dict with results for both games.
    """
    # Game 1: model_a goes first (blue), model_b second (red)
    result_1 = play_single_game(
        models[model_a_path], models[model_b_path], config.board_size,
        verbose, log_file, csv_file, play_config, model_a_path, model_b_path,
        model_a_label, model_b_label
    )
    
    # Game 2: model_b goes first (blue), model_a second (red)  
    result_2 = play_single_game(
        models[model_b_path], models[model_a_path], config.board_size,
        verbose, log_file, csv_file, play_config, model_b_path, model_a_path,
        model_b_label, model_a_label
    )
    
    return {
        'model_a_first': {
            'winner_position': result_1.winner,  # Piece.BLUE or Piece.RED based on color
            'winner_model': model_a_label if result_1.winner == Piece.BLUE else model_b_label,
            'loser_model': model_b_label if result_1.winner == Piece.BLUE else model_a_label,
            'trmph_str': result_1.trmph_str,
            'winner_char': result_1.winner_char,
            'swap_decision': result_1.swap_decision
        },
        'model_b_first': {
            'winner_position': result_2.winner,  # Piece.BLUE or Piece.RED based on color
            'winner_model': model_b_label if result_2.winner == Piece.BLUE else model_a_label,
            'loser_model': model_a_label if result_2.winner == Piece.BLUE else model_b_label,
            'trmph_str': result_2.trmph_str,
            'winner_char': result_2.winner_char,
            'swap_decision': result_2.swap_decision
        }
    }

def run_round_robin_tournament(
    config: TournamentConfig,
    verbose: int = 1,
    log_file: str = None,
    csv_file: str = None,
    play_config: Optional[TournamentPlayConfig] = None
) -> Tuple[TournamentResult, str, str]:
    """
    For each pair of models, play num_games games with each color assignment.
    Passes play_config to each game for pie rule, temperature, and randomness.
    """
    if play_config is None:
        play_config = TournamentPlayConfig()
    
    # Preload models for efficiency
    from hex_ai.inference.model_cache import preload_tournament_models, get_model_cache
    preload_tournament_models(config.checkpoint_paths)
    
    # Write header to .trmph file if specified
    actual_log_file = log_file
    if log_file:
        actual_log_file = write_tournament_trmph_header(log_file, config.checkpoint_paths, config.num_games, play_config, config.board_size)
    
    # Find available CSV filename if specified
    actual_csv_file = csv_file
    if csv_file:
        actual_csv_file = find_available_csv_filename(csv_file)
    
    # Get cached models
    model_cache = get_model_cache()
    models = {path: model_cache.get_simple_model(path) for path in config.checkpoint_paths}
    result = TournamentResult(config.player_labels)
    
    for model_a_path, model_b_path in itertools.combinations(config.checkpoint_paths, 2):
        # Get player labels for this pair
        model_a_label = config.player_labels[config.checkpoint_paths.index(model_a_path)]
        model_b_label = config.player_labels[config.checkpoint_paths.index(model_b_path)]
        
        print(f"\nPlaying {config.num_games} games: {model_a_label} vs {model_b_label}")
        
        for game_idx in range(config.num_games):
            # Play both games (each model goes first once)
            game_results = play_games_with_each_first(
                model_a_path, model_b_path, models, config, play_config, 
                verbose, actual_log_file, actual_csv_file, model_a_label, model_b_label
            )
            
            # Record results for both games using player labels
            # The game results now return player labels directly
            result.record_game(
                game_results['model_a_first']['winner_model'],
                game_results['model_a_first']['loser_model']
            )
            result.record_game(
                game_results['model_b_first']['winner_model'], 
                game_results['model_b_first']['loser_model']
            )
            
            if verbose >= 2:
                print(f"Game {game_idx+1} of {config.num_games} between {model_a_label} and {model_b_label} completed.")
                
                # Print details for both games
                for game_key, game_name in [('model_a_first', f'{model_a_label} first'), 
                                          ('model_b_first', f'{model_b_label} first')]:
                    game_data = game_results[game_key]
                    print(f"  {game_name}:")
                    print(f"    Trmph: {game_data['trmph_str']}")
                    print(f"    Winner: {game_data['winner_char']} ({game_data['winner_model']})")
                    print(f"    Swap: {game_data['swap_decision']}")

                print(f"Pie rule: {play_config.pie_rule}, Temperature: {play_config.temperature}, "
                      f"Random seed: {play_config.random_seed}")
                print(f"Strategy: {play_config.strategy}, Config: {play_config.strategy_config}")
                print(f"Search widths: {config.search_widths}")  # Legacy
            if verbose >= 1:
                print(f"{game_idx+1},", end="", flush=True)
        
        # Print match results after all games between this pair are complete
        if verbose >= 1:
            print()  # New line after game numbers
            
            # Calculate head-to-head stats for this match
            model_a_name = model_a_label
            model_b_name = model_b_label
            
            # Get results for this specific pair
            model_a_wins = result.results[model_a_label][model_b_label]['wins']
            model_b_wins = result.results[model_b_label][model_a_label]['wins']
            total_games = model_a_wins + model_b_wins
            
            if total_games > 0:
                from hex_ai.utils.tournament_stats import calculate_head_to_head_stats, print_head_to_head_stats
                stats = calculate_head_to_head_stats(model_a_name, model_b_name, model_a_wins, model_b_wins, total_games)
                print_head_to_head_stats(stats)
    
    print("\nDone.")
    return result, actual_log_file, actual_csv_file

# Example usage (to be moved to CLI or script):
if __name__ == "__main__":
    # Example: compare two checkpoints
    from hex_ai.inference.model_config import get_model_path
    checkpoints = [
        get_model_path("previous_best"),
        get_model_path("current_best")
    ]
    config = TournamentConfig(checkpoint_paths=checkpoints, num_games=4)
    result = run_round_robin_tournament(config, verbose=1)
    result.print_summary()
    result.print_elo() 