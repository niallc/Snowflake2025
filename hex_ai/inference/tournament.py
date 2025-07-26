import os
import itertools
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from hex_ai.inference.simple_model_inference import SimpleModelInference
from hex_ai.inference.game_engine import HexGameState
from hex_ai.inference.fixed_tree_search import minimax_policy_value_search
from hex_ai.value_utils import (
    Winner, 
    get_win_prob_from_model_output,
    # Add new utilities
    policy_logits_to_probs,
    get_legal_policy_probs,
    select_top_k_moves,
    sample_move_by_value,
    select_policy_move,  # Add the new public function
    apply_move_to_state,  # Add move application utilities
)
from hex_ai.config import BLUE_PLAYER, RED_PLAYER
from hex_ai.utils.tournament_logging import append_trmph_winner_line, log_game_csv
import random
from datetime import datetime
import csv
from pathlib import Path

# TODO: In the future, support different play configs (search widths, MCTS, etc.)
# TODO: Batch model inference for efficiency (currently sequential)
# TODO: Use incremental winner detection (UnionFind) for efficiency
# TODO: Add Swiss, knockout, and other tournament types
# TODO: Add logging and checkpointing for long tournaments

class TournamentConfig:
    def __init__(self,
                 checkpoint_paths: List[str],
                 board_size: int = 13,
                 search_widths: Optional[List[int]] = None,
                 num_games: int = 10):
        self.checkpoint_paths = checkpoint_paths
        self.board_size = board_size
        self.search_widths = search_widths
        self.num_games = num_games

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

    def print_summary(self):
        print("\nTournament Results:")
        for name in self.participants:
            print(f"{name}: {self.win_rates()[name]*100:.1f}% win rate ({sum(self.results[name][op]['wins'] for op in self.results[name])} wins / {sum(self.results[name][op]['games'] for op in self.results[name])} games)")
        print()

    # Optional: Elo calculation (simple version)
    def elo_ratings(self, k=32, base=1500) -> Dict[str, float]:
        ratings = {name: base for name in self.participants}
        for name in self.participants:
            for op in self.results[name]:
                games = self.results[name][op]['games']
                wins = self.results[name][op]['wins']
                losses = self.results[name][op]['losses']
                for _ in range(wins):
                    expected = 1 / (1 + 10 ** ((ratings[op] - ratings[name]) / 400))
                    ratings[name] += k * (1 - expected)
                    ratings[op] += k * (0 - (1 - expected))
                for _ in range(losses):
                    expected = 1 / (1 + 10 ** ((ratings[op] - ratings[name]) / 400))
                    ratings[name] += k * (0 - expected)
                    ratings[op] += k * (1 - (0 - expected))
        return ratings

    def print_elo(self):
        elos = self.elo_ratings()
        print("Elo Ratings:")
        for name, elo in sorted(elos.items(), key=lambda x: -x[1]):
            print(f"  {name}: {elo:.1f}")
        print()

class TournamentPlayConfig:
    """
    Configuration for tournament play, including randomness, temperature, pie rule, and reproducibility.
    If random_seed is None, use a time-based seed for uniqueness.
    """
    def __init__(
        self,
        temperature: float = 0.5,
        random_seed: Optional[int] = None,
        pie_rule: bool = False,
        pie_threshold: tuple = (0.495, 0.505),
        search_widths: Optional[list] = None
    ):
        self.temperature = temperature
        if random_seed is None:
            # Use a time-based seed, but ensure it's in [0, 2**32 - 1] for np.random.seed
            random_seed = int(datetime.now().timestamp() * 1e6) % (2**32)
        self.random_seed = random_seed
        self.pie_rule = pie_rule
        self.pie_threshold = pie_threshold  # (min, max) win prob for swap
        self.search_widths = search_widths
        random.seed(random_seed)
        np.random.seed(random_seed)

def play_single_game(model_1: SimpleModelInference, 
                     model_2: SimpleModelInference, 
                     board_size: int, 
                     search_widths: Optional[list] = None, 
                     verbose: int = 1, 
                     log_file: str = None, 
                     csv_file: str = None, 
                     play_config: Optional[TournamentPlayConfig] = None) -> str:
    """
    Play a single game between model_1 (Blue, first) and model_2 (Red, second).
    Supports the pie rule, configurable temperature, and logging.
    Returns the winner's name ("1" or "2").
    Uses Winner enums for player identity throughout.
    Verbosity levels:
      0: Silent (no output)
      1: High-level info (pie rule swaps, game results)
      2: Per-move details
    """
    if play_config is None:
        play_config = TournamentPlayConfig()
    state = HexGameState(board=np.zeros((board_size, board_size), dtype=np.int8), current_player=BLUE_PLAYER)
    move_num = 0
    move_sequence = []
    swap = False
    swap_decision = None
    # --- PIE RULE LOGIC ---
    if play_config.pie_rule:
        # First move by model_1 (Blue)
        if search_widths:
            move, _ = minimax_policy_value_search(state, model_1, search_widths)
        else:
            move = model_select_move(model_1, state, temperature=play_config.temperature)
        move_sequence.append(move)
        state = apply_move_to_state(state, *move)
        move_num += 1
        # Evaluate win prob for blue after first move
        policy_logits, value_logit = model_2.infer(state.board)
        win_prob_blue = get_win_prob_from_model_output(value_logit, Winner.BLUE)
        # Decide whether to swap
        min_thr, max_thr = play_config.pie_threshold
        if min_thr <= win_prob_blue <= max_thr:
            swap = True
            swap_decision = 'swap'
            # Swap model assignments: model_2 becomes blue, model_1 becomes red
            model_1, model_2 = model_2, model_1
        else:
            swap_decision = 'no_swap'
        if verbose >= 1:
            print(f"Pie rule: win_prob_blue={win_prob_blue:.3f}, swap={swap}")
    # --- MAIN GAME LOOP ---
    while not state.game_over:
        if state.current_player == BLUE_PLAYER:
            model = model_1
        else:
            model = model_2
        if search_widths:
            move, _ = minimax_policy_value_search(state, model, search_widths)
        else:
            move = model_select_move(model, state, temperature=play_config.temperature)
        if move is None:
            break  # No valid moves
        move_sequence.append(move)
        state = apply_move_to_state(state, *move)
        move_num += 1
        if verbose >= 2:
            # print(f"Move {move_num}: Player {state.current_player} played {move}")
            print("-", end="", flush=True)
    # Convert move_sequence to trmph string
    from hex_ai.utils.format_conversion import rowcol_to_trmph
    trmph_moves = ''.join([rowcol_to_trmph(r, c, board_size) for r, c in move_sequence])
    trmph_str = f"#{board_size},{trmph_moves}"
    # Determine winner for logging: 'b' if blue (first mover after swap) wins, 'r' if red wins
    if state.winner == "blue":
        winner_char = 'b'
        # Blue player is the first model (model_1) unless there was a swap
        result = "1" if not swap else "2"
    elif state.winner == "red":
        winner_char = 'r'
        # Red player is the second model (model_2) unless there was a swap
        result = "2" if not swap else "1"
    else:
        winner_char = 'd'  # draw (if ever possible)
        result = "draw"
    if verbose >= 1:
        # Get the model name (filename) from the checkpoint path
        model_name_1 = os.path.basename(getattr(model_1, 'checkpoint_path', str(model_1)))
        model_name_2 = os.path.basename(getattr(model_2, 'checkpoint_path', str(model_2)))
        print("".join([
            "\n*Winner*:", str(winner_char), "(Model ", str(result), "). Swapped=", str(swap), ".\n",
            "Model_1=", str(model_name_1), ",Model_2=", str(model_name_2)
        ]))
    # --- LOGGING ---
    if log_file:
        append_trmph_winner_line(trmph_str, winner_char, log_file)
    if csv_file:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        # Compose row with all config info
        row = {
            "timestamp": timestamp,
            "model_1": os.path.basename(getattr(model_1, 'checkpoint_path', str(model_1))),
            "model_2": os.path.basename(getattr(model_2, 'checkpoint_path', str(model_2))),
            "color_1": "blue",
            "trmph": trmph_str,
            "winner": winner_char,
            "pie_rule": play_config.pie_rule,
            "swap": swap_decision,
            "temperature": play_config.temperature,
            "search_widths": str(search_widths),
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
    return result, trmph_str, winner_char, swap_decision

# Helper: model_select_move
# TODO: Check duplication vs. scripts/play_vs_model_cli.py and select_policy_move
def model_select_move(model: SimpleModelInference,
                      state: HexGameState, top_k=20,
                      temperature=0.5):
    """
    Select a move using policy and value heads with centralized utilities.
    """
    # Get top-k moves using centralized utilities
    policy_logits, _ = model.infer(state.board)
    policy_probs = policy_logits_to_probs(policy_logits, temperature)
    legal_moves = state.get_legal_moves()
    legal_policy = get_legal_policy_probs(policy_probs, legal_moves, state.board.shape[0])
    
    if len(legal_policy) == 0:
        return None
    
    # Get top-k moves using centralized utility
    topk_moves = select_top_k_moves(legal_policy, legal_moves, top_k)
    
    # Evaluate each with value head
    move_values = []
    for move in topk_moves:
        temp_state = apply_move_to_state(state, *move)
        _, value_logit = model.infer(temp_state.board)
        move_values.append(value_logit)
    
    # Sample move using centralized utility
    chosen_idx = sample_move_by_value(move_values, temperature)
    return topk_moves[chosen_idx]

def play_games_with_each_first(
    model_a_path: str, 
    model_b_path: str, 
    models: Dict[str, SimpleModelInference], 
    config: TournamentConfig, 
    play_config: TournamentPlayConfig, 
    verbose: int, 
    log_file: Optional[str], 
    csv_file: Optional[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Play two games between model_a and model_b, with each going first once.
    Returns a dict with results for both games.
    
    Args:
        model_a_path: Path to first model checkpoint
        model_b_path: Path to second model checkpoint  
        models: Dictionary mapping checkpoint paths to model instances
        config: Tournament configuration
        play_config: Play configuration (temperature, pie rule, etc.)
        verbose: Verbosity level
        log_file: Optional log file path
        csv_file: Optional CSV file path
        
    Returns:
        Dictionary with results for both games, structured as:
        {
            'model_a_first': {winner_position, winner_model, loser_model, trmph_str, winner_char, swap_decision},
            'model_b_first': {winner_position, winner_model, loser_model, trmph_str, winner_char, swap_decision}
        }
    """
    # Game 1: model_a goes first (blue), model_b second (red)
    winner_1, trmph_str_1, winner_char_1, swap_decision_1 = play_single_game(
        models[model_a_path], models[model_b_path], config.board_size,
        config.search_widths, verbose, log_file, csv_file, play_config
    )
    
    # Game 2: model_b goes first (blue), model_a second (red)  
    winner_2, trmph_str_2, winner_char_2, swap_decision_2 = play_single_game(
        models[model_b_path], models[model_a_path], config.board_size,
        config.search_widths, verbose, log_file, csv_file, play_config
    )
    
    return {
        'model_a_first': {
            'winner_position': winner_1,  # "A" or "B" based on position
            'winner_model': model_a_path if winner_1 == "A" else model_b_path,
            'loser_model': model_b_path if winner_1 == "A" else model_a_path,
            'trmph_str': trmph_str_1,
            'winner_char': winner_char_1,
            'swap_decision': swap_decision_1
        },
        'model_b_first': {
            'winner_position': winner_2,  # "A" or "B" based on position
            'winner_model': model_b_path if winner_2 == "A" else model_a_path,
            'loser_model': model_a_path if winner_2 == "A" else model_b_path,
            'trmph_str': trmph_str_2,
            'winner_char': winner_char_2,
            'swap_decision': swap_decision_2
        }
    }


def run_round_robin_tournament(
    config: TournamentConfig,
    verbose: int = 1,
    log_file: str = None,
    csv_file: str = None,
    play_config: Optional[TournamentPlayConfig] = None
) -> TournamentResult:
    """
    For each pair of models, play num_games games with each color assignment.
    Passes play_config to each game for pie rule, temperature, and randomness.
    Verbosity levels:
      0: Silent
      1: High-level info (pie rule swaps, game results)
      2: Per-move details
    """
    if play_config is None:
        play_config = TournamentPlayConfig()
    
    models = {path: SimpleModelInference(path) for path in config.checkpoint_paths}
    result = TournamentResult(config.checkpoint_paths)
    
    for model_a_path, model_b_path in itertools.combinations(config.checkpoint_paths, 2):
        for game_idx in range(config.num_games):
            # Play both games (each model goes first once)
            game_results = play_games_with_each_first(
                model_a_path, model_b_path, models, config, play_config, 
                verbose, log_file, csv_file
            )
            
            # Record results for both games
            result.record_game(
                game_results['model_a_first']['winner_model'],
                game_results['model_a_first']['loser_model']
            )
            result.record_game(
                game_results['model_b_first']['winner_model'], 
                game_results['model_b_first']['loser_model']
            )
            
            if verbose >= 1:
                print(f"Game {game_idx+1} of {config.num_games} between {model_a_path} and {model_b_path} completed.")
                
                # Print details for both games
                for game_key, game_name in [('model_a_first', f'{model_a_path} first'), 
                                          ('model_b_first', f'{model_b_path} first')]:
                    game_data = game_results[game_key]
                    print(f"  {game_name}:")
                    print(f"    Trmph: {game_data['trmph_str']}")
                    print(f"    Winner: {game_data['winner_char']} ({game_data['winner_model']})")
                    print(f"    Swap: {game_data['swap_decision']}")
                
                print(f"Pie rule: {play_config.pie_rule}, Temperature: {play_config.temperature}, Random seed: {play_config.random_seed}")
                print(f"Search widths: {config.search_widths}")
    print("Done.")
    return result

# Example usage (to be moved to CLI or script):
if __name__ == "__main__":
    # Example: compare two checkpoints
    checkpoints = [
        "checkpoints/hyperparameter_tuning/loss_weight_sweep_exp0_bs256_98f719_20250724_233408/epoch1_mini30.pt",
        "checkpoints/hyperparameter_tuning/loss_weight_sweep_exp0_bs256_98f719_20250724_233408/epoch1_mini35.pt"
    ]
    config = TournamentConfig(checkpoint_paths=checkpoints, num_games=4)
    result = run_round_robin_tournament(config, verbose=1)
    result.print_summary()
    result.print_elo() 