import os
import itertools
from typing import List, Dict, Tuple, Optional
import numpy as np
from hex_ai.inference.simple_model_inference import SimpleModelInference
from hex_ai.inference.game_engine import HexGameState
from hex_ai.inference.fixed_tree_search import minimax_policy_value_search
from hex_ai.config import BLUE_PLAYER, RED_PLAYER
from hex_ai.utils.tournament_logging import append_trmph_winner_line, log_game_csv
import random
from hex_ai.value_utils import Winner
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
    def __init__(self, temperature: float = 0.5, random_seed: Optional[int] = None, pie_rule: bool = False, pie_threshold: tuple = (0.45, 0.55), search_widths: Optional[list] = None):
        self.temperature = temperature
        if random_seed is None:
            random_seed = int(datetime.now().strftime('%Y%m%d%H%M'))
        self.random_seed = random_seed
        self.pie_rule = pie_rule
        self.pie_threshold = pie_threshold  # (min, max) win prob for swap
        self.search_widths = search_widths
        random.seed(random_seed)
        np.random.seed(random_seed)

def play_single_game(model_a: SimpleModelInference, 
                     model_b: SimpleModelInference, 
                     board_size: int, 
                     search_widths: Optional[list] = None, 
                     verbose: int = 1, 
                     log_file: str = None, 
                     csv_file: str = None, 
                     play_config: Optional[TournamentPlayConfig] = None) -> str:
    """
    Play a single game between model_a (Blue, first) and model_b (Red, second).
    Supports the pie rule, configurable temperature, and logging.
    Returns the winner's name ("A" or "B").
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
        # First move by model_a (Blue)
        if search_widths:
            move, _ = minimax_policy_value_search(state, model_a, search_widths)
        else:
            move = model_select_move(model_a, state, temperature=play_config.temperature)
        move_sequence.append(move)
        state = state.make_move(*move)
        move_num += 1
        # Evaluate win prob for blue after first move
        policy_logits, value_logit = model_b.infer(state.board)
        from hex_ai.value_utils import get_win_prob_from_model_output
        win_prob_blue = get_win_prob_from_model_output(value_logit, Winner.BLUE)
        # Decide whether to swap
        min_thr, max_thr = play_config.pie_threshold
        if min_thr <= win_prob_blue <= max_thr:
            swap = True
            swap_decision = 'swap'
            # Swap model assignments: model_b becomes blue, model_a becomes red
            model_a, model_b = model_b, model_a
        else:
            swap_decision = 'no_swap'
        if verbose >= 1:
            print(f"Pie rule: win_prob_blue={win_prob_blue:.3f}, swap={swap}")
    # --- MAIN GAME LOOP ---
    while not state.game_over:
        if state.current_player == BLUE_PLAYER:
            model = model_a
        else:
            model = model_b
        if search_widths:
            move, _ = minimax_policy_value_search(state, model, search_widths)
        else:
            move = model_select_move(model, state, temperature=play_config.temperature)
        if move is None:
            break  # No valid moves
        move_sequence.append(move)
        state = state.make_move(*move)
        move_num += 1
        if verbose >= 2:
            print(f"Move {move_num}: Player {state.current_player} played {move}")
    # Convert move_sequence to trmph string
    from hex_ai.utils.format_conversion import rowcol_to_trmph
    trmph_moves = ''.join([rowcol_to_trmph(r, c, board_size) for r, c in move_sequence])
    trmph_str = f"#{board_size},{trmph_moves}"
    # Determine winner for logging: 'b' if blue (first mover after swap) wins, 'r' if red wins
    if state.winner == "blue":
        winner_char = 'b'
        result = "A" if not swap else "B"
    elif state.winner == "red":
        winner_char = 'r'
        result = "B" if not swap else "A"
    else:
        winner_char = 'd'  # draw (if ever possible)
        result = "draw"
    # --- LOGGING ---
    if log_file:
        append_trmph_winner_line(trmph_str, winner_char, log_file)
    if csv_file:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        # Compose row with all config info
        row = {
            "timestamp": timestamp,
            "model_a": getattr(model_a, 'checkpoint_path', str(model_a)),
            "model_b": getattr(model_b, 'checkpoint_path', str(model_b)),
            "color_a": "blue",
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
    return result

# Helper: model_select_move (copied from play_vs_model_cli.py for now)
def model_select_move(model: SimpleModelInference,
                      state: HexGameState, top_k=20,
                      temperature=0.5):
    board = state.board
    policy_logits, value_logit = model.infer(board)
    from hex_ai.value_utils import get_policy_probs_from_logits
    policy_probs = get_policy_probs_from_logits(policy_logits)
    legal_moves = state.get_legal_moves()
    move_indices = [row * state.board.shape[0] + col for row, col in legal_moves]
    legal_policy = np.array([policy_probs[idx] for idx in move_indices])
    if len(legal_policy) == 0:
        return None
    topk_idx = np.argsort(legal_policy)[::-1][:top_k]
    topk_moves = [legal_moves[i] for i in topk_idx]
    move_values = []
    for move in topk_moves:
        temp_state = state.make_move(*move)
        _, value_logit = model.infer(temp_state.board)
        move_values.append(value_logit)
    best_idx = np.argmax(move_values)
    if temperature > 0 and len(move_values) > 1:
        probs = np.exp(np.array(move_values) / temperature)
        probs /= probs.sum()
        chosen_idx = np.random.choice(len(move_values), p=probs)
    else:
        chosen_idx = best_idx
    return topk_moves[chosen_idx]

def run_round_robin_tournament(config: TournamentConfig, verbose: int = 1, log_file: str = None, csv_file: str = None, play_config: Optional[TournamentPlayConfig] = None) -> TournamentResult:
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
    for a, b in itertools.combinations(config.checkpoint_paths, 2):
        for game_idx in range(config.num_games):
            # A as blue, B as red
            winner = play_single_game(models[a], models[b], config.board_size, 
                                      config.search_widths, verbose, log_file, csv_file, 
                                      play_config)
            if winner == "A":
                result.record_game(a, b)
            elif winner == "B":
                result.record_game(b, a)
            # B as blue, A as red
            winner = play_single_game(models[b], models[a], config.board_size, 
                                      config.search_widths, verbose, log_file, csv_file, 
                                      play_config)
            if winner == "A":
                result.record_game(b, a)
            elif winner == "B":
                result.record_game(a, b)
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