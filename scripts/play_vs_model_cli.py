#!/usr/bin/env python3
import argparse
import numpy as np
import random
import operator
from functools import reduce
from hex_ai.inference.simple_model_inference import SimpleModelInference
from hex_ai.inference.board_display import display_hex_board
from hex_ai.inference.game_engine import HexGameState, apply_move_to_state_trmph, apply_move_to_state, select_top_value_head_move
from hex_ai.file_utils import GracefulShutdown
import sys
from hex_ai.inference.fixed_tree_search import minimax_policy_value_search
from hex_ai.value_utils import (
    Winner,
    ValuePerspective,
    model_output_to_prob,
    winner_to_color,
    get_win_prob_from_model_output,
    get_policy_probs_from_logits,
    temperature_scaled_softmax,
    # Add new utilities
    policy_logits_to_probs,
    get_legal_policy_probs,
    select_top_k_moves,
    sample_move_by_value,
    select_policy_move  # Add the new public function
)
from hex_ai.config import BLUE_PLAYER, RED_PLAYER
from hex_ai.inference.board_display import ansi_colored
from hex_ai.utils.format_conversion import trmph_move_to_rowcol, rowcol_to_trmph
from hex_ai.inference.model_config import get_model_path

DEFAULT_BOARD_SIZE = 13
DEFAULT_TOP_K = 20

DEFAULT_CHKPT_PATH = get_model_path("current_best")

def get_human_move(state: HexGameState, shutdown_handler: GracefulShutdown):
    while True:
        try:
            move = input("Enter your move (e.g., 'a1', 'm13', or 'q' to quit): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nKeyboard interrupt detected. Exiting game.")
            sys.exit(0)
        if move in ('q', 'quit', 'exit'):
            print("Quitting game by user request.")
            sys.exit(0)
        try:
            row, col = trmph_move_to_rowcol(move, board_size=DEFAULT_BOARD_SIZE)
            if state.is_valid_move(row, col):
                return row, col
            else:
                print("Invalid move: position occupied or out of bounds.")
        except Exception as e:
            print(f"Invalid input: {e}")

def main():
    parser = argparse.ArgumentParser(description="Play Hex against the trained model (CLI)")
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHKPT_PATH, help='Path to model checkpoint')
    parser.add_argument('--board-size', type=int, default=DEFAULT_BOARD_SIZE, help='Board size (default: 13)')
    parser.add_argument('--topk', type=int, default=DEFAULT_TOP_K, help='Top-k policy moves to consider (default: 20)')
    parser.add_argument('--temperature', type=float, default=0.5, help='Randomness for model move selection (default: 0.5)')
    parser.add_argument('--human-first', action='store_true', help='Human plays first (default: model plays first)')
    parser.add_argument('--search-widths', type=str, default=None, help='Comma-separated list of search widths for tree search (e.g., 20,10,10,5). If provided, use minimax tree search for model moves. WARNING: The number of leaf positions grows rapidly as the product of these numbers! (e.g., 20,10,10,5 = 10,000 leaves). Recommended max: 1,000,000.')
    parser.add_argument('--start-trmph', type=str, default=None, help='Optional starting position in trmph format (must start with "#13,", e.g., "#13,a1b2c3")')
    parser.add_argument('--use-policy-only', action='store_true', help='Use policy head only for model move selection (default: value head among top-k policy moves)')
    args = parser.parse_args()

    print(f"\nWelcome to Hex AI CLI! Board size: {args.board_size}x{args.board_size}")
    print("You will play against the model. Enter moves in trmph format (e.g., 'a1', 'm13').")
    print("The board and .trmph representation will be shown after each move.")
    print("You can also start from a specific position using --start-trmph.")
    print("  Example: --start-trmph '#13,a1b2c3' (must include the '#13,' prefix for 13x13)")
    print()
    print("You can use --search-widths to enable tree search for the model's move selection.")
    print("  Example: --search-widths 20,10,10,5 (expands up to 10,000 leaf positions)")
    print("  WARNING: The number of leaf positions is the product of the widths. If this exceeds 1,000,000, the program will refuse to run for safety.")
    print()

    model = SimpleModelInference(args.checkpoint)
    if args.search_widths:
        try:
            search_widths = [int(x) for x in args.search_widths.split(",") if x.strip()]
        except Exception:
            print("Error: --search-widths must be a comma-separated list of integers, e.g., 20,10,10,5")
            sys.exit(1)
        if not search_widths:
            print("Error: --search-widths must not be empty.")
            sys.exit(1)
        num_leaves = reduce(operator.mul, search_widths, 1)
        if num_leaves > 1_000_000:
            print(f"Error: The product of --search-widths is {num_leaves}, which exceeds the safety limit of 1,000,000. Please use smaller widths.")
            sys.exit(1)
    else:
        search_widths = [20, 10, 10, 5]
    if args.start_trmph:
        if not args.start_trmph.startswith("#13,"):
            print("Error: --start-trmph must start with '#13,'. Example: --start-trmph '#13,a1b2c3'")
            sys.exit(1)
        try:
            state = HexGameState.from_trmph(args.start_trmph)
        except Exception as e:
            print(f"Error parsing --start-trmph: {e}")
            print("Make sure your input matches the TRMPH format, e.g., '#13,a1b2c3'")
            sys.exit(1)
    else:
        state = HexGameState()
    human_player = 0 if args.human_first else 1
    move_num = 0

    shutdown_handler = GracefulShutdown()

    try:
        while not state.game_over:
            if shutdown_handler.shutdown_requested:
                print("\nGraceful shutdown requested. Exiting game loop.")
                break
            print(f"\nMove {move_num+1} - {'Human' if state.current_player == human_player else 'Model'} to play:")
            display_hex_board(state.board)
            print(f"Current .trmph: {state.to_trmph()}")
            # Show network's confidence (value head) for current player
            _, _, value_logit = model.simple_infer(state.board)
            current_winner = Winner.BLUE if state.current_player == BLUE_PLAYER else Winner.RED
            value = get_win_prob_from_model_output(value_logit, current_winner)
            player_str = current_winner.name.capitalize()
            print(f"Network confidence ({player_str} to play): {value:.3f} (probability {player_str} wins)")
            if state.current_player == human_player:
                row, col = get_human_move(state, shutdown_handler)
                state = apply_move_to_state(state, row, col)
            else:
                print("Model is thinking...")
                if args.search_widths:
                    move, move_value = minimax_policy_value_search(state, model, search_widths, batch_size=1000, use_alpha_beta=True, temperature=args.temperature)
                    num_leaves = np.prod(search_widths)
                    print(f"[Tree search] Evaluated up to {num_leaves} leaf positions (widths: {search_widths})")
                else:
                    if args.use_policy_only:
                        move = select_policy_move(state, model, args.temperature)
                    else:
                        move = select_top_value_head_move(model, state, top_k=args.topk, temperature=args.temperature)
                    move_value = None
                if move is None:
                    print("No valid moves left. Game over.")
                    break
                trmph_move = rowcol_to_trmph(move[0], move[1], board_size=args.board_size)
                # The model just played, so the player to move is now the human; the model's color is the opposite
                model_color_enum = Winner.RED if state.current_player == BLUE_PLAYER else Winner.BLUE
                colored_move = ansi_colored(trmph_move, winner_to_color(model_color_enum))
                print(f"Model plays: {colored_move}")
                # Print model's win probability for its own color after its move
                if move_value is not None:
                    win_prob = get_win_prob_from_model_output(move_value, model_color_enum)
                    print(f"Model's win probability ({model_color_enum.name.capitalize()}): {win_prob:.3f}")
                state = apply_move_to_state(state, *move)
            move_num += 1
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Exiting game.")
        sys.exit(0)
    # Final board
    print("\nFinal board:")
    display_hex_board(state.board)
    print(f"Final .trmph: {state.to_trmph()}")
    if state.winner:
        print(f"Game over! Winner: {state.winner.title()}")
    else:
        print("Game over! No winner detected.")

if __name__ == "__main__":
    main() 