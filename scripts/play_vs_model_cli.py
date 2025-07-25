#!/usr/bin/env python3
import argparse
import numpy as np
import random
from hex_ai.inference.simple_model_inference import SimpleModelInference
from hex_ai.inference.board_display import display_hex_board
from hex_ai.inference.game_engine import HexGameState

DEFAULT_BOARD_SIZE = 13
DEFAULT_TOP_K = 20

all_results_dir="checkpoints/hyperparameter_tuning/"
this_model_dir="loss_weight_sweep_exp0_bs256_98f719_20250724_233408"
checkpoint_file="epoch1_mini30.pt"
DEFAULT_CHKPT_PATH=f"{all_results_dir}/{this_model_dir}/{checkpoint_file}"

def get_human_move(state: HexGameState):
    while True:
        move = input("Enter your move (e.g., 'a1', 'm13'): ").strip().lower()
        try:
            from hex_ai.utils.format_conversion import trmph_move_to_rowcol
            row, col = trmph_move_to_rowcol(move, board_size=DEFAULT_BOARD_SIZE)
            if state.is_valid_move(row, col):
                return row, col
            else:
                print("Invalid move: position occupied or out of bounds.")
        except Exception as e:
            print(f"Invalid input: {e}")

def model_select_move(model: SimpleModelInference, state: HexGameState, top_k=DEFAULT_TOP_K, temperature=0.5):
    board = state.board
    # Model expects (N,N) np.ndarray or trmph string
    policy_probs, _, _ = model.infer(board)
    legal_moves = state.get_legal_moves()
    move_indices = [row * DEFAULT_BOARD_SIZE + col for row, col in legal_moves]
    legal_policy = np.array([policy_probs[idx] for idx in move_indices])
    if len(legal_policy) == 0:
        return None
    # Get top-k moves
    topk_idx = np.argsort(legal_policy)[::-1][:top_k]
    topk_moves = [legal_moves[i] for i in topk_idx]
    # Evaluate each with value head
    move_values = []
    for move in topk_moves:
        temp_state = state.make_move(*move)
        _, value, _ = model.infer(temp_state.board)
        move_values.append(value)
    # Pick the best, with some randomness
    best_idx = np.argmax(move_values)
    # Optionally add some randomness
    if temperature > 0 and len(move_values) > 1:
        probs = np.exp(np.array(move_values) / temperature)
        probs /= probs.sum()
        chosen_idx = np.random.choice(len(move_values), p=probs)
    else:
        chosen_idx = best_idx
    return topk_moves[chosen_idx]

def main():
    parser = argparse.ArgumentParser(description="Play Hex against the trained model (CLI)")
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHKPT_PATH, help='Path to model checkpoint')
    parser.add_argument('--board-size', type=int, default=DEFAULT_BOARD_SIZE, help='Board size (default: 13)')
    parser.add_argument('--topk', type=int, default=DEFAULT_TOP_K, help='Top-k policy moves to consider (default: 20)')
    parser.add_argument('--temperature', type=float, default=0.5, help='Randomness for model move selection (default: 0.5)')
    parser.add_argument('--human-first', action='store_true', help='Human plays first (default: model plays first)')
    args = parser.parse_args()

    print(f"\nWelcome to Hex AI CLI! Board size: {args.board_size}x{args.board_size}")
    print("You will play against the model. Enter moves in trmph format (e.g., 'a1', 'm13').")
    print("The board and .trmph representation will be shown after each move.\n")

    model = SimpleModelInference(args.checkpoint)
    state = HexGameState()
    human_player = 0 if args.human_first else 1
    move_num = 0

    while not state.game_over:
        print(f"\nMove {move_num+1} - {'Human' if state.current_player == human_player else 'Model'} to play:")
        display_hex_board(state.board)
        print(f"Current .trmph: {state.to_trmph()}")
        if state.current_player == human_player:
            row, col = get_human_move(state)
            state = state.make_move(row, col)
        else:
            print("Model is thinking...")
            move = model_select_move(model, state, top_k=args.topk, temperature=args.temperature)
            if move is None:
                print("No valid moves left. Game over.")
                break
            print(f"Model plays: {move}")
            state = state.make_move(*move)
        move_num += 1
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