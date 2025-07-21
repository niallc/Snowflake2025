#!/usr/bin/env python3
"""
Visualize Hex board augmentations for inspection.
Loads a few board states from a .pkl.gz file, applies create_augmented_boards,
and displays all 4 forms using display_hex_board.
"""
import sys
import os
import gzip
import pickle
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hex_ai.data_utils import create_augmented_boards, create_augmented_policies, create_augmented_values, create_augmented_player_to_move, get_player_to_move_from_board
from hex_ai.inference.board_display import display_hex_board
from hex_ai.utils.format_conversion import tensor_to_rowcol, tensor_to_trmph

DEFAULT_FILE = "data/processed/twoNetGames_13x13_mk45_d1b20_v1816_2s0_p2551k_vt25_pt10_processed.pkl.gz"
NUM_BOARDS = 4  # Number of board states to visualize

AUGMENT_LABELS = [
    "Original",
    "Rotated 180° (no color swap)",
    "Reflected Long Diagonal + Color Swap",
    "Reflected Short Diagonal + Color Swap"
]

def load_examples_from_pkl_gz(filepath, max_examples=3):
    examples = []
    with gzip.open(filepath, 'rb') as f:
        data = pickle.load(f)
        examples_data = data['examples']
        for i, example in enumerate(examples_data):
            if len(examples) >= max_examples:
                break
            board, policy, value = example
            # Debug: print type and shape
            print(f"Example {i}: board shape={board.shape}, policy shape={policy.shape}, value={value}")
            if isinstance(board, np.ndarray) and board.shape == (2, 13, 13):
                examples.append((board, policy, value))
            else:
                print(f"  Skipping example {i}: invalid board shape.")
    if not examples:
        print("No valid examples found in file!")
    return examples

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize Hex board augmentations.")
    parser.add_argument('--file', type=str, default=DEFAULT_FILE, help='Path to .pkl.gz file')
    parser.add_argument('--num', type=int, default=NUM_BOARDS, help='Number of boards to visualize')
    args = parser.parse_args()

    examples = load_examples_from_pkl_gz(args.file, max_examples=args.num)
    # Skip the first example, which is an empty board with nothing to process.
    for idx, (board, policy, value) in enumerate(examples[1:], start=1):
        print(f"\n=== Board {idx+1} (original position) ===")
        print(f"Value target: {value}")
        
        # Compute player-to-move for original board
        player_to_move = get_player_to_move_from_board(board)
        print(f"Player to move: {player_to_move} ({'Blue' if player_to_move == 0 else 'Red'})")
        
        # Show original board and policy
        print(f"\n--- {AUGMENT_LABELS[0]} ---")
        display_hex_board(board)
        
        # Find the (single) nonzero label in the policy vector
        nonzero_indices = np.where(policy > 1e-6)[0]
        if len(nonzero_indices) == 0:
            raise ValueError("No nonzero label found in policy vector!")
        if len(nonzero_indices) > 1:
            raise ValueError(f"More than one nonzero label found in policy vector! Indices: {nonzero_indices}, values: {policy[nonzero_indices]}")
        label_idx = nonzero_indices[0]
        row, col = tensor_to_rowcol(label_idx)
        trmph_move = tensor_to_trmph(label_idx)
        print(f"Policy label: move index {label_idx} -> ({row},{col}) -> {trmph_move} (value={policy[label_idx]:.4f})")
        
        # Show augmented boards and policies
        augmented_boards = create_augmented_boards(board)
        augmented_policies = create_augmented_policies(policy)
        augmented_values = create_augmented_values(value)
        augmented_players = create_augmented_player_to_move(player_to_move)
        
        for aug_idx, (aug_board, aug_policy, aug_value, aug_player) in enumerate(zip(augmented_boards[1:], augmented_policies[1:], augmented_values[1:], augmented_players[1:]), 1):  # Skip original
            print(f"\n--- {AUGMENT_LABELS[aug_idx]} ---")
            display_hex_board(aug_board)
            print(f"Value target: {aug_value}")
            print(f"Player to move: {aug_player} ({'Blue' if aug_player == 0 else 'Red'})")
            
            # Find the (single) nonzero label in the policy vector
            nonzero_indices = np.where(aug_policy > 1e-6)[0]
            if len(nonzero_indices) == 0:
                raise ValueError("No nonzero label found in policy vector for augmentation!")
            if len(nonzero_indices) > 1:
                raise ValueError(f"More than one nonzero label found in policy vector for augmentation! Indices: {nonzero_indices}, values: {aug_policy[nonzero_indices]}")
            label_idx = nonzero_indices[0]
            row, col = tensor_to_rowcol(label_idx)
            trmph_move = tensor_to_trmph(label_idx)
            print(f"Policy label: move index {label_idx} -> ({row},{col}) -> {trmph_move} (value={aug_policy[label_idx]:.4f})")

if __name__ == "__main__":
    main() 