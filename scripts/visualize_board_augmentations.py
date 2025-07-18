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

from hex_ai.data_utils import create_augmented_boards
from hex_ai.inference.board_display import display_hex_board

DEFAULT_FILE = "data/processed/twoNetGames_13x13_mk45_d1b20_v1816_2s0_p2551k_vt25_pt10_processed.pkl.gz"
NUM_BOARDS = 3  # Number of board states to visualize

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
    for idx, (board, policy, value) in enumerate(examples):
        print(f"\n=== Board {idx+1} (original position) ===")
        print(f"Value target: {value}")
        
        # Show original board and policy
        print(f"\n--- {AUGMENT_LABELS[0]} ---")
        display_hex_board(board)
        
        # Show top policy moves
        top_moves = np.argsort(policy)[-3:][::-1]  # Top 3 moves
        print("Top 3 policy moves:")
        for i, move_idx in enumerate(top_moves):
            print(f"  {i+1}. Move {move_idx}: {policy[move_idx]:.4f}")
        
        # Show augmented boards
        augmented_boards = create_augmented_boards(board)
        for aug_idx, aug_board in enumerate(augmented_boards[1:], 1):  # Skip original
            print(f"\n--- {AUGMENT_LABELS[aug_idx]} ---")
            display_hex_board(aug_board)

if __name__ == "__main__":
    main() 