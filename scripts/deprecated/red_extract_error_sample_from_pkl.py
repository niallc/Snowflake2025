#!/usr/bin/env python3
"""
Extract and display a record from a .pkl.gz file, given the file path and (optionally) an index or search string.
"""

import argparse
import numpy as np
from scripts.lib.data_loading_utils import load_examples_from_pkl
from scripts.lib.board_viz_utils import visualize_board_with_policy
from scripts.lib.consistency_checks import policy_on_empty_cell, player_to_move_channel_valid


def print_example(example, idx=None):
    board, policy, value = example
    print(f"{'='*40}")
    if idx is not None:
        print(f"Index: {idx}")
    print(f"Board shape: {board.shape}")
    print(f"Policy shape: {policy.shape if policy is not None else None}")
    print(f"Value: {value}")
    highlight_move, trmph_move = visualize_board_with_policy(board, policy)
    if highlight_move is not None:
        print(f"Policy target move: (row, col)={highlight_move}, trmph={trmph_move}")
        if not policy_on_empty_cell(board, highlight_move):
            print(f"[WARNING] Policy target move is on a non-empty cell!")
    else:
        print("Policy target move: None or not one-hot")
    if board.shape[0] == 3:
        valid, unique_vals = player_to_move_channel_valid(board[2])
        print(f"Player-to-move channel unique values: {unique_vals}")
        if not valid:
            print(f"[WARNING] Player-to-move channel has unexpected values!")
    print(f"{'='*40}")

def main():
    parser = argparse.ArgumentParser(description="Extract a record from a .pkl.gz file by index or search string.")
    parser.add_argument('file_path', type=str, help='Path to .pkl.gz file')
    parser.add_argument('--index', type=int, help='Index of the record to extract')
    parser.add_argument('--search-value', type=float, help='Search for a record with this value target')
    parser.add_argument('--search-policy', type=int, help='Search for a record with this policy index set to 1')
    parser.add_argument('--max', type=int, default=10, help='Max number of matches to print')
    args = parser.parse_args()

    examples = load_examples_from_pkl(args.file_path)
    print(f"Loaded {len(examples)} examples from {args.file_path}")

    matches = []
    if args.index is not None:
        if 0 <= args.index < len(examples):
            print_example(examples[args.index], idx=args.index)
        else:
            print(f"Index {args.index} out of range (0, {len(examples)-1})")
        return

    for idx, ex in enumerate(examples):
        board, policy, value = ex
        match = True
        if args.search_value is not None and not np.isclose(value, args.search_value):
            match = False
        if args.search_policy is not None and (policy is None or policy[args.search_policy] != 1.0):
            match = False
        if match:
            matches.append((idx, ex))
            if len(matches) >= args.max:
                break
    if matches:
        for idx, ex in matches:
            print_example(ex, idx=idx)
    else:
        print("No matching records found.")

if __name__ == "__main__":
    main() 