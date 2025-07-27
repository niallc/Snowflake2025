#!/usr/bin/env python3
"""
Inspect a batch of records from a .pkl.gz file, visualize each, run consistency checks, and print a summary of issues found.
"""

import argparse
import numpy as np
from hex_ai.data_utils import load_examples_from_pkl
from hex_ai.utils.board_visualization import visualize_board_with_policy
from hex_ai.utils.consistency_checks import policy_on_empty_cell, player_to_move_channel_valid


def inspect_batch(examples, batch_size=10, random_sample=True, only_show_issues=False):
    n = len(examples)
    indices = list(range(n))
    if random_sample:
        import random
        indices = random.sample(indices, min(batch_size, n))
    else:
        indices = indices[:batch_size]

    issues = []
    for idx in indices:
        board = examples[idx]['board']
        policy = examples[idx]['policy']
        value = examples[idx]['value']
        issue_found = False
        highlight_move, trmph_move = visualize_board_with_policy(board, policy)
        # Consistency check: is policy target on empty cell?
        policy_on_nonempty = False
        if highlight_move is not None:
            if not policy_on_empty_cell(board, highlight_move):
                policy_on_nonempty = True
                issue_found = True
        # Player-to-move channel check
        player_channel_issue = False
        if board.shape[0] == 3:
            valid, unique_vals = player_to_move_channel_valid(board[2])
            if not valid:
                player_channel_issue = True
                issue_found = True
        if only_show_issues and not issue_found:
            continue
        print(f"{'='*40}")
        print(f"Index: {idx}")
        print(f"Value: {value}")
        if highlight_move is not None:
            print(f"Policy target move: (row, col)={highlight_move}, trmph={trmph_move}")
        else:
            print("Policy target move: None or not one-hot")
        if policy_on_nonempty:
            print(f"[WARNING] Policy target move is on a non-empty cell!")
        if board.shape[0] == 3:
            print(f"Player-to-move channel unique values: {unique_vals}")
            if player_channel_issue:
                print(f"[WARNING] Player-to-move channel has unexpected values!")
        print(f"{'='*40}")
        if issue_found:
            issues.append(idx)
    print(f"\nSummary: {len(issues)} out of {len(indices)} records had issues.")
    if issues:
        print(f"Problematic indices: {issues}")

def main():
    parser = argparse.ArgumentParser(description="Inspect a batch of records from a .pkl.gz file.")
    parser.add_argument('file_path', type=str, help='Path to .pkl.gz file')
    parser.add_argument('--batch-size', type=int, default=10, help='Number of records to inspect')
    parser.add_argument('--random', action='store_true', help='Sample records randomly (default: sequential)')
    parser.add_argument('--only-issues', action='store_true', help='Only show records with detected issues')
    args = parser.parse_args()

    examples = load_examples_from_pkl(args.file_path)
    print(f"Loaded {len(examples)} examples from {args.file_path}")
    inspect_batch(examples, batch_size=args.batch_size, random_sample=args.random, only_show_issues=args.only_issues)

if __name__ == "__main__":
    main() 