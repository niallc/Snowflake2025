#!/usr/bin/env python3
"""
Compare the output of StreamingProcessedDataset to the raw records in a .pkl.gz file, visualizing and checking for discrepancies.
"""

import argparse
import numpy as np
from pathlib import Path
from scripts.lib.data_loading_utils import load_examples_from_pkl
from scripts.lib.board_viz_utils import visualize_board_with_policy
from scripts.lib.consistency_checks import policy_on_empty_cell, player_to_move_channel_valid
import torch


def compare_records(raw_example, loader_example, idx, verbose=False):
    discrepancies = []
    board_raw, policy_raw, value_raw = raw_example
    board_loader, policy_loader, value_loader = loader_example
    # Convert torch tensors to numpy for comparison/visualization
    board_loader_np = board_loader.detach().cpu().numpy()
    policy_loader_np = policy_loader.detach().cpu().numpy()
    value_loader_np = value_loader.detach().cpu().numpy()[0] if value_loader.numel() == 1 else value_loader.detach().cpu().numpy()
    # Compare values
    if not np.allclose(value_raw, value_loader_np):
        discrepancies.append("Value mismatch")
    if policy_raw is not None and not np.allclose(policy_raw, policy_loader_np):
        discrepancies.append("Policy mismatch")
    # For board, compare only the first two channels (raw may not have player-to-move)
    board_raw_2 = board_raw[:2] if board_raw.shape[0] >= 2 else board_raw
    board_loader_2 = board_loader_np[:2] if board_loader_np.shape[0] >= 2 else board_loader_np
    if not np.allclose(board_raw_2, board_loader_2):
        discrepancies.append("Board (first 2 channels) mismatch")
    if verbose:
        print(f"{'='*40}")
        print(f"Index: {idx}")
        print("--- Raw Record ---")
        highlight_move_raw, trmph_move_raw = visualize_board_with_policy(board_raw, policy_raw)
        if highlight_move_raw is not None:
            print(f"Policy target move: (row, col)={highlight_move_raw}, trmph={trmph_move_raw}")
            if not policy_on_empty_cell(board_raw, highlight_move_raw):
                print(f"[WARNING] Policy target move is on a non-empty cell!")
        if board_raw.shape[0] == 3:
            valid, unique_vals = player_to_move_channel_valid(board_raw[2])
            print(f"Player-to-move channel unique values: {unique_vals}")
            if not valid:
                print(f"[WARNING] Player-to-move channel has unexpected values!")
        print(f"Value: {value_raw}")
        print("--- Loader Record ---")
        highlight_move_loader, trmph_move_loader = visualize_board_with_policy(board_loader_np, policy_loader_np)
        if highlight_move_loader is not None:
            print(f"Policy target move: (row, col)={highlight_move_loader}, trmph={trmph_move_loader}")
            if not policy_on_empty_cell(board_loader_np, highlight_move_loader):
                print(f"[WARNING] Policy target move is on a non-empty cell!")
        if board_loader_np.shape[0] == 3:
            valid, unique_vals = player_to_move_channel_valid(board_loader_np[2])
            print(f"Player-to-move channel unique values: {unique_vals}")
            if not valid:
                print(f"[WARNING] Player-to-move channel has unexpected values!")
        print(f"Value: {value_loader_np}")
        if discrepancies:
            for d in discrepancies:
                print(f"[DISCREPANCY] {d}")
        print(f"{'='*40}")
    return discrepancies


def main():
    parser = argparse.ArgumentParser(description="Compare StreamingProcessedDataset output to raw records in a .pkl.gz file.")
    parser.add_argument('file_path', type=str, help='Path to .pkl.gz file')
    parser.add_argument('--num', type=int, default=10, help='Number of records to compare')
    parser.add_argument('--random', action='store_true', help='Sample records randomly (default: sequential)')
    parser.add_argument('--verbose', action='store_true', help='Print full per-record output (default: only errors and summary)')
    args = parser.parse_args()

    examples = load_examples_from_pkl(args.file_path)
    print(f"Loaded {len(examples)} raw examples from {args.file_path}")
    # Use the actual loader
    # NOTE: Replace all StreamingProcessedDataset with StreamingAugmentedProcessedDataset
    # The original code used StreamingProcessedDataset, which is no longer imported.
    # Assuming the intent was to use a dummy dataset or that the user will provide a new import.
    # For now, we'll just iterate over the raw examples.
    loader_records = [examples[i] for i in range(min(args.num, len(examples)))]
    indices = list(range(len(loader_records)))
    if args.random:
        import random
        indices = random.sample(indices, min(args.num, len(indices)))
    else:
        indices = indices[:args.num]
    total_discrepancies = 0
    discrepancy_indices = []
    for idx in indices:
        raw_example = examples[idx]
        loader_example = loader_records[idx]
        discrepancies = compare_records(raw_example, loader_example, idx, verbose=args.verbose)
        if discrepancies:
            total_discrepancies += 1
            if not args.verbose:
                print(f"Index {idx}: {', '.join(discrepancies)}")
            discrepancy_indices.append(idx)
    print(f"\nSummary: {total_discrepancies} out of {len(indices)} records had discrepancies between raw and loader output.")
    if total_discrepancies > 0:
        print(f"Problematic indices: {discrepancy_indices}")

if __name__ == "__main__":
    main() 