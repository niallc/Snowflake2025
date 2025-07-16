#!/usr/bin/env python3
"""
Extract and display a record from a .pkl.gz file, given the file path and (optionally) an index or search string.
"""

import argparse
import gzip
import pickle
import numpy as np
from pathlib import Path
import sys

def load_examples_from_pkl(file_path):
    with gzip.open(file_path, 'rb') as f:
        data = pickle.load(f)
    if 'examples' in data:
        return data['examples']
    else:
        raise ValueError(f"No 'examples' key found in {file_path}")

def print_example(example, idx=None):
    board, policy, value = example
    print(f"{'='*40}")
    if idx is not None:
        print(f"Index: {idx}")
    print(f"Board shape: {board.shape}")
    print(f"Policy shape: {policy.shape if policy is not None else None}")
    print(f"Value: {value}")
    print(f"Board array (first 2 channels):\n{board[:2]}")
    if board.shape[0] > 2:
        print(f"Player-to-move channel:\n{board[2]}")
    print(f"Policy (nonzero indices): {np.nonzero(policy)[0] if policy is not None else None}")
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