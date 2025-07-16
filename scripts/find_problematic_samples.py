#!/usr/bin/env python3
"""
Find specific problematic samples in a file that have invalid board states, and attempt to reconstruct the move sequence for each.
"""

import gzip
import pickle
import numpy as np
from pathlib import Path
from hex_ai.utils.format_conversion import rowcol_to_trmph


def reconstruct_move_sequence(examples):
    """
    Attempt to reconstruct the move sequence for each board state in examples.
    Returns a list of move lists (one per position), and a list of issues (if any).
    """
    move_sequences = []
    issues = []
    prev_board = np.zeros_like(examples[0][0])  # Start from empty board
    prev_moves = []
    for idx, (board_state, _, _) in enumerate(examples):
        # Find the difference between prev_board and board_state
        diff = board_state - prev_board
        # Check for impossible states: both red and blue at same location
        overlap = np.logical_and(board_state[0] == 1, board_state[1] == 1)
        if np.any(overlap):
            issues.append((idx, "Red and blue piece at same location!"))
        # Find new move: should be exactly one new piece (either blue or red)
        new_blues = np.argwhere(diff[0] == 1)
        new_reds = np.argwhere(diff[1] == 1)
        if len(new_blues) + len(new_reds) != 1:
            issues.append((idx, f"Expected 1 new move, found {len(new_blues)} blue and {len(new_reds)} red"))
        # Add the new move to the sequence
        moves = prev_moves.copy()
        if len(new_blues) == 1 and len(new_reds) == 0:
            row, col = new_blues[0]
            moves.append(rowcol_to_trmph(row, col))
        elif len(new_reds) == 1 and len(new_blues) == 0:
            row, col = new_reds[0]
            moves.append(rowcol_to_trmph(row, col))
        elif len(new_blues) == 0 and len(new_reds) == 0 and idx == 0:
            # Allow empty board for the first position
            pass
        else:
            # Ambiguous or missing move
            issues.append((idx, "Ambiguous or missing move when reconstructing sequence"))
        move_sequences.append(moves)
        prev_board = board_state.copy()
        prev_moves = moves
    return move_sequences, issues


def find_problematic_samples(file_path: str, max_samples: int = 1000):
    """Find samples with invalid board states and reconstruct their move sequences."""
    file_path = Path(file_path)
    with gzip.open(file_path, 'rb') as f:
        data = pickle.load(f)
    examples = data['examples']
    print(f"Analyzing {len(examples)} examples from {file_path.name}")
    # Reconstruct move sequences for all samples up to max_samples
    move_sequences, issues = reconstruct_move_sequence(examples[:max_samples])
    problematic_samples = []
    for i, example in enumerate(examples[:max_samples]):
        board_state, policy_target, value_target = example
        blue_count = int(np.sum(board_state[0]))
        red_count = int(np.sum(board_state[1]))
        # Check validity
        is_valid = (blue_count == red_count) or (blue_count == red_count + 1)
        if not is_valid:
            problematic_samples.append({
                'index': i,
                'blue_count': blue_count,
                'red_count': red_count,
                'total_pieces': blue_count + red_count,
                'blue_positions': list(zip(*np.where(board_state[0] == 1.0))),
                'red_positions': list(zip(*np.where(board_state[1] == 1.0))),
                'move_sequence': move_sequences[i] if i < len(move_sequences) else None
            })
    print(f"Found {len(problematic_samples)} problematic samples")
    for i, sample in enumerate(problematic_samples[:10]):  # Show first 10
        print(f"\nProblematic sample {i+1}:")
        print(f"  Index: {sample['index']}")
        print(f"  Blue count: {sample['blue_count']}")
        print(f"  Red count: {sample['red_count']}")
        print(f"  Total pieces: {sample['total_pieces']}")
        print(f"  Blue positions: {sample['blue_positions']}")
        print(f"  Red positions: {sample['red_positions']}")
        print(f"  Reconstructed move sequence: {''.join(sample['move_sequence']) if sample['move_sequence'] else 'N/A'}")
    if issues:
        print("\nConsistency issues detected during move sequence reconstruction:")
        for idx, msg in issues:
            print(f"  At sample {idx}: {msg}")
    return problematic_samples

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python scripts/find_problematic_samples.py <file_path>")
        sys.exit(1)
    file_path = sys.argv[1]
    find_problematic_samples(file_path) 