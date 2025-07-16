#!/usr/bin/env python3
"""
Find specific problematic samples in a file that have invalid board states.
"""

import gzip
import pickle
import numpy as np
from pathlib import Path

def find_problematic_samples(file_path: str, max_samples: int = 1000):
    """Find samples with invalid board states."""
    file_path = Path(file_path)
    
    with gzip.open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    examples = data['examples']
    print(f"Analyzing {len(examples)} examples from {file_path.name}")
    
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
                'red_positions': list(zip(*np.where(board_state[1] == 1.0)))
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
    
    return problematic_samples

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python scripts/find_problematic_samples.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    find_problematic_samples(file_path) 