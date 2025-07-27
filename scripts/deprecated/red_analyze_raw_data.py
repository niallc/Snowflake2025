#!/usr/bin/env python3
"""
Analyze raw data from .pkl.gz files to check color distributions and data quality.
"""

import argparse
import gzip
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json

def load_examples_from_pkl(file_path: Path) -> List[Tuple]:
    """Load examples from a .pkl.gz file."""
    with gzip.open(file_path, 'rb') as f:
        data = pickle.load(f)
    if 'examples' in data:
        return data['examples']
    else:
        raise ValueError(f"No 'examples' key found in {file_path}")

def analyze_board_state(board_state: np.ndarray) -> Dict:
    """Analyze a single board state."""
    if board_state.shape[0] != 2:
        return {'error': f"Expected 2 channels, got {board_state.shape[0]}"}
    
    blue_count = int(np.sum(board_state[0]))
    red_count = int(np.sum(board_state[1]))
    total_pieces = blue_count + red_count
    
    # Check validity
    is_valid = (blue_count == red_count) or (blue_count == red_count + 1)
    
    return {
        'blue_count': blue_count,
        'red_count': red_count,
        'total_pieces': total_pieces,
        'is_valid': is_valid,
        'blue_red_ratio': blue_count / max(red_count, 1),  # Avoid division by zero
        'first_move': total_pieces == 1,
        'empty_board': total_pieces == 0
    }

def analyze_file(file_path: Path, max_samples: int = 1000) -> Dict:
    """Analyze a single .pkl.gz file."""
    print(f"Analyzing {file_path}...")
    
    try:
        examples = load_examples_from_pkl(file_path)
        print(f"  Loaded {len(examples)} examples")
        
        # Analyze first max_samples
        samples_to_analyze = examples[:max_samples]
        analyses = []
        
        for i, example in enumerate(samples_to_analyze):
            board_state = example['board']
            policy_target = example['policy']
            value_target = example['value']
            analysis = analyze_board_state(board_state)
            analysis['sample_index'] = i
            analyses.append(analysis)
        
        # Aggregate statistics
        valid_count = sum(1 for a in analyses if a.get('is_valid', False))
        invalid_count = len(analyses) - valid_count
        
        blue_counts = [a['blue_count'] for a in analyses]
        red_counts = [a['red_count'] for a in analyses]
        total_pieces = [a['total_pieces'] for a in analyses]
        
        first_moves = [a for a in analyses if a['first_move']]
        empty_boards = [a for a in analyses if a['empty_board']]
        
        # Find problematic samples
        problematic = [a for a in analyses if not a.get('is_valid', False)]
        
        return {
            'file_path': str(file_path),
            'total_samples': len(examples),
            'analyzed_samples': len(analyses),
            'valid_samples': valid_count,
            'invalid_samples': invalid_count,
            'error_rate': invalid_count / len(analyses) if analyses else 0,
            'blue_count_stats': {
                'min': min(blue_counts),
                'max': max(blue_counts),
                'mean': np.mean(blue_counts),
                'median': np.median(blue_counts)
            },
            'red_count_stats': {
                'min': min(red_counts),
                'max': max(red_counts),
                'mean': np.mean(red_counts),
                'median': np.median(red_counts)
            },
            'total_pieces_stats': {
                'min': min(total_pieces),
                'max': max(total_pieces),
                'mean': np.mean(total_pieces),
                'median': np.median(total_pieces)
            },
            'first_moves': len(first_moves),
            'empty_boards': len(empty_boards),
            'problematic_samples': problematic[:10],  # First 10 problematic samples
            'analyses': analyses
        }
        
    except Exception as e:
        print(f"  Error analyzing {file_path}: {e}")
        return {
            'file_path': str(file_path),
            'error': str(e)
        }

def find_problematic_files(data_dir: str, max_files: int = 10, max_samples_per_file: int = 1000) -> Dict:
    """Find files with data quality issues."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory {data_dir} not found")
    
    # Find all .pkl.gz files
    data_files = list(data_path.glob("*.pkl.gz"))
    print(f"Found {len(data_files)} files to analyze")
    
    if max_files:
        data_files = data_files[:max_files]
        print(f"Analyzing first {max_files} files")
    
    results = []
    total_invalid = 0
    total_samples = 0
    
    for file_path in data_files:
        result = analyze_file(file_path, max_samples_per_file)
        results.append(result)
        
        if 'error' not in result:
            total_invalid += result['invalid_samples']
            total_samples += result['analyzed_samples']
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"Total samples analyzed: {total_samples}")
    print(f"Total invalid samples: {total_invalid}")
    print(f"Overall error rate: {total_invalid/total_samples:.2%}" if total_samples > 0 else "No samples")
    
    # Files with highest error rates
    files_with_errors = [r for r in results if 'error' not in r and r['invalid_samples'] > 0]
    if files_with_errors:
        print(f"\nFiles with errors:")
        for result in sorted(files_with_errors, key=lambda x: x['error_rate'], reverse=True)[:5]:
            print(f"  {Path(result['file_path']).name}: {result['invalid_samples']}/{result['analyzed_samples']} ({result['error_rate']:.2%})")
    
    return {
        'results': results,
        'total_samples': total_samples,
        'total_invalid': total_invalid,
        'overall_error_rate': total_invalid/total_samples if total_samples > 0 else 0
    }

def examine_specific_sample(file_path: str, sample_index: int = 0):
    """Examine a specific sample in detail."""
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"File {file_path} not found")
        return
    
    try:
        examples = load_examples_from_pkl(file_path)
        if sample_index >= len(examples):
            print(f"Sample index {sample_index} out of range (0-{len(examples)-1})")
            return
        
        board_state = examples[sample_index]['board']
        policy_target = examples[sample_index]['policy']
        value_target = examples[sample_index]['value']
        
        print(f"=== Sample {sample_index} from {file_path.name} ===")
        print(f"Board shape: {board_state.shape}")
        print(f"Policy target shape: {policy_target.shape if policy_target is not None else None}")
        print(f"Value target: {value_target}")
        
        analysis = analyze_board_state(board_state)
        print(f"Analysis: {analysis}")
        
        # Show board state
        blue_channel = board_state[0]
        red_channel = board_state[1]
        
        print(f"\nBlue channel sum: {np.sum(blue_channel)}")
        print(f"Red channel sum: {np.sum(red_channel)}")
        
        # Find piece positions
        blue_positions = list(zip(*np.where(blue_channel == 1.0)))
        red_positions = list(zip(*np.where(red_channel == 1.0)))
        
        print(f"Blue pieces: {blue_positions}")
        print(f"Red pieces: {red_positions}")
        
        # Show first few rows of each channel
        print(f"\nBlue channel (first 5x5):")
        print(blue_channel[:5, :5])
        print(f"\nRed channel (first 5x5):")
        print(red_channel[:5, :5])
        
    except Exception as e:
        print(f"Error examining sample: {e}")

def main():
    parser = argparse.ArgumentParser(description="Analyze raw data from .pkl.gz files")
    parser.add_argument('--data-dir', type=str, default='data/processed', 
                       help='Directory containing .pkl.gz files')
    parser.add_argument('--max-files', type=int, default=5, 
                       help='Maximum number of files to analyze')
    parser.add_argument('--max-samples', type=int, default=1000, 
                       help='Maximum samples per file to analyze')
    parser.add_argument('--examine-file', type=str, 
                       help='Examine a specific file in detail')
    parser.add_argument('--sample-index', type=int, default=0, 
                       help='Sample index to examine (with --examine-file)')
    parser.add_argument('--save-results', action='store_true', 
                       help='Save results to JSON file')
    
    args = parser.parse_args()
    
    if args.examine_file:
        examine_specific_sample(args.examine_file, args.sample_index)
        return
    
    # Analyze multiple files
    results = find_problematic_files(args.data_dir, args.max_files, args.max_samples)
    
    if args.save_results:
        output_file = Path("data_analysis_results.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main() 