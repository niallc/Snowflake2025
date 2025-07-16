#!/usr/bin/env python3
"""
Identify files with high error rates that should be excluded from training.
"""

import argparse
import gzip
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List

def analyze_file_error_rate(file_path: Path, max_samples: int = 1000) -> Dict:
    """Analyze error rate in a single file."""
    try:
        with gzip.open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        examples = data['examples']
        total_samples = min(len(examples), max_samples)
        
        invalid_count = 0
        
        for i, example in enumerate(examples[:max_samples]):
            board_state, policy_target, value_target = example
            
            blue_count = int(np.sum(board_state[0]))
            red_count = int(np.sum(board_state[1]))
            
            # Check validity
            is_valid = (blue_count == red_count) or (blue_count == red_count + 1)
            
            if not is_valid:
                invalid_count += 1
        
        error_rate = invalid_count / total_samples if total_samples > 0 else 0
        
        return {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'total_samples': len(examples),
            'analyzed_samples': total_samples,
            'invalid_samples': invalid_count,
            'error_rate': error_rate,
            'is_problematic': error_rate > 0.01  # 1% threshold
        }
        
    except Exception as e:
        return {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'error': str(e),
            'is_problematic': True
        }

def scan_all_files(data_dir: str, max_files: int = None, max_samples_per_file: int = 1000) -> Dict:
    """Scan all files in the data directory."""
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
    problematic_files = []
    
    for i, file_path in enumerate(data_files):
        print(f"Analyzing {i+1}/{len(data_files)}: {file_path.name}")
        result = analyze_file_error_rate(file_path, max_samples_per_file)
        results.append(result)
        
        if result.get('is_problematic', False):
            problematic_files.append(result)
    
    # Summary
    total_files = len(results)
    problematic_count = len(problematic_files)
    
    print(f"\n=== Summary ===")
    print(f"Total files analyzed: {total_files}")
    print(f"Problematic files: {problematic_count}")
    print(f"Problematic rate: {problematic_count/total_files:.2%}")
    
    if problematic_files:
        print(f"\nProblematic files:")
        for result in sorted(problematic_files, key=lambda x: x.get('error_rate', 0), reverse=True):
            if 'error' in result:
                print(f"  {result['file_name']}: ERROR - {result['error']}")
            else:
                print(f"  {result['file_name']}: {result['invalid_samples']}/{result['analyzed_samples']} ({result['error_rate']:.2%})")
    
    return {
        'results': results,
        'problematic_files': problematic_files,
        'total_files': total_files,
        'problematic_count': problematic_count
    }

def generate_exclusion_list(problematic_files: List[Dict], output_file: str = "exclude_files.txt"):
    """Generate a list of files to exclude from training."""
    with open(output_file, 'w') as f:
        for result in problematic_files:
            f.write(f"{result['file_name']}\n")
    
    print(f"\nExclusion list saved to {output_file}")
    print(f"Add this to your training script to exclude problematic files")

def main():
    parser = argparse.ArgumentParser(description="Identify files with high error rates")
    parser.add_argument('--data-dir', type=str, default='data/processed', 
                       help='Directory containing .pkl.gz files')
    parser.add_argument('--max-files', type=int, 
                       help='Maximum number of files to analyze')
    parser.add_argument('--max-samples', type=int, default=1000, 
                       help='Maximum samples per file to analyze')
    parser.add_argument('--generate-exclusion-list', action='store_true', 
                       help='Generate exclusion list file')
    parser.add_argument('--output-file', type=str, default='analysis/debugging/exclude_files.txt', 
                       help='Output file for exclusion list')
    
    args = parser.parse_args()
    
    results = scan_all_files(args.data_dir, args.max_files, args.max_samples)
    
    if args.generate_exclusion_list and results['problematic_files']:
        generate_exclusion_list(results['problematic_files'], args.output_file)

if __name__ == "__main__":
    main() 