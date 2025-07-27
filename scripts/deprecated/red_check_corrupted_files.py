#!/usr/bin/env python3
"""
Check for corrupted data files and report them.
"""

import gzip
import pickle
from pathlib import Path
from hex_ai.data_pipeline import discover_processed_files

def check_corrupted_files():
    """Check all data files for corruption."""
    print("Checking for corrupted data files...")
    
    data_files = discover_processed_files("data/processed")
    corrupted_files = []
    
    for i, file_path in enumerate(data_files):
        if i % 10 == 0:
            print(f"Checking file {i+1}/{len(data_files)}: {file_path.name}")
        
        try:
            with gzip.open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Basic validation
            if 'examples' not in data:
                print(f"  ⚠️  {file_path.name}: No 'examples' key")
                corrupted_files.append((file_path, "No 'examples' key"))
                continue
            
            examples = data['examples']
            if len(examples) == 0:
                print(f"  ⚠️  {file_path.name}: Empty examples list")
                corrupted_files.append((file_path, "Empty examples list"))
                continue
            
            # Check first example structure
            first_example = examples[0]
            if len(first_example) != 3:
                print(f"  ⚠️  {file_path.name}: Invalid example structure")
                corrupted_files.append((file_path, "Invalid example structure"))
                continue
            
            print(f"  ✓ {file_path.name}: {len(examples):,} examples")
            
        except Exception as e:
            print(f"  ❌ {file_path.name}: {e}")
            corrupted_files.append((file_path, str(e)))
    
    print(f"\nSummary:")
    print(f"Total files: {len(data_files)}")
    print(f"Corrupted files: {len(corrupted_files)}")
    
    if corrupted_files:
        print(f"\nCorrupted files:")
        for file_path, error in corrupted_files:
            print(f"  {file_path.name}: {error}")
        
        print(f"\nRecommendation: Remove or re-process corrupted files:")
        for file_path, error in corrupted_files:
            print(f"  rm '{file_path}'")
    else:
        print("✓ No corrupted files found!")

if __name__ == "__main__":
    check_corrupted_files() 