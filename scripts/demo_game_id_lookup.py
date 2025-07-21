#!/usr/bin/env python3
"""
Demonstration script showing how to use game_id lookup functionality.

This script shows how to:
1. Look up filenames from game_ids using the processing state
2. Get complete file information from game_ids
3. Trace examples back to their source files
"""

import sys
import os
import gzip
import pickle
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

from hex_ai.data_utils import get_filename_from_game_id_using_state, get_file_info_from_game_id_using_state


def main():
    """Demonstrate game_id lookup functionality."""
    print("Game ID Lookup Demonstration")
    print("=" * 50)
    
    # Path to processing state file
    state_file = Path("data/processed/data/processing_state.json")
    
    if not state_file.exists():
        print(f"❌ Processing state file not found: {state_file}")
        print("Please run process_all_trmph.py first to generate the state file.")
        return 1
    
    print(f"✅ Found processing state file: {state_file}")
    
    # Example 1: Basic filename lookup
    print("\n1. Basic Filename Lookup")
    print("-" * 30)
    
    test_game_ids = [
        (0, 1),    # First file, first line
        (0, 100),  # First file, 100th line
        (1, 1),    # Second file, first line
        (5, 1),    # 6th file, first line
    ]
    
    for game_id in test_game_ids:
        try:
            filename = get_filename_from_game_id_using_state(game_id, state_file)
            print(f"  game_id {game_id} -> {filename}")
        except ValueError as e:
            print(f"  game_id {game_id} -> ERROR: {e}")
    
    # Example 2: Complete file information lookup
    print("\n2. Complete File Information Lookup")
    print("-" * 40)
    
    game_id = (0, 1)
    try:
        file_info = get_file_info_from_game_id_using_state(game_id, state_file)
        print(f"  game_id {game_id} -> File: {Path(file_info['file']).name}")
        print(f"    Output: {Path(file_info['output']).name}")
        print(f"    Completed: {file_info['completed_at']}")
        print(f"    Stats: {file_info['stats']['all_games']} games, {file_info['stats']['examples_generated']} examples")
    except ValueError as e:
        print(f"  game_id {game_id} -> ERROR: {e}")
    
    # Example 3: Trace examples from processed files
    print("\n3. Tracing Examples from Processed Files")
    print("-" * 40)
    
    processed_data_dir = Path("data/processed/data")
    processed_files = list(processed_data_dir.glob("*_processed.pkl.gz"))
    
    if processed_files:
        # Load first processed file
        sample_file = processed_files[0]
        print(f"  Loading: {sample_file.name}")
        
        try:
            with gzip.open(sample_file, 'rb') as f:
                data = pickle.load(f)
            
            examples = data['examples']
            print(f"  Found {len(examples)} examples")
            
            # Check first few examples for game_ids
            examples_with_game_id = 0
            for i, example in enumerate(examples[:10]):
                metadata = example.get('metadata', {})
                game_id = metadata.get('game_id')
                
                if game_id is not None:
                    examples_with_game_id += 1
                    try:
                        filename = get_filename_from_game_id_using_state(game_id, state_file)
                        print(f"    Example {i}: game_id {game_id} -> {filename}")
                    except ValueError as e:
                        print(f"    Example {i}: game_id {game_id} -> ERROR: {e}")
                else:
                    print(f"    Example {i}: No game_id (processed before fix)")
            
            if examples_with_game_id == 0:
                print("  ℹ️  No examples with game_id found (likely processed before the fix)")
                print("  ℹ️  New processing runs will include game_id values")
            
        except Exception as e:
            print(f"  ❌ Error loading {sample_file}: {e}")
    else:
        print("  ❌ No processed files found")
    
    # Example 4: Show how to use in practice
    print("\n4. Practical Usage Example")
    print("-" * 30)
    
    print("  # To look up a filename from a game_id:")
    print("  from hex_ai.data_utils import get_filename_from_game_id_using_state")
    print("  filename = get_filename_from_game_id_using_state((0, 1), Path('data/processed/data/processing_state.json'))")
    print("  print(f'game_id (0, 1) comes from: {filename}')")
    print()
    print("  # To get complete file information:")
    print("  from hex_ai.data_utils import get_file_info_from_game_id_using_state")
    print("  file_info = get_file_info_from_game_id_using_state((0, 1), Path('data/processed/data/processing_state.json'))")
    print("  print(f'File path: {file_info[\"file\"]}')")
    print("  print(f'Examples generated: {file_info[\"stats\"][\"examples_generated\"]}')")
    
    print("\n" + "=" * 50)
    print("✅ Demonstration complete!")
    return 0


if __name__ == "__main__":
    exit(main()) 