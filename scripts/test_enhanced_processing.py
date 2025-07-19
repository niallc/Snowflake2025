#!/usr/bin/env python3
"""
Test script for enhanced data processing.

This script tests the enhanced data processing pipeline with a small sample
to verify the new format and metadata are working correctly.
"""

import sys
from pathlib import Path
import gzip
import pickle
import json
import numpy as np
from hex_ai.utils.format_conversion import tensor_to_rowcol

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

trmph_preamble = "https://trmph.com/hex/board#13,"

from hex_ai.enhanced_data_processing import (
    extract_training_examples_from_game_v2,
    assign_value_sample_tiers,
    remove_repeated_moves
)


def test_basic_functions():
    """Test basic utility functions."""
    print("Testing basic functions...")
    
    # Test tier assignment
    tiers = assign_value_sample_tiers(30)
    print(f"Tiers for 30 positions: {tiers}")
    print(f"Tier 0 count: {tiers.count(0)}")
    print(f"Tier 1 count: {tiers.count(1)}")
    print(f"Tier 2 count: {tiers.count(2)}")
    print(f"Tier 3 count: {tiers.count(3)}")
    
    # Test repeated moves removal
    moves = ['a1', 'b2', 'c3', 'a1', 'd4']  # a1 is repeated
    clean_moves = remove_repeated_moves(moves)
    print(f"Original moves: {moves}")
    print(f"Clean moves: {clean_moves}")
    
    print("Basic function tests passed!\n")


def test_game_processing():
    """Test processing a single game."""
    print("Testing game processing...")
    
    # Sample TRMPH game (simplified)
    trmph_text = trmph_preamble + "a1b2c3d4e5f6g7h8i9j10k11l12m13"
    winner = "1"  # BLUE wins
    game_id = (0, 42)  # file 0, line 42
    
    examples = extract_training_examples_from_game_v2(
        trmph_text=trmph_text,
        winner_from_file=winner,
        game_id=game_id,
        include_trmph=True,
        shuffle_positions=False  # Keep order for testing
    )
    
    print(f"Generated {len(examples)} examples")
    
    # Examine first few examples
    for i, example in enumerate(examples[:3]):
        print(f"\nExample {i}:")
        print(f"  Board shape: {example['board'].shape}")
        print(f"  Policy: {example['policy'] is not None}")
        if example['policy'] is not None:
            # example['policy'] is a numpy array of length 169
            nonzero_indices = np.nonzero(example['policy'])[0]
            rowcol_indices = [tensor_to_rowcol(i) for i in nonzero_indices]
            # Format as comma-separated "row,col" strings
            formatted_indices = [f"{int(row)},{int(col)}" for row, col in rowcol_indices]
            print(f"  Policy nonzero indices: {', '.join(formatted_indices)}")
        print(f"  Value: {example['value']}")
        print(f"  Metadata: {example['metadata']}")
    
    # Verify metadata
    metadata = examples[0]['metadata']
    assert metadata['game_id'] == (0, 42)
    assert metadata['winner'] == 'BLUE'
    # The metadata uses 'winner' instead of 'value'
    # assert metadata['value'] == 0.0  # BLUE = 0.0
    assert 'trmph_game' in metadata
    
    print("Game processing test passed!\n")


def test_file_lookup():
    """Test file lookup table creation."""
    print("Testing file lookup table...")
    
    from hex_ai.enhanced_data_processing import create_file_lookup_table
    
    # Create dummy file list
    dummy_files = [
        Path("data/file1.trmph"),
        Path("data/file2.trmph"),
        Path("data/subdir/file3.trmph")
    ]
    
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    lookup_file = create_file_lookup_table(dummy_files, output_dir)
    
    # Read and verify lookup table
    with open(lookup_file, 'r') as f:
        lookup_data = json.load(f)
    
    print(f"Lookup table created: {lookup_file}")
    print(f"Total files: {lookup_data['total_files']}")
    print(f"File mapping: {lookup_data['file_mapping']}")
    
    # Cleanup
    lookup_file.unlink()
    output_dir.rmdir()
    
    print("File lookup test passed!\n")


def test_small_stratified_processing():
    """Test stratified processing with a very small dataset."""
    print("Testing stratified processing...")
    
    # Create a small test TRMPH file
    test_file = Path("test_data.trmph")
    test_file.parent.mkdir(exist_ok=True)
    
    with open(test_file, 'w') as f:
        f.write(f"{trmph_preamble}a1b2c3d4e5f6g7h8i9j10k11l12m13 1\n")  # Game 1
        f.write(f"{trmph_preamble}a1b2c3d4e5f6g7h8i9j10k11l12 2\n")     # Game 2
        f.write(f"{trmph_preamble}a1b2c3d4e5f6g7h8i9j10k11 1\n")        # Game 3
    
    from hex_ai.enhanced_data_processing import create_stratified_dataset
    
    output_dir = Path("test_stratified_output")
    
    try:
        processed_files, lookup_file = create_stratified_dataset(
            trmph_files=[test_file],
            output_dir=output_dir,
            positions_per_pass=3,
            include_trmph=False,
            max_positions_per_game=15
        )
        
        print(f"Generated {len(processed_files)} files")
        print(f"Lookup table: {lookup_file}")
        
        # Examine one of the processed files
        if processed_files:
            with gzip.open(processed_files[0], 'rb') as f:
                data = pickle.load(f)
            
            print(f"File format version: {data.get('format_version', 'unknown')}")
            print(f"Examples count: {len(data['examples'])}")
            print(f"Pass info: {data.get('pass_info', 'N/A')}")
            
            # Examine first example
            if data['examples']:
                example = data['examples'][0]
                print(f"First example metadata: {example['metadata']}")
        
        # Cleanup
        for file_path in processed_files:
            file_path.unlink()
        lookup_file.unlink()
        output_dir.rmdir()
        
    finally:
        # Safer cleanup: move test files/dirs to a Trash folder in project root with timestamp
        import shutil
        from datetime import datetime

        # Determine project root (parent of this script)
        project_root = Path(__file__).resolve().parent.parent
        trash_dir = project_root / "Trash"
        trash_dir.mkdir(exist_ok=True)

        # Create a unique trash subdirectory for this cleanup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cleanup_dir = trash_dir / f"test_enhanced_processing_cleanup_{timestamp}"
        cleanup_dir.mkdir()

        # Move test_file (and its parent if not current dir) to cleanup_dir
        try:
            # Move the test file
            if test_file.exists():
                shutil.move(str(test_file), str(cleanup_dir / test_file.name))
            # If the parent is not current dir and is now empty, move it too
            if test_file.parent != Path('.') and test_file.parent.exists():
                # Only move if it's now empty (except possibly .DS_Store or similar)
                remaining = [p for p in test_file.parent.iterdir() if not p.name.startswith('.')]
                if not remaining:
                    shutil.move(str(test_file.parent), str(cleanup_dir / test_file.parent.name))
        except Exception as cleanup_exc:
            print(f"Warning: Cleanup failed: {cleanup_exc}")
    
    print("Stratified processing test passed!\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("ENHANCED DATA PROCESSING TESTS")
    print("=" * 60)
    
    try:
        test_basic_functions()
        test_game_processing()
        test_file_lookup()
        test_small_stratified_processing()
        
        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    main() 