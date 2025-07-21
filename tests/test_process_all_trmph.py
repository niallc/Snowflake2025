#!/usr/bin/env python3
"""
Tests for process_all_trmph.py to verify game_id population and file lookup functionality.
"""

import sys
import os
import tempfile
import shutil
import gzip
import pickle
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hex_ai.data_utils import create_file_lookup_table, get_filename_from_game_id_using_state, get_file_info_from_game_id_using_state


def create_test_trmph_files(temp_dir: Path, num_files: int = 3) -> List[Path]:
    """Create test TRMPH files for processing."""
    trmph_files = []
    
    for i in range(num_files):
        trmph_file = temp_dir / f"test_file_{i}.trmph"
        with open(trmph_file, 'w') as f:
            # Write a few simple game records with proper TRMPH format
            f.write("http://www.trmph.com/hex/board#13,a1b2c3 1\n")
            f.write("http://www.trmph.com/hex/board#13,a1b2c3d4e5f6g7h8i9j10k11l12m13 2\n")
            f.write("http://www.trmph.com/hex/board#13,a1b2c3d4e5f6g7h8i9j10k11l12 1\n")
        
        trmph_files.append(trmph_file)
    
    return trmph_files


def test_game_id_population():
    """Test that process_all_trmph.py properly writes game_id values."""
    print("Testing game_id population in process_all_trmph.py...")
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    data_dir = temp_dir / "data"
    data_dir.mkdir()
    output_dir = temp_dir / "output"
    
    try:
        # Create test TRMPH files
        trmph_files = create_test_trmph_files(data_dir, num_files=3)
        print(f"Created {len(trmph_files)} test TRMPH files")
        
        # Run process_all_trmph.py
        cmd = [
            sys.executable, "scripts/process_all_trmph.py",
            "--data-dir", str(data_dir),
            "--output-dir", str(output_dir),
            "--max-files", "3"  # Limit to 3 files for testing
        ]
        
        # Set PYTHONPATH to include the project root
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path.cwd())
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd(), env=env)
        
        if result.returncode != 0:
            print(f"Command failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            assert False, "process_all_trmph.py failed"
        
        print("✓ process_all_trmph.py completed successfully")
        
        # Check that output files were created
        processed_files = list(output_dir.glob("*_processed.pkl.gz"))
        assert len(processed_files) > 0, f"No processed files found in {output_dir}"
        print(f"✓ Found {len(processed_files)} processed files")
        
        # Check that game_id is populated in all examples
        all_game_ids = set()
        file_game_ids = {}  # Track which game_ids come from which files
        
        for processed_file in processed_files:
            with gzip.open(processed_file, 'rb') as f:
                data = pickle.load(f)
            
            examples = data['examples']
            print(f"  File {processed_file.name}: {len(examples)} examples")
            
            for example in examples:
                metadata = example['metadata']
                game_id = metadata.get('game_id')
                
                # Check that game_id is populated
                assert game_id is not None, f"game_id is None for example: {metadata}"
                assert isinstance(game_id, tuple), f"game_id should be tuple, got {type(game_id)}: {game_id}"
                assert len(game_id) == 2, f"game_id should have 2 elements, got {len(game_id)}: {game_id}"
                
                all_game_ids.add(game_id)
                
                # Track which file this game_id comes from
                file_idx = game_id[0]
                if file_idx not in file_game_ids:
                    file_game_ids[file_idx] = []
                file_game_ids[file_idx].append(game_id)
        
        print(f"✓ Found {len(all_game_ids)} unique game_ids: {sorted(all_game_ids)}")
        
        # Verify we have game_ids from all expected files (0, 1, 2)
        expected_file_indices = {0, 1, 2}
        actual_file_indices = set(file_game_ids.keys())
        assert actual_file_indices == expected_file_indices, f"Expected file indices {expected_file_indices}, got {actual_file_indices}"
        
        print("✓ game_id population test passed!")
        
    finally:
        shutil.rmtree(temp_dir)


def test_file_lookup_table():
    """Test file lookup table creation and lookup functionality."""
    print("Testing file lookup table functionality...")
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    data_dir = temp_dir / "data"
    data_dir.mkdir()
    output_dir = temp_dir / "output"
    
    try:
        # Create test TRMPH files
        trmph_files = create_test_trmph_files(data_dir, num_files=3)
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create file lookup table
        lookup_file = create_file_lookup_table(trmph_files, output_dir)
        assert lookup_file.exists(), f"Lookup file not created: {lookup_file}"
        
        # Load lookup table
        with open(lookup_file, 'r') as f:
            lookup_data = json.load(f)
        
        # Verify lookup table structure
        assert 'file_mapping' in lookup_data, "Missing file_mapping in lookup data"
        assert 'created_at' in lookup_data, "Missing created_at in lookup data"
        assert 'total_files' in lookup_data, "Missing total_files in lookup data"
        assert lookup_data['total_files'] == 3, f"Expected 3 files, got {lookup_data['total_files']}"
        
        file_mapping = lookup_data['file_mapping']
        assert len(file_mapping) == 3, f"Expected 3 mappings, got {len(file_mapping)}"
        
        # Verify mappings
        for file_idx in range(3):
            assert str(file_idx) in file_mapping, f"Missing mapping for file_idx {file_idx}"
            expected_filename = f"test_file_{file_idx}.trmph"
            actual_path = file_mapping[str(file_idx)]
            assert actual_path.endswith(expected_filename), f"Expected {expected_filename}, got {actual_path}"
        
        print("✓ File lookup table creation test passed!")
        
    finally:
        shutil.rmtree(temp_dir)


def test_game_id_to_filename_lookup():
    """Test that we can recover the correct filename from a game_id."""
    print("Testing game_id to filename lookup...")
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    data_dir = temp_dir / "data"
    data_dir.mkdir()
    output_dir = temp_dir / "output"
    
    try:
        # Create test TRMPH files
        trmph_files = create_test_trmph_files(data_dir, num_files=3)
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create file lookup table
        lookup_file = create_file_lookup_table(trmph_files, output_dir)
        
        # Load lookup table
        with open(lookup_file, 'r') as f:
            lookup_data = json.load(f)
        
        file_mapping = lookup_data['file_mapping']
        
        # Test lookup function
        def get_filename_from_game_id(game_id: tuple, file_mapping: dict) -> str:
            """Get filename from game_id using file mapping."""
            file_idx, line_idx = game_id
            file_path = file_mapping[str(file_idx)]
            return Path(file_path).name
        
        # Test some game_ids
        test_cases = [
            ((0, 1), "test_file_0.trmph"),
            ((1, 2), "test_file_1.trmph"),
            ((2, 3), "test_file_2.trmph"),
        ]
        
        for game_id, expected_filename in test_cases:
            actual_filename = get_filename_from_game_id(game_id, file_mapping)
            assert actual_filename == expected_filename, f"Expected {expected_filename}, got {actual_filename} for game_id {game_id}"
            print(f"  ✓ game_id {game_id} -> {actual_filename}")
        
        print("✓ game_id to filename lookup test passed!")
        
    finally:
        shutil.rmtree(temp_dir)


def test_integrated_workflow():
    """Test the complete workflow: processing + lookup table + filename recovery."""
    print("Testing integrated workflow...")
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    data_dir = temp_dir / "data"
    data_dir.mkdir()
    output_dir = temp_dir / "output"
    
    try:
        # Create test TRMPH files
        trmph_files = create_test_trmph_files(data_dir, num_files=2)
        
        # Run process_all_trmph.py
        cmd = [
            sys.executable, "scripts/process_all_trmph.py",
            "--data-dir", str(data_dir),
            "--output-dir", str(output_dir),
            "--max-files", "2"
        ]
        
        # Set PYTHONPATH to include the project root
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path.cwd())
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd(), env=env)
        assert result.returncode == 0, f"process_all_trmph.py failed: {result.stderr}"
        
        # Create file lookup table
        lookup_file = create_file_lookup_table(trmph_files, output_dir)
        
        # Load lookup table
        with open(lookup_file, 'r') as f:
            lookup_data = json.load(f)
        
        file_mapping = lookup_data['file_mapping']
        
        # Load processed data and verify game_id -> filename mapping
        processed_files = list(output_dir.glob("*_processed.pkl.gz"))
        assert len(processed_files) > 0, "No processed files found"
        
        for processed_file in processed_files:
            with gzip.open(processed_file, 'rb') as f:
                data = pickle.load(f)
            
            examples = data['examples']
            
            for example in examples:
                metadata = example['metadata']
                game_id = metadata['game_id']
                
                # Recover filename from game_id
                file_idx, line_idx = game_id
                file_path = file_mapping[str(file_idx)]
                filename = Path(file_path).name
                
                # Verify the filename makes sense
                assert filename.startswith("test_file_"), f"Unexpected filename: {filename}"
                assert file_idx in [0, 1], f"Unexpected file_idx: {file_idx}"
                
                print(f"  ✓ game_id {game_id} -> {filename}")
        
        print("✓ Integrated workflow test passed!")
        
    finally:
        shutil.rmtree(temp_dir)


def test_real_processing_state_lookup():
    """Test that we can use the actual processing_state.json to lookup filenames from game_ids."""
    print("Testing real processing state lookup...")
    
    # Check if processing state file exists
    state_file = Path("data/processed/data/processing_state.json")
    if not state_file.exists():
        print("  ⚠️  No processing_state.json found - skipping real data test")
        return
    
    try:
        # Load processing state
        with open(state_file, 'r') as f:
            state_data = json.load(f)
        
        # Extract processed files array
        processed_files = state_data.get('processed_files', [])
        if not processed_files:
            print("  ⚠️  No processed files in state - skipping real data test")
            return
        
        print(f"  Found {len(processed_files)} processed files in state")
        
        # Use the utility function from data_utils
        state_file_path = Path("data/processed/data/processing_state.json")
        
        # Test with some sample game_ids
        test_cases = [
            (0, 1),  # First file, first line
            (0, 100),  # First file, 100th line
            (1, 1),   # Second file, first line
            (min(5, len(processed_files)-1), 1),  # 5th file (or last if fewer files)
        ]
        
        for file_idx, line_idx in test_cases:
            if file_idx < len(processed_files):
                game_id = (file_idx, line_idx)
                filename = get_filename_from_game_id_using_state(game_id, state_file_path)
                expected_file_info = processed_files[file_idx]
                expected_filename = Path(expected_file_info['file']).name
                
                assert filename == expected_filename, f"Expected {expected_filename}, got {filename} for game_id {game_id}"
                print(f"  ✓ game_id {game_id} -> {filename}")
        
        # Test with a real processed file if available
        processed_data_dir = Path("data/processed/data")
        processed_pkl_files = list(processed_data_dir.glob("*_processed.pkl.gz"))
        
        if processed_pkl_files:
            # Load one processed file and check game_ids
            sample_file = processed_pkl_files[0]
            print(f"  Testing with real processed file: {sample_file.name}")
            
            with gzip.open(sample_file, 'rb') as f:
                data = pickle.load(f)
            
            examples = data['examples']
            if examples:
                # Check first few examples
                for i, example in enumerate(examples[:5]):
                    metadata = example['metadata']
                    game_id = metadata.get('game_id')
                    
                    if game_id is not None:
                        file_idx, line_idx = game_id
                        filename = get_filename_from_game_id_using_state(game_id, state_file_path)
                        print(f"    ✓ Example {i}: game_id {game_id} -> {filename}")
                    else:
                        print(f"    ⚠️  Example {i}: No game_id found")
        
        print("✓ Real processing state lookup test passed!")
        
    except Exception as e:
        print(f"  ✗ Real processing state lookup test failed: {e}")
        raise


def main():
    """Run all tests."""
    print("Running process_all_trmph.py tests...")
    print("=" * 50)
    
    tests = [
        test_game_id_population,
        test_file_lookup_table,
        test_game_id_to_filename_lookup,
        test_integrated_workflow,
        test_real_processing_state_lookup
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
    
    print("=" * 50)
    print(f"Tests completed: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("All tests passed! ✓")
        return 0
    else:
        print("Some tests failed! ✗")
        return 1


if __name__ == "__main__":
    exit(main()) 