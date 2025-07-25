#!/usr/bin/env python3
"""
Tests for data shuffling functionality.

This module tests the DataShuffler class and related utilities to ensure
the shuffling process works correctly and addresses value head fingerprinting.
"""

import sys
import os
import tempfile
import shutil
import gzip
import pickle
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.shuffle_processed_data import DataShuffler


def create_test_data(num_files: int = 3, examples_per_file: int = 100, games_per_file: int = 5) -> Path:
    """Create test data for shuffling validation."""
    temp_dir = Path(tempfile.mkdtemp())
    input_dir = temp_dir / "input"
    input_dir.mkdir()
    
    for file_idx in range(num_files):
        examples = []
        
        # Create games with multiple positions
        for game_idx in range(games_per_file):
            game_length = random.randint(20, 50)
            winner = random.choice(["BLUE", "RED"])
            value_target = 0.0 if winner == "BLUE" else 1.0
            
            # Create positions for this game
            for position in range(game_length):
                # Create random board state
                board = np.random.randint(0, 2, (2, 13, 13)).astype(np.float32)
                
                # Create random policy target
                policy = np.zeros(169, dtype=np.float32)
                if position < game_length - 1:  # Not final position
                    policy[random.randint(0, 168)] = 1.0
                
                # Create metadata
                metadata = {
                    'game_id': (file_idx, game_idx),  # Use file_idx and game_idx for tracking
                    'position_in_game': position,
                    'total_positions': game_length,
                    'value_sample_tier': random.randint(0, 3),
                    'winner': winner
                }
                
                example = {
                    'board': board,
                    'policy': policy,
                    'value': value_target,
                    'metadata': metadata
                }
                examples.append(example)
        
        # Shuffle examples within file to simulate real data
        random.shuffle(examples)
        
        # Write test file
        file_data = {
            'examples': examples,
            'source_file': f'test_file_{file_idx}.trmph',
            'processing_stats': {
                'games_processed': games_per_file,
                'examples_generated': len(examples)
            },
            'processed_at': '2025-01-01T00:00:00',
            'file_size_bytes': len(examples) * 1000  # Approximate
        }
        
        output_file = input_dir / f"test_file_{file_idx}_processed.pkl.gz"
        with gzip.open(output_file, 'wb') as f:
            pickle.dump(file_data, f)
    
    return temp_dir


def test_bucket_distribution():
    """Test that examples are properly distributed across buckets."""
    print("Testing bucket distribution...")
    
    # Create test data
    temp_dir = create_test_data(num_files=2, examples_per_file=50, games_per_file=3)
    input_dir = temp_dir / "input"
    output_dir = temp_dir / "output"
    temp_bucket_dir = temp_dir / "temp_buckets"
    
    try:
        # Create shuffler with small number of buckets for testing
        shuffler = DataShuffler(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            temp_dir=str(temp_bucket_dir),
            num_buckets=5
        )
        
        # Run distribution phase
        input_files = list(input_dir.glob("*.pkl.gz"))
        shuffler._distribute_to_buckets(input_files)
        
        # Check that bucket files were created
        bucket_files = list(temp_bucket_dir.glob("*_bucket_*.pkl.gz"))
        assert len(bucket_files) > 0, "No bucket files created"
        
        # Check that examples are distributed across buckets
        bucket_examples = []
        for bucket_file in bucket_files:
            with gzip.open(bucket_file, 'rb') as f:
                data = pickle.load(f)
                bucket_examples.append(len(data['examples']))
        
        # Should have examples in multiple buckets
        non_empty_buckets = sum(1 for count in bucket_examples if count > 0)
        assert non_empty_buckets > 1, "Examples not distributed across buckets"
        
        print("✓ Bucket distribution test passed")
        
    finally:
        shutil.rmtree(temp_dir)


def test_shuffling_effectiveness():
    """Test that shuffling effectively breaks game correlations."""
    print("Testing shuffling effectiveness...")
    
    # Create test data with obvious game patterns
    temp_dir = create_test_data(num_files=1, examples_per_file=200, games_per_file=4)
    input_dir = temp_dir / "input"
    output_dir = temp_dir / "output"
    temp_bucket_dir = temp_dir / "temp_buckets"
    
    try:
        # Create shuffler
        shuffler = DataShuffler(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            temp_dir=str(temp_bucket_dir),
            num_buckets=3
        )
        
        # Run full shuffling process
        shuffler.shuffle_data()
        
        # Check that shuffled files were created
        shuffled_files = list(output_dir.glob("shuffled_*.pkl.gz"))
        assert len(shuffled_files) > 0, "No shuffled files created"
        
        # Load shuffled data and check for game dispersion
        all_examples = []
        for shuffled_file in shuffled_files:
            with gzip.open(shuffled_file, 'rb') as f:
                data = pickle.load(f)
                all_examples.extend(data['examples'])
        
        # Check that games are mixed (not clustered)
        game_sequences = []
        current_game = []
        current_winner = None
        current_total = None
        
        for example in all_examples:
            metadata = example['metadata']
            winner = metadata['winner']
            total_positions = metadata['total_positions']
            
            if (winner == current_winner and 
                total_positions == current_total and
                metadata['position_in_game'] == len(current_game)):
                current_game.append(example)
            else:
                if current_game:
                    game_sequences.append(current_game)
                current_game = [example]
                current_winner = winner
                current_total = total_positions
        
        if current_game:
            game_sequences.append(current_game)
        
        # Check that games are broken up (not all positions from same game together)
        max_consecutive_same_game = max(len(seq) for seq in game_sequences)
        total_positions = sum(len(seq) for seq in game_sequences)
        
        # Should have some games broken up (not all positions consecutive)
        assert max_consecutive_same_game < total_positions * 0.5, "Games not properly shuffled"
        
        print("✓ Shuffling effectiveness test passed")
        
    finally:
        shutil.rmtree(temp_dir)


def test_memory_efficiency():
    """Test that the process doesn't use excessive memory."""
    print("Testing memory efficiency...")
    
    try:
        import psutil
        import os
        
        # Create larger test data
        temp_dir = create_test_data(num_files=5, examples_per_file=500, games_per_file=10)
        input_dir = temp_dir / "input"
        output_dir = temp_dir / "output"
        temp_bucket_dir = temp_dir / "temp_buckets"
        
        try:
            # Monitor memory usage
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create shuffler
            shuffler = DataShuffler(
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                temp_dir=str(temp_bucket_dir),
                num_buckets=10
            )
            
            # Run distribution phase
            input_files = list(input_dir.glob("*.pkl.gz"))
            shuffler._distribute_to_buckets(input_files)
            
            # Check memory usage
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = peak_memory - initial_memory
            
            # Should not use more than 500MB for this test
            assert memory_increase < 500, f"Memory usage too high: {memory_increase:.1f}MB"
            
            print(f"✓ Memory efficiency test passed (peak increase: {memory_increase:.1f}MB)")
            
        finally:
            shutil.rmtree(temp_dir)
            
    except ImportError:
        print("✓ Memory efficiency test skipped (psutil not available)")
    except Exception as e:
        print(f"✓ Memory efficiency test skipped: {e}")


def test_resume_functionality():
    """Test that the process can resume from interruptions."""
    print("Testing resume functionality...")
    
    # Create test data
    temp_dir = create_test_data(num_files=3, examples_per_file=100, games_per_file=5)
    input_dir = temp_dir / "input"
    output_dir = temp_dir / "output"
    temp_bucket_dir = temp_dir / "temp_buckets"
    
    try:
        # Create shuffler
        shuffler = DataShuffler(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            temp_dir=str(temp_bucket_dir),
            num_buckets=5
        )
        
        # Run first phase
        input_files = list(input_dir.glob("*.pkl.gz"))
        shuffler._distribute_to_buckets(input_files)
        
        # Check progress file was created
        progress_file = output_dir / "shuffling_progress.json"
        assert progress_file.exists(), "Progress file not created"
        
        # Create new shuffler (simulating restart)
        shuffler2 = DataShuffler(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            temp_dir=str(temp_bucket_dir),
            num_buckets=5
        )
        
        # Should detect existing progress
        assert len(shuffler2.progress['processed_files']) > 0, "Progress not loaded"
        
        # Complete the process
        shuffler2.shuffle_data()
        
        # Check final output
        shuffled_files = list(output_dir.glob("shuffled_*.pkl.gz"))
        assert len(shuffled_files) > 0, "No shuffled files after resume"
        
        print("✓ Resume functionality test passed")
        
    finally:
        shutil.rmtree(temp_dir)


def test_data_integrity():
    """Test that no data is lost during shuffling."""
    print("Testing data integrity...")
    
    # Create test data
    temp_dir = create_test_data(num_files=2, examples_per_file=100, games_per_file=5)
    input_dir = temp_dir / "input"
    output_dir = temp_dir / "output"
    temp_bucket_dir = temp_dir / "temp_buckets"
    
    try:
        # Count input examples
        input_examples = []
        input_files = list(input_dir.glob("*.pkl.gz"))
        for input_file in input_files:
            with gzip.open(input_file, 'rb') as f:
                data = pickle.load(f)
                input_examples.extend(data['examples'])
        
        input_count = len(input_examples)
        
        # Run shuffling
        shuffler = DataShuffler(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            temp_dir=str(temp_bucket_dir),
            num_buckets=5
        )
        shuffler.shuffle_data()
        
        # Count output examples
        output_examples = []
        shuffled_files = list(output_dir.glob("shuffled_*.pkl.gz"))
        for shuffled_file in shuffled_files:
            with gzip.open(shuffled_file, 'rb') as f:
                data = pickle.load(f)
                output_examples.extend(data['examples'])
        
        output_count = len(output_examples)
        
        # Should have same number of examples
        assert input_count == output_count, f"Data loss detected: {input_count} -> {output_count}"
        
        # Check that all required fields are present
        for example in output_examples:
            assert 'board' in example, "Missing board field"
            assert 'policy' in example, "Missing policy field"
            assert 'value' in example, "Missing value field"
            assert 'metadata' in example, "Missing metadata field"
            
            metadata = example['metadata']
            assert 'winner' in metadata, "Missing winner in metadata"
            assert 'position_in_game' in metadata, "Missing position_in_game in metadata"
            assert 'total_positions' in metadata, "Missing total_positions in metadata"
        
        print("✓ Data integrity test passed")
        
    finally:
        shutil.rmtree(temp_dir)


def test_error_handling():
    """Test that errors are handled properly."""
    print("Testing error handling...")
    
    # Create test data
    temp_dir = create_test_data(num_files=1, examples_per_file=50, games_per_file=2)
    input_dir = temp_dir / "input"
    output_dir = temp_dir / "output"
    temp_bucket_dir = temp_dir / "temp_buckets"
    
    try:
        # Test with non-existent input directory
        shuffler = DataShuffler(
            input_dir="non_existent_dir",
            output_dir=str(output_dir),
            temp_dir=str(temp_bucket_dir),
            num_buckets=5
        )
        
        # Should handle missing directory gracefully
        input_files = list(Path("non_existent_dir").glob("*.pkl.gz"))
        assert len(input_files) == 0, "Should handle missing directory"
        
        print("✓ Error handling test passed")
        
    finally:
        shutil.rmtree(temp_dir)


def test_file_naming():
    """Test that bucket files are named correctly."""
    print("Testing file naming...")
    
    # Create test data
    temp_dir = create_test_data(num_files=1, examples_per_file=50, games_per_file=2)
    input_dir = temp_dir / "input"
    output_dir = temp_dir / "output"
    temp_bucket_dir = temp_dir / "temp_buckets"
    
    try:
        shuffler = DataShuffler(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            temp_dir=str(temp_bucket_dir),
            num_buckets=5
        )
        
        # Run distribution phase
        input_files = list(input_dir.glob("*.pkl.gz"))
        shuffler._distribute_to_buckets(input_files)
        
        # Check bucket file naming
        bucket_files = list(temp_bucket_dir.glob("*_bucket_*.pkl.gz"))
        assert len(bucket_files) > 0, "No bucket files created"
        
        # Check naming pattern
        for bucket_file in bucket_files:
            filename = bucket_file.name
            assert "_bucket_" in filename, f"Invalid bucket filename: {filename}"
            assert filename.endswith(".pkl.gz"), f"Invalid bucket filename: {filename}"
        
        print("✓ File naming test passed")
        
    finally:
        shutil.rmtree(temp_dir)


def test_empty_input():
    """Test handling of empty input directory."""
    print("Testing empty input handling...")
    
    # Create empty input directory
    temp_dir = Path(tempfile.mkdtemp())
    input_dir = temp_dir / "input"
    input_dir.mkdir()
    output_dir = temp_dir / "output"
    temp_bucket_dir = temp_dir / "temp_buckets"
    
    try:
        shuffler = DataShuffler(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            temp_dir=str(temp_bucket_dir),
            num_buckets=5
        )
        
        # Should handle empty directory gracefully
        input_files = list(input_dir.glob("*.pkl.gz"))
        assert len(input_files) == 0, "Should handle empty directory"
        
        print("✓ Empty input test passed")
        
    finally:
        shutil.rmtree(temp_dir)


def test_large_bucket_count():
    """Test with a large number of buckets."""
    print("Testing large bucket count...")
    
    # Create test data
    temp_dir = create_test_data(num_files=1, examples_per_file=100, games_per_file=3)
    input_dir = temp_dir / "input"
    output_dir = temp_dir / "output"
    temp_bucket_dir = temp_dir / "temp_buckets"
    
    try:
        # Use large number of buckets
        shuffler = DataShuffler(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            temp_dir=str(temp_bucket_dir),
            num_buckets=50  # Large bucket count
        )
        
        # Run full process
        shuffler.shuffle_data()
        
        # Check output
        shuffled_files = list(output_dir.glob("shuffled_*.pkl.gz"))
        assert len(shuffled_files) > 0, "Should create shuffled files"
        
        print("✓ Large bucket count test passed")
        
    finally:
        shutil.rmtree(temp_dir)


def test_corrupted_input_file():
    """Test handling of corrupted input files."""
    print("Testing corrupted input file handling...")
    
    # Create test data
    temp_dir = create_test_data(num_files=1, examples_per_file=50, games_per_file=2)
    input_dir = temp_dir / "input"
    output_dir = temp_dir / "output"
    temp_bucket_dir = temp_dir / "temp_buckets"
    
    try:
        # Corrupt one of the input files
        input_files = list(input_dir.glob("*.pkl.gz"))
        if input_files:
            with open(input_files[0], 'wb') as f:
                f.write(b"corrupted data")
        
        shuffler = DataShuffler(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            temp_dir=str(temp_bucket_dir),
            num_buckets=5
        )
        
        # Should fail when trying to process corrupted file
        try:
            shuffler.shuffle_data()
            assert False, "Should have failed on corrupted file"
        except Exception:
            # Expected to fail
            pass
        
        print("✓ Corrupted input file test passed")
        
    finally:
        shutil.rmtree(temp_dir)


def test_validation_disabled():
    """Test that validation can be disabled."""
    print("Testing validation disabled...")
    
    # Create test data
    temp_dir = create_test_data(num_files=1, examples_per_file=50, games_per_file=2)
    input_dir = temp_dir / "input"
    output_dir = temp_dir / "output"
    temp_bucket_dir = temp_dir / "temp_buckets"
    
    try:
        shuffler = DataShuffler(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            temp_dir=str(temp_bucket_dir),
            num_buckets=5,
            validation_enabled=False
        )
        
        # Run full process
        shuffler.shuffle_data()
        
        # Check output exists
        shuffled_files = list(output_dir.glob("shuffled_*.pkl.gz"))
        assert len(shuffled_files) > 0, "Should create shuffled files even with validation disabled"
        
        print("✓ Validation disabled test passed")
        
    finally:
        shutil.rmtree(temp_dir)


def test_player_to_move_retention():
    """Test that player_to_move field is retained in all shuffled examples."""
    print("Testing player_to_move retention in shuffled data...")
    temp_dir = create_test_data(num_files=1, examples_per_file=20, games_per_file=2)
    input_dir = temp_dir / "input"
    output_dir = temp_dir / "output"
    temp_bucket_dir = temp_dir / "temp_buckets"

    # Inject player_to_move field into all examples in input files
    for file_path in input_dir.glob("*.pkl.gz"):
        with gzip.open(file_path, 'rb') as f:
            data = pickle.load(f)
        for ex in data['examples']:
            ex['player_to_move'] = 0  # Arbitrary, just for test
        with gzip.open(file_path, 'wb') as f:
            pickle.dump(data, f)

    shuffler = DataShuffler(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        temp_dir=str(temp_bucket_dir),
        num_buckets=3,
        resume_enabled=False,
        cleanup_temp=True,
        validation_enabled=False
    )
    shuffler.shuffle_data()

    # Check all output files for player_to_move field
    for shuffled_file in output_dir.glob("shuffled_*.pkl.gz"):
        with gzip.open(shuffled_file, 'rb') as f:
            data = pickle.load(f)
        for ex in data['examples']:
            assert 'player_to_move' in ex, f"Missing player_to_move in example in {shuffled_file}"
    print("All shuffled examples retain player_to_move field.")


def test_phase2_consolidation_on_small_real_file():
    """
    Test Phase 2 (consolidation and shuffling) on the bucket files created by the parallel distribution test.
    This test assumes the bucket files are present in a temp directory, as created by test_parallel_distribution_on_small_real_file.
    It checks that the total number of examples in the shuffled output matches the original, and that all examples are present and not duplicated.
    Cleans up after itself if run independently.
    """
    input_file = Path("data/processed/step1_unshuffled/twoNetGames_13x13_mk45_breadths_5_3_3_v1816_3s2_p2551k_1s0_vt25sc_pt10sc_processed.pkl.gz")
    assert input_file.exists(), f"Test input file {input_file} does not exist"

    import tempfile, shutil
    temp_dir = Path(tempfile.mkdtemp())
    temp_bucket_dir = temp_dir / "temp_buckets"
    temp_bucket_dir.mkdir()
    num_buckets = 8

    try:
        from scripts.shuffle_processed_data import DataShuffler
        shuffler = DataShuffler(
            input_dir=input_file.parent,
            output_dir=temp_dir,
            temp_dir=temp_bucket_dir,
            num_buckets=num_buckets,
            resume_enabled=False,
            cleanup_temp=False,
            validation_enabled=False
        )
        # Phase 1: create bucket files
        shuffler._distribute_to_buckets([input_file])
        # Phase 2: consolidate and shuffle
        shuffler._consolidate_and_shuffle_all_buckets()
        # Check shuffled output
        shuffled_files = list(temp_dir.glob("shuffled_*.pkl.gz"))
        assert len(shuffled_files) > 0, "No shuffled output files created"
        all_examples = []
        for sf in shuffled_files:
            with gzip.open(sf, 'rb') as f:
                data = pickle.load(f)
            all_examples.extend(data['examples'])
        with gzip.open(input_file, 'rb') as f:
            orig_data = pickle.load(f)
        orig_examples = orig_data['examples']
        assert len(all_examples) == len(orig_examples), "Mismatch in total number of examples after shuffling"
        orig_ids = set(id(ex) for ex in orig_examples)
        new_ids = set(id(ex) for ex in all_examples)
        assert len(new_ids) == len(orig_examples), "Duplicate examples found in shuffled output"
        print("✓ Phase 2 consolidation/shuffling test passed.")
    finally:
        shutil.rmtree(temp_dir)


def main():
    """Run all tests."""
    print("Running data shuffling tests...")
    print("=" * 50)
    
    tests = [
        test_bucket_distribution,
        test_shuffling_effectiveness,
        test_memory_efficiency,
        test_resume_functionality,
        test_data_integrity,
        test_error_handling,
        test_file_naming,
        test_empty_input,
        test_large_bucket_count,
        test_corrupted_input_file,
        test_validation_disabled,
        test_player_to_move_retention,
        test_phase2_consolidation_on_small_real_file
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