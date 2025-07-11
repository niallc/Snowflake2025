#!/usr/bin/env python3
"""
Test script to compare processing speeds.
"""

import time
from pathlib import Path
from fast_data_processing import process_file_fast
from hex_ai.data_processing import DataProcessor

def test_processing_speeds():
    """Compare original vs fast processing on a small subset."""
    print("Testing processing speeds...")
    
    # Find a small .trmph file for testing
    source_dir = Path("data/twoNetGames")
    trmph_files = list(source_dir.glob("*.trmph"))
    
    if not trmph_files:
        print("No .trmph files found!")
        return
    
    # Use a medium-sized file for testing (not the smallest, not the largest)
    sorted_files = sorted(trmph_files, key=lambda f: f.stat().st_size)
    test_file = sorted_files[len(sorted_files) // 4]  # 25th percentile
    print(f"Testing with file: {test_file.name} ({test_file.stat().st_size / 1024:.1f} KB)")
    
    # Test original processing
    print("\n" + "="*50)
    print("TESTING ORIGINAL PROCESSING")
    print("="*50)
    
    start_time = time.time()
    processor = DataProcessor("data/processed_test_original")
    original_shards = processor.process_file(test_file, games_per_shard=100, compress=True)
    original_time = time.time() - start_time
    
    print(f"Original processing time: {original_time:.2f} seconds")
    print(f"Shards created: {len(original_shards)}")
    
    # Test fast processing
    print("\n" + "="*50)
    print("TESTING FAST PROCESSING")
    print("="*50)
    
    start_time = time.time()
    fast_shards = process_file_fast(test_file, Path("data/processed_test_fast"), 
                                   games_per_shard=100, compress=True, num_workers=4)
    fast_time = time.time() - start_time
    
    print(f"Fast processing time: {fast_time:.2f} seconds")
    print(f"Shards created: {len(fast_shards)}")
    
    # Compare results
    print("\n" + "="*50)
    print("SPEED COMPARISON")
    print("="*50)
    
    if original_time > 0:
        speedup = original_time / fast_time
        print(f"Speedup: {speedup:.2f}x faster")
        print(f"Time saved: {original_time - fast_time:.2f} seconds")
    
    # Cleanup test directories
    import shutil
    shutil.rmtree("data/processed_test_original", ignore_errors=True)
    shutil.rmtree("data/processed_test_fast", ignore_errors=True)

if __name__ == "__main__":
    test_processing_speeds() 