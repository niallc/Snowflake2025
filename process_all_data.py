#!/usr/bin/env python3
"""
Process all .trmph files from data/twoNetGames/ into a complete processed dataset.

This script will:
1. Find all .trmph files in data/twoNetGames/
2. Process each file into sharded, compressed training data
3. Store everything in data/processed/ for future training
"""

import os
import time
from pathlib import Path
from hex_ai.data_processing import DataProcessor


def process_all_data():
    """Process all .trmph files into a complete dataset."""
    print("Starting complete data processing...")
    
    # Setup paths
    source_dir = Path("data/twoNetGames")
    processed_dir = Path("data/processed")
    
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory {source_dir} not found")
    
    # Create processed directory
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all .trmph files
    trmph_files = list(source_dir.glob("*.trmph"))
    
    if not trmph_files:
        raise FileNotFoundError(f"No .trmph files found in {source_dir}")
    
    print(f"Found {len(trmph_files)} .trmph files to process")
    
    # Initialize processor
    processor = DataProcessor(str(processed_dir))
    
    # Process all files
    total_shards = 0
    start_time = time.time()
    
    for i, trmph_file in enumerate(trmph_files, 1):
        print(f"\nProcessing file {i}/{len(trmph_files)}: {trmph_file.name}")
        
        try:
            shard_files = processor.process_file(
                trmph_file, 
                games_per_shard=1000, 
                compress=True
            )
            total_shards += len(shard_files)
            print(f"  Created {len(shard_files)} shards")
            
        except Exception as e:
            print(f"  Error processing {trmph_file.name}: {e}")
            continue
    
    processing_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("DATA PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total files processed: {len(trmph_files)}")
    print(f"Total shards created: {total_shards}")
    print(f"Processing time: {processing_time:.1f} seconds")
    print(f"Processed data location: {processed_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    process_all_data() 